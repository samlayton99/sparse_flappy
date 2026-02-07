"""Training script with wandb logging and DDP support.

Usage:
    # Train with config file
    torchrun --nproc_per_node=8 train.py --config hyperparameters/base_model.py --run_name my_run
    
    # Train with config file and save weights
    torchrun --nproc_per_node=8 train.py --config hyperparameters/moe_model.py --run_name moe_run --save
"""

import argparse
import importlib.util
import os
import signal
import time
import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from tqdm import tqdm

# Global flag for graceful shutdown (Ctrl+C saves checkpoint)
_shutdown_requested = False

# Peak TFLOPS for MFU calculation (BF16 tensor core ops)
# H100 SXM: 990 TFLOPS BF16, A100: 312 TFLOPS BF16
GPU_PEAK_TFLOPS = 990

from data import get_dataloader
from model import create_model


def load_config(config_path: str) -> dict:
    """Load configuration from a Python file.
    
    Args:
        config_path: Path to the config .py file
        
    Returns:
        CONFIG dict from the file
    """
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if not hasattr(config_module, "CONFIG"):
        raise ValueError(f"Config file {config_path} must define a CONFIG dict")
    
    return config_module.CONFIG


def setup_ddp():
    """Initialize DDP. Returns local_rank."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown with checkpoint save."""
    global _shutdown_requested
    if is_main():
        print("\n⚠️  Interrupt received, will save checkpoint and exit after current step...")
    _shutdown_requested = True


PAD_TOKEN_ID = 50256  # GPT-2 <|endoftext|> used as pad token


def compute_loss(model, batch, device):
    """Compute cross-entropy loss for a batch using BF16 autocast.
    
    Also adds MoE auxiliary loss if the model supports it.
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids, attention_mask)
        
        # Flatten for cross-entropy
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=PAD_TOKEN_ID,
        )
        
        # Add MoE auxiliary loss if model supports it
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        if hasattr(raw_model, "get_aux_loss"):
            aux_loss = raw_model.get_aux_loss()
            loss = loss + aux_loss
    
    return loss


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup then linear decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    return max_lr - (max_lr - min_lr) * decay_ratio


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 20):
    """Evaluate model on validation set (lightweight, uses BF16).

    Returns a dict with keys:
        loss      – cross-entropy (uses clustered-decode logits when block_decode is on)
        accuracy  – top-1 token prediction accuracy (ignoring pad)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )

        total_loss += loss.item()

        # Top-1 accuracy (ignoring pad positions)
        preds = logits.argmax(dim=-1)                   # (B, T)
        mask = labels != PAD_TOKEN_ID
        total_correct += ((preds == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

        n_batches += 1
        if n_batches >= max_batches:
            break

    model.train()

    if n_batches == 0:
        return {"loss": float("inf"), "accuracy": 0.0}

    avg_loss = total_loss / n_batches
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_steps: int = 1000,
    eval_every: int = 50,
    max_lr: float = 1e-3,
    warmup_steps: int = 200,
):
    """Main training loop with logging every eval_every steps."""
    model.train()
    raw_model = model.module if hasattr(model, "module") else model
    # Unwrap torch.compile if present
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    num_params = raw_model.count_parameters(count_zeros=True)
    min_lr = max_lr * 0.1  # decay to 10% of peak
    
    # Check if model has sparsity features (SparseDyadicTransformer)
    has_sparse_attn = hasattr(raw_model, 'l1_lambda') and hasattr(raw_model, 'get_sparse_params')
    has_sparse_ff = hasattr(raw_model, 'ff_l1_lambda') and hasattr(raw_model, 'get_ff_params')
    # MoETransformer: l1_lambda targets FFN (not attention), uses get_ff_params
    has_moe_l1 = (hasattr(raw_model, 'l1_lambda') and hasattr(raw_model, 'get_ff_params')
                  and not hasattr(raw_model, 'get_sparse_params'))
    has_pruning = hasattr(raw_model, 'prune_every_n') and hasattr(raw_model, 'magnitude_prune')
    
    train_iter = iter(train_loader)
    running_loss = 0.0
    total_tokens = 0
    epoch = 0
    t0 = time.time()
    
    pbar = tqdm(range(num_steps), desc="Training", disable=not is_main())
    for step in pbar:
        # Check for graceful shutdown request (Ctrl+C)
        if _shutdown_requested:
            if is_main():
                print(f"Stopping training at step {step} due to interrupt...")
            break
        
        # Update learning rate (ratio-based for multi-group support)
        lr = get_lr(step, warmup_steps, num_steps, max_lr, min_lr)
        lr_scale = lr / max_lr if max_lr > 0 else 1.0
        for param_group in optimizer.param_groups:
            base_lr = param_group.get("initial_lr", max_lr)
            param_group["lr"] = base_lr * lr_scale
        
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            epoch += 1
        
        B, T = batch["input_ids"].shape
        tokens_this_step = B * T * dist.get_world_size()
        total_tokens += tokens_this_step
        
        optimizer.zero_grad()
        loss = compute_loss(model, batch, device)
        
        # L1 regularization on attention sparse params (M, V) — mutually exclusive
        if has_sparse_attn and raw_model.l1_lambda > 0:
            l1_penalty = sum(p.abs().sum() for p in raw_model.get_sparse_params())
            loss = loss + raw_model.l1_lambda * l1_penalty
        
        # L1 regularization on FFN weights (w1, w2, w3) — SparseDyadic variant
        if has_sparse_ff and raw_model.ff_l1_lambda > 0:
            ff_l1_penalty = sum(p.abs().sum() for p in raw_model.get_ff_params())
            loss = loss + raw_model.ff_l1_lambda * ff_l1_penalty
        
        # L1 regularization on MoE expert FFN weights
        if has_moe_l1 and raw_model.l1_lambda > 0:
            moe_l1_penalty = sum(p.abs().sum() for p in raw_model.get_ff_params())
            loss = loss + raw_model.l1_lambda * moe_l1_penalty
        
        loss.backward()
        optimizer.step()
        
        # Magnitude pruning on M, V (zeros entries + clears optimizer momentum)
        if has_pruning and raw_model.prune_every_n > 0:
            if (step + 1) % raw_model.prune_every_n == 0:
                raw_model.magnitude_prune(optimizer=optimizer)
        
        loss_val = loss.item()
        running_loss += loss_val
        
        # Log per-step training loss (every step for smooth curves)
        if is_main():
            wandb.log({
                "train/loss_step": loss_val,
                "train/lr_step": lr,
            }, step=step)
        
        if (step + 1) % eval_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            # MFU: 6*params*tokens / (peak_flops * time)
            # Uses BF16 peak TFLOPS since we run with autocast(bf16)
            tokens_interval = tokens_this_step * eval_every
            mfu = 6 * num_params * tokens_interval / (GPU_PEAK_TFLOPS * 1e12 * dt * dist.get_world_size())
            
            # Total FLOPs: 6 * params * tokens (forward + backward)
            total_flops = 6 * num_params * total_tokens
            
            train_loss = running_loss / eval_every
            val_metrics = evaluate(model, val_loader, device)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            nonzero_params = raw_model.count_parameters(count_zeros=False)
            
            # Calculate additional metrics
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            val_ppl = torch.exp(torch.tensor(val_loss)).item() if val_loss != float('inf') else float('inf')
            tokens_per_sec = tokens_interval / dt
            
            if is_main():
                postfix = {
                    "train": f"{train_loss:.3f}", 
                    "val": f"{val_loss:.3f}",
                    "acc": f"{val_acc:.3f}",
                    "ppl": f"{val_ppl:.1f}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                }
                # Add sparsity to progress bar
                if hasattr(raw_model, "get_sparsity_stats"):
                    sp = raw_model.get_sparsity_stats()
                    # Support both attn_density (SparseDyadic) and ff_density (MoE) keys
                    density = sp.get("attn_density", sp.get("ff_density", None))
                    if density is not None:
                        postfix["density"] = f"{density:.3f}"
                pbar.set_postfix(postfix)
                
                log_dict = {
                    # Training metrics
                    "train/loss": train_loss,
                    "train/perplexity": train_ppl,
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/total_tokens": total_tokens,
                    "train/total_flops": total_flops,
                    "train/epoch": epoch,
                    # Validation metrics
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/perplexity": val_ppl,
                    # Model stats
                    "params/total": num_params,
                    "params/nonzero": nonzero_params,
                    "params/sparsity": 1.0 - (nonzero_params / num_params),
                    # Performance
                    "perf/mfu": mfu,
                    "perf/tokens_per_sec": tokens_per_sec,
                    "step": step + 1,
                }
                
                # Log MoE expert utilization stats if model supports it
                if hasattr(raw_model, "get_expert_stats"):
                    expert_stats = raw_model.get_expert_stats()
                    if expert_stats:
                        log_dict["moe/expert_util_min"] = expert_stats["min"]
                        log_dict["moe/expert_util_max"] = expert_stats["max"]
                        log_dict["moe/expert_util_std"] = expert_stats["std"]
                        # Log per-expert utilization
                        if "mean_utilization" in expert_stats:
                            for i, util in enumerate(expert_stats["mean_utilization"].tolist()):
                                log_dict[f"moe/expert_{i}_util"] = util
                
                # Log MoH head utilization stats if model supports it
                if hasattr(raw_model, "get_head_stats"):
                    head_stats = raw_model.get_head_stats()
                    if head_stats:
                        log_dict["moh/head_util_min"] = head_stats["min"]
                        log_dict["moh/head_util_max"] = head_stats["max"]
                        log_dict["moh/head_util_std"] = head_stats["std"]
                        # Log per-head utilization
                        if "mean_utilization" in head_stats:
                            for i, util in enumerate(head_stats["mean_utilization"].tolist()):
                                log_dict[f"moh/head_{i}_util"] = util
                
                # Log sparsity stats (works for both SparseDyadic and MoE)
                if hasattr(raw_model, "get_sparsity_stats"):
                    sp_stats = raw_model.get_sparsity_stats()
                    for k, v in sp_stats.items():
                        if isinstance(v, (int, float)):
                            log_dict[f"sparse/{k}"] = v
                
                wandb.log(log_dict)
                
                if val_loss < 3.0:
                    print(f"\n🎉 Goal achieved! val_loss={val_loss:.4f} < 3.0 (acc={val_acc:.3f}) with {nonzero_params:,} non-zero params")
            
            running_loss = 0.0
            t0 = time.time()
    
    return model, step


def main():
    parser = argparse.ArgumentParser(description="Train transformer models with config files")
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config .py file (e.g., hyperparameters/base_model.py)")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name for this run (used for wandb and checkpoint naming)")
    
    # Optional: save weights
    parser.add_argument("--save", action="store_true", default=False,
                        help="Save model weights at end/interrupt (default: False)")
    
    # Optional: override config values
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_lr", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default="weightless")
    
    args = parser.parse_args()
    
    # Load config from file
    config = load_config(args.config)
    if is_main():
        print(f"Loaded config from: {args.config}")
        print(f"Model class: {config.get('model_class', 'SimpleTransformer')}")
    
    # Override config with any CLI arguments that were explicitly set
    cli_overrides = {
        "batch_size": args.batch_size,
        "max_lr": args.max_lr,
        "num_steps": args.num_steps,
        "eval_every": args.eval_every,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value
            if is_main():
                print(f"  Override: {key}={value}")
    
    # Autoscale max_lr: config max_lr is at d_model=2048, scale by sqrt(2048/d_model)
    base_lr = config["max_lr"]
    d_model = config["d_model"]
    config["max_lr"] = base_lr * (2048 / d_model) ** 0.5
    if is_main():
        print(f"Autoscaled max_lr from {base_lr} to {config['max_lr']:.2e} (d_model={d_model})")

    # Enable flash attention and bf16 optimizations
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    
    # DDP setup
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize wandb (only on main)
    if is_main():
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=config,
        )
    
    # Data (each rank gets different data via streaming)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if is_main():
        print("Setting up data loaders...")
    train_loader = get_dataloader(split="train", batch_size=config["batch_size"], streaming=True, rank=rank, world_size=world_size)
    val_loader = get_dataloader(split="test", batch_size=config["batch_size"], streaming=True, rank=rank, world_size=world_size)
    
    # Build model kwargs from config (filter to model-specific params)
    model_class = config.get("model_class", "SimpleTransformer")
    model_kwargs = {
        "d_model": config["d_model"],
        "n_layers": config["n_layers"],
        "n_heads": config["n_heads"],
        "dropout": config.get("dropout", 0.1),
        "block_decode": config.get("block_decode", False),
        "head_k": config.get("head_k", 1),
        "head_balance_tolerance": config.get("head_balance_tolerance", 10),
    }
    
    # Add model-specific parameters
    if model_class == "SimpleTransformer":
        model_kwargs["d_ff"] = config.get("d_ff", config["d_model"] * 4)
    elif model_class == "MoETransformer":
        model_kwargs["d_expert"] = config.get("d_expert", config["d_model"])
        model_kwargs["num_experts"] = config.get("num_experts", 8)
        model_kwargs["top_k"] = config.get("top_k", 2)
        model_kwargs["aux_loss_weight"] = config.get("aux_loss_weight", 0.01)
        model_kwargs["z_loss_weight"] = config.get("z_loss_weight", 0.001)
        model_kwargs["l1_lambda"] = config.get("l1_lambda", 0.0)
        model_kwargs["prune_threshold"] = config.get("prune_threshold", 0.0)
        model_kwargs["prune_every_n"] = config.get("prune_every_n", 0)
    elif model_class == "SparseDyadicTransformer":
        model_kwargs["d_ff"] = config.get("d_ff", config["d_model"] * 2)
        model_kwargs["l1_lambda"] = config.get("l1_lambda", 0.0)
        model_kwargs["ff_l1_lambda"] = config.get("ff_l1_lambda", 0.0)
        model_kwargs["prune_threshold"] = config.get("prune_threshold", 0.0)
        model_kwargs["prune_every_n"] = config.get("prune_every_n", 0)
        model_kwargs["use_gradient_checkpointing"] = config.get("use_gradient_checkpointing", True)
    elif model_class == "FinalSparseTransformer":
        # MoE FFN config
        model_kwargs["d_expert"] = config.get("d_expert", config["d_model"])
        model_kwargs["num_experts"] = config.get("num_experts", 8)
        model_kwargs["top_k"] = config.get("top_k", 2)
        model_kwargs["aux_loss_weight"] = config.get("aux_loss_weight", 0.01)
        model_kwargs["z_loss_weight"] = config.get("z_loss_weight", 0.001)
        # Sparsity config (both attention M/V and FFN experts)
        model_kwargs["l1_lambda"] = config.get("l1_lambda", 0.0)
        model_kwargs["ff_l1_lambda"] = config.get("ff_l1_lambda", 0.0)
        model_kwargs["prune_threshold"] = config.get("prune_threshold", 0.0)
        model_kwargs["prune_every_n"] = config.get("prune_every_n", 0)
        model_kwargs["use_gradient_checkpointing"] = config.get("use_gradient_checkpointing", True)
    
    # Model (using BF16 autocast for forward/backward)
    if is_main():
        print(f"Creating {model_class} (BF16 + torch.compile)...")
        if config.get("block_decode", False):
            print(f"  block_decode=True, head_k={config.get('head_k', 1)}, "
                  f"head_balance_tolerance={config.get('head_balance_tolerance', 10)} "
                  f"(clustered eval-time decode)")
    model = create_model(model_class=model_class, **model_kwargs)
    model.to(device)
    # Disable DDP optimizer for models that access params outside forward()
    # (L1 regularization, gradient checkpointing higher-order ops).
    # Single-bucket gradient sync is slightly less efficient but avoids errors.
    needs_ddp_disable = (
        (model_class == "SparseDyadicTransformer" and config.get("use_gradient_checkpointing", True))
        or (model_class == "MoETransformer" and config.get("l1_lambda", 0.0) > 0)
        or model_class == "FinalSparseTransformer"
    )
    if needs_ddp_disable:
        torch._dynamo.config.optimize_ddp = False
        if is_main():
            print("Disabled DDPOptimizer (required for out-of-forward param access + compile)")
    model = torch.compile(model)
    # find_unused_parameters=True is required for MoE models where not all experts
    # receive tokens in every batch (dynamic routing causes some params to be unused)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # Static graph allows params to be marked ready multiple times (needed when
    # L1 regularization accesses params outside forward, creating a second
    # autograd path to the same parameters during backward).
    needs_static_graph = (
        (model_class == "MoETransformer" and config.get("l1_lambda", 0.0) > 0)
        or model_class == "FinalSparseTransformer"
    )
    if needs_static_graph:
        model._set_static_graph()
        if is_main():
            print("Enabled DDP static graph (required for L1 on params outside forward)")
    
    raw_model = model.module
    # Handle compiled model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    total_params = raw_model.count_parameters(count_zeros=True)
    nonzero_params = raw_model.count_parameters(count_zeros=False)
    if is_main():
        print(f"Total parameters: {total_params:,}")
        print(f"Non-zero parameters: {nonzero_params:,}")
        wandb.log({
            "params/total": total_params,
            "params/nonzero": nonzero_params,
        })
    
    # Optimizer — separate param groups for SparseDyadicTransformer
    if model_class == "SparseDyadicTransformer":
        # Sparse attention params (M, V): configurable weight_decay and LR
        sparse_param_ids = set(id(p) for p in raw_model.get_sparse_params())
        sparse_lr = config.get("sparse_lr") or config["max_lr"]
        
        # Identify LayerNorm and bias params (should not get weight decay)
        no_decay_ids = set()
        for name, p in raw_model.named_parameters():
            if "ln" in name or "norm" in name or name.endswith(".bias"):
                no_decay_ids.add(id(p))
        
        sparse_group = {
            "params": [p for p in model.parameters() if id(p) in sparse_param_ids],
            "weight_decay": config.get("sparse_weight_decay", 0.0),
            "initial_lr": sparse_lr,
        }
        # Decayed group: everything that isn't sparse and isn't LayerNorm/bias
        other_decay_group = {
            "params": [p for p in model.parameters()
                       if id(p) not in sparse_param_ids and id(p) not in no_decay_ids],
            "weight_decay": config.get("weight_decay", 0.3),
            "initial_lr": config["max_lr"],
        }
        # No-decay group: LayerNorm and bias params
        no_decay_group = {
            "params": [p for p in model.parameters()
                       if id(p) not in sparse_param_ids and id(p) in no_decay_ids],
            "weight_decay": 0.0,
            "initial_lr": config["max_lr"],
        }
        
        optimizer = torch.optim.AdamW(
            [sparse_group, other_decay_group, no_decay_group],
            lr=config["max_lr"],
            betas=(0.9, 0.95),
        )
        if is_main():
            n_sparse = sum(p.numel() for p in sparse_group["params"])
            n_decay = sum(p.numel() for p in other_decay_group["params"])
            n_no_decay = sum(p.numel() for p in no_decay_group["params"])
            print(f"Optimizer: 3 param groups")
            print(f"  Sparse (M, V):    {n_sparse:,} params, wd={sparse_group['weight_decay']}, lr={sparse_lr:.2e}")
            print(f"  Decayed (FFN+emb): {n_decay:,} params, wd={other_decay_group['weight_decay']}, lr={config['max_lr']:.2e}")
            print(f"  No-decay (LN+bias): {n_no_decay:,} params, wd=0.0, lr={config['max_lr']:.2e}")
    elif model_class == "MoETransformer":
        # 4 mutually-exclusive param groups:
        #   1. Attention (qkv, proj)      → attn_weight_decay
        #   2. FFN expert weights          → ff_weight_decay (L1 handles sparsity)
        #   3. Embeddings / output head    → embed_weight_decay
        #   4. LayerNorm + biases          → 0
        ff_ids = set(id(p) for p in raw_model.get_ff_params())
        attn_ids = set(id(p) for p in raw_model.get_attn_params())
        # Also include MoE router weights in attention group
        for layer in raw_model.layers:
            attn_ids.add(id(layer.moe.router.weight))
        
        no_decay_ids = set()
        embed_ids = set()
        for name, p in raw_model.named_parameters():
            if "ln" in name or "norm" in name or name.endswith(".bias"):
                no_decay_ids.add(id(p))
            elif "token_emb" in name or "head" in name:
                embed_ids.add(id(p))
        
        attn_group = {
            "params": [p for p in model.parameters() if id(p) in attn_ids],
            "weight_decay": config.get("attn_weight_decay", 0.1),
            "initial_lr": config["max_lr"],
        }
        ff_group = {
            "params": [p for p in model.parameters() if id(p) in ff_ids],
            "weight_decay": config.get("ff_weight_decay", 0.0),
            "initial_lr": config["max_lr"],
        }
        embed_group = {
            "params": [p for p in model.parameters()
                       if id(p) in embed_ids and id(p) not in no_decay_ids],
            "weight_decay": config.get("embed_weight_decay", 0.1),
            "initial_lr": config["max_lr"],
        }
        no_decay_group = {
            "params": [p for p in model.parameters() if id(p) in no_decay_ids],
            "weight_decay": 0.0,
            "initial_lr": config["max_lr"],
        }
        
        optimizer = torch.optim.AdamW(
            [attn_group, ff_group, embed_group, no_decay_group],
            lr=config["max_lr"],
            betas=(0.9, 0.95),
        )
        if is_main():
            n_attn = sum(p.numel() for p in attn_group["params"])
            n_ff = sum(p.numel() for p in ff_group["params"])
            n_embed = sum(p.numel() for p in embed_group["params"])
            n_nd = sum(p.numel() for p in no_decay_group["params"])
            print(f"Optimizer: 4 param groups")
            print(f"  Attention (qkv+proj+router): {n_attn:,} params, wd={attn_group['weight_decay']}")
            print(f"  FFN experts (w1,w2,w3):      {n_ff:,} params, wd={ff_group['weight_decay']}")
            print(f"  Embeddings+head:             {n_embed:,} params, wd={embed_group['weight_decay']}")
            print(f"  No-decay (LN+bias):          {n_nd:,} params, wd=0.0")
    elif model_class == "FinalSparseTransformer":
        # 5 mutually-exclusive param groups:
        #   1. Sparse attention (M, V)     → attn_weight_decay, optional sparse_lr
        #   2. MoE routers                 → attn_weight_decay (grouped with attention)
        #   3. FFN expert weights           → ff_weight_decay
        #   4. Embeddings / output head    → embed_weight_decay
        #   5. LayerNorm + biases          → 0
        sparse_ids = set(id(p) for p in raw_model.get_sparse_params())
        ff_ids = set(id(p) for p in raw_model.get_ff_params())
        router_ids = set(id(p) for p in raw_model.get_attn_params())
        sparse_lr = config.get("sparse_lr") or config["max_lr"]

        no_decay_ids = set()
        embed_ids = set()
        for name, p in raw_model.named_parameters():
            if "ln" in name or "norm" in name or name.endswith(".bias"):
                no_decay_ids.add(id(p))
            elif "token_emb" in name or "head" in name:
                embed_ids.add(id(p))

        sparse_group = {
            "params": [p for p in model.parameters() if id(p) in sparse_ids],
            "weight_decay": config.get("attn_weight_decay", 0.0),
            "initial_lr": sparse_lr,
        }
        router_group = {
            "params": [p for p in model.parameters() if id(p) in router_ids],
            "weight_decay": config.get("attn_weight_decay", 0.0),
            "initial_lr": config["max_lr"],
        }
        ff_group = {
            "params": [p for p in model.parameters() if id(p) in ff_ids],
            "weight_decay": config.get("ff_weight_decay", 0.0),
            "initial_lr": config["max_lr"],
        }
        embed_group = {
            "params": [p for p in model.parameters()
                       if id(p) in embed_ids and id(p) not in no_decay_ids],
            "weight_decay": config.get("embed_weight_decay", 0.1),
            "initial_lr": config["max_lr"],
        }
        no_decay_group = {
            "params": [p for p in model.parameters() if id(p) in no_decay_ids],
            "weight_decay": 0.0,
            "initial_lr": config["max_lr"],
        }

        optimizer = torch.optim.AdamW(
            [sparse_group, router_group, ff_group, embed_group, no_decay_group],
            lr=config["max_lr"],
            betas=(0.9, 0.95),
        )
        if is_main():
            n_sparse = sum(p.numel() for p in sparse_group["params"])
            n_router = sum(p.numel() for p in router_group["params"])
            n_ff = sum(p.numel() for p in ff_group["params"])
            n_embed = sum(p.numel() for p in embed_group["params"])
            n_nd = sum(p.numel() for p in no_decay_group["params"])
            print(f"Optimizer: 5 param groups")
            print(f"  Sparse attn (M, V):          {n_sparse:,} params, wd={sparse_group['weight_decay']}, lr={sparse_lr:.2e}")
            print(f"  MoE routers:                 {n_router:,} params, wd={router_group['weight_decay']}")
            print(f"  FFN experts (w1,w2,w3):      {n_ff:,} params, wd={ff_group['weight_decay']}")
            print(f"  Embeddings+head:             {n_embed:,} params, wd={embed_group['weight_decay']}")
            print(f"  No-decay (LN+bias):          {n_nd:,} params, wd=0.0")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["max_lr"],
            betas=(0.9, 0.95),
            weight_decay=config.get("weight_decay", 0.3),
        )
    
    # Register signal handlers for graceful shutdown (Ctrl+C saves checkpoint)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    # Train
    model, final_step = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_steps=config["num_steps"],
        eval_every=config["eval_every"],
        max_lr=config["max_lr"],
        warmup_steps=config.get("warmup_steps", 200),
    )
    
    # Save model weights (only on main rank)
    # Always save for models with sparsity support; otherwise require --save flag
    should_save = args.save or model_class in ("SparseDyadicTransformer", "MoETransformer", "FinalSparseTransformer")
    if is_main() and should_save:
        # Create saved_models directory
        os.makedirs("saved_models", exist_ok=True)
        
        raw_model = model.module if hasattr(model, "module") else model
        # Get the underlying model from compiled version if needed
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        
        # Save weights with config (no optimizer state)
        checkpoint_path = f"saved_models/{args.run_name}_step{final_step}.pt"
        checkpoint = {
            "model_state_dict": raw_model.state_dict(),
            "step": final_step,
            "config": config,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"\n✅ Model saved to {checkpoint_path} (step {final_step})")
    elif is_main() and not should_save:
        print("\n⚠️  Model NOT saved (use --save to save weights)")
    
    if is_main():
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

