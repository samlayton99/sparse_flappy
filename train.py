"""Training script with wandb logging and DDP support."""

import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from tqdm import tqdm

# Peak TFLOPS for MFU calculation (BF16 tensor core ops)
# H100 SXM: 990 TFLOPS BF16, A100: 312 TFLOPS BF16
GPU_PEAK_TFLOPS = 990

from data import get_dataloader
from model import create_model


def setup_ddp():
    """Initialize DDP. Returns local_rank."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


PAD_TOKEN_ID = 50256  # GPT-2 <|endoftext|> used as pad token


def compute_loss(model, batch, device):
    """Compute cross-entropy loss for a batch using BF16 autocast."""
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
    return loss


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup then linear decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    return max_lr - (max_lr - min_lr) * decay_ratio


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 20):
    """Evaluate model on validation set (lightweight, uses BF16)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for batch in val_loader:
        loss = compute_loss(model, batch, device)  # autocast handled in compute_loss
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= max_batches:
            break
    
    model.train()
    return total_loss / n_batches


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_steps: int = 10000,
    eval_every: int = 50,
    max_lr: float = 1e-3,
    warmup_steps: int = 200,
):
    """Main training loop with logging every eval_every steps."""
    model.train()
    raw_model = model.module if hasattr(model, "module") else model
    num_params = raw_model.count_parameters(count_zeros=True)
    min_lr = max_lr * 0.1  # decay to 10% of peak
    
    train_iter = iter(train_loader)
    running_loss = 0.0
    total_tokens = 0
    epoch = 0
    t0 = time.time()
    
    pbar = tqdm(range(num_steps), desc="Training", disable=not is_main())
    for step in pbar:
        # Update learning rate
        lr = get_lr(step, warmup_steps, num_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
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
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
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
            val_loss = evaluate(model, val_loader, device)
            nonzero_params = raw_model.count_parameters(count_zeros=False)
            
            if is_main():
                pbar.set_postfix({"train": f"{train_loss:.3f}", "val": f"{val_loss:.3f}", "mfu": f"{mfu:.1%}"})
                wandb.log({
                    "train/loss": train_loss,
                    "train/lr": lr,
                    "train/total_tokens": total_tokens,
                    "train/total_flops": total_flops,
                    "train/epoch": epoch,
                    "val/loss": val_loss,
                    "params/nonzero": nonzero_params,
                    "mfu": mfu,
                    "step": step + 1,
                })
                if val_loss < 3.0:
                    print(f"\n🎉 Goal achieved! val_loss={val_loss:.4f} < 3.0 with {nonzero_params:,} non-zero params")
            
            running_loss = 0.0
            t0 = time.time()
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32) # per-device batch size if DDP, be aware 
    parser.add_argument("--max_lr", type=float, default=5e-4)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=2048)
    parser.add_argument("--n_layers", type=int, default=16)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=4096)
    parser.add_argument("--wandb_project", type=str, default="weightless")
    args = parser.parse_args()

    # Autoscale max_lr: args.max_lr is at d_model=2048, scale by sqrt(2048/d_model)
    # If d_model increases, lr decreases.
    base_lr = args.max_lr
    args.max_lr = base_lr * (2048 / args.d_model) ** 0.5
    if is_main():
        print(f"Autoscaled max_lr from {base_lr} to {args.max_lr:.2e} (d_model={args.d_model})")

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
            entity="flappy-external",
            config=vars(args),
        )
    
    # Data (each rank gets different data via streaming)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if is_main():
        print("Setting up data loaders...")
    train_loader = get_dataloader(split="train", batch_size=args.batch_size, streaming=True, rank=rank, world_size=world_size)
    val_loader = get_dataloader(split="test", batch_size=args.batch_size, streaming=True, rank=rank, world_size=world_size)
    
    # Model (using BF16 autocast for forward/backward)
    if is_main():
        print("Creating model (BF16 + torch.compile)...")
    model = create_model(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )
    model.to(device)
    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])
    
    raw_model = model.module
    total_params = raw_model.count_parameters(count_zeros=True)
    nonzero_params = raw_model.count_parameters(count_zeros=False)
    if is_main():
        print(f"Total parameters: {total_params:,}")
        print(f"Non-zero parameters: {nonzero_params:,}")
        wandb.log({
            "params/total": total_params,
            "params/nonzero": nonzero_params,
        })
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.max_lr, 
        betas=(0.9, 0.95), 
        weight_decay=0.3,
    )
    
    # Train
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        max_lr=args.max_lr,
    )
    
    if is_main():
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

