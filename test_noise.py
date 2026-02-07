"""Test layer sensitivity to noise injection.

Injects Gaussian noise at attention and/or FFN outputs to measure
how sensitive each layer is to perturbations.

Noise formula: output + eps * v * ||output|| / ||v||
where eps is relative magnitude and v is Gaussian noise.

Usage:
    # Single GPU
    python test_noise.py --checkpoint model_weightless.pt --eps 0.1 0.5 1.0
    
    # Multi-GPU (parallel)
    python test_noise.py --checkpoint model_weightless.pt --eps 0.1 0.5 1.0 --gpus 0 1 2 3
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from functools import partial
import multiprocessing as mp

import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import create_model
from data import get_dataloader


# =============================================================================
# Noise Injection
# =============================================================================

# Global counter for debugging hook execution
_hook_call_counter = {"count": 0}


def create_noise_hook(eps: float, debug: bool = False):
    """Create a forward hook that adds relative Gaussian noise.
    
    Noise formula: output + eps * v * ||output|| / ||v||
    
    Noise is computed in fp32 for precision, then cast back to original dtype.
    This prevents small epsilon values from being rounded away in bf16.
    
    Args:
        eps: Relative noise magnitude
        debug: If True, prints debug info on first call
        
    Returns:
        Hook function
    """
    def hook(module, input, output):
        _hook_call_counter["count"] += 1
        
        if debug and _hook_call_counter["count"] <= 3:
            print(f"    [DEBUG] Hook #{_hook_call_counter['count']}: "
                  f"module={type(module).__name__}, output.shape={output.shape}, eps={eps}")
        
        if eps <= 0:
            return output
        
        # Compute in fp32 for precision, then cast back to original dtype
        # This prevents small epsilon values from being rounded away in bf16
        orig_dtype = output.dtype
        output_fp32 = output.float()
        
        # Generate Gaussian noise in fp32
        v = torch.randn(output_fp32.shape, device=output.device, dtype=torch.float32)
        
        # Compute norms in fp32
        out_norm = output_fp32.norm(dim=-1, keepdim=True)
        v_norm = v.norm(dim=-1, keepdim=True)
        
        # Relative noise: eps * v * ||output|| / ||v||
        noise = eps * v * out_norm / (v_norm + 1e-8)
        
        # Add noise and cast back to original dtype
        return (output_fp32 + noise).to(orig_dtype)
    
    return hook


def reset_hook_counter():
    """Reset the global hook call counter."""
    _hook_call_counter["count"] = 0


def get_hook_count():
    """Get the current hook call count."""
    return _hook_call_counter["count"]


def register_noise_hooks(
    model,
    targets: List[Tuple[int, str]],
    eps: float,
    debug: bool = False,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register noise hooks on specified modules.
    
    Args:
        model: SimpleTransformer or MoETransformer model
        targets: List of (layer_idx, module_type) tuples
                 module_type is "attn", "ff", or "moe"
        eps: Noise magnitude
        debug: If True, enable debug output
        
    Returns:
        List of hook handles (for removal)
    """
    hooks = []
    reset_hook_counter()  # Reset counter for each registration
    hook_fn = create_noise_hook(eps, debug=debug)
    
    for layer_idx, module_type in targets:
        layer = model.layers[layer_idx]
        
        if module_type == "attn":
            module = layer.attn
        elif module_type == "ff":
            # Support both SimpleTransformer (.ff) and MoETransformer (.moe)
            if hasattr(layer, "ff"):
                module = layer.ff
            elif hasattr(layer, "moe"):
                module = layer.moe
            else:
                raise ValueError(f"Layer {layer_idx} has neither 'ff' nor 'moe' attribute")
        elif module_type == "moe":
            # Explicit MoE targeting
            if hasattr(layer, "moe"):
                module = layer.moe
            else:
                raise ValueError(f"Layer {layer_idx} has no 'moe' attribute")
        else:
            raise ValueError(f"Unknown module type: {module_type}. Use 'attn', 'ff', or 'moe'")
        
        handle = module.register_forward_hook(hook_fn)
        hooks.append(handle)
    
    if debug and len(hooks) > 0:
        print(f"  [DEBUG] Registered {len(hooks)} hooks for eps={eps}")
    
    return hooks


def clear_hooks(hooks: List[torch.utils.hooks.RemovableHandle]):
    """Remove all hooks."""
    for hook in hooks:
        hook.remove()


# =============================================================================
# Evaluation
# =============================================================================

PAD_TOKEN_ID = 50256  # GPT-2 <|endoftext|>


@torch.no_grad()
def evaluate_with_noise(
    model,
    dataloader,
    device: torch.device,
    eps: float,
    targets: List[Tuple[int, str]],
    num_samples: int,
    batch_size: int,
    debug: bool = False,
) -> float:
    """Evaluate model with noise injected at specified targets.
    
    Args:
        model: SimpleTransformer or MoETransformer model
        dataloader: Validation dataloader
        device: Device
        eps: Noise magnitude
        targets: List of (layer_idx, module_type) tuples
        num_samples: Total samples to evaluate
        batch_size: Batch size
        debug: If True, verify hooks are firing
        
    Returns:
        Average cross-entropy loss
    """
    model.eval()
    
    # Register hooks
    hooks = register_noise_hooks(model, targets, eps, debug=debug)
    
    try:
        total_loss = 0.0
        total_tokens = 0
        samples_seen = 0
        
        for batch in dataloader:
            if samples_seen >= num_samples:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass (hooks will inject noise)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
            
            # Compute loss (ignore padding)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
                reduction="sum",
            )
            
            # Count non-pad tokens
            n_tokens = (labels != PAD_TOKEN_ID).sum().item()
            
            total_loss += loss.item()
            total_tokens += n_tokens
            samples_seen += input_ids.size(0)
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        
        # Sanity check: verify hooks actually fired
        hook_count = get_hook_count()
        expected_hooks = len(targets)
        if debug and expected_hooks > 0:
            if hook_count == 0:
                print(f"  [WARNING] No hooks fired! Expected {expected_hooks} per forward pass")
            else:
                print(f"  [DEBUG] Hooks fired {hook_count} times total")
        
        return avg_loss
    
    finally:
        clear_hooks(hooks)


# =============================================================================
# Layer Sweep
# =============================================================================

def run_layer_sweep(
    model,
    dataloader,
    device: torch.device,
    eps_values: List[float],
    num_samples: int,
    batch_size: int,
    n_layers: int,
) -> Dict:
    """Run noise injection sweep across all layers and module types.
    
    Uses a SINGLE shuffled dataloader that advances across all experiments,
    ensuring each (hookpoint, eps) combination sees DIFFERENT random data.
    
    Args:
        model: SimpleTransformer model
        dataloader: Ignored (for backward compatibility) - we create our own
        device: Device
        eps_values: List of noise magnitudes to test
        num_samples: Samples per evaluation
        batch_size: Batch size
        n_layers: Number of transformer layers
        
    Returns:
        Dict with results: {
            "baseline": baseline_loss,
            "results": [{"layer": i, "module": "attn"/"ff", "eps": e, "loss": l}, ...]
        }
    """
    results = []
    
    # Create a SINGLE shuffled dataloader that advances across ALL experiments
    # This ensures each experiment sees different random data
    print("Creating shuffled dataloader for sweep...")
    global_dataloader = iter(get_dataloader(
        split="test", batch_size=batch_size, shuffle=True
    ))
    
    # First get baseline (no noise) - uses first N batches
    print("Computing baseline (no noise)...")
    baseline = evaluate_with_noise(
        model, global_dataloader, device,
        eps=0.0, targets=[], 
        num_samples=num_samples, batch_size=batch_size
    )
    print(f"Baseline loss: {baseline:.4f}")
    
    # Test each layer and module type
    total_tests = n_layers * 2 * len(eps_values)
    pbar = tqdm(total=total_tests, desc="Layer sweep")
    
    for layer_idx in range(n_layers):
        for module_type in ["attn", "ff"]:
            for eps in eps_values:
                # Use the global dataloader - it advances to next batches
                # DO NOT create a new dataloader here!
                targets = [(layer_idx, module_type)]
                loss = evaluate_with_noise(
                    model, global_dataloader, device,
                    eps=eps, targets=targets,
                    num_samples=num_samples, batch_size=batch_size
                )
                
                results.append({
                    "layer": layer_idx,
                    "module": module_type,
                    "eps": eps,
                    "loss": loss,
                    "loss_increase": loss - baseline,
                    "loss_ratio": loss / baseline if baseline > 0 else float('inf'),
                })
                
                pbar.set_postfix({
                    "layer": layer_idx,
                    "module": module_type,
                    "eps": f"{eps:.2f}",
                    "loss": f"{loss:.3f}",
                })
                pbar.update(1)
    
    pbar.close()
    
    return {
        "baseline": baseline,
        "n_layers": n_layers,
        "eps_values": eps_values,
        "num_samples": num_samples,
        "results": results,
    }


# =============================================================================
# Multi-GPU Parallel Execution
# =============================================================================

def gpu_worker(
    gpu_id: int,
    tasks: List[Tuple[int, str, float]],  # List of (layer_idx, module_type, eps)
    checkpoint_path: str,
    num_samples: int,
    batch_size: int,
    worker_id: int,
    total_workers: int,
    worker_seed: int,
) -> List[Dict]:
    """Worker function that runs on a single GPU.
    
    Uses a SINGLE shuffled dataloader that advances across all tasks assigned
    to this worker. Each worker gets a unique seed for different random shuffles.
    
    Args:
        gpu_id: CUDA device ID to use
        tasks: List of (layer_idx, module_type, eps) tuples to evaluate
        checkpoint_path: Path to model checkpoint
        num_samples: Samples per evaluation
        batch_size: Batch size
        worker_id: Worker index (for progress display)
        total_workers: Total number of workers
        worker_seed: Unique seed for this worker's random shuffling
        
    Returns:
        List of result dicts
    """
    # Set the GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")  # Will be the gpu_id due to CUDA_VISIBLE_DEVICES
    
    # Set unique seed for this worker to get different shuffles
    torch.manual_seed(worker_seed)
    import random
    random.seed(worker_seed)
    
    # Load model on this GPU
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {
        "d_model": 2048, "n_layers": 16, "n_heads": 8, "d_ff": 4096
    })
    model = create_model(**config)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    results = []
    desc = f"GPU {gpu_id} [{worker_id+1}/{total_workers}]"
    
    # Create a SINGLE shuffled dataloader that advances across ALL tasks
    # This ensures each task sees different random data
    global_dataloader = iter(get_dataloader(split="test", batch_size=batch_size, shuffle=True))
    
    for layer_idx, module_type, eps in tqdm(tasks, desc=desc, position=worker_id):
        # Use the global dataloader - it advances to next batches
        # DO NOT create a new dataloader here!
        targets = [(layer_idx, module_type)]
        loss = evaluate_with_noise(
            model, global_dataloader, device,
            eps=eps, targets=targets,
            num_samples=num_samples, batch_size=batch_size
        )
        
        results.append({
            "layer": layer_idx,
            "module": module_type,
            "eps": eps,
            "loss": loss,
        })
    
    return results


def run_layer_sweep_parallel(
    checkpoint_path: str,
    eps_values: List[float],
    num_samples: int,
    batch_size: int,
    n_layers: int,
    gpu_ids: List[int],
) -> Dict:
    """Run noise injection sweep in parallel across multiple GPUs.
    
    Args:
        checkpoint_path: Path to model checkpoint
        eps_values: List of noise magnitudes to test
        num_samples: Samples per evaluation
        batch_size: Batch size
        n_layers: Number of transformer layers
        gpu_ids: List of GPU IDs to use
        
    Returns:
        Dict with results
    """
    # Generate all tasks: (layer_idx, module_type, eps)
    all_tasks = []
    for layer_idx in range(n_layers):
        for module_type in ["attn", "ff"]:
            for eps in eps_values:
                all_tasks.append((layer_idx, module_type, eps))
    
    # Compute baseline on GPU 0 first (single process)
    print(f"Computing baseline on GPU {gpu_ids[0]}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    device = torch.device("cuda:0")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {
        "d_model": 2048, "n_layers": 16, "n_heads": 8, "d_ff": 4096
    })
    model = create_model(**config)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Use shuffled dataloader for baseline too
    baseline_dataloader = iter(get_dataloader(split="test", batch_size=batch_size, shuffle=True))
    baseline = evaluate_with_noise(
        model, baseline_dataloader, device,
        eps=0.0, targets=[],
        num_samples=num_samples, batch_size=batch_size
    )
    print(f"Baseline loss: {baseline:.4f}")
    
    # Clean up before spawning workers
    del model
    torch.cuda.empty_cache()
    
    # Split tasks across GPUs
    num_gpus = len(gpu_ids)
    tasks_per_gpu = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        tasks_per_gpu[i % num_gpus].append(task)
    
    print(f"\nDistributing {len(all_tasks)} tasks across {num_gpus} GPUs...")
    for i, (gpu_id, tasks) in enumerate(zip(gpu_ids, tasks_per_gpu)):
        print(f"  GPU {gpu_id}: {len(tasks)} tasks")
    
    # Run workers in parallel using multiprocessing
    # Use spawn to avoid CUDA context issues
    # Each worker gets a unique seed for different random shuffles
    ctx = mp.get_context("spawn")
    base_seed = int(time.time())  # Base seed from current time
    
    with ctx.Pool(num_gpus) as pool:
        worker_args = [
            (gpu_ids[i], tasks_per_gpu[i], checkpoint_path, num_samples, batch_size, i, num_gpus, base_seed + i)
            for i in range(num_gpus)
        ]
        all_results = pool.starmap(gpu_worker, worker_args)
    
    # Merge results from all workers
    results = []
    for worker_results in all_results:
        results.extend(worker_results)
    
    # Add baseline info to results
    for r in results:
        r["loss_increase"] = r["loss"] - baseline
        r["loss_ratio"] = r["loss"] / baseline if baseline > 0 else float('inf')
    
    # Sort results by layer, module, eps for consistent output
    results.sort(key=lambda x: (x["layer"], x["module"], x["eps"]))
    
    return {
        "baseline": baseline,
        "n_layers": n_layers,
        "eps_values": eps_values,
        "num_samples": num_samples,
        "results": results,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_sensitivity(
    data: Dict,
    output_path: str = "noise_sensitivity.png",
):
    """Plot layer sensitivity to noise.
    
    Args:
        data: Results from run_layer_sweep
        output_path: Where to save the plot
    """
    n_layers = data["n_layers"]
    eps_values = data["eps_values"]
    baseline = data["baseline"]
    results = data["results"]
    
    # Organize data by eps and module type
    fig, axes = plt.subplots(1, len(eps_values), figsize=(5*len(eps_values), 5), squeeze=False)
    
    for idx, eps in enumerate(eps_values):
        ax = axes[0, idx]
        
        # Filter results for this eps
        eps_results = [r for r in results if r["eps"] == eps]
        
        # Separate attn and ff
        attn_losses = [r["loss"] for r in eps_results if r["module"] == "attn"]
        ff_losses = [r["loss"] for r in eps_results if r["module"] == "ff"]
        layers = list(range(n_layers))
        
        # Plot
        ax.plot(layers, attn_losses, 'o--', label='Attention', alpha=0.8, markersize=4)
        ax.plot(layers, ff_losses, 's-', label='FFN', alpha=0.8, markersize=4)
        ax.axhline(y=baseline, color='gray', linestyle=':', label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Validation Loss')
        ax.set_title(f'eps = {eps}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Layer Sensitivity to Noise Injection', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_sensitivity_combined(
    data: Dict,
    output_path: str = "noise_sensitivity_combined.png",
):
    """Plot all eps values on a single plot with loss ratio.
    
    Args:
        data: Results from run_layer_sweep
        output_path: Where to save the plot
    """
    n_layers = data["n_layers"]
    eps_values = data["eps_values"]
    results = data["results"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors for different eps values
    colors = plt.cm.viridis([i / len(eps_values) for i in range(len(eps_values))])
    
    layers = list(range(n_layers))
    
    for idx, eps in enumerate(eps_values):
        eps_results = [r for r in results if r["eps"] == eps]
        
        # Attention
        attn_ratios = [r["loss_ratio"] for r in eps_results if r["module"] == "attn"]
        ax1.plot(layers, attn_ratios, 'o--', color=colors[idx], 
                 label=f'eps={eps}', alpha=0.8, markersize=4)
        
        # FFN
        ff_ratios = [r["loss_ratio"] for r in eps_results if r["module"] == "ff"]
        ax2.plot(layers, ff_ratios, 's-', color=colors[idx], 
                 label=f'eps={eps}', alpha=0.8, markersize=4)
    
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Loss Ratio (noisy / baseline)')
    ax1.set_title('Attention Sensitivity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Loss Ratio (noisy / baseline)')
    ax2.set_title('FFN Sensitivity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Layer Sensitivity: Loss Ratio vs Layer', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def print_summary(data: Dict):
    """Print a summary table of results."""
    n_layers = data["n_layers"]
    eps_values = data["eps_values"]
    baseline = data["baseline"]
    results = data["results"]
    
    print("\n" + "=" * 70)
    print("NOISE SENSITIVITY SUMMARY")
    print("=" * 70)
    print(f"Baseline loss: {baseline:.4f}")
    print(f"Layers: {n_layers}, Eps values: {eps_values}")
    print()
    
    # Find most sensitive layers
    for eps in eps_values:
        eps_results = [r for r in results if r["eps"] == eps]
        
        # Sort by loss increase
        sorted_results = sorted(eps_results, key=lambda x: x["loss_increase"], reverse=True)
        
        print(f"eps = {eps}:")
        print(f"  Top 3 most sensitive:")
        for r in sorted_results[:3]:
            print(f"    Layer {r['layer']:2d} {r['module']:4s}: loss={r['loss']:.4f} (+{r['loss_increase']:.4f}, {r['loss_ratio']:.2f}x)")
        print()


# =============================================================================
# Main
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use defaults
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Assume default config
        config = {
            "d_model": 2048,
            "n_layers": 16,
            "n_heads": 8,
            "d_ff": 4096,
        }
    
    model = create_model(**config)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Test layer sensitivity to noise")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                        help="Noise magnitudes to test")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples per evaluation")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory for output files")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output to verify hooks are firing")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU IDs to use for parallel execution (e.g., --gpus 0 1 2 3)")
    args = parser.parse_args()
    
    # Handle multi-GPU parallel execution
    if args.gpus and len(args.gpus) > 1:
        print(f"Running in parallel mode on GPUs: {args.gpus}")
        
        # Quick check: load model once to get n_layers
        temp_device = torch.device(f"cuda:{args.gpus[0]}")
        checkpoint = torch.load(args.checkpoint, map_location=temp_device, weights_only=False)
        config = checkpoint.get("config", {"n_layers": 16})
        n_layers = config.get("n_layers", 16)
        del checkpoint
        torch.cuda.empty_cache()
        
        # Run parallel sweep
        results = run_layer_sweep_parallel(
            checkpoint_path=args.checkpoint,
            eps_values=args.eps,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            n_layers=n_layers,
            gpu_ids=args.gpus,
        )
        
        # Save and plot results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "noise_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        plot_sensitivity(results, str(output_dir / "noise_sensitivity.png"))
        plot_sensitivity_combined(results, str(output_dir / "noise_sensitivity_combined.png"))
        print_summary(results)
        return
    
    # Single GPU mode
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
        device = torch.device("cuda:0")
        print(f"Device: cuda (GPU {args.gpus[0]})")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    n_layers = len(model.layers)
    
    # Detect model type and FFN module name
    layer0 = model.layers[0]
    if hasattr(layer0, "ff"):
        ffn_attr = "ff"
        model_type = "SimpleTransformer"
    elif hasattr(layer0, "moe"):
        ffn_attr = "moe"
        model_type = "MoETransformer"
    else:
        raise ValueError(f"Unknown layer type: {type(layer0)}")
    print(f"Model type: {model_type} (FFN attr: {ffn_attr})")
    print(f"Model loaded: {n_layers} layers, d_model={config.get('d_model', '?')}")
    
    # Get dataloader
    print("Setting up dataloader...")
    dataloader = get_dataloader(split="test", batch_size=args.batch_size)
    
    # Quick sanity check: verify hooks work before full sweep
    if args.debug:
        print("\n[DEBUG] Running sanity check...")
        test_dataloader = iter(get_dataloader(split="test", batch_size=4))
        test_batch = next(test_dataloader)
        input_ids = test_batch["input_ids"].to(device)
        
        # Test without hooks
        reset_hook_counter()
        with torch.no_grad():
            out_clean = model(input_ids).clone()
        
        # Test with hooks
        reset_hook_counter()
        hook_fn = create_noise_hook(1.0, debug=True)
        handle = layer0.attn.register_forward_hook(hook_fn)
        with torch.no_grad():
            out_noisy = model(input_ids).clone()
        handle.remove()
        
        diff = (out_noisy - out_clean).abs().max().item()
        hooks_fired = get_hook_count()
        print(f"[DEBUG] Sanity check: hooks_fired={hooks_fired}, max_diff={diff:.4f}")
        if hooks_fired == 0:
            print("[ERROR] Hooks are NOT firing! Something is wrong.")
        elif diff < 0.001:
            print("[WARNING] Output barely changed despite hooks - eps might be too small or issue exists")
        else:
            print("[DEBUG] Sanity check PASSED - hooks are working correctly")
        print()
    
    # Run sweep
    print(f"Running noise sweep with eps={args.eps}, {args.num_samples} samples...")
    results = run_layer_sweep(
        model=model,
        dataloader=dataloader,
        device=device,
        eps_values=args.eps,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        n_layers=n_layers,
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "noise_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Plot
    plot_sensitivity(results, str(output_dir / "noise_sensitivity.png"))
    plot_sensitivity_combined(results, str(output_dir / "noise_sensitivity_combined.png"))
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
