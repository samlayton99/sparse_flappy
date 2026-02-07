"""Training script for sparsity experiments.

Experiments:
A) SwiGLUMLP with L1 regularization on all weights
B) LowRankSwiGLUMLP with strong L1 on S matrices only
C) Placeholder for future method

Usage:
    # Run with config file (edit toy_hyperparameters.py to change settings)
    python train.py --config toy_hyperparameters.py

    # Override specific params from CLI
    python train.py --config toy_hyperparameters.py --lr 5e-4 --n_epochs 200

    # Run without config (all defaults or CLI args)
    python train.py --experiment A --n_epochs 50
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_generation import get_dataloaders, get_real_dataloaders
from toy_models import (
    SwiGLUMLP,
    LowRankSwiGLUMLP,
    PureABSwiGLUMLP,
    compute_sparsity,
    compute_S_sparsity,
    compute_effective_sparsity,
    compute_param_efficiency,
    count_parameters,
    magnitude_prune,
)


# =============================================================================
# Training Utilities
# =============================================================================

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def l1_regularization(model: nn.Module) -> torch.Tensor:
    """Compute L1 regularization over all model weights."""
    l1_loss = 0.0
    for p in model.parameters():
        l1_loss = l1_loss + p.abs().sum()
    return l1_loss


def l1_regularization_S_only(model: LowRankSwiGLUMLP) -> torch.Tensor:
    """Compute L1 regularization only on S matrices (sparse residuals)."""
    l1_loss = 0.0
    for S in model.get_S_matrices():
        l1_loss = l1_loss + S.abs().sum()
    return l1_loss


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    device: torch.device,
    l1_weight: float,
    use_S_only_regularization: bool = False,
    prune_threshold: float = 0.0,
    prune_every_n: int = 0,
    global_step: int = 0,
) -> Tuple[float, float, int]:
    """Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device
        l1_weight: L1 regularization weight
        use_S_only_regularization: If True, only regularize S matrices
        prune_threshold: If > 0, zero out params below this magnitude
        prune_every_n: Apply pruning every N gradient steps (0 = disabled)
        global_step: Current global step count (updated and returned)
        
    Returns:
        Tuple of (avg_loss, avg_accuracy, updated_global_step)
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        logits = model(x)
        ce_loss = F.cross_entropy(logits, y)
        
        # L1 regularization
        if l1_weight > 0:
            if use_S_only_regularization and hasattr(model, 'get_S_matrices'):
                l1_loss = l1_regularization_S_only(model)
            else:
                l1_loss = l1_regularization(model)
            loss = ce_loss + l1_weight * l1_loss
        else:
            loss = ce_loss
        
        loss.backward()
        optimizer.step()
        global_step += 1
        
        # Magnitude pruning: zero out parameters below threshold every N steps
        # Also zeros Adam state so momentum doesn't immediately restore pruned weights
        if prune_every_n > 0 and prune_threshold > 0 and global_step % prune_every_n == 0:
            magnitude_prune(model, threshold=prune_threshold, optimizer=optimizer)
        
        total_loss += ce_loss.item()  # Track CE loss only for comparison
        total_acc += compute_accuracy(logits, y)
        n_batches += 1
    
    return total_loss / n_batches, total_acc / n_batches, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation set.
    
    Returns:
        Tuple of (avg_loss, avg_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        total_loss += loss.item()
        total_acc += compute_accuracy(logits, y)
        n_batches += 1
    
    return total_loss / n_batches, total_acc / n_batches


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    l1_weight: float = 0.0,
    use_S_only_regularization: bool = False,
    prune_threshold: float = 0.0,
    prune_every_n: int = 0,
    verbose: bool = True,
) -> Dict:
    """Train a model and return results.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device
        n_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: AdamW weight decay
        l1_weight: L1 regularization weight
        use_S_only_regularization: If True, only regularize S matrices
        prune_threshold: Zero out params with abs < threshold (0 = disabled)
        prune_every_n: Apply pruning every N gradient steps (0 = disabled)
        verbose: Show progress bar
    
    Returns:
        Dict with training history and final metrics
    """
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "sparsity": [],
    }
    
    best_val_acc = 0.0
    global_step = 0
    
    pbar = tqdm(range(n_epochs), disable=not verbose)
    for epoch in pbar:
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, optimizer, device,
            l1_weight=l1_weight,
            use_S_only_regularization=use_S_only_regularization,
            prune_threshold=prune_threshold,
            prune_every_n=prune_every_n,
            global_step=global_step,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        
        # Compute sparsity using true param efficiency:
        # Dense: 1 - nnz(W)/numel(W)
        # AB+S:  1 - (|A|+|B|+nnz(S))/dim(S)
        param_eff = compute_param_efficiency(model)
        sparsity = param_eff["param_sparsity"]
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["sparsity"].append(sparsity)
        
        best_val_acc = max(best_val_acc, val_acc)
        
        postfix = {
            "loss": f"{val_loss:.4f}",
            "acc": f"{val_acc:.2%}",
            "sparse": f"{sparsity:.1%}",
            "active": f"{param_eff['active_params']:,}",
            "step": global_step,
        }
        # Show S sparsity for LowRank models
        if hasattr(model, 'get_S_matrices'):
            s_sp = compute_S_sparsity(model)
            postfix["S_sp"] = f"{s_sp:.1%}"
        pbar.set_postfix(postfix)
    
    # Final metrics from the last epoch
    final_sparsity = history["sparsity"][-1]
    final_val_loss = history["val_loss"][-1]
    final_val_acc = history["val_acc"][-1]
    
    # Compute S sparsity (meaningful for LowRank models, 0 for dense)
    s_sparsity = compute_S_sparsity(model)
    
    # Compute true parameter efficiency
    param_eff = compute_param_efficiency(model)
    
    return {
        "history": history,
        "final_sparsity": final_sparsity,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "best_val_acc": best_val_acc,
        "n_parameters": count_parameters(model),
        "S_sparsity": s_sparsity,
        "active_params": param_eff["active_params"],
        "dense_params": param_eff["dense_params"],
        "compression_ratio": param_eff["compression_ratio"],
        "param_sparsity": param_eff["param_sparsity"],
        "model": model,  # Return the trained model for saving
    }


# =============================================================================
# Experiments
# =============================================================================

def run_experiment_A(
    train_loader,
    val_loader,
    device: torch.device,
    l1_weights: List[float],
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    prune_threshold: float = 1e-4,
    prune_every_n: int = 50,
    input_dim: int = 512,
    hidden_dim: int = 1024,
    n_classes: int = 20,
    output_dir: str = "toy_data",
) -> List[Dict]:
    """Experiment A: SwiGLUMLP with L1 regularization on all weights.
    
    Also applies magnitude pruning every N gradient steps.
    
    Args:
        l1_weights: List of L1 regularization strengths to try
        weight_decay: AdamW weight decay
        prune_threshold: Zero out params with abs < threshold
        prune_every_n: Apply pruning every N gradient steps (0 = disabled)
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        n_classes: Number of output classes
        
    Returns:
        List of results for each L1 weight
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT A: SwiGLUMLP with L1 regularization + magnitude pruning")
    print(f"  Model: input={input_dim}, hidden={hidden_dim}, classes={n_classes}")
    print(f"  Prune threshold: {prune_threshold}, every {prune_every_n} steps")
    print("=" * 60)
    
    results = []
    
    for l1_weight in l1_weights:
        print(f"\n--- L1 weight: {l1_weight} ---")
        
        model = SwiGLUMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
        )
        
        result = train_model(
            model, train_loader, val_loader, device,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            l1_weight=l1_weight,
            use_S_only_regularization=False,
            prune_threshold=prune_threshold,
            prune_every_n=prune_every_n,
        )
        # Extract and save model, then remove from result dict for JSON serialization
        trained_model = result.pop("model")
        
        result["l1_weight"] = l1_weight
        result["prune_threshold"] = prune_threshold
        result["prune_every_n"] = prune_every_n
        result["experiment"] = "A"
        result["model_type"] = "SwiGLUMLP"
        
        # Save model weights
        model_save_path = Path(output_dir) / f"exp_a_l1_{l1_weight}.pt" if output_dir else None
        if model_save_path:
            torch.save(trained_model.state_dict(), model_save_path)
            result["model_path"] = str(model_save_path)
            print(f"  Saved model to {model_save_path}")
        
        print(f"Final: loss={result['final_val_loss']:.4f}, "
              f"acc={result['final_val_acc']:.2%}, "
              f"sparsity={result['final_sparsity']:.1%}, "
              f"active_params={result['active_params']:,}/{result['dense_params']:,} "
              f"({result['param_sparsity']:.1%} param reduction)")
        
        results.append(result)
    
    return results


def run_experiment_B(
    train_loader,
    val_loader,
    device: torch.device,
    l1_weights: List[float],
    rank: int = 32,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    prune_threshold: float = 1e-4,
    prune_every_n: int = 50,
    input_dim: int = 512,
    hidden_dim: int = 1024,
    n_classes: int = 20,
    output_dir: str = "toy_data",
) -> List[Dict]:
    """Experiment B: LowRankSwiGLUMLP with L1 on S matrices only.
    
    Also applies magnitude pruning every N gradient steps.
    
    Args:
        l1_weights: List of L1 regularization strengths for S matrices
        rank: Rank for the AB decomposition
        weight_decay: AdamW weight decay
        prune_threshold: Zero out params with abs < threshold
        prune_every_n: Apply pruning every N gradient steps (0 = disabled)
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        n_classes: Number of output classes
        
    Returns:
        List of results for each L1 weight
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: LowRankSwiGLUMLP (AB + S) with L1 on S only + magnitude pruning")
    print(f"  Model: input={input_dim}, hidden={hidden_dim}, classes={n_classes}, rank={rank}")
    print(f"  Prune threshold: {prune_threshold}, every {prune_every_n} steps")
    print("=" * 60)
    
    results = []
    
    for l1_weight in l1_weights:
        print(f"\n--- L1 weight (S only): {l1_weight}, rank: {rank} ---")
        
        model = LowRankSwiGLUMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            rank=rank,
        )
        
        result = train_model(
            model, train_loader, val_loader, device,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            l1_weight=l1_weight,
            use_S_only_regularization=True,
            prune_threshold=prune_threshold,
            prune_every_n=prune_every_n,
        )
        # Extract and save model, then remove from result dict for JSON serialization
        trained_model = result.pop("model")
        
        result["l1_weight"] = l1_weight
        result["rank"] = rank
        result["prune_threshold"] = prune_threshold
        result["prune_every_n"] = prune_every_n
        result["experiment"] = "B"
        result["model_type"] = "LowRankSwiGLUMLP"
        
        # Save model weights
        model_save_path = Path(output_dir) / f"exp_b_l1_{l1_weight}_rank_{rank}.pt" if output_dir else None
        if model_save_path:
            torch.save(trained_model.state_dict(), model_save_path)
            result["model_path"] = str(model_save_path)
            print(f"  Saved model to {model_save_path}")
        
        print(f"Final: loss={result['final_val_loss']:.4f}, "
              f"acc={result['final_val_acc']:.2%}, "
              f"S_sparsity={result['S_sparsity']:.1%}, "
              f"active_params={result['active_params']:,}/{result['dense_params']:,} "
              f"({result['param_sparsity']:.1%} param reduction)")
        
        results.append(result)
    
    return results


def run_experiment_C(
    train_loader,
    val_loader,
    device: torch.device,
    ranks: List[int] = None,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    input_dim: int = 512,
    hidden_dim: int = 1024,
    n_classes: int = 20,
    output_dir: str = "toy_data",
) -> List[Dict]:
    """Experiment C: Pure low-rank AB (no sparse residual), sweep over ranks.
    
    Each weight matrix W = A @ B with rank r. No L1, no pruning — sparsity
    is purely structural: 1 - (|A| + |B|) / (out * in).
    
    Args:
        ranks: List of rank values to sweep
        n_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: AdamW weight decay
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        n_classes: Number of output classes
        output_dir: Directory to save model checkpoints
        
    Returns:
        List of results for each rank
    """
    if ranks is None:
        ranks = [4, 8, 16, 32, 64, 128, 256]
    
    print("\n" + "=" * 60)
    print("EXPERIMENT C: PureABSwiGLUMLP — low-rank AB, rank sweep")
    print(f"  Model: input={input_dim}, hidden={hidden_dim}, classes={n_classes}")
    print(f"  Ranks to sweep: {ranks}")
    print("=" * 60)
    
    results = []
    
    for rank in ranks:
        print(f"\n--- Rank: {rank} ---")
        
        model = PureABSwiGLUMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            rank=rank,
        )
        
        # No L1 regularization, no pruning — sparsity is structural
        result = train_model(
            model, train_loader, val_loader, device,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            l1_weight=0.0,
            use_S_only_regularization=False,
            prune_threshold=0.0,
            prune_every_n=0,
        )
        # Extract and save model
        trained_model = result.pop("model")
        
        result["rank"] = rank
        result["experiment"] = "C"
        result["model_type"] = "PureABSwiGLUMLP"
        
        # Save model weights
        model_save_path = Path(output_dir) / f"exp_c_rank_{rank}.pt" if output_dir else None
        if model_save_path:
            torch.save(trained_model.state_dict(), model_save_path)
            result["model_path"] = str(model_save_path)
            print(f"  Saved model to {model_save_path}")
        
        print(f"Final: loss={result['final_val_loss']:.4f}, "
              f"acc={result['final_val_acc']:.2%}, "
              f"active_params={result['active_params']:,}/{result['dense_params']:,} "
              f"({result['param_sparsity']:.1%} param reduction)")
        
        results.append(result)
    
    return results


# =============================================================================
# Main
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load CONFIG dict from a Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def main():
    parser = argparse.ArgumentParser(description="Run sparsity experiments")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config .py file (e.g. toy_hyperparameters.py)")
    
    # All CLI args act as overrides on top of config file defaults
    parser.add_argument("--experiment", type=str, choices=["A", "B", "C", "all"], default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--prune_threshold", type=float, default=None)
    parser.add_argument("--prune_every_n", type=int, default=None)
    args = parser.parse_args()
    
    # Load config file if provided, otherwise use defaults
    if args.config:
        print(f"Loading config from {args.config}")
        cfg = load_config(args.config)
    else:
        cfg = {}
    
    # Resolve parameters: CLI override > config file > hardcoded default
    def resolve(cli_val, cfg_key, default):
        if cli_val is not None:
            return cli_val
        return cfg.get(cfg_key, default)
    
    # Data source: "real" (pre-computed embeddings) or "synthetic" (random network)
    data_source = cfg.get("data_source", "synthetic")
    
    # Synthetic data generation params (only used when data_source == "synthetic")
    data_input_dim = cfg.get("data_input_dim", 512)
    data_hidden_dim = cfg.get("data_hidden_dim", 1024)
    data_n_layers = cfg.get("data_n_layers", 3)
    data_n_classes = cfg.get("data_n_classes", 20)
    
    # Model params
    model_input_dim = cfg.get("model_input_dim", data_input_dim)
    model_hidden_dim = cfg.get("model_hidden_dim", 1024)
    model_n_classes = cfg.get("model_n_classes", data_n_classes)
    
    # Training params
    experiment = resolve(args.experiment, "experiment", "all")
    n_epochs = resolve(args.n_epochs, "n_epochs", 100)
    lr = resolve(args.lr, "lr", 1e-3)
    batch_size = resolve(args.batch_size, "batch_size", 128)
    rank = resolve(args.rank, "rank", 32)
    n_train = resolve(args.n_train, "data_n_train", 10000)
    n_val = resolve(args.n_val, "data_n_val", 2000)
    seed = resolve(args.seed, "data_seed", 42)
    output_dir = resolve(args.output_dir, "output_dir", "toy_data")
    prune_threshold = resolve(args.prune_threshold, "prune_threshold", 1e-4)
    prune_every_n = resolve(args.prune_every_n, "prune_every_n", 50)
    l1_weights = cfg.get("l1_weights", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    weight_decay = cfg.get("weight_decay", 0.01)
    real_data_dir = cfg.get("real_data_dir", "real_data")
    rank_sweep = cfg.get("rank_sweep", [4, 8, 16, 32, 64, 128, 256])
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(seed)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data based on source
    if data_source == "real":
        print(f"\n--- Configuration (real data) ---")
        print(f"  Data source: {data_source} ({real_data_dir}/)")
        
        train_loader, val_loader, meta = get_real_dataloaders(
            data_dir=real_data_dir,
            batch_size=batch_size,
            n_train=n_train if n_train != 10000 else None,  # Only subsample if explicitly set
            n_val=n_val if n_val != 2000 else None,
            seed=seed,
        )
        
        # Override model dims from the actual data
        model_input_dim = meta.get("embed_dim", model_input_dim)
        model_n_classes = meta.get("n_classes", model_n_classes)
        
        print(f"  Model: input_dim={model_input_dim}, hidden_dim={model_hidden_dim}, "
              f"n_classes={model_n_classes}, rank={rank}")
        print(f"  Training: epochs={n_epochs}, lr={lr}, batch_size={batch_size}, wd={weight_decay}")
        print(f"  Sparsity: l1_weights={l1_weights}")
        print(f"  Pruning: threshold={prune_threshold}, every_n={prune_every_n}")
    else:
        print(f"\n--- Configuration (synthetic data) ---")
        print(f"  Data: input_dim={data_input_dim}, hidden_dim={data_hidden_dim}, "
              f"n_layers={data_n_layers}, n_classes={data_n_classes}")
        print(f"  Model: input_dim={model_input_dim}, hidden_dim={model_hidden_dim}, "
              f"n_classes={model_n_classes}, rank={rank}")
        print(f"  Training: epochs={n_epochs}, lr={lr}, batch_size={batch_size}, wd={weight_decay}")
        print(f"  Sparsity: l1_weights={l1_weights}")
        print(f"  Pruning: threshold={prune_threshold}, every_n={prune_every_n}")
        print(f"  Data: n_train={n_train}, n_val={n_val}, seed={seed}")
        
        print("\nGenerating synthetic data...")
        train_loader, val_loader, _ = get_dataloaders(
            input_dim=data_input_dim,
            hidden_dim=data_hidden_dim,
            n_layers=data_n_layers,
            n_classes=data_n_classes,
            n_train=n_train,
            n_val=n_val,
            batch_size=batch_size,
            seed=seed,
        )
    
    print(f"  Output: {output_dir}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    all_results = {}
    
    # Run experiments
    if experiment in ["A", "all"]:
        results_A = run_experiment_A(
            train_loader, val_loader, device,
            l1_weights=l1_weights,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            prune_threshold=prune_threshold,
            prune_every_n=prune_every_n,
            input_dim=model_input_dim,
            hidden_dim=model_hidden_dim,
            n_classes=model_n_classes,
            output_dir=str(output_dir),
        )
        all_results["A"] = results_A
        
        # Save results (filter non-serializable keys)
        with open(output_dir / "experiment_a_results.json", "w") as f:
            serializable = []
            for r in results_A:
                d = {k: v for k, v in r.items() if k != "history"}
                # Handle inf/nan for JSON
                if d.get("compression_ratio") == float('inf'):
                    d["compression_ratio"] = None
                serializable.append(d)
            json.dump(serializable, f, indent=2)
    
    if experiment in ["B", "all"]:
        results_B = run_experiment_B(
            train_loader, val_loader, device,
            l1_weights=l1_weights,
            rank=rank,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            prune_threshold=prune_threshold,
            prune_every_n=prune_every_n,
            input_dim=model_input_dim,
            hidden_dim=model_hidden_dim,
            n_classes=model_n_classes,
            output_dir=str(output_dir),
        )
        all_results["B"] = results_B
        
        # Save results (filter non-serializable keys)
        with open(output_dir / "experiment_b_results.json", "w") as f:
            serializable = []
            for r in results_B:
                d = {k: v for k, v in r.items() if k != "history"}
                if d.get("compression_ratio") == float('inf'):
                    d["compression_ratio"] = None
                serializable.append(d)
            json.dump(serializable, f, indent=2)
    
    if experiment in ["C", "all"]:
        results_C = run_experiment_C(
            train_loader, val_loader, device,
            ranks=rank_sweep,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            input_dim=model_input_dim,
            hidden_dim=model_hidden_dim,
            n_classes=model_n_classes,
            output_dir=str(output_dir),
        )
        all_results["C"] = results_C
        
        if results_C:
            with open(output_dir / "experiment_c_results.json", "w") as f:
                serializable = []
                for r in results_C:
                    d = {k: v for k, v in r.items() if k != "history"}
                    if d.get("compression_ratio") == float('inf'):
                        d["compression_ratio"] = None
                    serializable.append(d)
                json.dump(serializable, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for exp_name, results in all_results.items():
        if results:
            print(f"\nExperiment {exp_name}:")
            for r in results:
                extra = ""
                if r.get("S_sparsity", 0) > 0:
                    extra += f", S_sparsity={r['S_sparsity']:.1%}"
                extra += f", active={r.get('active_params', '?'):,}/{r.get('dense_params', '?'):,}"
                # Use rank as the sweep variable for Experiment C
                if r.get("experiment") == "C":
                    label = f"rank={r.get('rank', '?')}"
                else:
                    label = f"L1={r.get('l1_weight', 'N/A')}"
                print(f"  {label}: "
                      f"acc={r['final_val_acc']:.2%}, "
                      f"sparsity={r.get('param_sparsity', r.get('final_sparsity', 0)):.1%}"
                      f"{extra}")
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
