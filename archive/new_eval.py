"""Evaluation script with memory access reporting.

Loads a checkpoint, evaluates validation loss, and reports memory accesses in MB.

Usage:
    python new_eval.py --checkpoint checkpoint_weightless.pt --batch_size 32
    python new_eval.py --checkpoint checkpoint_weightless.pt --batch_size 64 --max_batches 100
"""

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import get_dataloader
from model_test import create_model, SEQ_LEN


PAD_TOKEN_ID = 50256  # GPT-2 <|endoftext|> used as pad token


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Tuple of (model, config, step)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError("Checkpoint does not contain 'config' key. "
                        "Make sure to use a checkpoint saved with the updated train.py")
    
    # Get step if available
    step = checkpoint.get("step", None)
    
    # Create model with config
    model = create_model(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
    )
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume checkpoint is just state dict
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config, step


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device, max_batches: int = None):
    """Evaluate model on validation set.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader for validation set
        device: Device
        max_batches: Maximum batches to evaluate (None = all)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    
    desc = f"Evaluating (max {max_batches} batches)" if max_batches else "Evaluating"
    
    for batch in tqdm(dataloader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
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
        n_batches += 1
        
        if max_batches and n_batches >= max_batches:
            break
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # Count parameters
    total_params = model.count_parameters(count_zeros=True)
    nonzero_params = model.count_parameters(count_zeros=False)
    sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
    
    return {
        "val_loss": avg_loss,
        "perplexity": perplexity,
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "sparsity": sparsity,
        "n_batches": n_batches,
        "n_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with memory access reporting")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Maximum batches to evaluate (None = all)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load checkpoint
    model, config, step = load_checkpoint(args.checkpoint, device)
    
    print(f"Model config: d_model={config['d_model']}, n_layers={config['n_layers']}, "
          f"n_heads={config['n_heads']}, d_ff={config['d_ff']}")
    if step is not None:
        print(f"Checkpoint from step: {step}")
    
    # Load validation data
    print("\nLoading validation data...")
    val_loader = get_dataloader(
        split="test",
        batch_size=args.batch_size,
        streaming=True,
    )
    
    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluate(model, val_loader, device, max_batches=args.max_batches)
    
    # Compute memory access
    memory_mb = model.count_memory_mb(
        batch_size=args.batch_size,
        seq_len=SEQ_LEN,
        dtype=torch.bfloat16,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint:         {args.checkpoint}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Batches evaluated:  {metrics['n_batches']}")
    print(f"Tokens evaluated:   {metrics['n_tokens']:,}")
    print("-" * 60)
    print(f"Validation Loss:    {metrics['val_loss']:.4f}")
    print(f"Perplexity:         {metrics['perplexity']:.2f}")
    print("-" * 60)
    print(f"Total Parameters:   {metrics['total_params']:,}")
    print(f"Non-zero Params:    {metrics['nonzero_params']:,}")
    print(f"Sparsity:           {metrics['sparsity']:.2%}")
    print("-" * 60)
    print(f"Memory Access:      {memory_mb:.2f} MB per forward pass")
    print(f"                    (batch_size={args.batch_size}, seq_len={SEQ_LEN})")
    print("=" * 60)
    
    # Goal check
    if metrics['val_loss'] < 3.0:
        print(f"\nGOAL ACHIEVED: val_loss={metrics['val_loss']:.4f} < 3.0")
        print(f"  with {metrics['nonzero_params']:,} non-zero parameters")
        print(f"  and {memory_mb:.2f} MB memory access per forward pass")
    else:
        print(f"\nGoal not yet achieved: val_loss={metrics['val_loss']:.4f} >= 3.0")
    
    return metrics, memory_mb


if __name__ == "__main__":
    main()
