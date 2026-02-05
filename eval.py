"""Evaluation script for trained models."""

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import get_dataloader
from model import create_model


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model and return metrics.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader
        device: Device
        max_batches: Max batches to evaluate (None = all)
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        logits = model(input_ids, attention_mask)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )
        
        # Count valid tokens
        n_tokens = attention_mask.sum().item()
        
        total_loss += loss.item()
        total_tokens += n_tokens
        n_batches += 1
        
    avg_loss = total_loss / total_tokens
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
    parser = argparse.ArgumentParser(description="Evaluate model on FineWeb")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    args = parser.parse_args()
    
    device = torch.device("cuda")
    
    # Load model
    print("Loading model...")
    model = create_model(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )
    raise NotImplementedError("Evaluation currently only evaluates a random init model")
    
    # Data
    print("Loading validation data...")
    val_loader = get_dataloader(split="test", batch_size=args.batch_size, streaming=True)
    
    # Evaluate
    print("Evaluating...")
    metrics = evaluate(model, val_loader, device)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Validation Loss:    {metrics['val_loss']:.4f}")
    print(f"Perplexity:         {metrics['perplexity']:.2f}")
    print(f"Total Parameters:   {metrics['total_params']:,}")
    print(f"Non-zero Params:    {metrics['nonzero_params']:,}")
    print(f"Sparsity:           {metrics['sparsity']:.2%}")
    print(f"Evaluated Tokens:   {metrics['n_tokens']:,}")
    print("=" * 50)
    
    # Check goal
    if metrics['val_loss'] < 3.0:
        print(f"✅ GOAL ACHIEVED: val_loss={metrics['val_loss']:.4f} < 3.0")
        print(f"   with only {metrics['nonzero_params']:,} non-zero parameters!")
    else:
        print(f"❌ Goal not yet achieved: val_loss={metrics['val_loss']:.4f} >= 3.0")
        print(f"   Keep optimizing your model architecture!")
    
    return metrics


if __name__ == "__main__":
    main()

