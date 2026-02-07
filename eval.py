"""Evaluation script for FinalSparseTransformer checkpoints.

Loads a saved checkpoint, optionally prunes near-zero weights, evaluates on
the FineWeb validation set, and computes per-token active nonzero parameter
counts with segment breakdown.  Results are appended to a persistent CSV file.

Two evaluation modes (controlled by --full_vocab):
  Default (block_decode): clustered two-stage head.  Reports loss over
      reachable tokens only, plus coverage.
  --full_vocab:  standard full-vocabulary softmax (F.linear).  Reports the
      true cross-entropy over all 50 257 vocab entries.

Usage:
    python eval.py --checkpoint saved_models/final_sparse_v1_step2000.pt
    python eval.py --checkpoint saved_models/final_sparse_v1_step2000.pt --full_vocab
    python eval.py --checkpoint saved_models/final_sparse_v1_step2000.pt \
                   --prune_threshold 1e-3 --max_batches 50
"""

import argparse
import csv
import math
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import get_dataloader
from model import create_model

EVAL_THRESHOLD = 4.0
head_k = 128


# ---------------------------------------------------------------------------
# Model reconstruction from checkpoint config
# ---------------------------------------------------------------------------

def build_model_kwargs(config: dict, block_decode: bool = True) -> dict:
    """Build model constructor kwargs from a saved config dict.

    Mirrors the FinalSparseTransformer branch in train.py.
    *block_decode* controls whether the clustered head is used at eval time.
    """
    kwargs = {
        "d_model": config["d_model"],
        "n_layers": config["n_layers"],
        "n_heads": config["n_heads"],
        "dropout": config.get("dropout", 0.1),
        "block_decode": block_decode,
        "head_k": head_k,
        "head_balance_tolerance": config.get("head_balance_tolerance", 10),
        # MoE FFN config
        "d_expert": config.get("d_expert", config["d_model"]),
        "num_experts": config.get("num_experts", 8),
        "top_k": config.get("top_k", 2),
        "aux_loss_weight": config.get("aux_loss_weight", 0.01),
        "z_loss_weight": config.get("z_loss_weight", 0.001),
        # Sparsity config (stored but not used at eval; needed for constructor)
        "l1_lambda": config.get("l1_lambda", 0.0),
        "ff_l1_lambda": config.get("ff_l1_lambda", 0.0),
        "prune_threshold": config.get("prune_threshold", 0.0),
        "prune_every_n": config.get("prune_every_n", 0),
        # Disable gradient checkpointing for eval
        "use_gradient_checkpointing": False,
    }
    return kwargs


# ---------------------------------------------------------------------------
# Post-load weight pruning
# ---------------------------------------------------------------------------

def prune_weights(model: torch.nn.Module, threshold: float):
    """Zero out all parameter entries with magnitude below *threshold*."""
    if threshold <= 0:
        return
    with torch.no_grad():
        for param in model.parameters():
            mask = param.data.abs() < threshold
            param.data[mask] = 0.0


# ---------------------------------------------------------------------------
# Evaluation — block_decode mode (clustered head, with coverage)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_block_decode(model, dataloader, device, max_batches=None):
    """Evaluate using the clustered two-stage head.

    Logits for vocab entries outside the selected clusters are -inf.
    Loss is computed only over *reachable* tokens (where the true label had
    a finite logit).  Coverage reports the fraction of reachable tokens.
    """
    model.eval()

    total_loss = 0.0
    total_reachable = 0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating (block_decode)"):
        if max_batches is not None and n_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)

        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )

        attn_mask_flat = attention_mask.reshape(-1).bool()
        per_token_loss = per_token_loss[attn_mask_flat]

        finite_mask = torch.isfinite(per_token_loss)
        total_loss += per_token_loss[finite_mask].sum().item()
        total_reachable += int(finite_mask.sum().item())

        n_tokens = int(attn_mask_flat.sum().item())
        total_tokens += n_tokens

        preds = logits.argmax(dim=-1)
        mask = attention_mask.bool()
        total_correct += (preds[mask] == labels[mask]).sum().item()

        n_batches += 1

    avg_loss = total_loss / total_reachable if total_reachable > 0 else float("inf")
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    coverage = total_reachable / total_tokens if total_tokens > 0 else 0.0

    return {
        "val_loss": avg_loss,
        "accuracy": accuracy,
        "coverage": coverage,
        "perplexity": math.exp(avg_loss) if math.isfinite(avg_loss) else float("inf"),
        "n_batches": n_batches,
        "n_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Evaluation — full vocab mode (standard F.linear head)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_full_vocab(model, dataloader, device, max_batches=None):
    """Evaluate using the full-vocabulary softmax (no clustering).

    Uses F.linear(x, weight, bias) directly — the same weight matrix, but
    scores all vocab entries.  Gives the true cross-entropy loss.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating (full_vocab)"):
        if max_batches is not None and n_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)

        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )

        attn_mask_flat = attention_mask.reshape(-1).bool()
        per_token_loss = per_token_loss[attn_mask_flat]

        total_loss += per_token_loss.sum().item()
        n_tokens = int(attn_mask_flat.sum().item())
        total_tokens += n_tokens

        preds = logits.argmax(dim=-1)
        mask = attention_mask.bool()
        total_correct += (preds[mask] == labels[mask]).sum().item()

        n_batches += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return {
        "val_loss": avg_loss,
        "accuracy": accuracy,
        "coverage": 1.0,  # full vocab — always 100%
        "perplexity": math.exp(avg_loss) if math.isfinite(avg_loss) else float("inf"),
        "n_batches": n_batches,
        "n_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Active nonzero params per token
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_active_params_per_token(model) -> dict:
    """Count the nonzero parameters activated per single token.

    Segments:
        attention   – M and V matrices across all layers
        ffn         – MoE router + expert weights (scaled by top_k/num_experts)
        ln_embed    – all LayerNorms + one embedding row + final LN
        output_head – GroupedHead centroids + selected weight rows (block_decode)

    Returns dict with total_active, segment counts, percentages, and bytes.
    """
    d_model = model.d_model
    n_layers = model.n_layers

    # Accumulators for the four segments
    seg_attention = 0
    seg_ffn = 0
    seg_ln_embed = 0
    seg_output_head = 0

    # ------------------------------------------------------------------
    # Per layer (summed across all layers)
    # ------------------------------------------------------------------
    for layer in model.layers:
        # LayerNorm 1 (always dense: weight + bias)
        seg_ln_embed += layer.ln1.weight.numel() + layer.ln1.bias.numel()

        # Sparse Dyadic Attention: M and V matrices (count nonzero)
        seg_attention += int((layer.attn.M != 0).sum().item())
        seg_attention += int((layer.attn.V != 0).sum().item())

        # LayerNorm 2
        seg_ln_embed += layer.ln2.weight.numel() + layer.ln2.bias.numel()

        # MoE Router (always fully read, treated as dense)
        seg_ffn += layer.moe.router.weight.numel()

        # MoE Experts: nonzero params * top_k / num_experts
        moe = layer.moe
        num_experts = len(moe.experts)
        top_k_experts = moe.top_k
        expert_nz = 0
        for expert in moe.experts:
            expert_nz += int((expert.w1.weight != 0).sum().item())
            expert_nz += int((expert.w2.weight != 0).sum().item())
            expert_nz += int((expert.w3.weight != 0).sum().item())
        seg_ffn += expert_nz * top_k_experts / num_experts

    # ------------------------------------------------------------------
    # Once (not per layer)
    # ------------------------------------------------------------------

    # Embedding table: one row lookup per token
    seg_ln_embed += d_model

    # Final LayerNorm
    seg_ln_embed += model.ln_f.weight.numel() + model.ln_f.bias.numel()

    # Output head (GroupedHead with block_decode)
    head = model.head
    if head.block_decode and head._clusters_valid:
        # Centroid routing: full centroid matrix is read
        seg_output_head += int((head.centroids != 0).sum().item())

        # Selected weight rows: average nonzero per cluster * k
        n_clusters = head.centroids.shape[0]
        cluster_sizes = head.cluster_sizes
        organized_weight = head.organized_weight

        total_cluster_nz = 0
        for c in range(n_clusters):
            cs = int(cluster_sizes[c].item())
            total_cluster_nz += int((organized_weight[c, :cs] != 0).sum().item())
        avg_nz_per_cluster = total_cluster_nz / n_clusters
        seg_output_head += avg_nz_per_cluster * head.k
    else:
        # Fallback: full head weight matrix
        seg_output_head += int((head.weight != 0).sum().item())
        if head.bias is not None:
            seg_output_head += int((head.bias != 0).sum().item())

    # ------------------------------------------------------------------
    # Totals and percentages
    # ------------------------------------------------------------------
    total_active = seg_attention + seg_ffn + seg_ln_embed + seg_output_head

    result = {
        "total_active": total_active,
        "seg_attention": seg_attention,
        "seg_ffn": seg_ffn,
        "seg_ln_embed": seg_ln_embed,
        "seg_output_head": seg_output_head,
    }

    if total_active > 0:
        result["pct_attention"] = seg_attention / total_active * 100
        result["pct_ffn"] = seg_ffn / total_active * 100
        result["pct_ln_embed"] = seg_ln_embed / total_active * 100
        result["pct_output_head"] = seg_output_head / total_active * 100
    else:
        result["pct_attention"] = 0.0
        result["pct_ffn"] = 0.0
        result["pct_ln_embed"] = 0.0
        result["pct_output_head"] = 0.0

    return result


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "run_name",
    "checkpoint",
    "step",
    "eval_mode",
    "val_loss",
    "accuracy",
    "coverage",
    "perplexity",
    "total_params",
    "nonzero_params",
    "active_params_per_token",
    "active_MB",
    "pct_attention",
    "pct_ffn",
    "pct_ln_embed",
    "pct_output_head",
    "dtype",
    "timestamp",
]


def append_csv(path: str, row: dict):
    """Append a single result row to *path*, creating the file if needed."""
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a FinalSparseTransformer checkpoint on FineWeb",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to saved .pt checkpoint",
    )
    parser.add_argument(
        "--full_vocab", action="store_true", default=False,
        help="Use full-vocabulary softmax instead of block_decode clustering",
    )
    parser.add_argument(
        "--prune_threshold", type=float, default=5e-3,
        help="Post-load pruning: zero out |w| < threshold (0 = disabled)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--max_batches", type=int, default=None,
        help="Cap number of eval batches (None = all)",
    )
    parser.add_argument(
        "--results_csv", type=str, default="eval_results.csv",
        help="Path to CSV file for appending results",
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Name for this eval run (defaults to checkpoint filename)",
    )
    args = parser.parse_args()

    use_block_decode = not args.full_vocab
    eval_mode = "block_decode" if use_block_decode else "full_vocab"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or os.path.splitext(os.path.basename(args.checkpoint))[0]

    # ------------------------------------------------------------------
    # Stage 1-2: Load checkpoint and reconstruct model
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    step = checkpoint.get("step", 0)

    model_class = config.get("model_class", "")
    assert model_class == "FinalSparseTransformer", (
        f"eval.py currently only supports FinalSparseTransformer, got: {model_class}"
    )

    model_kwargs = build_model_kwargs(config, block_decode=use_block_decode)
    print(f"Reconstructing {model_class} (d_model={config['d_model']}, "
          f"n_layers={config['n_layers']}, n_heads={config['n_heads']}, "
          f"num_experts={config.get('num_experts', 8)}, top_k={config.get('top_k', 2)}, "
          f"head_k={config.get('head_k', 1)}, eval_mode={eval_mode})")

    model = create_model(model_class, **model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).to(torch.bfloat16)
    print(f"  Model loaded (step {step})")

    # ------------------------------------------------------------------
    # Stage 3: Post-load pruning
    # ------------------------------------------------------------------
    if args.prune_threshold > 0:
        print(f"Pruning weights with |w| < {args.prune_threshold} ...")
        prune_weights(model, args.prune_threshold)
        print("  Pruning done.")

    # ------------------------------------------------------------------
    # Stage 4: Evaluate
    # ------------------------------------------------------------------
    bytes_per_param = 2  # BF16
    dtype_str = "bfloat16"

    print("Loading validation data...")
    val_loader = get_dataloader(split="test", batch_size=args.batch_size, streaming=True)

    print(f"Evaluating ({eval_mode})...")
    if use_block_decode:
        metrics = evaluate_block_decode(model, val_loader, device, max_batches=args.max_batches)
    else:
        metrics = evaluate_full_vocab(model, val_loader, device, max_batches=args.max_batches)

    # Global parameter counts
    total_params = model.count_parameters(count_zeros=True)
    nonzero_params = model.count_parameters(count_zeros=False)
    sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0

    # ------------------------------------------------------------------
    # Stage 5: Active nonzero params per token
    # ------------------------------------------------------------------
    print("Computing active params per token...")
    active = compute_active_params_per_token(model)
    active_mb = active["total_active"] * bytes_per_param / 1e6

    # ------------------------------------------------------------------
    # Stage 6: Print results and save CSV
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Run name:              {run_name}")
    print(f"  Checkpoint:            {args.checkpoint}")
    print(f"  Step:                  {step}")
    print(f"  Eval mode:             {eval_mode}")
    print(f"  Dtype:                 {dtype_str}")
    if args.prune_threshold > 0:
        print(f"  Post-load prune:       |w| < {args.prune_threshold}")
    print("-" * 60)
    print(f"  Val Loss:              {metrics['val_loss']:.4f}")
    print(f"  Accuracy (top-1):      {metrics['accuracy']:.4%}")
    if use_block_decode:
        print(f"  Coverage:              {metrics['coverage']:.4%}")
    print(f"  Perplexity:            {metrics['perplexity']:.2f}")
    print(f"  Eval batches:          {metrics['n_batches']}")
    print(f"  Eval tokens:           {metrics['n_tokens']:,}")
    print("-" * 60)
    print(f"  Total params:          {total_params:,}")
    print(f"  Nonzero params:        {nonzero_params:,}")
    print(f"  Sparsity:              {sparsity:.2%}")
    print("-" * 60)
    print(f"  Active params/token:   {active['total_active']:,.0f}")
    print(f"  Active MB:             {active_mb:.2f} MB")
    print("-" * 60)
    print("  Segment breakdown (active params per token):")
    print(f"    Attention (M, V):    {active['seg_attention']:>12,.0f}  ({active['pct_attention']:.1f}%)")
    print(f"    FFN (router+experts):{active['seg_ffn']:>12,.0f}  ({active['pct_ffn']:.1f}%)")
    print(f"    LN / Embedding:      {active['seg_ln_embed']:>12,.0f}  ({active['pct_ln_embed']:.1f}%)")
    print(f"    Output Head (block): {active['seg_output_head']:>12,.0f}  ({active['pct_output_head']:.1f}%)")
    print("=" * 60)

    # Goal check
    if metrics["val_loss"] < EVAL_THRESHOLD:
        print(f"  GOAL ACHIEVED: val_loss={metrics['val_loss']:.4f} < {EVAL_THRESHOLD}")
        print(f"  Active params/token: {active['total_active']:,.0f}  |  Active MB: {active_mb:.2f}")
    else:
        print(f"  Goal not yet achieved: val_loss={metrics['val_loss']:.4f} >= {EVAL_THRESHOLD}")
    print()

    # Build CSV row
    csv_row = {
        "run_name": run_name,
        "checkpoint": args.checkpoint,
        "step": step,
        "eval_mode": eval_mode,
        "val_loss": f"{metrics['val_loss']:.6f}",
        "accuracy": f"{metrics['accuracy']:.6f}",
        "coverage": f"{metrics['coverage']:.6f}",
        "perplexity": f"{metrics['perplexity']:.4f}",
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "active_params_per_token": f"{active['total_active']:.0f}",
        "active_MB": f"{active_mb:.4f}",
        "pct_attention": f"{active['pct_attention']:.2f}",
        "pct_ffn": f"{active['pct_ffn']:.2f}",
        "pct_ln_embed": f"{active['pct_ln_embed']:.2f}",
        "pct_output_head": f"{active['pct_output_head']:.2f}",
        "dtype": dtype_str,
        "timestamp": datetime.now().isoformat(),
    }
    append_csv(args.results_csv, csv_row)
    print(f"Results appended to {args.results_csv}")

    return metrics


if __name__ == "__main__":
    main()
