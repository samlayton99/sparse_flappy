"""Sparse Dyadic Attention transformer configuration.

Uses block-dyadic attention with full-rank sparse M (metric) and V (value-map)
matrices per head. Two independent L1 regularization channels:
  - l1_lambda:    on attention M, V matrices (induces attention sparsity)
  - ff_l1_lambda: on FFN w1, w2, w3 weights (induces FFN sparsity)
These are mutually exclusive — no parameter is penalized by both.

Weight decay should NOT be applied to M and V (handled via separate optimizer
param groups with sparse_weight_decay=0.0).

Gradient checkpointing is enabled by default (required for batch_size >= 32
due to the larger activation footprint from d-dimensional heads).

Usage:
    torchrun --nproc_per_node=8 train.py \\
        --config hyperparameters/sparse_dyadic_model.py \\
        --run_name sparse_dyadic_v1 --save
"""

CONFIG = {
    # Model architecture
    "model_class": "SparseDyadicTransformer",
    "d_model": 2048,
    "n_layers": 16,
    "n_heads": 8,
    "d_ff": 4096,
    "dropout": 0.1,

    # --- Sparsity controls ---
    # L1 on attention M, V (metric + value-map matrices)
    "l1_lambda": 1e-6,          # L1 penalty strength (0 = disabled)
    # L1 on FFN weights w1, w2, w3 (mutually exclusive with l1_lambda)
    "ff_l1_lambda": 0.0,        # L1 penalty on FFN weights (0 = disabled)
    # Magnitude pruning (only applies to M, V)
    "prune_threshold": 1e-3,    # zero out |w| < threshold (0 = disabled)
    "prune_every_n": 100,       # prune every N steps (0 = disabled)
    # Optimizer group settings for M, V
    "sparse_weight_decay": 0.05, # weight decay for M, V (0 = let L1 handle it)
    "sparse_lr": None,          # None = use max_lr; set float to override

    # Gradient checkpointing (activation recomputation to save memory)
    "use_gradient_checkpointing": True,

    # Output head
    "block_decode": False,   # True = clustered two-stage decode at eval time
    "head_k": 1,             # Number of top-k clusters selected during eval decode
    "head_balance_tolerance": 10,  # Max ± deviation from target cluster size

    # Training parameters
    "batch_size": 64,
    "max_lr": 5e-4,
    "num_steps": 2000,
    "eval_every": 50,
    "warmup_steps": 500,
    "weight_decay": 0.3,        # applies to FFN, embeddings, LayerNorm only
}
