"""Final Sparse Transformer configuration.

Combines sparse dyadic attention (block-dyadic M/V matrices) with MoE FFN
(top-k expert routing) for maximum sparsity control and model capacity.

Two independent L1 regularization channels:
  - l1_lambda:     on attention M, V matrices (induces attention sparsity)
  - ff_l1_lambda:  on expert FFN w1, w2, w3 weights (induces FFN sparsity)
These are mutually exclusive — no parameter is penalized by both.

Weight decay is split into 5 mutually-exclusive optimizer groups:
  1. Sparse attention (M, V)           → attn_weight_decay, optional sparse_lr
  2. MoE router weights                → attn_weight_decay
  3. FFN expert weights (w1, w2, w3)   → ff_weight_decay
  4. Embeddings / output head          → embed_weight_decay
  5. LayerNorm & biases                → 0 (no decay)

Gradient checkpointing is enabled by default (required due to the large
activation footprint from d-dimensional attention heads).

Usage:
    torchrun --nproc_per_node=8 train.py \\
        --config hyperparameters/final_sparse.py \\
        --run_name final_sparse_v1
"""

CONFIG = {
    # Model architecture
    "model_class": "FinalSparseTransformer",
    "d_model": 768,
    "n_layers": 8,
    "n_heads": 8,
    "dropout": 0.1,

    # MoE FFN config
    "d_expert": 512,            # Hidden dim per expert
    "num_experts": 8,           # Number of expert FFNs
    "top_k": 2,                 # Experts activated per token
    "aux_loss_weight": 0.01,    # Load balancing loss weight
    "z_loss_weight": 0.001,     # Router z-loss weight (stabilizes training)

    # --- Sparsity controls ---
    # L1 on attention M, V (metric + value-map matrices)
    "l1_lambda": 3.5e-6,          # L1 penalty on attention M, V (0 = disabled)
    # L1 on expert FFN weights w1, w2, w3 (mutually exclusive with l1_lambda)
    "ff_l1_lambda": 1e-6,        # L1 penalty on FFN weights (0 = disabled)
    # Magnitude pruning (applies to BOTH M/V and expert FFN weights)
    "prune_threshold": 5e-3,    # zero out |w| < threshold (0 = disabled)
    "prune_every_n": 100,       # prune every N steps (0 = disabled)

    # --- Per-group weight decay ---
    "attn_weight_decay": 0.125,  # M, V + MoE router weights
    "ff_weight_decay": 0.01,     # expert FFN weights (0 = let L1 handle it)
    "embed_weight_decay": 0.2,  # token_emb + output head weight
    # LayerNorm + biases always get 0 weight decay (handled in train.py)
    "sparse_lr": None,          # None = use max_lr; set float to override LR for M, V

    # Gradient checkpointing (activation recomputation to save memory)
    "use_gradient_checkpointing": False,

    # Output head
    "block_decode": False,      # True = clustered two-stage decode at eval time
    "head_k": 2,                # Number of top-k clusters selected during eval decode
    "head_balance_tolerance": 10,  # Max ± deviation from target cluster size

    # Training parameters
    "batch_size": 64,
    "max_lr": 5e-4,
    "num_steps": 5000,
    "eval_every": 50,
    "warmup_steps": 500,
}
