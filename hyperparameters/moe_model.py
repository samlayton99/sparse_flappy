"""Mixture of Experts transformer configuration.

Uses MoE layers with top-k routing instead of dense FFN.
With 8 experts and top_k=2, only 25% of FFN params are active per token,
while maintaining model capacity.

Sparsity:
  - l1_lambda on expert FFN weights only (w1, w2, w3 across all experts)
  - Magnitude pruning on the same FFN weights
  - Weight decay is split into 4 mutually-exclusive groups:
      1. Attention (qkv, proj)           → attn_weight_decay
      2. FFN expert weights (w1, w2, w3) → ff_weight_decay
      3. Embeddings / output head        → embed_weight_decay
      4. LayerNorm & biases              → 0 (no decay)
  - L1 applies ONLY to group 2 (FFN); no double-regularization.
"""

CONFIG = {
    # Model architecture
    "model_class": "MoETransformer",
    "d_model": 2048,
    "n_layers": 16,
    "n_heads": 8,
    "d_expert": 512,        # Hidden dim per expert
    "num_experts": 8,       # Number of expert FFNs
    "top_k": 2,             # Experts activated per token
    "aux_loss_weight": 0.01,  # Load balancing loss weight
    "z_loss_weight": 0.001,   # Router z-loss weight (stabilizes training)
    "dropout": 0.1,

    # --- Sparsity controls ---
    "l1_lambda": 1e-6,          # L1 penalty on expert FFN weights (0 = disabled)
    "prune_threshold": 1e-3,     # magnitude prune cutoff (0 = disabled)
    "prune_every_n": 100,         # prune every N steps (0 = disabled)

    # --- Per-group weight decay ---
    "attn_weight_decay": 0.1,   # qkv + proj weights
    "ff_weight_decay": 0.05,     # expert FFN weights (0 = let L1 handle it)
    "embed_weight_decay": 0.1,  # token_emb + output head weight
    # LayerNorm + biases always get 0 weight decay (handled in train.py)

    # Output head
    "block_decode": False,   # True = clustered two-stage decode at eval time
    "head_k": 1,             # Number of top-k clusters selected during eval decode
    "head_balance_tolerance": 10,  # Max ± deviation from target cluster size

    # Training parameters
    "batch_size": 48,
    "max_lr": 5e-4,
    "num_steps": 2000,
    "eval_every": 50,
    "warmup_steps": 200,
}
