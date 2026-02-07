"""Mixture of Heads transformer configuration.

Uses MoH attention with per-sequence head routing.
With 16 heads and k_active=4, only 25% of attention heads are active per sequence,
reducing attention FLOPs while maintaining model capacity.
"""

CONFIG = {
    # Model architecture
    "model_class": "MoHTransformer",
    "d_model": 2048,
    "n_layers": 16,
    "n_heads": 8,          # Total number of attention heads
    "k_active": 4,          # Heads activated per sequence (25% sparsity)
    "d_ff": 4096,           # Dense FFN hidden dim
    "aux_loss_weight": 0.01,  # Head load balancing loss weight
    "z_loss_weight": 0.001,   # Router z-loss weight (stabilizes training)
    "dropout": 0.1,

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
    "weight_decay": 0.3,
}
