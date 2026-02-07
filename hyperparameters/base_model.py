"""Base dense transformer configuration.

This is the standard SimpleTransformer with dense SwiGLU FFN layers.
Good baseline for comparison with sparse/MoE models.
"""

CONFIG = {
    # Model architecture
    "model_class": "SimpleTransformer",
    "d_model": 2048,
    "n_layers": 16,
    "n_heads": 8,
    "d_ff": 4096,
    "dropout": 0.1,
    
    # Output head
    "block_decode": True,   # True = clustered two-stage decode at eval time
    "head_k": 8,            # Number of top-k clusters selected during eval decode
    "head_balance_tolerance": 10,  # Max ± deviation from target cluster size

    # Training parameters
    "batch_size": 32,
    "max_lr": 5e-4,
    "num_steps": 2000,
    "eval_every": 50,
    "warmup_steps": 200,
    "weight_decay": 0.3,
}
