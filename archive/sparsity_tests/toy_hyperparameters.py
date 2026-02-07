"""Central hyperparameter configuration for sparsity experiments.

Edit this file to control all experiment parameters from one place.
Then run:
    python train.py --config toy_hyperparameters.py

For real data, first run:
    python prepare_real_data.py
"""

CONFIG = {
    # =========================================================================
    # Data Source: "real" (AG News embeddings) or "synthetic" (random network)
    # =========================================================================
    "data_source": "real",          # "real" or "synthetic"
    "real_data_dir": "real_data",   # Directory with cached embeddings (for "real")

    # =========================================================================
    # Synthetic Data Generation (only used when data_source == "synthetic")
    # =========================================================================
    "data_input_dim": 16,
    "data_hidden_dim": 64,
    "data_n_layers": 1,
    "data_n_classes": 5,
    "data_n_train": 50000,
    "data_n_val": 10000,
    "data_seed": 42,

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # For "real" data, model_input_dim and model_n_classes are auto-detected
    # from the cached embeddings (1024 dims, 4 classes for AG News + bge-large).
    "model_input_dim": 1024,        # Auto-overridden for real data
    "model_hidden_dim": 2048,       # Hidden dimension of the trained models
    "model_n_classes": 4,           # Auto-overridden for real data
    "rank": 32,                     # Low-rank k for the AB decomposition (Experiment B)
    "rank_sweep": [8, 16, 32, 64, 128, 256],  # Ranks to sweep for Experiment C (pure AB)

    # =========================================================================
    # Training
    # =========================================================================
    "n_epochs": 20,                 # Number of training epochs
    "lr": 1e-3,                     # Learning rate
    "batch_size": 256,              # Batch size
    "weight_decay": 0.01,           # AdamW weight decay

    # =========================================================================
    # Sparsity / Regularization
    # =========================================================================
    "l1_weights": [0.0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],

    # Magnitude pruning: zero out params with |w| < threshold every N steps
    "prune_threshold": 1e-3,        # Pruning threshold (0 = disabled)
    "prune_every_n": 0,           # Apply pruning every N gradient steps (0 = disabled)

    # =========================================================================
    # Experiment Selection
    # =========================================================================
    "experiment": "all",            # "A", "B", "C", or "all"

    # =========================================================================
    # Output
    # =========================================================================
    "output_dir": "toy_data",       # Directory to save results
}
