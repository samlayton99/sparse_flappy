"""Mixture of Heads + Mixture of Experts transformer configuration.

Combines MoH attention (sparse head selection) with MoE FFN (sparse expert selection)
for maximum sparsity while maintaining model capacity.

With 16 heads, k_active=4: 25% attention head sparsity
With 8 experts, top_k=2: 25% expert sparsity

Combined, this model has significantly fewer active parameters than a dense model
of equivalent total capacity.
"""

CONFIG = {
    # Model architecture
    "model_class": "MoH_MoETransformer",
    "d_model": 2048,
    "n_layers": 16,
    
    # MoH attention parameters
    "n_heads": 16,          # Total number of attention heads
    "k_active": 4,          # Heads activated per sequence (25% sparsity)
    "attn_aux_loss_weight": 0.01,  # Head load balancing loss weight
    "attn_z_loss_weight": 0.001,   # Attention router z-loss weight
    
    # MoE FFN parameters
    "d_expert": 512,        # Hidden dim per expert
    "num_experts": 8,       # Number of expert FFNs
    "top_k": 2,             # Experts activated per token
    "moe_aux_loss_weight": 0.01,   # Expert load balancing loss weight
    "moe_z_loss_weight": 0.001,    # MoE router z-loss weight
    
    "dropout": 0.1,

    # Output head
    "block_decode": False,   # True = clustered two-stage decode at eval time
    "head_k": 1,             # Number of top-k clusters selected during eval decode
    "head_balance_tolerance": 10,  # Max ± deviation from target cluster size

    # Training parameters
    "batch_size": 32,
    "max_lr": 5e-4,
    "num_steps": 2000,
    "eval_every": 50,
    "warmup_steps": 200,
    "weight_decay": 0.3,
}
