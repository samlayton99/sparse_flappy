"""Starter model for the FineWeb challenge.

Your goal: achieve val loss < 3.0 with the fewest non-zero parameters.
Modify this model architecture to be as sparse/efficient as possible.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from rope import RotaryPositionalEmbedding

# GPT-2 tokenizer vocab size
VOCAB_SIZE = 50257
SEQ_LEN = 512  # 513 - 1 for causal LM


class SimpleTransformer(nn.Module):
    """A minimal transformer for language modeling.
    
    This is a basic starter - you should modify/replace this
    to minimize parameter count while achieving val loss < 3.0.
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        block_decode: bool = False,
        head_k: int = 1,
        head_balance_tolerance: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Token embeddings (no learned positional embedding - using RoPE)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE for positional encoding (applied in attention)
        head_dim = d_model // n_heads
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=head_dim,
            max_seq_len=max_seq_len,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = GroupedHead(d_model, vocab_size, k=head_k, bias=False, block_decode=block_decode, balance_tolerance=head_balance_tolerance)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        self._init_weights(n_layers)
    
    def _init_weights(self, n_layers):
        for name, module in self.named_modules():
            if isinstance(module, GroupedHead):
                continue  # GroupedHead handles its own init; weight-tied anyway
            elif isinstance(module, nn.Linear):
                # LeCun init: std = 1/sqrt(fan_in)
                fan_in = module.weight.shape[1]
                std = fan_in ** -0.5
                # Scale down residual projections (attn.proj and ff second linear)
                if "proj" in name or ".w2" in name:
                    std = std / (2 * n_layers) ** 0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=module.weight.shape[1] ** -0.5)
    
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        
        # Token embeddings (positional info added via RoPE in attention)
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        
        # Position indices for RoPE
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask, self.rope, positions)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def count_parameters(self, count_zeros: bool = False):
        """Count model parameters.
        
        Args:
            count_zeros: If False, only count non-zero parameters
        
        Returns:
            Total parameter count
        """
        if count_zeros:
            return sum(p.numel() for p in self.parameters())
        else:
            return sum((p != 0).sum().item() for p in self.parameters())


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = SwiGLUFF(d_model, d_ff)
    
    def forward(self, x, causal_mask, attention_mask, rope, positions):
        x = x + self.attn(self.ln1(x), causal_mask, attention_mask, rope, positions)
        x = x + self.ff(self.ln2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE and Flash Attention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, causal_mask, attention_mask, rope, positions):
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)
        
        # Apply RoPE to Q and K
        q = rope(q, positions)
        k = rope(k, positions)
        
        # Flash Attention via scaled_dot_product_attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MixtureOfHeadsAttention(nn.Module):
    """Mixture of Heads (MoH) attention with per-sequence head routing.
    
    Dynamically selects a subset of attention heads for each sequence,
    reducing active FLOPs while maintaining model capacity. Uses RoPE-aware
    routing: applies RoPE to activations before pooling for head selection.
    
    The key insight is that attention can be expressed as a blocked outer product
    (diadic form) of concatenated heads with output matrices. This allows us to
    skip computing unused heads entirely.
    
    Args:
        d_model: Model dimension
        num_heads: Total number of attention heads
        k_active: Number of heads to select per sequence
        dropout: Dropout rate for attention
        aux_loss_weight: Weight for load balancing auxiliary loss
        z_loss_weight: Weight for router z-loss (prevents large logits)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        k_active: int,
        dropout: float = 0.1,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert k_active <= num_heads, "k_active must be <= num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_active = k_active
        self.d_head = d_model // num_heads
        self.dropout = dropout
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight
        
        # Router components:
        # 1. Project to head_dim for RoPE application
        self.router_proj = nn.Linear(d_model, self.d_head, bias=False)
        # 2. Route from pooled RoPE'd features to head scores
        self.router = nn.Linear(self.d_head, num_heads, bias=False)
        
        # Per-head weight matrices (stored separately for efficient gathering)
        # Shape: (num_heads, d_model, d_head) for Q, K, V
        # Shape: (num_heads, d_head, d_model) for O
        self.W_q = nn.Parameter(torch.empty(num_heads, d_model, self.d_head))
        self.W_k = nn.Parameter(torch.empty(num_heads, d_model, self.d_head))
        self.W_v = nn.Parameter(torch.empty(num_heads, d_model, self.d_head))
        self.W_o = nn.Parameter(torch.empty(num_heads, self.d_head, d_model))
        
        # Initialize weights
        self._init_weights()
        
        # Track losses and stats (set during forward, read during training)
        self.aux_loss = 0.0
        self.z_loss = 0.0
        self.head_utilization = None  # (num_heads,) tensor of usage fractions
    
    def _init_weights(self):
        """Initialize weight matrices with scaled normal distribution."""
        # Standard deviation for Q, K, V: 1/sqrt(d_model)
        std_qkv = self.d_model ** -0.5
        nn.init.normal_(self.W_q, mean=0.0, std=std_qkv)
        nn.init.normal_(self.W_k, mean=0.0, std=std_qkv)
        nn.init.normal_(self.W_v, mean=0.0, std=std_qkv)
        
        # Output projection: scaled down (will be further scaled by model init)
        std_o = self.d_head ** -0.5
        nn.init.normal_(self.W_o, mean=0.0, std=std_o)
        
        # Router initialization
        nn.init.normal_(self.router_proj.weight, mean=0.0, std=self.d_model ** -0.5)
        nn.init.normal_(self.router.weight, mean=0.0, std=self.d_head ** -0.5)
    
    def forward(self, x, causal_mask, attention_mask, rope, positions):
        """Forward pass with sparse head computation.
        
        Args:
            x: Input tensor (B, T, d_model)
            causal_mask: Causal attention mask (unused, we use is_causal=True)
            attention_mask: Optional padding mask
            rope: RotaryPositionalEmbedding module
            positions: Position indices (T,)
            
        Returns:
            Output tensor (B, T, d_model)
        """
        B, T, D = x.shape
        
        # =====================================================================
        # Step 1: Routing with RoPE-aware pooling
        # =====================================================================
        # Project to head_dim for RoPE
        x_for_rope = self.router_proj(x)  # (B, T, d_head)
        
        # Apply RoPE to get positional information into the routing decision
        # Need to add a "heads" dimension for the rope module
        x_rotated = rope(x_for_rope.unsqueeze(1), positions).squeeze(1)  # (B, T, d_head)
        
        # Pool over sequence and compute router logits
        x_pooled = x_rotated.mean(dim=1)  # (B, d_head)
        router_logits = self.router(x_pooled)  # (B, num_heads)
        
        # Select top-k heads
        top_k_logits, top_k_indices = torch.topk(router_logits, self.k_active, dim=1)
        # top_k_indices: (B, k_active)
        
        # Compute routing weights (softmax over selected heads)
        routing_weights = F.softmax(top_k_logits, dim=-1)  # (B, k_active)
        
        # =====================================================================
        # Step 2: Compute auxiliary losses (only during training)
        # =====================================================================
        if self.training:
            router_probs = F.softmax(router_logits, dim=-1)  # (B, num_heads)
            
            # Fraction of sequences using each head
            head_mask = F.one_hot(top_k_indices, self.num_heads).float()  # (B, k, H)
            seqs_per_head = head_mask.sum(dim=(0, 1))  # (H,)
            load_fraction = seqs_per_head / (B * self.k_active)
            
            # Mean router probability per head
            prob_per_head = router_probs.mean(dim=0)  # (H,)
            
            # Load balancing loss: encourages uniform head utilization
            self.aux_loss = self.aux_loss_weight * self.num_heads * (load_fraction * prob_per_head).sum()
            
            # Router z-loss: prevents router logits from becoming too large
            log_z = torch.logsumexp(router_logits, dim=-1)  # (B,)
            self.z_loss = self.z_loss_weight * (log_z ** 2).mean()
            
            # Track head utilization for logging
            self.head_utilization = load_fraction.detach()
        
        # =====================================================================
        # Step 3: Gather weights for selected heads
        # =====================================================================
        # We need to gather the weight matrices for the top-k heads per sample
        # top_k_indices: (B, k)
        # W_q: (H, D, d_h) -> gather to (B, k, D, d_h)
        
        # Expand indices for gathering: (B, k) -> (B, k, D, d_h)
        idx_qkv = top_k_indices.view(B, self.k_active, 1, 1).expand(-1, -1, D, self.d_head)
        
        # Gather Q, K, V weights
        W_q_batch = self.W_q.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_qkv)  # (B, k, D, d_h)
        W_k_batch = self.W_k.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_qkv)
        W_v_batch = self.W_v.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_qkv)
        
        # Gather O weights: (H, d_h, D) -> (B, k, d_h, D)
        idx_o = top_k_indices.view(B, self.k_active, 1, 1).expand(-1, -1, self.d_head, D)
        W_o_batch = self.W_o.unsqueeze(0).expand(B, -1, -1, -1).gather(1, idx_o)  # (B, k, d_h, D)
        
        # =====================================================================
        # Step 4: Sparse projection via einsum
        # =====================================================================
        # x: (B, T, D), W: (B, k, D, d_h) -> (B, k, T, d_h)
        q = torch.einsum('btd,bkdh->bkth', x, W_q_batch)
        k = torch.einsum('btd,bkdh->bkth', x, W_k_batch)
        v = torch.einsum('btd,bkdh->bkth', x, W_v_batch)
        
        # =====================================================================
        # Step 5: Apply RoPE to Q and K
        # =====================================================================
        # rope expects (*, n_heads, T, d_head), we have (B, k, T, d_h)
        q = rope(q, positions)
        k = rope(k, positions)
        
        # =====================================================================
        # Step 6: Scaled dot-product attention (manual, since shapes vary per sample)
        # =====================================================================
        # q, k, v: (B, k, T, d_h)
        scale = self.d_head ** -0.5
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, k, T, T)
        
        # Apply causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        if self.training and self.dropout > 0:
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=True)
        
        # Apply attention to values
        context = torch.matmul(attn_probs, v)  # (B, k, T, d_h)
        
        # =====================================================================
        # Step 7: Output projection and aggregation
        # =====================================================================
        # Project each head's output: (B, k, T, d_h) x (B, k, d_h, D) -> (B, k, T, D)
        out_per_head = torch.einsum('bkth,bkhd->bktd', context, W_o_batch)
        
        # Weight by routing probabilities (soft routing)
        routing_weights_expanded = routing_weights.view(B, self.k_active, 1, 1)
        out_per_head = out_per_head * routing_weights_expanded
        
        # Sum over heads
        output = out_per_head.sum(dim=1)  # (B, T, D)
        
        return output
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get the total auxiliary loss (load balancing + z-loss) from the last forward pass."""
        return self.aux_loss + self.z_loss
    
    def get_head_stats(self) -> dict:
        """Get head utilization statistics from the last forward pass.
        
        Returns:
            Dict with 'utilization' (tensor), 'min', 'max', 'std' keys
        """
        if self.head_utilization is None:
            return {}
        util = self.head_utilization
        return {
            "utilization": util,
            "min": util.min().item(),
            "max": util.max().item(),
            "std": util.std().item(),
        }


class SwiGLUFF(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
            bias=False,
        )
        self.w2 = nn.Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
            bias=False,
        )
        self.w3 = nn.Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
            bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFFN(nn.Module):
    """Mixture of Experts Feed-Forward Network.
    
    Uses top-k routing to select experts per token. Each expert is a smaller
    SwiGLU FFN. Includes auxiliary load balancing loss and router z-loss
    to prevent expert collapse and ensure training stability.
    
    Args:
        d_model: Model dimension
        d_expert: Hidden dimension per expert (typically d_ff / num_experts or smaller)
        num_experts: Number of expert FFNs
        top_k: Number of experts to route each token to (1 or 2 typical)
        aux_loss_weight: Weight for load balancing auxiliary loss
        z_loss_weight: Weight for router z-loss (prevents large logits)
        device: Device to create parameters on
        dtype: Data type for parameters
    """
    
    def __init__(
        self,
        d_model: int,
        d_expert: int,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_expert = d_expert
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight
        
        # Router: projects input to expert logits
        self.router = nn.Linear(d_model, num_experts, bias=False, device=device, dtype=dtype)
        
        # Expert FFNs: each expert is a SwiGLU with smaller hidden dim
        # Using nn.ModuleList for simplicity; can be optimized with grouped GEMM
        self.experts = nn.ModuleList([
            SwiGLUFF(d_model, d_expert, device=device, dtype=dtype)
            for _ in range(num_experts)
        ])
        
        # Track losses and stats (set during forward, read during training)
        self.aux_loss = 0.0
        self.z_loss = 0.0
        self.expert_utilization = None  # (num_experts,) tensor of usage fractions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with top-k expert routing.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            Output tensor of shape (B, T, d_model)
        """
        B, T, C = x.shape
        
        # Flatten batch and sequence dimensions for routing
        x_flat = x.view(-1, C)  # (B*T, d_model)
        num_tokens = x_flat.shape[0]
        
        # Compute router logits and probabilities
        router_logits = self.router(x_flat)  # (B*T, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Normalize top-k probabilities to sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary losses and stats
        if self.training:
            # Fraction of tokens routed to each expert
            expert_mask = F.one_hot(top_k_indices, self.num_experts).float()  # (B*T, k, E)
            tokens_per_expert = expert_mask.sum(dim=(0, 1))  # (E,)
            load_fraction = tokens_per_expert / (num_tokens * self.top_k)
            
            # Mean router probability per expert
            prob_per_expert = router_probs.mean(dim=0)  # (E,)
            
            # Load balancing loss: encourages uniform expert utilization
            self.aux_loss = self.aux_loss_weight * self.num_experts * (load_fraction * prob_per_expert).sum()
            
            # Router z-loss: prevents router logits from becoming too large
            # This stabilizes training by keeping logits in a reasonable range
            log_z = torch.logsumexp(router_logits, dim=-1)  # (B*T,)
            self.z_loss = self.z_loss_weight * (log_z ** 2).mean()
            
            # Track expert utilization for logging (detached to avoid graph issues)
            self.expert_utilization = load_fraction.detach()
        
        # Compute expert outputs
        # For efficiency with small top_k, iterate over experts
        output = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert (across all top-k slots)
            expert_mask = (top_k_indices == expert_idx)  # (B*T, k)
            
            if not expert_mask.any():
                continue
            
            # Get token indices and their weights for this expert
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Get the routing weights for these tokens to this expert
            weights = (top_k_probs * expert_mask.float()).sum(dim=-1)[token_indices]  # (num_selected,)
            
            # Process tokens through expert
            expert_input = x_flat[token_indices]  # (num_selected, d_model)
            expert_output = self.experts[expert_idx](expert_input)  # (num_selected, d_model)
            
            # Weighted add to output
            output[token_indices] += weights.unsqueeze(-1) * expert_output
        
        return output.view(B, T, C)
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get the total auxiliary loss (load balancing + z-loss) from the last forward pass."""
        return self.aux_loss + self.z_loss
    
    def get_expert_stats(self) -> dict:
        """Get expert utilization statistics from the last forward pass.
        
        Returns:
            Dict with 'utilization' (tensor), 'min', 'max', 'std' keys
        """
        if self.expert_utilization is None:
            return {}
        util = self.expert_utilization
        return {
            "utilization": util,
            "min": util.min().item(),
            "max": util.max().item(),
            "std": util.std().item(),
        }


class MoETransformerBlock(nn.Module):
    """Transformer block using Mixture of Experts instead of dense FFN."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_expert: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoEFFN(
            d_model=d_model,
            d_expert=d_expert,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight,
            z_loss_weight=z_loss_weight,
        )
    
    def forward(self, x, causal_mask, attention_mask, rope, positions):
        x = x + self.attn(self.ln1(x), causal_mask, attention_mask, rope, positions)
        x = x + self.moe(self.ln2(x))
        return x
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get auxiliary loss from MoE layer."""
        return self.moe.get_aux_loss()
    
    def get_expert_stats(self) -> dict:
        """Get expert utilization stats from MoE layer."""
        return self.moe.get_expert_stats()


class MoHTransformerBlock(nn.Module):
    """Transformer block using Mixture of Heads attention with dense FFN.
    
    Uses MoH attention (sparse head selection) with standard SwiGLU FFN.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        k_active: int,
        d_ff: int,
        dropout: float = 0.1,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MixtureOfHeadsAttention(
            d_model=d_model,
            num_heads=n_heads,
            k_active=k_active,
            dropout=dropout,
            aux_loss_weight=aux_loss_weight,
            z_loss_weight=z_loss_weight,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = SwiGLUFF(d_model, d_ff)
    
    def forward(self, x, causal_mask, attention_mask, rope, positions):
        x = x + self.attn(self.ln1(x), causal_mask, attention_mask, rope, positions)
        x = x + self.ff(self.ln2(x))
        return x
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get auxiliary loss from MoH attention layer."""
        return self.attn.get_aux_loss()
    
    def get_head_stats(self) -> dict:
        """Get head utilization stats from MoH attention layer."""
        return self.attn.get_head_stats()


class MoH_MoETransformerBlock(nn.Module):
    """Transformer block using both MoH attention AND MoE FFN.
    
    Combines sparse head selection (MoH) with sparse expert selection (MoE)
    for maximum sparsity while maintaining model capacity.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        k_active: int,
        d_expert: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        attn_aux_loss_weight: float = 0.01,
        attn_z_loss_weight: float = 0.001,
        moe_aux_loss_weight: float = 0.01,
        moe_z_loss_weight: float = 0.001,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MixtureOfHeadsAttention(
            d_model=d_model,
            num_heads=n_heads,
            k_active=k_active,
            dropout=dropout,
            aux_loss_weight=attn_aux_loss_weight,
            z_loss_weight=attn_z_loss_weight,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoEFFN(
            d_model=d_model,
            d_expert=d_expert,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_weight=moe_aux_loss_weight,
            z_loss_weight=moe_z_loss_weight,
        )
    
    def forward(self, x, causal_mask, attention_mask, rope, positions):
        x = x + self.attn(self.ln1(x), causal_mask, attention_mask, rope, positions)
        x = x + self.moe(self.ln2(x))
        return x
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get combined auxiliary loss from both MoH and MoE layers."""
        return self.attn.get_aux_loss() + self.moe.get_aux_loss()
    
    def get_head_stats(self) -> dict:
        """Get head utilization stats from MoH attention layer."""
        return self.attn.get_head_stats()
    
    def get_expert_stats(self) -> dict:
        """Get expert utilization stats from MoE FFN layer."""
        return self.moe.get_expert_stats()


class MoETransformer(nn.Module):
    """Mixture of Experts Transformer for language modeling.
    
    Similar to SimpleTransformer but uses MoE layers instead of dense FFN.
    Each layer has multiple expert FFNs with top-k routing.

    Sparsity support:
        - l1_lambda on expert FFN weights (w1, w2, w3 across all experts)
        - Magnitude pruning on the same FFN weights
        - Separate weight_decay groups: attention, FFN, embeddings, no-decay (LN/bias)
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_expert: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        block_decode: bool = False,
        head_k: int = 1,
        head_balance_tolerance: int = 10,
        # Sparsity config
        l1_lambda: float = 0.0,
        prune_threshold: float = 0.0,
        prune_every_n: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_experts = num_experts
        self.l1_lambda = l1_lambda
        self.prune_threshold = prune_threshold
        self.prune_every_n = prune_every_n
        
        # Token embeddings (no learned positional embedding - using RoPE)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE for positional encoding (applied in attention)
        head_dim = d_model // n_heads
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=head_dim,
            max_seq_len=max_seq_len,
        )
        
        # MoE Transformer layers
        self.layers = nn.ModuleList([
            MoETransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_expert=d_expert,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                aux_loss_weight=aux_loss_weight,
                z_loss_weight=z_loss_weight,
            )
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = GroupedHead(d_model, vocab_size, k=head_k, bias=False, block_decode=block_decode, balance_tolerance=head_balance_tolerance)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        self._init_weights(n_layers)
    
    def _init_weights(self, n_layers):
        for name, module in self.named_modules():
            if isinstance(module, GroupedHead):
                continue  # GroupedHead handles its own init; weight-tied anyway
            elif isinstance(module, nn.Linear):
                # LeCun init: std = 1/sqrt(fan_in)
                fan_in = module.weight.shape[1]
                std = fan_in ** -0.5
                # Scale down residual projections (attn.proj and ff second linear)
                if "proj" in name or ".w2" in name:
                    std = std / (2 * n_layers) ** 0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=module.weight.shape[1] ** -0.5)
    
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        
        # Token embeddings (positional info added via RoPE in attention)
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        
        # Position indices for RoPE
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask, self.rope, positions)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get total auxiliary loss from all MoE layers."""
        total_aux_loss = 0.0
        for layer in self.layers:
            total_aux_loss = total_aux_loss + layer.get_aux_loss()
        return total_aux_loss
    
    def get_expert_stats(self) -> dict:
        """Get aggregated expert utilization statistics across all layers.
        
        Returns:
            Dict with aggregated stats: min/max/mean utilization, std across experts
        """
        all_utils = []
        for layer in self.layers:
            stats = layer.get_expert_stats()
            if stats and "utilization" in stats:
                all_utils.append(stats["utilization"])
        
        if not all_utils:
            return {}
        
        # Stack utilizations: (n_layers, num_experts)
        stacked = torch.stack(all_utils)
        mean_per_expert = stacked.mean(dim=0)  # Average across layers
        
        return {
            "mean_utilization": mean_per_expert,  # (num_experts,)
            "min": mean_per_expert.min().item(),
            "max": mean_per_expert.max().item(),
            "std": mean_per_expert.std().item(),
            "per_layer": [u.tolist() for u in all_utils],  # For detailed logging
        }
    
    # ----- Sparsity API -----

    def get_ff_params(self):
        """Collect all expert FFN weight parameters (w1, w2, w3) across all layers.

        Does NOT include the MoE router weights — only expert body weights.
        """
        params = []
        for layer in self.layers:
            for expert in layer.moe.experts:
                params.extend([expert.w1.weight, expert.w2.weight, expert.w3.weight])
        return params

    def get_attn_params(self):
        """Collect attention projection weights (qkv, proj) across all layers."""
        params = []
        for layer in self.layers:
            params.append(layer.attn.qkv.weight)
            params.append(layer.attn.proj.weight)
        return params

    def magnitude_prune(self, optimizer=None):
        """Zero out expert FFN entries below prune_threshold.

        Clears AdamW optimizer momentum for pruned entries to prevent bounce-back.
        """
        if self.prune_threshold <= 0:
            return
        with torch.no_grad():
            for p in self.get_ff_params():
                mask = p.abs() < self.prune_threshold
                p[mask] = 0.0
                if optimizer is not None:
                    state = optimizer.state.get(p)
                    if state:
                        if "exp_avg" in state:
                            state["exp_avg"][mask] = 0.0
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"][mask] = 0.0

    def get_sparsity_stats(self):
        """Compute sparsity statistics for expert FFN weights."""
        total = 0
        nonzero = 0
        with torch.no_grad():
            for p in self.get_ff_params():
                total += p.numel()
                nonzero += (p != 0).sum().item()
        return {
            "ff_sparsity": 1.0 - nonzero / total if total > 0 else 0.0,
            "ff_density": nonzero / total if total > 0 else 1.0,
            "ff_nonzero": nonzero,
            "ff_total": total,
        }

    def count_parameters(self, count_zeros: bool = False):
        """Count model parameters.
        
        Args:
            count_zeros: If False, only count non-zero parameters
        
        Returns:
            Total parameter count
        """
        if count_zeros:
            return sum(p.numel() for p in self.parameters())
        else:
            return sum((p != 0).sum().item() for p in self.parameters())


class MoHTransformer(nn.Module):
    """Mixture of Heads Transformer for language modeling.
    
    Uses MoH attention (sparse head selection) with dense SwiGLU FFN.
    Each layer dynamically selects a subset of attention heads per sequence.
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_heads: int = 8,
        k_active: int = 2,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        block_decode: bool = False,
        head_k: int = 1,
        head_balance_tolerance: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.k_active = k_active
        
        # Token embeddings (no learned positional embedding - using RoPE)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE for positional encoding (applied in attention)
        head_dim = d_model // n_heads
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=head_dim,
            max_seq_len=max_seq_len,
        )
        
        # MoH Transformer layers
        self.layers = nn.ModuleList([
            MoHTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                k_active=k_active,
                d_ff=d_ff,
                dropout=dropout,
                aux_loss_weight=aux_loss_weight,
                z_loss_weight=z_loss_weight,
            )
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = GroupedHead(d_model, vocab_size, k=head_k, bias=False, block_decode=block_decode, balance_tolerance=head_balance_tolerance)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        self._init_weights(n_layers)
    
    def _init_weights(self, n_layers):
        for name, module in self.named_modules():
            if isinstance(module, GroupedHead):
                continue  # GroupedHead handles its own init; weight-tied anyway
            elif isinstance(module, nn.Linear):
                # LeCun init: std = 1/sqrt(fan_in)
                fan_in = module.weight.shape[1]
                std = fan_in ** -0.5
                # Scale down residual projections
                if "proj" in name or ".w2" in name:
                    std = std / (2 * n_layers) ** 0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=module.weight.shape[1] ** -0.5)
    
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        
        # Token embeddings (positional info added via RoPE in attention)
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        
        # Position indices for RoPE
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask, self.rope, positions)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get total auxiliary loss from all MoH layers."""
        total_aux_loss = 0.0
        for layer in self.layers:
            total_aux_loss = total_aux_loss + layer.get_aux_loss()
        return total_aux_loss
    
    def get_head_stats(self) -> dict:
        """Get aggregated head utilization statistics across all layers.
        
        Returns:
            Dict with aggregated stats: min/max/mean utilization, std across heads
        """
        all_utils = []
        for layer in self.layers:
            stats = layer.get_head_stats()
            if stats and "utilization" in stats:
                all_utils.append(stats["utilization"])
        
        if not all_utils:
            return {}
        
        # Stack utilizations: (n_layers, num_heads)
        stacked = torch.stack(all_utils)
        mean_per_head = stacked.mean(dim=0)  # Average across layers
        
        return {
            "mean_utilization": mean_per_head,  # (num_heads,)
            "min": mean_per_head.min().item(),
            "max": mean_per_head.max().item(),
            "std": mean_per_head.std().item(),
            "per_layer": [u.tolist() for u in all_utils],  # For detailed logging
        }
    
    def count_parameters(self, count_zeros: bool = False):
        """Count model parameters.
        
        Args:
            count_zeros: If False, only count non-zero parameters
        
        Returns:
            Total parameter count
        """
        if count_zeros:
            return sum(p.numel() for p in self.parameters())
        else:
            return sum((p != 0).sum().item() for p in self.parameters())


class MoH_MoETransformer(nn.Module):
    """Mixture of Heads + Mixture of Experts Transformer for language modeling.
    
    Combines MoH attention (sparse head selection) with MoE FFN (sparse expert selection)
    for maximum sparsity while maintaining model capacity.
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_heads: int = 8,
        k_active: int = 2,
        n_layers: int = 4,
        d_expert: int = 256,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        attn_aux_loss_weight: float = 0.01,
        attn_z_loss_weight: float = 0.001,
        moe_aux_loss_weight: float = 0.01,
        moe_z_loss_weight: float = 0.001,
        block_decode: bool = False,
        head_k: int = 1,
        head_balance_tolerance: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.k_active = k_active
        self.num_experts = num_experts
        
        # Token embeddings (no learned positional embedding - using RoPE)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE for positional encoding (applied in attention)
        head_dim = d_model // n_heads
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=head_dim,
            max_seq_len=max_seq_len,
        )
        
        # MoH + MoE Transformer layers
        self.layers = nn.ModuleList([
            MoH_MoETransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                k_active=k_active,
                d_expert=d_expert,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                attn_aux_loss_weight=attn_aux_loss_weight,
                attn_z_loss_weight=attn_z_loss_weight,
                moe_aux_loss_weight=moe_aux_loss_weight,
                moe_z_loss_weight=moe_z_loss_weight,
            )
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = GroupedHead(d_model, vocab_size, k=head_k, bias=False, block_decode=block_decode, balance_tolerance=head_balance_tolerance)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        self._init_weights(n_layers)
    
    def _init_weights(self, n_layers):
        for name, module in self.named_modules():
            if isinstance(module, GroupedHead):
                continue  # GroupedHead handles its own init; weight-tied anyway
            elif isinstance(module, nn.Linear):
                # LeCun init: std = 1/sqrt(fan_in)
                fan_in = module.weight.shape[1]
                std = fan_in ** -0.5
                # Scale down residual projections
                if "proj" in name or ".w2" in name:
                    std = std / (2 * n_layers) ** 0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=module.weight.shape[1] ** -0.5)
    
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        
        # Token embeddings (positional info added via RoPE in attention)
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        
        # Position indices for RoPE
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask, attention_mask, self.rope, positions)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get total auxiliary loss from all MoH and MoE layers."""
        total_aux_loss = 0.0
        for layer in self.layers:
            total_aux_loss = total_aux_loss + layer.get_aux_loss()
        return total_aux_loss
    
    def get_head_stats(self) -> dict:
        """Get aggregated head utilization statistics across all layers.
        
        Returns:
            Dict with aggregated stats: min/max/mean utilization, std across heads
        """
        all_utils = []
        for layer in self.layers:
            stats = layer.get_head_stats()
            if stats and "utilization" in stats:
                all_utils.append(stats["utilization"])
        
        if not all_utils:
            return {}
        
        # Stack utilizations: (n_layers, num_heads)
        stacked = torch.stack(all_utils)
        mean_per_head = stacked.mean(dim=0)  # Average across layers
        
        return {
            "mean_utilization": mean_per_head,  # (num_heads,)
            "min": mean_per_head.min().item(),
            "max": mean_per_head.max().item(),
            "std": mean_per_head.std().item(),
            "per_layer": [u.tolist() for u in all_utils],  # For detailed logging
        }
    
    def get_expert_stats(self) -> dict:
        """Get aggregated expert utilization statistics across all layers.
        
        Returns:
            Dict with aggregated stats: min/max/mean utilization, std across experts
        """
        all_utils = []
        for layer in self.layers:
            stats = layer.get_expert_stats()
            if stats and "utilization" in stats:
                all_utils.append(stats["utilization"])
        
        if not all_utils:
            return {}
        
        # Stack utilizations: (n_layers, num_experts)
        stacked = torch.stack(all_utils)
        mean_per_expert = stacked.mean(dim=0)  # Average across layers
        
        return {
            "mean_utilization": mean_per_expert,  # (num_experts,)
            "min": mean_per_expert.min().item(),
            "max": mean_per_expert.max().item(),
            "std": mean_per_expert.std().item(),
            "per_layer": [u.tolist() for u in all_utils],  # For detailed logging
        }
    
    def count_parameters(self, count_zeros: bool = False):
        """Count model parameters.
        
        Args:
            count_zeros: If False, only count non-zero parameters
        
        Returns:
            Total parameter count
        """
        if count_zeros:
            return sum(p.numel() for p in self.parameters())
        else:
            return sum((p != 0).sum().item() for p in self.parameters())


# =============================================================================
# Sparse Dyadic Attention
# =============================================================================

class SparseDyadicAttention(nn.Module):
    """Block-dyadic attention with full-rank sparse metric and value-map matrices.

    Each head i computes:
        head_i(X) = softmax( RoPE(X @ M_i) · RoPE(X)^T / sqrt(d) ) · (X @ V_i)

    Output = sum over heads.

    M and V are stored as (d_model, n_heads * d_model) for batched matmul.
    Sparsity is induced externally via L1 regularization on M and V only.

    Note: head_dim = d_model (NOT d_model // n_heads), so Flash Attention is
    not available (requires head_dim <= 256). We use manual attention instead.

    Args:
        d_model: Model dimension (also the per-head dimension)
        n_heads: Number of attention heads
        dropout: Dropout rate for attention weights
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        # Core sparse parameters: metric (M) and value-map (V)
        # Stored as (d_model, n_heads * d_model) for single-matmul efficiency.
        # Conceptually, each head i owns M[:, i*d:(i+1)*d] and V[:, i*d:(i+1)*d].
        self.M = nn.Parameter(torch.empty(d_model, n_heads * d_model))
        self.V = nn.Parameter(torch.empty(d_model, n_heads * d_model))

        self._init_weights()

    def _init_weights(self):
        """Scaled normal init: std = 1/sqrt(d_model) per head slice."""
        std = self.d_model ** -0.5
        nn.init.normal_(self.M, mean=0.0, std=std)
        nn.init.normal_(self.V, mean=0.0, std=std)

    def forward(self, x, causal_mask, attention_mask, rope, positions):
        """
        Args:
            x: (B, T, d_model) — input from LayerNorm
            causal_mask: unused (we build our own), kept for interface compat
            attention_mask: unused, kept for interface compat
            rope: RotaryPositionalEmbedding with d_key = d_model
            positions: (T,) position indices

        Returns:
            (B, T, d_model)
        """
        B, T, d = x.shape
        h = self.n_heads

        # --- Step 1: Project Q and val via two big matmuls ---
        # x @ M  -> (B, T, h*d), then reshape to (B, h, T, d)
        Q = (x @ self.M).view(B, T, h, d).transpose(1, 2)    # (B, h, T, d)
        val = (x @ self.V).view(B, T, h, d).transpose(1, 2)   # (B, h, T, d)

        # --- Step 2: K = X, shared across heads (zero-copy expand) ---
        K = x.unsqueeze(1).expand(-1, h, -1, -1)               # (B, h, T, d)

        # --- Step 3: Apply full-dim RoPE to Q and K ---
        Q = rope(Q, positions)
        K = rope(K, positions)

        # --- Step 4: Compute attention scores (manual — no Flash for d > 256) ---
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * scale  # (B, h, T, T)

        # Causal mask: prevent attending to future positions
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        # --- Step 5: Apply attention to values ---
        out = torch.matmul(attn, val)  # (B, h, T, d)

        # --- Step 6: Sum over heads (block dyadic summation) ---
        out = out.sum(dim=1)  # (B, T, d)

        return out

    # ----- Sparsity introspection API -----

    def get_sparse_params(self):
        """Return the M and V parameter tensors for targeted regularization."""
        return [self.M, self.V]

    def get_sparsity_stats(self):
        """Compute sparsity statistics for the metric and value-map matrices."""
        with torch.no_grad():
            m_total = self.M.numel()
            m_nonzero = (self.M != 0).sum().item()
            v_total = self.V.numel()
            v_nonzero = (self.V != 0).sum().item()
        return {
            "M_sparsity": 1.0 - m_nonzero / m_total,
            "V_sparsity": 1.0 - v_nonzero / v_total,
            "M_nonzero": m_nonzero,
            "V_nonzero": v_nonzero,
            "total_attn_params": m_total + v_total,
            "nonzero_attn_params": m_nonzero + v_nonzero,
        }


class SparseDyadicTransformerBlock(nn.Module):
    """Transformer block: SparseDyadicAttention + SwiGLU FFN (pre-norm)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SparseDyadicAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = SwiGLUFF(d_model, d_ff)

    def forward(self, x, causal_mask, attention_mask, rope, positions):
        x = x + self.attn(self.ln1(x), causal_mask, attention_mask, rope, positions)
        x = x + self.ff(self.ln2(x))
        return x


class SparseDyadicTransformer(nn.Module):
    """Transformer with sparse dyadic attention for language modeling.

    Key differences from SimpleTransformer:
    - Attention uses full-rank sparse M_i and V_i matrices per head (block dyadic form)
    - RoPE operates on full d_model dimensions (not d_model // n_heads)
    - Gradient checkpointing is built in (needed due to larger activation footprint)
    - Provides get_sparse_params() / get_sparsity_stats() for targeted L1 and pruning

    Training note: M and V should typically have zero weight_decay in the optimizer
    (use separate param groups). L1 regularization and magnitude pruning are applied
    externally by the training loop via get_sparse_params() and magnitude_prune().

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension (also the per-head key/value dimension)
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        d_ff: FFN hidden dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        rope_theta: RoPE base frequency
        l1_lambda: L1 penalty on attention M, V matrices (0 = disabled)
        ff_l1_lambda: L1 penalty on FFN weights w1, w2, w3 (0 = disabled).
            Mutually exclusive with l1_lambda (no parameter is in both sets).
        prune_threshold: Magnitude pruning cutoff for M, V (0 = disabled)
        prune_every_n: Prune every N optimizer steps (0 = disabled)
        use_gradient_checkpointing: Wrap layer forwards in activation checkpointing
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 2048,
        n_heads: int = 8,
        n_layers: int = 16,
        d_ff: int = 4096,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        # Sparsity config (stored as attributes, consumed by training loop)
        l1_lambda: float = 0.0,
        ff_l1_lambda: float = 0.0,
        prune_threshold: float = 0.0,
        prune_every_n: int = 0,
        use_gradient_checkpointing: bool = True,
        block_decode: bool = False,
        head_k: int = 1,
        head_balance_tolerance: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Sparsity config
        self.l1_lambda = l1_lambda        # L1 on attention M, V matrices
        self.ff_l1_lambda = ff_l1_lambda  # L1 on FFN weight matrices (mutually exclusive with above)
        self.prune_threshold = prune_threshold
        self.prune_every_n = prune_every_n
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # RoPE with d_key = d_model (full dimension, NOT d_model // n_heads)
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=d_model,
            max_seq_len=max_seq_len,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            SparseDyadicTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = GroupedHead(d_model, vocab_size, k=head_k, bias=False, block_decode=block_decode, balance_tolerance=head_balance_tolerance)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights(n_layers)

    def _init_weights(self, n_layers):
        """Initialize non-attention weights (FFN, embeddings, LayerNorm).

        SparseDyadicAttention handles its own M/V init internally.
        """
        for name, module in self.named_modules():
            if isinstance(module, (SparseDyadicAttention, GroupedHead)):
                # SparseDyadicAttention and GroupedHead handle their own init
                continue
            elif isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                std = fan_in ** -0.5
                # Scale down residual projections (FFN w2)
                if ".w2" in name:
                    std = std / (2 * n_layers) ** 0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=module.weight.shape[1] ** -0.5
                )

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids)
        x = self.dropout_layer(x)

        positions = torch.arange(0, T, dtype=torch.long, device=device)

        # Causal mask (passed for interface compat; attention builds its own)
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = grad_checkpoint(
                    layer, x, causal_mask, attention_mask, self.rope, positions,
                    use_reentrant=False,
                )
            else:
                x = layer(x, causal_mask, attention_mask, self.rope, positions)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    # ----- Sparsity API (consumed by training loop) -----

    def get_sparse_params(self):
        """Collect all sparse M and V parameters across all layers.

        Returns:
            List of nn.Parameter tensors (M and V from each layer's attention).
        """
        params = []
        for layer in self.layers:
            params.extend(layer.attn.get_sparse_params())
        return params

    def get_ff_params(self):
        """Collect all FFN weight parameters across all layers.

        Returns w1, w2, w3 weight tensors from each layer's SwiGLU FFN.
        These are disjoint from get_sparse_params() (no double-counting).
        """
        params = []
        for layer in self.layers:
            params.extend([
                layer.ff.w1.weight,
                layer.ff.w2.weight,
                layer.ff.w3.weight,
            ])
        return params

    def get_sparsity_stats(self):
        """Aggregate sparsity statistics across all layers.

        Returns:
            Dict with overall and per-layer sparsity metrics.
        """
        total_params = 0
        total_nonzero = 0
        per_layer = []
        for layer in self.layers:
            stats = layer.attn.get_sparsity_stats()
            total_params += stats["total_attn_params"]
            total_nonzero += stats["nonzero_attn_params"]
            per_layer.append(stats)
        return {
            "attn_sparsity": 1.0 - total_nonzero / total_params if total_params > 0 else 0.0,
            "attn_density": total_nonzero / total_params if total_params > 0 else 1.0,
            "attn_nonzero": total_nonzero,
            "attn_total": total_params,
            "per_layer": per_layer,
        }

    def magnitude_prune(self, optimizer=None):
        """Zero out M and V entries below prune_threshold.

        Also clears corresponding AdamW optimizer momentum states to prevent
        pruned weights from "bouncing back" through accumulated gradients.

        Args:
            optimizer: Optional AdamW optimizer. If provided, exp_avg and
                       exp_avg_sq are zeroed for pruned entries.
        """
        if self.prune_threshold <= 0:
            return
        with torch.no_grad():
            for layer in self.layers:
                for p in layer.attn.get_sparse_params():
                    mask = p.abs() < self.prune_threshold
                    p[mask] = 0.0
                    if optimizer is not None:
                        state = optimizer.state.get(p)
                        if state:
                            if "exp_avg" in state:
                                state["exp_avg"][mask] = 0.0
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"][mask] = 0.0

    def count_parameters(self, count_zeros: bool = False):
        """Count model parameters.

        Args:
            count_zeros: If False, only count non-zero parameters.

        Returns:
            Total parameter count.
        """
        if count_zeros:
            return sum(p.numel() for p in self.parameters())
        else:
            return sum((p != 0).sum().item() for p in self.parameters())


# =============================================================================
# Final Sparse Transformer (SparseDyadic Attention + MoE FFN)
# =============================================================================

class FinalSparseTransformerBlock(nn.Module):
    """Transformer block: SparseDyadicAttention + MoE FFN (pre-norm)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_expert: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SparseDyadicAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoEFFN(
            d_model=d_model,
            d_expert=d_expert,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight,
            z_loss_weight=z_loss_weight,
        )

    def forward(self, x, causal_mask, attention_mask, rope, positions):
        x = x + self.attn(self.ln1(x), causal_mask, attention_mask, rope, positions)
        x = x + self.moe(self.ln2(x))
        return x

    def get_aux_loss(self) -> torch.Tensor:
        return self.moe.get_aux_loss()

    def get_expert_stats(self) -> dict:
        return self.moe.get_expert_stats()


class FinalSparseTransformer(nn.Module):
    """Sparse Dyadic Attention + MoE FFN transformer for language modeling.

    Combines:
    - SparseDyadicAttention: full-rank sparse M (metric) and V (value-map)
      matrices per head with L1 regularization and magnitude pruning.
    - MoE FFN: top-k expert routing with load-balancing loss, L1 on expert
      weights, and magnitude pruning.
    - Gradient checkpointing for memory efficiency.
    - GroupedHead for optional clustered eval-time decode.

    Two independent L1 regularization channels:
      - l1_lambda:     on attention M, V matrices
      - ff_l1_lambda:  on expert FFN w1, w2, w3 weights
    These are mutually exclusive — no parameter is in both sets.

    Weight decay is split into 4 mutually exclusive groups (handled by train.py):
      1. Attention (M, V, router)       → attn_weight_decay
      2. FFN expert weights (w1,w2,w3)  → ff_weight_decay
      3. Embeddings / output head       → embed_weight_decay
      4. LayerNorm & biases             → 0 (no decay)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 2048,
        n_heads: int = 8,
        n_layers: int = 16,
        d_expert: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        dropout: float = 0.1,
        max_seq_len: int = SEQ_LEN,
        rope_theta: float = 10000.0,
        # Sparsity config
        l1_lambda: float = 0.0,
        ff_l1_lambda: float = 0.0,
        prune_threshold: float = 0.0,
        prune_every_n: int = 0,
        use_gradient_checkpointing: bool = True,
        # Output head
        block_decode: bool = False,
        head_k: int = 1,
        head_balance_tolerance: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Sparsity config (consumed by training loop)
        self.l1_lambda = l1_lambda
        self.ff_l1_lambda = ff_l1_lambda
        self.prune_threshold = prune_threshold
        self.prune_every_n = prune_every_n
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # RoPE with d_key = d_model (full dimension for sparse dyadic attention)
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_key=d_model,
            max_seq_len=max_seq_len,
        )

        # Transformer layers (SparseDyadicAttention + MoE FFN)
        self.layers = nn.ModuleList([
            FinalSparseTransformerBlock(
                d_model, n_heads, d_expert, num_experts, top_k,
                dropout, aux_loss_weight, z_loss_weight,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = GroupedHead(
            d_model, vocab_size, k=head_k, bias=False,
            block_decode=block_decode, balance_tolerance=head_balance_tolerance,
        )

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights(n_layers)

    def _init_weights(self, n_layers):
        """Initialize non-attention weights.

        SparseDyadicAttention handles its own M/V init internally.
        """
        for name, module in self.named_modules():
            if isinstance(module, (SparseDyadicAttention, GroupedHead)):
                continue
            elif isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                std = fan_in ** -0.5
                # Scale down residual projections (FFN w2)
                if ".w2" in name:
                    std = std / (2 * n_layers) ** 0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=module.weight.shape[1] ** -0.5
                )

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids)
        x = self.dropout_layer(x)

        positions = torch.arange(0, T, dtype=torch.long, device=device)

        # Causal mask (passed for interface compat; attention builds its own)
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = grad_checkpoint(
                    layer, x, causal_mask, attention_mask, self.rope, positions,
                    use_reentrant=False,
                )
            else:
                x = layer(x, causal_mask, attention_mask, self.rope, positions)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    # ----- MoE API -----

    def get_aux_loss(self) -> torch.Tensor:
        """Get total auxiliary loss from all MoE layers."""
        total = 0.0
        for layer in self.layers:
            total = total + layer.get_aux_loss()
        return total

    def get_expert_stats(self) -> dict:
        """Get aggregated expert utilization statistics across all layers."""
        all_utils = []
        for layer in self.layers:
            stats = layer.get_expert_stats()
            if stats and "utilization" in stats:
                all_utils.append(stats["utilization"])

        if not all_utils:
            return {}

        stacked = torch.stack(all_utils)
        mean_per_expert = stacked.mean(dim=0)

        return {
            "mean_utilization": mean_per_expert,
            "min": mean_per_expert.min().item(),
            "max": mean_per_expert.max().item(),
            "std": mean_per_expert.std().item(),
            "per_layer": [u.tolist() for u in all_utils],
        }

    # ----- Sparsity API (consumed by training loop) -----

    def get_sparse_params(self):
        """Collect all sparse M and V parameters across all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.attn.get_sparse_params())
        return params

    def get_ff_params(self):
        """Collect all expert FFN weight parameters (w1, w2, w3) across all layers.

        Does NOT include MoE router weights — only expert body weights.
        """
        params = []
        for layer in self.layers:
            for expert in layer.moe.experts:
                params.extend([expert.w1.weight, expert.w2.weight, expert.w3.weight])
        return params

    def get_attn_params(self):
        """Collect MoE router weights across all layers (for weight decay grouping)."""
        params = []
        for layer in self.layers:
            params.append(layer.moe.router.weight)
        return params

    def get_sparsity_stats(self):
        """Aggregate sparsity statistics for both attention (M, V) and FFN experts."""
        # Attention sparsity (M, V)
        attn_total = 0
        attn_nonzero = 0
        for layer in self.layers:
            stats = layer.attn.get_sparsity_stats()
            attn_total += stats["total_attn_params"]
            attn_nonzero += stats["nonzero_attn_params"]

        # FFN sparsity (expert w1, w2, w3)
        ff_total = 0
        ff_nonzero = 0
        with torch.no_grad():
            for p in self.get_ff_params():
                ff_total += p.numel()
                ff_nonzero += (p != 0).sum().item()

        return {
            "attn_sparsity": 1.0 - attn_nonzero / attn_total if attn_total > 0 else 0.0,
            "attn_density": attn_nonzero / attn_total if attn_total > 0 else 1.0,
            "attn_nonzero": attn_nonzero,
            "attn_total": attn_total,
            "ff_sparsity": 1.0 - ff_nonzero / ff_total if ff_total > 0 else 0.0,
            "ff_density": ff_nonzero / ff_total if ff_total > 0 else 1.0,
            "ff_nonzero": ff_nonzero,
            "ff_total": ff_total,
        }

    def magnitude_prune(self, optimizer=None):
        """Zero out M/V and expert FFN entries below prune_threshold.

        Also clears corresponding AdamW optimizer momentum states to prevent
        pruned weights from bouncing back through accumulated gradients.
        """
        if self.prune_threshold <= 0:
            return
        with torch.no_grad():
            # Prune attention M, V
            for layer in self.layers:
                for p in layer.attn.get_sparse_params():
                    mask = p.abs() < self.prune_threshold
                    p[mask] = 0.0
                    if optimizer is not None:
                        state = optimizer.state.get(p)
                        if state:
                            if "exp_avg" in state:
                                state["exp_avg"][mask] = 0.0
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"][mask] = 0.0
            # Prune expert FFN weights
            for p in self.get_ff_params():
                mask = p.abs() < self.prune_threshold
                p[mask] = 0.0
                if optimizer is not None:
                    state = optimizer.state.get(p)
                    if state:
                        if "exp_avg" in state:
                            state["exp_avg"][mask] = 0.0
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"][mask] = 0.0

    def count_parameters(self, count_zeros: bool = False):
        if count_zeros:
            return sum(p.numel() for p in self.parameters())
        else:
            return sum((p != 0).sum().item() for p in self.parameters())


# =============================================================================
# Grouped Head (clustered output projection)
# =============================================================================

class GroupedHead(nn.Module):
    """Output head that clusters vocab rows for efficient MoE-style inference.

    During **training**, acts as a standard ``nn.Linear(d_model, vocab_size)``.

    At **eval** time, ``build_clusters()`` partitions the learned weight rows
    into balanced clusters via k-means.  The forward pass then:

    1. Scores input against cluster centroids  →  top-k cluster selection
    2. Dispatches tokens to selected clusters (MoE-style sort + padded bmm)
    3. Scatters sub-logits back into a full ``(B, T, V)`` tensor

    This reduces the effective output-head matmul from ``O(V)`` to
    ``O(sqrt(V * k))`` per token.

    Args:
        d_model:    Input feature dimension.
        vocab_size: Number of output classes / vocabulary entries.
        k:          Number of clusters each token is routed to at eval time.
        bias:       Whether to include a bias vector.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = VOCAB_SIZE,
        k: int = 1,
        bias: bool = False,
        block_decode: bool = False,
        balance_tolerance: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.k = k
        self.block_decode = block_decode
        self.balance_tolerance = balance_tolerance
        self.n_clusters = math.ceil(math.sqrt(vocab_size * k))

        # Trainable parameters — identical to nn.Linear(d_model, vocab_size)
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self.bias = nn.Parameter(torch.zeros(vocab_size)) if bias else None

        # Eval-time buffers (populated by build_clusters)
        self.register_buffer("centroids", None)
        self.register_buffer("organized_weight", None)
        self.register_buffer("organized_bias", None)
        self.register_buffer("cluster_members", None)   # (n_clusters, max_cs) vocab ids
        self.register_buffer("cluster_sizes", None)      # (n_clusters,)

        self._clusters_valid = False

        # Init weights (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = d_model
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._clusters_valid = False
        elif self.block_decode and not self._clusters_valid:
            # PyTorch's nn.Module.eval() calls self.train(False) recursively,
            # so clustering must be triggered here (not in eval()) to fire
            # when a parent module calls .eval().
            self.build_clusters()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            logits: (B, T, vocab_size)
        """
        if self.training or not self.block_decode or not self._clusters_valid:
            return F.linear(x, self.weight, self.bias)
        return self._forward_clustered(x)

    def _forward_clustered(self, x: torch.Tensor) -> torch.Tensor:
        """MoE-style dispatch through cluster centroids."""
        B, T, d = x.shape
        N = B * T
        V = self.vocab_size
        k = self.k
        max_cs = self.cluster_members.shape[1]

        x_flat = x.reshape(N, d)

        # --- Stage 1: route to top-k clusters ---
        centroid_scores = x_flat @ self.centroids.T            # (N, n_clusters)
        _, top_clusters = centroid_scores.topk(k, dim=-1)      # (N, k)

        # --- Stage 2: MoE dispatch (replicate → sort → pad → bmm) ---
        token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
        cluster_ids = top_clusters.reshape(-1)                 # (N*k,)

        # Sort by cluster to group tokens heading to the same cluster
        sort_order = cluster_ids.argsort(stable=True)
        sorted_token_ids = token_ids[sort_order]
        sorted_cluster_ids = cluster_ids[sort_order]

        sorted_x = x_flat[sorted_token_ids]                   # (N*k, d)

        # Compute within-group positions (fully vectorized)
        unique_clusters, counts = torch.unique_consecutive(
            sorted_cluster_ids, return_counts=True,
        )
        n_active = unique_clusters.shape[0]
        max_count = int(counts.max().item())

        cluster_seq_idx = torch.repeat_interleave(
            torch.arange(n_active, device=x.device), counts,
        )
        group_starts = torch.zeros(n_active, dtype=torch.long, device=x.device)
        group_starts[1:] = counts[:-1].cumsum(0)
        within_pos = torch.arange(N * k, device=x.device) - group_starts[cluster_seq_idx]

        # Pad into (n_active, max_count, d)
        padded_x = x_flat.new_zeros(n_active, max_count, d)
        padded_x[cluster_seq_idx, within_pos] = sorted_x

        # Gather organized weights for active clusters: (n_active, max_cs, d)
        active_weights = self.organized_weight[unique_clusters]

        # --- Stage 3: batched matmul ---
        # (n_active, max_count, d) @ (n_active, d, max_cs) → (n_active, max_count, max_cs)
        sub_logits = torch.bmm(padded_x, active_weights.transpose(1, 2))

        # Add bias if present
        if self.organized_bias is not None:
            sub_logits = sub_logits + self.organized_bias[unique_clusters].unsqueeze(1)

        # --- Stage 4: scatter back into (N, V) ---
        valid_logits = sub_logits[cluster_seq_idx, within_pos]        # (N*k, max_cs)
        valid_vocab = self.cluster_members[sorted_cluster_ids]         # (N*k, max_cs)

        output = x_flat.new_full((N, V), float("-inf"))
        row_idx = sorted_token_ids.unsqueeze(1).expand(-1, max_cs)     # (N*k, max_cs)
        flat_idx = row_idx * V + valid_vocab                           # (N*k, max_cs)

        # Mask out padding entries (cluster_members == -1)
        valid_mask = valid_vocab >= 0
        output.view(-1).scatter_(
            0,
            flat_idx[valid_mask].view(-1),
            valid_logits[valid_mask].view(-1),
        )

        return output.view(B, T, V)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_clusters(self, max_iter: int = 50):
        """Run balanced k-means on weight rows and populate eval buffers.

        Args:
            max_iter: Maximum k-means iterations.

        Uses ``self.balance_tolerance`` (set at init) for the ± cluster size cap.
        """
        W = self.weight.data.float()  # (V, d)
        V, d = W.shape
        K = self.n_clusters
        target = V // K
        max_size = target + self.balance_tolerance

        # ---- k-means++ init ----
        centroids = self._kmeans_pp_init(W, K)

        for _ in range(max_iter):
            assignments = self._balanced_assign(W, centroids, max_size)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for c in range(K):
                mask = assignments == c
                if mask.any():
                    new_centroids[c] = W[mask].mean(dim=0)
                else:
                    new_centroids[c] = centroids[c]

            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        assignments = self._balanced_assign(W, centroids, max_size)

        # ---- Build organized buffers ----
        cluster_lists: list[list[int]] = [[] for _ in range(K)]
        for v_idx in range(V):
            cluster_lists[int(assignments[v_idx].item())].append(v_idx)

        actual_sizes = [len(cl) for cl in cluster_lists]
        max_cs = max(actual_sizes)

        members = torch.full((K, max_cs), -1, dtype=torch.long, device=W.device)
        org_weight = torch.zeros(K, max_cs, d, dtype=self.weight.dtype, device=self.weight.device)
        org_bias = None
        if self.bias is not None:
            org_bias = torch.zeros(K, max_cs, dtype=self.bias.dtype, device=self.bias.device)

        for c, idxs in enumerate(cluster_lists):
            n = len(idxs)
            t_idxs = torch.tensor(idxs, dtype=torch.long, device=W.device)
            members[c, :n] = t_idxs
            org_weight[c, :n] = self.weight.data[t_idxs]
            if org_bias is not None:
                org_bias[c, :n] = self.bias.data[t_idxs]

        sizes = torch.tensor(actual_sizes, dtype=torch.long, device=W.device)

        # Store as buffers (persist across .to() / .cuda() / state_dict)
        self.centroids = centroids.to(dtype=self.weight.dtype)
        self.organized_weight = org_weight
        self.organized_bias = org_bias
        self.cluster_members = members
        self.cluster_sizes = sizes
        self._clusters_valid = True

    # ---- k-means helpers ----

    @staticmethod
    def _kmeans_pp_init(X: torch.Tensor, K: int) -> torch.Tensor:
        """K-means++ centroid initialization."""
        V, d = X.shape
        centroids = torch.empty(K, d, device=X.device, dtype=X.dtype)
        idx = torch.randint(V, (1,)).item()
        centroids[0] = X[idx]

        for i in range(1, K):
            dists = torch.cdist(X, centroids[:i]).min(dim=1).values  # (V,)
            probs = dists / dists.sum()
            idx = torch.multinomial(probs, 1).item()
            centroids[i] = X[idx]
        return centroids

    @staticmethod
    def _balanced_assign(
        X: torch.Tensor,
        centroids: torch.Tensor,
        max_size: int,
    ) -> torch.Tensor:
        """Assign each row to the nearest centroid that has capacity.

        Iteratively assigns unassigned points to nearest non-full cluster.
        Typically converges in 2-3 passes.

        Args:
            X:         (V, d) data points
            centroids: (K, d) cluster centers
            max_size:  Maximum number of members per cluster

        Returns:
            assignments: (V,) cluster index per point
        """
        V = X.shape[0]
        K = centroids.shape[0]
        device = X.device

        distances = torch.cdist(X, centroids)  # (V, K)
        assignments = torch.full((V,), -1, dtype=torch.long, device=device)
        cluster_counts = torch.zeros(K, dtype=torch.long, device=device)
        remaining = torch.arange(V, device=device)

        while remaining.numel() > 0:
            sub_dists = distances[remaining]  # (n_rem, K)

            # Mask out full clusters
            full_mask = cluster_counts >= max_size
            if full_mask.all():
                # All clusters full — force-assign to least-full
                least_full = cluster_counts.argmin()
                assignments[remaining] = least_full
                break
            sub_dists[:, full_mask] = float("inf")

            nearest = sub_dists.argmin(dim=1)  # (n_rem,)
            assignments[remaining] = nearest

            # Resolve over-assignments: for each cluster, keep closest
            new_remaining = []
            for c in range(K):
                if full_mask[c]:
                    continue
                mask_c = nearest == c
                if not mask_c.any():
                    continue
                capacity = int(max_size - cluster_counts[c].item())
                indices_in_rem = mask_c.nonzero(as_tuple=True)[0]
                if indices_in_rem.numel() <= capacity:
                    cluster_counts[c] += indices_in_rem.numel()
                else:
                    c_dists = sub_dists[indices_in_rem, c]
                    keep = c_dists.argsort()[:capacity]
                    reject = c_dists.argsort()[capacity:]
                    cluster_counts[c] = max_size
                    reject_global = remaining[indices_in_rem[reject]]
                    assignments[reject_global] = -1
                    new_remaining.append(reject_global)

            if new_remaining:
                remaining = torch.cat(new_remaining)
            else:
                break

        return assignments

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_cluster_stats(self) -> dict:
        """Return clustering statistics (call after build_clusters)."""
        if not self._clusters_valid or self.cluster_sizes is None:
            return {}
        sizes = self.cluster_sizes.float()
        return {
            "n_clusters": self.n_clusters,
            "k": self.k,
            "rows_per_token": self.k * int(sizes.float().mean().item()),
            "cluster_size_mean": sizes.mean().item(),
            "cluster_size_min": int(sizes.min().item()),
            "cluster_size_max": int(sizes.max().item()),
            "cluster_size_std": sizes.std().item(),
        }


# Model registry for dynamic model creation
MODEL_REGISTRY = {
    "SimpleTransformer": SimpleTransformer,
    "MoETransformer": MoETransformer,
    "MoHTransformer": MoHTransformer,
    "MoH_MoETransformer": MoH_MoETransformer,
    "SparseDyadicTransformer": SparseDyadicTransformer,
    "FinalSparseTransformer": FinalSparseTransformer,
}


def create_model(model_class: str = "SimpleTransformer", **kwargs):
    """Factory function to create a model.
    
    Args:
        model_class: Name of the model class to instantiate
        **kwargs: Model-specific arguments
        
    Returns:
        Instantiated model
    """
    if model_class not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class: {model_class}. Available: {list(MODEL_REGISTRY.keys())}")
    
    # Filter kwargs to only include valid arguments for the model class
    model_cls = MODEL_REGISTRY[model_class]
    return model_cls(**kwargs)


if __name__ == "__main__":
    print("Testing SimpleTransformer:")
    model = create_model("SimpleTransformer", d_model=256, n_layers=4, n_heads=4, d_ff=512)
    total_params = model.count_parameters(count_zeros=True)
    nonzero_params = model.count_parameters(count_zeros=False)
    print(f"  Total params: {total_params:,}")
    print(f"  Nonzero params: {nonzero_params:,}")
    
    print("\nTesting MoETransformer:")
    moe_model = create_model("MoETransformer", d_model=256, n_layers=4, n_heads=4, d_expert=256, num_experts=8, top_k=2)
    total_params = moe_model.count_parameters(count_zeros=True)
    nonzero_params = moe_model.count_parameters(count_zeros=False)
    print(f"  Total params: {total_params:,}")
    print(f"  Nonzero params: {nonzero_params:,}")
    
    print("\nTesting MoHTransformer:")
    moh_model = create_model("MoHTransformer", d_model=256, n_layers=4, n_heads=8, k_active=2, d_ff=512)
    total_params = moh_model.count_parameters(count_zeros=True)
    nonzero_params = moh_model.count_parameters(count_zeros=False)
    print(f"  Total params: {total_params:,}")
    print(f"  Nonzero params: {nonzero_params:,}")
    print(f"  Active heads: {moh_model.k_active} of {moh_model.n_heads}")
    
    print("\nTesting MoH_MoETransformer:")
    moh_moe_model = create_model("MoH_MoETransformer", d_model=256, n_layers=4, n_heads=8, k_active=2, d_expert=256, num_experts=8, top_k=2)
    total_params = moh_moe_model.count_parameters(count_zeros=True)
    nonzero_params = moh_moe_model.count_parameters(count_zeros=False)
    print(f"  Total params: {total_params:,}")
    print(f"  Nonzero params: {nonzero_params:,}")
    print(f"  Active heads: {moh_moe_model.k_active} of {moh_moe_model.n_heads}")
    print(f"  Active experts: {moh_moe_model.layers[0].moe.top_k} of {moh_moe_model.num_experts}")
    
    print("\nTesting SparseDyadicTransformer:")
    sd_model = create_model(
        "SparseDyadicTransformer",
        d_model=128, n_layers=2, n_heads=4, d_ff=256,
        use_gradient_checkpointing=False,
    )
    total_params = sd_model.count_parameters(count_zeros=True)
    nonzero_params = sd_model.count_parameters(count_zeros=False)
    print(f"  Total params: {total_params:,}")
    print(f"  Nonzero params: {nonzero_params:,}")
    # Quick forward pass sanity check
    dummy_ids = torch.randint(0, VOCAB_SIZE, (2, 32))
    logits = sd_model(dummy_ids)
    print(f"  Forward pass: input {dummy_ids.shape} -> logits {logits.shape}")
    sp_stats = sd_model.get_sparsity_stats()
    print(f"  Attn density: {sp_stats['attn_density']:.4f}")
    print(f"  Sparse params (M+V): {len(sd_model.get_sparse_params())} tensors, "
          f"{sp_stats['attn_total']:,} elements")

    print("\nTesting GroupedHead:")
    d_test, v_test, k_test = 128, 5003, 2
    gh = GroupedHead(d_model=d_test, vocab_size=v_test, k=k_test, bias=True, block_decode=True)
    # Training forward — standard linear
    x_test = torch.randn(2, 16, d_test)
    logits_train = gh(x_test)
    print(f"  Train forward: input {x_test.shape} -> logits {logits_train.shape}")
    assert logits_train.shape == (2, 16, v_test), "Shape mismatch in train mode"
    # Switch to eval — triggers clustering
    gh.eval()
    stats = gh.get_cluster_stats()
    print(f"  n_clusters={stats['n_clusters']}, k={stats['k']}, "
          f"rows_per_token={stats['rows_per_token']}")
    print(f"  cluster sizes: mean={stats['cluster_size_mean']:.1f}, "
          f"min={stats['cluster_size_min']}, max={stats['cluster_size_max']}")
    logits_eval = gh(x_test)
    print(f"  Eval forward:  input {x_test.shape} -> logits {logits_eval.shape}")
    assert logits_eval.shape == (2, 16, v_test), "Shape mismatch in eval mode"
    # Check that non-selected entries are -inf and selected entries are finite
    n_inf = (logits_eval == float("-inf")).sum().item()
    n_finite = (logits_eval != float("-inf")).sum().item()
    total = logits_eval.numel()
    print(f"  Eval logits: {n_finite}/{total} finite ({100*n_finite/total:.1f}%), "
          f"{n_inf}/{total} masked")
    # Back to train should invalidate clusters
    gh.train()
    assert not gh._clusters_valid, "Clusters should be invalidated after .train()"
    print("  Lifecycle: .train() correctly invalidates clusters")
    print("  All GroupedHead tests passed!")

