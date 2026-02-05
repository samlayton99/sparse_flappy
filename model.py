"""Starter model for the FineWeb challenge.

Your goal: achieve val loss < 3.0 with the fewest non-zero parameters.
Modify this model architecture to be as sparse/efficient as possible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        self._init_weights(n_layers)
    
    def _init_weights(self, n_layers):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
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

def create_model(**kwargs):
    """Factory function to create a model."""
    return SimpleTransformer(**kwargs)


if __name__ == "__main__":
    model = create_model()
    total_params = model.count_parameters(count_zeros=True)
    nonzero_params = model.count_parameters(count_zeros=False)
    

