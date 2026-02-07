"""Model with memory access counting for the FineWeb challenge.

This is a copy of model.py with added memory counting infrastructure.
When count_memory=True in forward(), it calculates total memory accesses in MB.

Memory counting philosophy:
- Count parameter reads
- Count activation reads and writes
- Count reshapes/permutes as single touch of all elements
- Datatype-aware (tracks bytes based on tensor dtype)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from math import prod

from rope import RotaryPositionalEmbedding

# GPT-2 tokenizer vocab size
VOCAB_SIZE = 50257
SEQ_LEN = 512  # 513 - 1 for causal LM


# =============================================================================
# Memory Counter Infrastructure
# =============================================================================

def dtype_to_bytes(dtype: torch.dtype) -> int:
    """Convert torch dtype to bytes per element."""
    dtype_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float64: 8,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    if dtype not in dtype_bytes:
        raise ValueError(f"Unknown dtype: {dtype}")
    return dtype_bytes[dtype]


class MemoryCounter:
    """Accumulates memory access counts with dtype awareness.
    
    Usage:
        counter = MemoryCounter()
        counter.add(num_elements, dtype)
        print(f"Total: {counter.mb:.2f} MB")
    """
    
    def __init__(self):
        self.total_bytes = 0
        self.total_elements = 0
        # Breakdown by category for debugging
        self.breakdown = {
            "params": 0,
            "activations": 0,
            "attention_matrix": 0,
        }
    
    def add(self, num_elements: int, dtype: torch.dtype, category: str = "activations"):
        """Add memory access.
        
        Args:
            num_elements: Number of elements accessed
            dtype: Data type of the elements
            category: One of "params", "activations", "attention_matrix"
        """
        bytes_accessed = num_elements * dtype_to_bytes(dtype)
        self.total_bytes += bytes_accessed
        self.total_elements += num_elements
        if category in self.breakdown:
            self.breakdown[category] += bytes_accessed
    
    def reset(self):
        """Reset all counters."""
        self.total_bytes = 0
        self.total_elements = 0
        for key in self.breakdown:
            self.breakdown[key] = 0
    
    @property
    def mb(self) -> float:
        """Total memory accessed in megabytes."""
        return self.total_bytes / (1024 * 1024)
    
    @property
    def gb(self) -> float:
        """Total memory accessed in gigabytes."""
        return self.total_bytes / (1024 * 1024 * 1024)
    
    def summary(self) -> str:
        """Return a formatted summary of memory accesses."""
        lines = [
            f"Total Memory Accesses: {self.mb:.2f} MB ({self.total_elements:,} elements)",
            f"  Parameters:       {self.breakdown['params'] / (1024*1024):.2f} MB",
            f"  Activations:      {self.breakdown['activations'] / (1024*1024):.2f} MB", 
            f"  Attention Matrix: {self.breakdown['attention_matrix'] / (1024*1024):.2f} MB",
        ]
        return "\n".join(lines)


# =============================================================================
# Helper Functions for Composable Memory Counting
# =============================================================================

def count_matmul(
    counter: MemoryCounter,
    input_elements: int,
    weight_elements: int, 
    output_elements: int,
    activation_dtype: torch.dtype,
    weight_dtype: torch.dtype = None,
):
    """Count memory for matrix multiplication: read input, read weights, write output.
    
    Args:
        counter: MemoryCounter to accumulate into
        input_elements: Number of elements in input tensor
        weight_elements: Number of elements in weight tensor
        output_elements: Number of elements in output tensor
        activation_dtype: Dtype of activations
        weight_dtype: Dtype of weights (defaults to activation_dtype)
    """
    if weight_dtype is None:
        weight_dtype = activation_dtype
    
    counter.add(input_elements, activation_dtype, "activations")
    counter.add(weight_elements, weight_dtype, "params")
    counter.add(output_elements, activation_dtype, "activations")


def count_elementwise(
    counter: MemoryCounter,
    num_elements: int,
    dtype: torch.dtype,
    n_reads: int = 2,
    n_writes: int = 1,
):
    """Count memory for elementwise operations.
    
    Args:
        counter: MemoryCounter to accumulate into
        num_elements: Number of elements in the tensor
        dtype: Data type
        n_reads: Number of input tensors to read
        n_writes: Number of output tensors to write
    """
    counter.add(num_elements * n_reads, dtype, "activations")
    counter.add(num_elements * n_writes, dtype, "activations")


def count_reshape(
    counter: MemoryCounter,
    num_elements: int,
    dtype: torch.dtype,
):
    """Count memory for reshape/permute/transpose (touch all elements once).
    
    Args:
        counter: MemoryCounter to accumulate into
        num_elements: Number of elements in the tensor
        dtype: Data type
    """
    counter.add(num_elements, dtype, "activations")


def count_attention_scores(
    counter: MemoryCounter,
    B: int,
    H: int,
    T: int,
    dtype: torch.dtype,
):
    """Count memory for materialized attention matrix (T² scaling).
    
    This counts:
    - Write attention scores S (BHT²)
    - Read/write softmax P (2 * BHT²)
    - Read P for output (BHT²)
    
    Total: 4 * BHT²
    
    Args:
        counter: MemoryCounter to accumulate into
        B: Batch size
        H: Number of heads
        T: Sequence length
        dtype: Data type
    """
    attn_matrix_elements = B * H * T * T
    # Write S, read S for softmax, write P, read P for output
    counter.add(4 * attn_matrix_elements, dtype, "attention_matrix")


def count_layernorm(
    counter: MemoryCounter,
    E: int,  # B * T
    C: int,  # d_model
    dtype: torch.dtype,
):
    """Count memory for LayerNorm: read x, write x, read gamma+beta.
    
    Total: 2EC + 2C
    
    Args:
        counter: MemoryCounter to accumulate into
        E: Number of token positions (B * T)
        C: Hidden dimension (d_model)
        dtype: Data type
    """
    counter.add(2 * E * C, dtype, "activations")  # read + write
    counter.add(2 * C, dtype, "params")  # gamma + beta


def count_residual_add(
    counter: MemoryCounter,
    num_elements: int,
    dtype: torch.dtype,
):
    """Count memory for residual addition: read x, read y, write out.
    
    Total: 3 * num_elements
    
    Args:
        counter: MemoryCounter to accumulate into
        num_elements: Number of elements (E * C)
        dtype: Data type
    """
    counter.add(3 * num_elements, dtype, "activations")


def count_embedding_lookup(
    counter: MemoryCounter,
    B: int,
    T: int,
    C: int,
    dtype: torch.dtype,
):
    """Count memory for embedding lookup: read embeddings, write output.
    
    Note: This is a simplified count. In practice, only accessed rows are read,
    but we count the output size as a proxy.
    
    Args:
        counter: MemoryCounter to accumulate into
        B: Batch size
        T: Sequence length
        C: Embedding dimension
        dtype: Data type
    """
    # Read indices (small, ignore) + read embedding vectors + write output
    counter.add(B * T * C, dtype, "params")  # embedding lookups
    counter.add(B * T * C, dtype, "activations")  # write output


def count_lowrank_proj(
    counter: MemoryCounter,
    E: int,  # B * T
    C: int,  # input/output dim
    R: int,  # rank
    dtype: torch.dtype,
):
    """Count memory for low-rank projection (AB decomposition).
    
    x @ A: read x (EC), read A (CR), write intermediate (ER)
    intermediate @ B: read intermediate (ER), read B (RC), write output (EC)
    
    Total: 2EC + 2ER + 2CR
    
    Args:
        counter: MemoryCounter to accumulate into
        E: Number of token positions (B * T)
        C: Hidden dimension
        R: Rank of decomposition
        dtype: Data type
    """
    # First matmul: x @ A
    counter.add(E * C, dtype, "activations")  # read x
    counter.add(C * R, dtype, "params")  # read A
    counter.add(E * R, dtype, "activations")  # write intermediate
    
    # Second matmul: intermediate @ B
    counter.add(E * R, dtype, "activations")  # read intermediate
    counter.add(R * C, dtype, "params")  # read B
    counter.add(E * C, dtype, "activations")  # write output


# =============================================================================
# Model Components with Memory Counting
# =============================================================================

class SimpleTransformer(nn.Module):
    """A minimal transformer for language modeling with memory counting.
    
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
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        
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
    
    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        count_memory: bool = False,
        memory_dtype: torch.dtype = torch.bfloat16,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """Forward pass with optional memory counting.
        
        Args:
            input_ids: Input token IDs [B, T]
            attention_mask: Optional attention mask
            count_memory: If True, also return memory accesses in MB
            memory_dtype: Dtype to use for memory calculations (default: bfloat16)
        
        Returns:
            logits: Output logits [B, T, vocab_size]
            memory_mb: (only if count_memory=True) Total memory accesses in MB
        """
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
        
        if count_memory:
            counter = MemoryCounter()
            self._count_total_memory(B, T, counter, memory_dtype)
            return logits, counter.mb
        
        return logits
    
    def _count_total_memory(
        self, 
        B: int, 
        T: int, 
        counter: MemoryCounter, 
        dtype: torch.dtype,
    ):
        """Count total memory accesses for a forward pass.
        
        Args:
            B: Batch size
            T: Sequence length
            counter: MemoryCounter to accumulate into
            dtype: Data type for memory calculations
        """
        C = self.d_model
        E = B * T
        
        # Embedding lookup
        count_embedding_lookup(counter, B, T, C, dtype)
        
        # Transformer layers
        for layer in self.layers:
            layer._count_memory(B, T, counter, dtype)
        
        # Final LayerNorm
        count_layernorm(counter, E, C, dtype)
        
        # Output head (weight-tied with embedding, so no extra params)
        # read x (EC), read weight (vocab*C but tied), write logits (E*vocab)
        counter.add(E * C, dtype, "activations")  # read x
        # Weight is tied to embedding, already counted
        counter.add(E * self.vocab_size, dtype, "activations")  # write logits
    
    def count_memory_mb(
        self, 
        batch_size: int, 
        seq_len: int, 
        dtype: torch.dtype = torch.bfloat16,
    ) -> float:
        """Convenience method to get memory accesses without running forward.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            dtype: Data type for memory calculations
        
        Returns:
            Total memory accesses in MB
        """
        counter = MemoryCounter()
        self._count_total_memory(batch_size, seq_len, counter, dtype)
        return counter.mb
    
    def count_memory_detailed(
        self, 
        batch_size: int, 
        seq_len: int, 
        dtype: torch.dtype = torch.bfloat16,
    ) -> MemoryCounter:
        """Get detailed memory breakdown without running forward.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            dtype: Data type for memory calculations
        
        Returns:
            MemoryCounter with full breakdown
        """
        counter = MemoryCounter()
        self._count_total_memory(batch_size, seq_len, counter, dtype)
        return counter
    
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
    """Single transformer block with pre-norm and memory counting."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = SwiGLUFF(d_model, d_ff)
    
    def forward(self, x, causal_mask, attention_mask, rope, positions):
        x = x + self.attn(self.ln1(x), causal_mask, attention_mask, rope, positions)
        x = x + self.ff(self.ln2(x))
        return x
    
    def _count_memory(
        self, 
        B: int, 
        T: int, 
        counter: MemoryCounter, 
        dtype: torch.dtype,
    ):
        """Count memory accesses for this block.
        
        Block does: LN1 + MHA + ResAdd + LN2 + SwiGLU + ResAdd
        Total: 4BHT² + 31EC + 8EF + 3CF + 4C² + 4C
        
        Args:
            B: Batch size
            T: Sequence length
            counter: MemoryCounter to accumulate into
            dtype: Data type
        """
        C = self.d_model
        E = B * T
        
        # LN1
        count_layernorm(counter, E, C, dtype)
        
        # MHA
        self.attn._count_memory(B, T, counter, dtype)
        
        # Residual add 1
        count_residual_add(counter, E * C, dtype)
        
        # LN2
        count_layernorm(counter, E, C, dtype)
        
        # SwiGLU FFN
        self.ff._count_memory(B, T, counter, dtype)
        
        # Residual add 2
        count_residual_add(counter, E * C, dtype)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE, Flash Attention, and memory counting."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
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
    
    def _count_memory(
        self, 
        B: int, 
        T: int, 
        counter: MemoryCounter, 
        dtype: torch.dtype,
    ):
        """Count memory accesses for multi-head attention.
        
        Total: 4BHT² + 18EC + 4C²
        
        Breakdown:
        - QKV projection: 4EC + 3C²
        - Reshape QKV: 3EC
        - RoPE on q,k: 4EC
        - Attention scores (S = qk^T): 2EC + BHT²
        - Softmax: 2BHT²
        - Attention output (O = PV): BHT² + 2EC
        - Reshape O: EC
        - Output projection: 2EC + C²
        
        Args:
            B: Batch size
            T: Sequence length
            counter: MemoryCounter to accumulate into
            dtype: Data type
        """
        C = self.d_model
        H = self.n_heads
        E = B * T
        
        # (a) QKV projection: x @ W_qkv
        # read x (EC), read params (3C²), write qkv (3EC)
        counter.add(E * C, dtype, "activations")
        counter.add(3 * C * C, dtype, "params")
        counter.add(3 * E * C, dtype, "activations")
        
        # (b) Reshape/permute QKV (touch once)
        counter.add(3 * E * C, dtype, "activations")
        
        # (c) RoPE on q and k (read+write both)
        # Each of q, k has EC elements
        counter.add(4 * E * C, dtype, "activations")
        
        # (d) Attention scores S = qk^T (explicitly materialized)
        # read q (EC), read k (EC), write S (BHT²)
        counter.add(E * C, dtype, "activations")
        counter.add(E * C, dtype, "activations")
        counter.add(B * H * T * T, dtype, "attention_matrix")
        
        # (e) Softmax: read S, write P
        counter.add(2 * B * H * T * T, dtype, "attention_matrix")
        
        # (f) Attention output O = PV
        # read P (BHT²), read V (EC), write O (EC)
        counter.add(B * H * T * T, dtype, "attention_matrix")
        counter.add(E * C, dtype, "activations")
        counter.add(E * C, dtype, "activations")
        
        # (g) Reshape/transpose O -> (B, T, C) (touch once)
        counter.add(E * C, dtype, "activations")
        
        # (h) Output projection: O @ W_proj
        # read O (EC), read params (C²), write output (EC)
        counter.add(E * C, dtype, "activations")
        counter.add(C * C, dtype, "params")
        counter.add(E * C, dtype, "activations")


class SwiGLUFF(nn.Module):
    """SwiGLU feedforward with memory counting."""
    
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
    
    def _count_memory(
        self, 
        B: int, 
        T: int, 
        counter: MemoryCounter, 
        dtype: torch.dtype,
    ):
        """Count memory accesses for SwiGLU FFN.
        
        Total: 3EC + 8EF + 3CF
        
        Breakdown:
        - w1: x -> h1: read x (EC), read w1 (CF), write h1 (EF)
        - w3: x -> h3: read x (EC), read w3 (CF), write h3 (EF)
        - silu(h1): read h1 (EF), write (EF)
        - h1 * h3: read h1 (EF), read h3 (EF), write (EF)
        - w2: h -> out: read h (EF), read w2 (FC), write out (EC)
        
        Args:
            B: Batch size
            T: Sequence length
            counter: MemoryCounter to accumulate into
            dtype: Data type
        """
        C = self.d_model
        F = self.d_ff
        E = B * T
        
        # w1: x -> h1
        counter.add(E * C, dtype, "activations")  # read x
        counter.add(C * F, dtype, "params")  # read w1
        counter.add(E * F, dtype, "activations")  # write h1
        
        # w3: x -> h3
        counter.add(E * C, dtype, "activations")  # read x
        counter.add(C * F, dtype, "params")  # read w3
        counter.add(E * F, dtype, "activations")  # write h3
        
        # silu(h1): read + write
        counter.add(2 * E * F, dtype, "activations")
        
        # h1 * h3: read both, write result
        counter.add(3 * E * F, dtype, "activations")
        
        # w2: h -> out
        counter.add(E * F, dtype, "activations")  # read h
        counter.add(F * C, dtype, "params")  # read w2
        counter.add(E * C, dtype, "activations")  # write out


def create_model(**kwargs):
    """Factory function to create a model."""
    return SimpleTransformer(**kwargs)


if __name__ == "__main__":
    # Test memory counting
    print("=" * 60)
    print("Memory Counter Test")
    print("=" * 60)
    
    # Create model with default params
    model = create_model()
    total_params = model.count_parameters(count_zeros=True)
    nonzero_params = model.count_parameters(count_zeros=False)
    
    print(f"\nModel Configuration:")
    print(f"  d_model: {model.d_model}")
    print(f"  n_heads: {model.n_heads}")
    print(f"  n_layers: {model.n_layers}")
    print(f"  d_ff: {model.d_ff}")
    print(f"  Total params: {total_params:,}")
    print(f"  Nonzero params: {nonzero_params:,}")
    
    # Test memory counting for different batch sizes and sequence lengths
    print(f"\nMemory Access Counts (dtype=bfloat16):")
    print("-" * 60)
    
    for B in [1, 4, 32]:
        for T in [128, 512]:
            counter = model.count_memory_detailed(B, T, dtype=torch.bfloat16)
            print(f"B={B:2d}, T={T:3d}: {counter.mb:8.2f} MB")
            
    # Detailed breakdown for B=1, T=512
    print(f"\nDetailed Breakdown (B=1, T=512, bfloat16):")
    print("-" * 60)
    counter = model.count_memory_detailed(1, 512, dtype=torch.bfloat16)
    print(counter.summary())
    
    # Compare dtypes
    print(f"\nDtype Comparison (B=32, T=512):")
    print("-" * 60)
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        mb = model.count_memory_mb(32, 512, dtype=dtype)
        print(f"  {str(dtype):15s}: {mb:8.2f} MB")
    
    # Test with actual forward pass
    print(f"\nForward Pass with Memory Counting (B=4, T=128):")
    print("-" * 60)
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randint(0, 50257, (4, 128))
        logits, memory_mb = model(dummy_input, count_memory=True)
        print(f"  Output shape: {logits.shape}")
        print(f"  Memory accesses: {memory_mb:.2f} MB")
