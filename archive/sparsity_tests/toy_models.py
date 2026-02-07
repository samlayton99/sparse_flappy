"""Toy models for sparsity experiments.

Contains:
1. SwiGLUMLP - Single-layer dense SwiGLU + classifier
2. LowRankSwiGLUMLP - Single-layer SwiGLU with AB + S decomposition
3. PureABSwiGLUMLP - Single-layer SwiGLU with pure low-rank AB (no S)

Note: These are intentionally single-layer models to keep the experiments
focused on sparsity methods. The data-generating network uses multiple layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# =============================================================================
# Standard Dense SwiGLU
# =============================================================================

class SwiGLUMLP(nn.Module):
    """Single-layer SwiGLU MLP for classification.
    
    forward: classifier(W2(silu(W1(x)) * W3(x)))
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension for SwiGLU
        n_classes: Number of output classes
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        n_classes: int = 20,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        
        # Single SwiGLU layer
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.ln = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = self.ln(x)
        return self.classifier(x)
    
    def get_weight_matrices(self) -> List[torch.Tensor]:
        """Return list of all weight matrices for regularization."""
        return [self.w1.weight, self.w2.weight, self.w3.weight, self.classifier.weight]


# =============================================================================
# Low-Rank + Sparse (AB + S) Decomposition
# =============================================================================

class LowRankLinear(nn.Module):
    """Linear layer with AB + S decomposition.
    
    W = A @ B + S
    
    where:
    - A: (out_features, rank) - low-rank factor
    - B: (rank, in_features) - low-rank factor
    - S: (out_features, in_features) - sparse residual
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank component (k)
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Low-rank factors A @ B
        self.A = nn.Parameter(torch.empty(out_features, rank))
        self.B = nn.Parameter(torch.empty(rank, in_features))
        
        # Sparse residual S
        self.S = nn.Parameter(torch.zeros(out_features, in_features))
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize A, B with small values so A @ B starts near zero
        # This lets the sparse S matrix dominate initially
        nn.init.normal_(self.A, mean=0.0, std=0.01)
        nn.init.normal_(self.B, mean=0.0, std=0.01)
        # S starts at zero (will be regularized to stay sparse)
        nn.init.zeros_(self.S)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute W = A @ B + S
        W = self.A @ self.B + self.S
        return F.linear(x, W)
    
    def get_effective_weight(self) -> torch.Tensor:
        """Return the effective weight matrix W = A @ B + S."""
        return self.A @ self.B + self.S


class LowRankSwiGLUMLP(nn.Module):
    """Single-layer SwiGLU with AB + S decomposition for sparsity.
    
    Each weight matrix W is decomposed as W = A @ B + S where:
    - AB is a low-rank component (captures main structure)
    - S is a sparse residual (captures fine details)
    
    During training, we apply strong L1 regularization on S to encourage sparsity.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension for SwiGLU
        n_classes: Number of output classes
        rank: Rank k for the AB decomposition
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        n_classes: int = 20,
        rank: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.rank = rank
        
        # Single SwiGLU layer with low-rank decomposition
        self.w1 = LowRankLinear(input_dim, hidden_dim, rank=rank)
        self.w2 = LowRankLinear(hidden_dim, hidden_dim, rank=rank)
        self.w3 = LowRankLinear(input_dim, hidden_dim, rank=rank)
        self.ln = nn.LayerNorm(hidden_dim)
        self.classifier = LowRankLinear(hidden_dim, n_classes, rank=min(rank, n_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = self.ln(x)
        return self.classifier(x)
    
    def get_S_matrices(self) -> List[torch.Tensor]:
        """Return list of all sparse S matrices for regularization."""
        return [self.w1.S, self.w2.S, self.w3.S, self.classifier.S]
    
    def get_AB_matrices(self) -> List[tuple]:
        """Return list of all (A, B) tuples for low-rank components."""
        return [
            (self.w1.A, self.w1.B),
            (self.w2.A, self.w2.B),
            (self.w3.A, self.w3.B),
            (self.classifier.A, self.classifier.B),
        ]
    
    def get_effective_weights(self) -> List[torch.Tensor]:
        """Return list of all effective weight matrices (A @ B + S)."""
        return [
            self.w1.get_effective_weight(),
            self.w2.get_effective_weight(),
            self.w3.get_effective_weight(),
            self.classifier.get_effective_weight(),
        ]


# =============================================================================
# Pure Low-Rank (AB only, no S) Decomposition
# =============================================================================

class PureABLinear(nn.Module):
    """Linear layer with pure low-rank AB decomposition (no sparse residual).
    
    W = A @ B
    
    where:
    - A: (out_features, rank)
    - B: (rank, in_features)
    
    Sparsity = 1 - (|A| + |B|) / (out * in)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank approximation
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.A = nn.Parameter(torch.empty(out_features, rank))
        self.B = nn.Parameter(torch.empty(rank, in_features))
        
        self._init_weights()
    
    def _init_weights(self):
        # Xavier-style init scaled for the factored form
        nn.init.kaiming_normal_(self.A, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.B, mode='fan_in', nonlinearity='linear')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.A @ self.B)
    
    def get_effective_weight(self) -> torch.Tensor:
        return self.A @ self.B


class PureABSwiGLUMLP(nn.Module):
    """Single-layer SwiGLU with pure low-rank AB decomposition (no sparse residual).
    
    Each weight matrix W is represented as W = A @ B with rank r.
    No L1 regularization needed — sparsity is purely structural: (|A|+|B|) / dim(W).
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension for SwiGLU
        n_classes: Number of output classes
        rank: Rank for the AB decomposition
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        n_classes: int = 20,
        rank: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.rank = rank
        
        self.w1 = PureABLinear(input_dim, hidden_dim, rank=rank)
        self.w2 = PureABLinear(hidden_dim, hidden_dim, rank=rank)
        self.w3 = PureABLinear(input_dim, hidden_dim, rank=rank)
        self.ln = nn.LayerNorm(hidden_dim)
        self.classifier = PureABLinear(hidden_dim, n_classes, rank=min(rank, n_classes))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x = self.ln(x)
        return self.classifier(x)
    
    def get_AB_matrices(self) -> List[tuple]:
        """Return list of all (A, B) tuples."""
        return [
            (self.w1.A, self.w1.B),
            (self.w2.A, self.w2.B),
            (self.w3.A, self.w3.B),
            (self.classifier.A, self.classifier.B),
        ]


# =============================================================================
# Magnitude Pruning
# =============================================================================

@torch.no_grad()
def magnitude_prune(model: nn.Module, threshold: float = 1e-4,
                    optimizer: torch.optim.Optimizer = None,
                    exclude_types: tuple = (nn.LayerNorm,)):
    """Zero out parameters with absolute value below threshold.
    
    For LowRank models, ONLY prunes S matrices (not A or B), since pruning
    low-rank factors destroys structure non-locally.
    
    Also zeros the corresponding Adam optimizer state (exp_avg, exp_avg_sq)
    so momentum doesn't immediately restore pruned weights.
    
    Skips parameters belonging to excluded module types (e.g. LayerNorm).
    
    Args:
        model: PyTorch model
        threshold: Magnitude threshold below which parameters are zeroed
        optimizer: If provided, also zero the optimizer state for pruned indices
        exclude_types: Tuple of module types whose parameters should NOT be pruned
    """
    # Collect parameter ids to exclude: LayerNorm params + A/B of LowRankLinear
    excluded_ids = set()
    for m in model.modules():
        if isinstance(m, exclude_types):
            for p in m.parameters():
                excluded_ids.add(id(p))
        # For LowRankLinear, only allow pruning S -- skip A and B
        if isinstance(m, LowRankLinear):
            excluded_ids.add(id(m.A))
            excluded_ids.add(id(m.B))
    
    for p in model.parameters():
        if id(p) in excluded_ids:
            continue
        mask = p.abs() < threshold
        p.data[mask] = 0.0
        
        # Zero out Adam state so momentum doesn't restore pruned weights
        if optimizer is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param is p and param in optimizer.state:
                        state = optimizer.state[param]
                        if 'exp_avg' in state:
                            state['exp_avg'][mask] = 0.0
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'][mask] = 0.0
                        break


# =============================================================================
# Sparsity Utilities
# =============================================================================

def compute_sparsity(model: nn.Module, threshold: float = 1e-4) -> float:
    """Compute the sparsity of a model (fraction of near-zero weights).
    
    Args:
        model: PyTorch model
        threshold: Values with abs < threshold are considered zero
        
    Returns:
        Sparsity as a fraction (0.0 to 1.0)
    """
    total = 0
    zeros = 0
    
    for p in model.parameters():
        total += p.numel()
        zeros += (p.abs() < threshold).sum().item()
    
    return zeros / total if total > 0 else 0.0


def compute_S_sparsity(model: nn.Module, threshold: float = 1e-4) -> float:
    """Compute sparsity of ONLY the S matrices in a LowRank model.
    
    Returns fraction of S entries that are near-zero. This is the meaningful
    sparsity metric for AB+S models since A@B is always dense.
    
    Args:
        model: LowRankSwiGLUMLP model
        threshold: Values with abs < threshold are considered zero
        
    Returns:
        Sparsity of S as a fraction (0.0 to 1.0), or 0.0 if not a LowRank model
    """
    if not hasattr(model, 'get_S_matrices'):
        return 0.0
    
    total = 0
    zeros = 0
    for S in model.get_S_matrices():
        total += S.numel()
        zeros += (S.abs() < threshold).sum().item()
    
    return zeros / total if total > 0 else 0.0


def compute_effective_sparsity(model: nn.Module, threshold: float = 1e-4) -> float:
    """Compute sparsity of effective weight matrices for LowRank models.
    
    For LowRankSwiGLUMLP, computes sparsity of A @ B + S matrices.
    For regular models, falls back to compute_sparsity.
    
    Args:
        model: PyTorch model
        threshold: Values with abs < threshold are considered zero
        
    Returns:
        Sparsity as a fraction (0.0 to 1.0)
    """
    if hasattr(model, 'get_effective_weights'):
        weights = model.get_effective_weights()
        total = sum(w.numel() for w in weights)
        zeros = sum((w.abs() < threshold).sum().item() for w in weights)
        return zeros / total if total > 0 else 0.0
    else:
        return compute_sparsity(model, threshold)


def compute_param_efficiency(model: nn.Module, threshold: float = 1e-4) -> dict:
    """Compute the TRUE parameter efficiency for any model (weight matrices only).
    
    For SwiGLUMLP (dense):
        sparsity = 1 - nnz(W) / numel(W)  across all weight matrices
    
    For LowRankSwiGLUMLP (AB+S):
        sparsity = 1 - (numel(A) + numel(B) + nnz(S)) / dim(S)
        where dim(S) = out_features * in_features = the full dense weight size
    
    Only counts weight matrices — LayerNorm parameters are excluded since
    they are small overhead, not part of the compression story.
    
    Returns a dict with:
        - dense_params: total elements in the equivalent dense weight matrices
        - active_params: non-zero params you actually need to store
        - compression_ratio: dense_params / active_params
        - param_sparsity: 1 - active_params / dense_params
    """
    if hasattr(model, 'get_S_matrices') and hasattr(model, 'get_AB_matrices'):
        # LowRank AB+S model: 1 - (|A| + |B| + nnz(S)) / dim(S) per weight matrix
        dense_params = 0
        active_params = 0
        
        for m in model.modules():
            if isinstance(m, LowRankLinear):
                dim_S = m.in_features * m.out_features       # full dense size
                numel_A = m.A.numel()                        # out * rank
                numel_B = m.B.numel()                        # rank * in
                nnz_S = (m.S.abs() >= threshold).sum().item() # non-zero entries in S
                
                dense_params += dim_S
                active_params += numel_A + numel_B + nnz_S
        
        return {
            "dense_params": dense_params,
            "active_params": active_params,
            "compression_ratio": dense_params / active_params if active_params > 0 else float('inf'),
            "param_sparsity": 1.0 - active_params / dense_params if dense_params > 0 else 0.0,
        }
    elif hasattr(model, 'get_AB_matrices') and not hasattr(model, 'get_S_matrices'):
        # Pure AB model: 1 - (|A| + |B|) / (out * in) per weight matrix
        dense_params = 0
        active_params = 0
        
        for m in model.modules():
            if isinstance(m, PureABLinear):
                dim_W = m.in_features * m.out_features  # full dense size
                numel_A = m.A.numel()                   # out * rank
                numel_B = m.B.numel()                   # rank * in
                
                dense_params += dim_W
                active_params += numel_A + numel_B
        
        return {
            "dense_params": dense_params,
            "active_params": active_params,
            "compression_ratio": dense_params / active_params if active_params > 0 else float('inf'),
            "param_sparsity": 1.0 - active_params / dense_params if dense_params > 0 else 0.0,
        }
    else:
        # Dense model: 1 - nnz(W) / numel(W) across all weight matrices (skip LayerNorm)
        dense_params = 0
        active_params = 0
        
        ln_ids = set()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    ln_ids.add(id(p))
        
        for p in model.parameters():
            if id(p) in ln_ids:
                continue
            dense_params += p.numel()
            active_params += (p.abs() >= threshold).sum().item()
        
        return {
            "dense_params": dense_params,
            "active_params": active_params,
            "compression_ratio": dense_params / active_params if active_params > 0 else float('inf'),
            "param_sparsity": 1.0 - active_params / dense_params if dense_params > 0 else 0.0,
        }


def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Quick test
    print("Testing toy models (single-layer)...")
    
    # Test SwiGLUMLP
    print("\n1. SwiGLUMLP:")
    model1 = SwiGLUMLP(input_dim=512, hidden_dim=1024, n_classes=20)
    x = torch.randn(4, 512)
    out = model1(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    print(f"   Parameters: {count_parameters(model1):,}")
    print(f"   Sparsity: {compute_sparsity(model1):.2%}")
    
    # Test LowRankSwiGLUMLP
    print("\n2. LowRankSwiGLUMLP (rank=32):")
    model2 = LowRankSwiGLUMLP(input_dim=512, hidden_dim=1024, n_classes=20, rank=32)
    out = model2(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    print(f"   Parameters: {count_parameters(model2):,}")
    print(f"   S matrices: {len(model2.get_S_matrices())}")
    print(f"   AB pairs: {len(model2.get_AB_matrices())}")
    print(f"   Effective sparsity: {compute_effective_sparsity(model2):.2%}")
    
    # Test magnitude pruning
    print("\n3. Magnitude pruning test:")
    print(f"   Before pruning: {compute_sparsity(model1):.2%} sparse")
    magnitude_prune(model1, threshold=0.1)
    print(f"   After pruning (threshold=0.1): {compute_sparsity(model1):.2%} sparse")
    
    print("\nAll tests passed!")
