"""Synthetic data generation for sparsity experiments.

Generates classification data by passing random inputs through a frozen
random SwiGLU network to get labels. This creates a learnable but non-trivial
mapping that we can use to test different sparsity methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Optional, Tuple


class SwiGLUBlock(nn.Module):
    """Single SwiGLU block: W2(silu(W1(x)) * W3(x))"""
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RandomSwiGLUClassifier(nn.Module):
    """Multi-layer SwiGLU classifier with random frozen weights.
    
    Used to generate synthetic classification data.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        n_layers: int = 3,
        n_classes: int = 20,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        # Build layers
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else hidden_dim
            layers.append(SwiGLUBlock(in_dim, hidden_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, n_classes, bias=False)
        
        # Initialize with Xavier scale for well-conditioned mappings
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.classifier(x)


class DataGenerator:
    """Generates synthetic classification data from a frozen random network.
    
    The idea: create a fixed random SwiGLU network, pass random x vectors through it,
    and use the argmax of the output as labels. This creates a learnable classification
    task that we can use to test sparsity methods.
    
    Args:
        input_dim: Dimension of input vectors
        hidden_dim: Hidden dimension of the generator network
        n_layers: Number of SwiGLU layers in the generator
        n_classes: Number of output classes
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        n_layers: int = 3,
        n_classes: int = 20,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.seed = seed
        
        # Set seed and build frozen network
        torch.manual_seed(seed)
        self.net = RandomSwiGLUClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_classes=n_classes,
        )
        
        # Freeze the network
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()
    
    def generate(
        self,
        n_samples: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate data on
            
        Returns:
            Tuple of (x, y) where:
                x: (n_samples, input_dim) random input vectors
                y: (n_samples,) class labels (0 to n_classes-1)
        """
        self.net.to(device)
        
        # Generate random inputs
        x = torch.randn(n_samples, self.input_dim, device=device)
        
        # Get labels from frozen network
        with torch.no_grad():
            logits = self.net(x)
            y = logits.argmax(dim=-1)
        
        return x, y
    
    def generate_dataset(
        self,
        n_train: int = 10000,
        n_val: int = 2000,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[TensorDataset, TensorDataset]:
        """Generate train and validation datasets.
        
        Args:
            n_train: Number of training samples
            n_val: Number of validation samples
            device: Device to generate data on
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Generate training data
        torch.manual_seed(self.seed + 1)  # Different seed for train data
        x_train, y_train = self.generate(n_train, device)
        
        # Generate validation data
        torch.manual_seed(self.seed + 2)  # Different seed for val data
        x_val, y_val = self.generate(n_val, device)
        
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        
        return train_dataset, val_dataset


def get_dataloaders(
    input_dim: int = 512,
    hidden_dim: int = 1024,
    n_layers: int = 3,
    n_classes: int = 20,
    n_train: int = 10000,
    n_val: int = 2000,
    batch_size: int = 128,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Tuple[DataLoader, DataLoader, DataGenerator]:
    """Convenience function to get train and val dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader, generator)
    """
    generator = DataGenerator(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_classes=n_classes,
        seed=seed,
    )
    
    train_dataset, val_dataset = generator.generate_dataset(
        n_train=n_train,
        n_val=n_val,
        device=device,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return train_loader, val_loader, generator


def get_real_dataloaders(
    data_dir: str = "real_data",
    batch_size: int = 256,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, dict]:
    """Load pre-computed real embeddings from disk.

    Embeddings must first be generated with prepare_real_data.py.

    Args:
        data_dir: Directory containing cached .pt files
        batch_size: Batch size for dataloaders
        n_train: If set, subsample training data to this many samples
        n_val: If set, subsample validation/test data to this many samples
        seed: Random seed for subsampling

    Returns:
        Tuple of (train_loader, val_loader, metadata_dict)
    """
    from pathlib import Path

    data_path = Path(data_dir)
    meta_path = data_path / "metadata.pt"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"No cached embeddings found in {data_dir}/. "
            f"Run 'python prepare_real_data.py' first to generate them."
        )

    print(f"Loading cached embeddings from {data_dir}/...")
    meta = torch.load(meta_path, weights_only=True)
    x_train = torch.load(data_path / "train_embeddings.pt", weights_only=True)
    y_train = torch.load(data_path / "train_labels.pt", weights_only=True)
    x_val = torch.load(data_path / "test_embeddings.pt", weights_only=True)
    y_val = torch.load(data_path / "test_labels.pt", weights_only=True)

    print(f"  Model: {meta.get('model', 'unknown')} ({meta.get('embed_dim', '?')}-dim)")
    print(f"  Train: {x_train.shape}, Val: {x_val.shape}")
    print(f"  Classes: {meta.get('n_classes', '?')} ({meta.get('class_names', [])})")

    # Optional subsampling
    if n_train is not None and n_train < len(x_train):
        torch.manual_seed(seed)
        perm = torch.randperm(len(x_train))[:n_train]
        x_train = x_train[perm]
        y_train = y_train[perm]
        print(f"  Subsampled train to {n_train} samples")

    if n_val is not None and n_val < len(x_val):
        torch.manual_seed(seed + 1)
        perm = torch.randperm(len(x_val))[:n_val]
        x_val = x_val[perm]
        y_val = y_val[perm]
        print(f"  Subsampled val to {n_val} samples")

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, meta


if __name__ == "__main__":
    # Quick test
    print("Testing data generation...")

    generator = DataGenerator(
        input_dim=512,
        hidden_dim=1024,
        n_layers=3,
        n_classes=20,
        seed=42,
    )

    x, y = generator.generate(1000)
    print(f"Generated data: x.shape={x.shape}, y.shape={y.shape}")
    print(f"Label distribution: {torch.bincount(y, minlength=20)}")

    train_loader, val_loader, _ = get_dataloaders(
        n_train=10000,
        n_val=2000,
        batch_size=128,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Check a batch
    x_batch, y_batch = next(iter(train_loader))
    print(f"Batch shapes: x={x_batch.shape}, y={y_batch.shape}")
    print("Data generation test passed!")
