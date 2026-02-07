"""Prepare real text classification data for sparsity experiments.

Generates 1024-dim embeddings from AG News using BAAI/bge-large-en-v1.5,
a top-ranked sentence embedding model. Embeddings are cached to disk so
training iterations are instant.

AG News: 4 classes (World, Sports, Business, Sci/Tech), 120K train, 7.6K test.

Usage:
    python prepare_real_data.py                          # default: bge-large (1024-dim)
    python prepare_real_data.py --model bge-base         # smaller: bge-base (768-dim)
    python prepare_real_data.py --output_dir real_data   # custom output directory
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Model configs: name -> (huggingface_id, embedding_dim)
MODEL_CONFIGS = {
    "bge-large": ("BAAI/bge-large-en-v1.5", 1024),
    "bge-base": ("BAAI/bge-base-en-v1.5", 768),
}


def load_ag_news():
    """Load AG News dataset from HuggingFace."""
    from datasets import load_dataset

    print("Downloading AG News dataset...")
    ds = load_dataset("fancyzhx/ag_news")

    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]

    print(f"  Train: {len(train_texts)} samples")
    print(f"  Test:  {len(test_texts)} samples")
    print(f"  Classes: {sorted(set(train_labels))} (0=World, 1=Sports, 2=Business, 3=Sci/Tech)")

    # Check class balance
    from collections import Counter
    dist = Counter(train_labels)
    print(f"  Train distribution: {dict(sorted(dist.items()))}")

    return train_texts, train_labels, test_texts, test_labels


def compute_embeddings(texts, model, tokenizer, device, batch_size=128, max_length=256):
    """Compute embeddings for a list of texts using a pre-trained model.

    Uses CLS token pooling and L2 normalization (standard for BGE models).

    Args:
        texts: List of strings
        model: Pre-trained transformer model
        tokenizer: Corresponding tokenizer
        device: torch device
        batch_size: Inference batch size
        max_length: Max token length (256 is plenty for news headlines+snippets)

    Returns:
        Tensor of shape (len(texts), embedding_dim)
    """
    model.eval()
    model.to(device)
    all_embeddings = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i : i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            # CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]
            # L2 normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Prepare AG News embeddings for sparsity experiments")
    parser.add_argument(
        "--model",
        type=str,
        default="bge-large",
        choices=list(MODEL_CONFIGS.keys()),
        help="Embedding model to use (default: bge-large, 1024-dim)",
    )
    parser.add_argument("--output_dir", type=str, default="real_data", help="Output directory for cached embeddings")
    parser.add_argument("--batch_size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected if not set)")
    args = parser.parse_args()

    # Resolve model
    hf_model_id, embed_dim = MODEL_CONFIGS[args.model]
    print(f"Model: {args.model} ({hf_model_id}), embedding dim: {embed_dim}")

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already computed
    train_emb_path = output_dir / "train_embeddings.pt"
    test_emb_path = output_dir / "test_embeddings.pt"
    meta_path = output_dir / "metadata.pt"

    if train_emb_path.exists() and test_emb_path.exists() and meta_path.exists():
        meta = torch.load(meta_path, weights_only=True)
        print(f"\nEmbeddings already exist in {output_dir}/")
        print(f"  Model: {meta.get('model', 'unknown')}")
        print(f"  Embedding dim: {meta.get('embed_dim', 'unknown')}")
        print(f"  Train samples: {meta.get('n_train', 'unknown')}")
        print(f"  Test samples: {meta.get('n_test', 'unknown')}")
        print("Delete the directory to recompute.")
        return

    # Load dataset
    train_texts, train_labels, test_texts, test_labels = load_ag_news()

    # Load model and tokenizer
    from transformers import AutoModel, AutoTokenizer

    print(f"\nLoading model: {hf_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModel.from_pretrained(hf_model_id)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute embeddings
    print(f"\nComputing train embeddings ({len(train_texts)} samples)...")
    t0 = time.time()
    train_embeddings = compute_embeddings(
        train_texts, model, tokenizer, device, batch_size=args.batch_size
    )
    t1 = time.time()
    print(f"  Done in {t1 - t0:.1f}s, shape: {train_embeddings.shape}")

    print(f"\nComputing test embeddings ({len(test_texts)} samples)...")
    test_embeddings = compute_embeddings(
        test_texts, model, tokenizer, device, batch_size=args.batch_size
    )
    t2 = time.time()
    print(f"  Done in {t2 - t1:.1f}s, shape: {test_embeddings.shape}")

    # Convert labels to tensors
    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long)

    # Save
    print(f"\nSaving to {output_dir}/...")
    torch.save(train_embeddings, train_emb_path)
    torch.save(test_embeddings, test_emb_path)
    torch.save(train_labels_t, output_dir / "train_labels.pt")
    torch.save(test_labels_t, output_dir / "test_labels.pt")
    torch.save(
        {
            "model": args.model,
            "hf_model_id": hf_model_id,
            "embed_dim": embed_dim,
            "n_train": len(train_texts),
            "n_test": len(test_texts),
            "n_classes": len(set(train_labels)),
            "class_names": ["World", "Sports", "Business", "Sci/Tech"],
        },
        meta_path,
    )

    # Print summary
    train_size_mb = train_embeddings.numel() * 4 / 1024 / 1024
    test_size_mb = test_embeddings.numel() * 4 / 1024 / 1024
    print(f"\n=== Summary ===")
    print(f"  Train: {train_embeddings.shape} ({train_size_mb:.1f} MB)")
    print(f"  Test:  {test_embeddings.shape} ({test_size_mb:.1f} MB)")
    print(f"  Labels: {len(set(train_labels))} classes")
    print(f"  Total disk: {train_size_mb + test_size_mb:.1f} MB")
    print(f"\nDone! Run training with:")
    print(f"  python train.py --config toy_hyperparameters.py")


if __name__ == "__main__":
    main()
