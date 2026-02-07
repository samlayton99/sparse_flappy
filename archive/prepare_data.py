"""Prepare tokenized FineWeb-Edu dataset with GPT-2 tokenizer.

Creates local parquet files matching the expected format:
- input_ids: list of 513 token IDs (including EOS)
- pad_mask: list of 513 booleans (1=real token, 0=padding)

Usage:
    python prepare_data.py --max_tokens 1_000_000_000  # 1B tokens (~5GB)
    python prepare_data.py --max_tokens 2_000_000_000  # 2B tokens (~10GB)
"""

import argparse
import os
from pathlib import Path
from typing import Iterator, List
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset

# Constants
SEQ_LEN = 513  # Total sequence length including EOS
EOS_TOKEN_ID = 50256  # GPT-2 <|endoftext|>
PAD_TOKEN_ID = 50256  # Use EOS as pad (standard practice for GPT-2)
SEQUENCES_PER_FILE = 50_000  # ~25MB per file after compression


def create_tokenizer() -> GPT2TokenizerFast:
    """Create GPT-2 tokenizer with proper settings."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS as pad
    return tokenizer


def pack_tokens(
    token_stream: Iterator[List[int]], 
    seq_len: int = SEQ_LEN,
) -> Iterator[dict]:
    """Pack tokens into fixed-length sequences.
    
    Concatenates documents with EOS separators, then chunks into seq_len sequences.
    Each sequence ends with EOS and is padded if needed.
    
    Yields:
        dict with 'input_ids' and 'pad_mask'
    """
    buffer = []
    
    for tokens in token_stream:
        # Add document tokens + EOS separator
        buffer.extend(tokens)
        buffer.append(EOS_TOKEN_ID)
        
        # Yield complete sequences
        while len(buffer) >= seq_len:
            seq_tokens = buffer[:seq_len]
            buffer = buffer[seq_len:]
            
            yield {
                "input_ids": seq_tokens,
                "pad_mask": [1] * seq_len,  # All real tokens
            }
    
    # Handle remaining tokens (pad the last sequence)
    if buffer:
        pad_len = seq_len - len(buffer)
        yield {
            "input_ids": buffer + [PAD_TOKEN_ID] * pad_len,
            "pad_mask": [1] * len(buffer) + [0] * pad_len,
        }


def tokenize_documents(
    dataset_iter,
    tokenizer: GPT2TokenizerFast,
    text_column: str = "text",
    batch_size: int = 1000,
) -> Iterator[List[int]]:
    """Tokenize documents from dataset iterator.
    
    Yields:
        List of token IDs for each document
    """
    batch = []
    
    for example in dataset_iter:
        batch.append(example[text_column])
        
        if len(batch) >= batch_size:
            # Batch tokenize for efficiency
            encoded = tokenizer(
                batch, 
                add_special_tokens=False,  # We add EOS manually
                return_attention_mask=False,
            )
            for ids in encoded["input_ids"]:
                yield ids
            batch = []
    
    # Process remaining
    if batch:
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        for ids in encoded["input_ids"]:
            yield ids


def save_parquet_file(
    sequences: List[dict],
    output_path: Path,
    split: str,
    file_idx: int,
    total_files: int,
):
    """Save sequences to a parquet file."""
    # Create PyArrow table
    table = pa.table({
        "input_ids": [seq["input_ids"] for seq in sequences],
        "pad_mask": [seq["pad_mask"] for seq in sequences],
    })
    
    # Save with compression
    filename = f"{split}-{file_idx:05d}-of-{total_files:05d}.parquet"
    pq.write_table(
        table, 
        output_path / filename,
        compression="zstd",
        compression_level=3,
    )
    return filename


def prepare_dataset(
    output_dir: str = "./data/fineweb-edu-gpt2",
    max_tokens: int = 1_000_000_000,  # 1B tokens default
    split: str = "train",
    hf_dataset: str = "HuggingFaceFW/fineweb-edu",
    hf_subset: str = "sample-10BT",
):
    """Prepare the tokenized dataset.
    
    Args:
        output_dir: Where to save the parquet files
        max_tokens: Maximum number of tokens to process
        split: "train" or "test"
        hf_dataset: HuggingFace dataset name
        hf_subset: Dataset subset/config name
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    print(f"Max tokens: {max_tokens:,}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Source: {hf_dataset}/{hf_subset}")
    print()
    
    # Create tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = create_tokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {EOS_TOKEN_ID})")
    print(f"  Pad token: {tokenizer.pad_token} (ID: {PAD_TOKEN_ID})")
    print()
    
    # Load dataset in streaming mode
    print(f"Loading {hf_dataset} ({hf_subset}) in streaming mode...")
    dataset = load_dataset(
        hf_dataset,
        hf_subset,
        split="train",  # FineWeb-Edu only has train split
        streaming=True,
        trust_remote_code=True,
    )
    
    # Process
    print("Tokenizing and packing sequences...")
    tokenized = tokenize_documents(dataset, tokenizer)
    packed = pack_tokens(tokenized, seq_len=SEQ_LEN)
    
    # Collect and save
    sequences = []
    total_tokens = 0
    total_sequences = 0
    file_idx = 0
    saved_files = []
    
    max_sequences = max_tokens // SEQ_LEN + 1
    
    pbar = tqdm(total=max_sequences, desc="Sequences", unit="seq")
    
    for seq in packed:
        sequences.append(seq)
        real_tokens = sum(seq["pad_mask"])
        total_tokens += real_tokens
        total_sequences += 1
        pbar.update(1)
        
        # Save file when buffer is full
        if len(sequences) >= SEQUENCES_PER_FILE:
            # We don't know total files yet, will rename later
            filename = save_parquet_file(
                sequences, output_path, split, file_idx, 99999
            )
            saved_files.append(filename)
            sequences = []
            file_idx += 1
            pbar.set_postfix({
                "tokens": f"{total_tokens/1e9:.2f}B",
                "files": file_idx,
            })
        
        # Check token limit
        if total_tokens >= max_tokens:
            break
    
    pbar.close()
    
    # Save remaining sequences
    if sequences:
        filename = save_parquet_file(
            sequences, output_path, split, file_idx, 99999
        )
        saved_files.append(filename)
        file_idx += 1
    
    total_files = file_idx
    
    # Rename files with correct total
    print(f"\nRenaming {total_files} files with correct total...")
    for i, old_name in enumerate(saved_files):
        old_path = output_path / old_name
        new_name = f"{split}-{i:05d}-of-{total_files:05d}.parquet"
        new_path = output_path / new_name
        old_path.rename(new_path)
    
    # Summary
    print()
    print("=" * 60)
    print("DATASET PREPARED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Total sequences: {total_sequences:,}")
    print(f"  Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"  Files created: {total_files}")
    print(f"  Output directory: {output_path}")
    print()
    
    # Estimate size
    total_size = sum(f.stat().st_size for f in output_path.glob("*.parquet"))
    print(f"  Total size: {total_size / 1e9:.2f} GB")
    print()
    
    # Print update instructions
    print("To use this dataset, update data.py:")
    print('  DATASET_REPO = "local"  # or comment out HF loading')
    print(f'  LOCAL_DATA_PATH = "{output_path}"')
    

def prepare_test_split(
    train_dir: str = "./data/fineweb-edu-gpt2",
    test_ratio: float = 0.01,
):
    """Create a test split by moving some train files.
    
    Args:
        train_dir: Directory with train parquet files
        test_ratio: Fraction of files to use for test (default 1%)
    """
    train_path = Path(train_dir)
    train_files = sorted(train_path.glob("train-*.parquet"))
    
    if not train_files:
        print(f"No train files found in {train_path}")
        return
    
    n_test = max(1, int(len(train_files) * test_ratio))
    test_files = train_files[-n_test:]  # Take last N files
    
    print(f"Moving {n_test} files to test split...")
    
    for i, old_path in enumerate(test_files):
        new_name = f"test-{i:05d}-of-{n_test:05d}.parquet"
        new_path = train_path / new_name
        old_path.rename(new_path)
        print(f"  {old_path.name} -> {new_name}")
    
    # Rename remaining train files
    remaining_train = sorted(train_path.glob("train-*.parquet"))
    n_train = len(remaining_train)
    
    print(f"\nRenaming {n_train} train files...")
    for i, old_path in enumerate(remaining_train):
        new_name = f"train-{i:05d}-of-{n_train:05d}.parquet"
        new_path = train_path / new_name
        if old_path != new_path:
            old_path.rename(new_path)
    
    print(f"\nDone! Train: {n_train} files, Test: {n_test} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu dataset with GPT-2 tokenizer")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data/fineweb-edu-gpt2",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1_000_000_000,
        help="Maximum tokens to process (default: 1B)"
    )
    parser.add_argument(
        "--hf_subset",
        type=str,
        default="sample-10BT",
        help="HuggingFace dataset subset (default: sample-10BT)"
    )
    parser.add_argument(
        "--create_test_split",
        action="store_true",
        help="After processing, create a test split from train files"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.01,
        help="Fraction of files for test split (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    # Prepare main dataset
    prepare_dataset(
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        hf_subset=args.hf_subset,
    )
    
    # Optionally create test split
    if args.create_test_split:
        print("\n" + "=" * 60)
        print("CREATING TEST SPLIT")
        print("=" * 60)
        prepare_test_split(
            train_dir=args.output_dir,
            test_ratio=args.test_ratio,
        )
