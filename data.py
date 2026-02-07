"""Data loading utilities for FineWeb-edu-gpt2 dataset."""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, IterableDataset
from huggingface_hub import HfFileSystem

# Configuration: Set LOCAL_DATA_PATH to use local files, or None for HuggingFace
LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH", "./data/fineweb-edu-gpt2")

# HuggingFace remote dataset (fallback if local not found)
DATASET_REPO = "flappingairplanes/fineweb-edu-gpt2"
SUBSET = "sample-10BT_max_length_513"


def get_parquet_files(split: str = "train"):
    """Get list of parquet file paths for a split.
    
    Checks local path first, falls back to HuggingFace Hub.
    """
    # Try local files first
    local_path = Path(LOCAL_DATA_PATH) if LOCAL_DATA_PATH else None
    if local_path and local_path.exists():
        files = sorted(local_path.glob(f"{split}-*.parquet"))
        if files:
            return [str(f) for f in files]
        print(f"Warning: No {split} files found in {local_path}, trying HuggingFace...")
    
    # Fall back to HuggingFace
    fs = HfFileSystem()
    path = f"datasets/{DATASET_REPO}/{SUBSET}"
    files = fs.ls(path, detail=False)
    # Filter by split prefix (e.g., "train-00000-of-00061.parquet")
    split_files = [f for f in files if f.endswith(".parquet") and f"/{split}-" in f]
    return sorted(split_files)


class StreamingParquetDataset(IterableDataset):
    """Stream parquet files from local disk or HuggingFace Hub."""
    
    def __init__(self, split: str = "train", shuffle: bool = False, rank: int = 0, world_size: int = 1):
        self.files = get_parquet_files(split)
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        # Check if files are local or remote
        self.is_local = len(self.files) > 0 and not self.files[0].startswith("datasets/")
    
    def __iter__(self):
        import pyarrow.parquet as pq
        
        # If enough files, shard by files (efficient for training)
        # If fewer files than ranks, all ranks read all files but shard rows
        if len(self.files) >= self.world_size:
            files = self.files[self.rank::self.world_size]
            shard_rows = False
        else:
            files = self.files  # all ranks read all files
            shard_rows = True
        
        if self.shuffle:
            import random
            files = files.copy()
            random.shuffle(files)
        
        for file_path in files:
            if self.is_local:
                # Local file
                table = pq.read_table(file_path)
            else:
                # Remote HuggingFace file
                from huggingface_hub import HfFileSystem
                fs = HfFileSystem()
                with fs.open(file_path, "rb") as f:
                    table = pq.read_table(f)
            
            for i in range(len(table)):
                # If sharding by rows, only yield rows for this rank
                if shard_rows and (i % self.world_size != self.rank):
                    continue
                row = {col: table[col][i].as_py() for col in table.column_names}
                yield row


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    pad_mask = torch.tensor([item["pad_mask"] for item in batch], dtype=torch.long)
    
    # For language modeling: inputs are all tokens except last, labels are all except first
    return {
        "input_ids": input_ids[:, :-1],
        "labels": input_ids[:, 1:],
        "attention_mask": pad_mask[:, :-1],
    }


def get_dataloader(
    split: str = "train",
    batch_size: int = 32,
    streaming: bool = True,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = None,
):
    """Get a DataLoader for the dataset.
    
    Args:
        split: "train" or "test"
        batch_size: Batch size
        streaming: Whether to stream (always True for this dataset)
        num_workers: Number of data loading workers
        rank: DDP rank for sharding
        world_size: DDP world size for sharding
        shuffle: Whether to shuffle. If None, defaults to True for train, False for test.
    
    Returns:
        DataLoader
    """
    if shuffle is None:
        shuffle = (split == "train")
    dataset = StreamingParquetDataset(split=split, shuffle=shuffle, rank=rank, world_size=world_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=8 if num_workers > 0 else None,
    )


if __name__ == "__main__":
    # Quick test
    print("Loading dataset...")
    loader = get_dataloader(split="train", batch_size=4)
    batch = next(iter(loader))
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
