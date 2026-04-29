"""PyTorch dataset wrapping tokenised request prompts (10 classes)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from attachiq.config import REQUEST_CLASSES


class TextPromptDataset(Dataset):
    def __init__(self, csv_path: str | Path, tokenizer, max_length: int = 128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_idx = {c: i for i, c in enumerate(REQUEST_CLASSES)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = self.label_to_idx[row["label"]]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "text": text,
        }
