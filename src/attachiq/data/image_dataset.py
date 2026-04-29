"""PyTorch dataset for grayscale 224x224 document images.

Training augmentations only (no horizontal flip):
- small rotation
- slight translation/scale
- brightness/contrast jitter
- Gaussian noise
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from attachiq.config import DOCUMENT_CLASSES

IMAGE_SIZE = 224
NORM_MEAN = (0.5,)
NORM_STD = (0.5,)


class _AddGaussianNoise:
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(t + torch.randn_like(t) * self.std, 0.0, 1.0)


def _train_transforms() -> T.Compose:
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomAffine(degrees=4, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            _AddGaussianNoise(std=0.015),
            T.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def _eval_transforms() -> T.Compose:
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def get_transforms(train: bool) -> T.Compose:
    return _train_transforms() if train else _eval_transforms()


class DocImageDataset(Dataset):
    def __init__(self, csv_path: str | Path, train: bool = False):
        self.df = pd.read_csv(csv_path)
        self.transform = get_transforms(train=train)
        self.label_to_idx = {c: i for i, c in enumerate(DOCUMENT_CLASSES)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("L")
        x = self.transform(image)
        y = self.label_to_idx[row["label"]]
        return {"image": x, "labels": torch.tensor(y, dtype=torch.long), "image_path": row["image_path"]}
