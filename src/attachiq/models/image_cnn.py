"""Scratch CNNs for 8-class document image classification.

No pretrained backbone. Three architectures, all using the same
Conv-BN-ReLU-MaxPool block:

  baseline (DocImageCNN)        — 4 blocks @ 32/64/128/256, ~390K params
  wide     (DocImageCNNWide)    — 4 blocks @ 64/128/256/384, ~1.3M params
  deep     (DocImageCNNDeep)    — 5 blocks @ 64/128/256/384/384, ~2.6M params

The architecture is selected by name via ``build_image_model("baseline" |
"wide" | "deep")``. The on-disk checkpoint records its architecture in
``arch.txt`` so ``load_image_model`` instantiates the right class.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from attachiq.config import NUM_DOC


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class DocImageCNN(nn.Module):
    """Baseline 4-block CNN at channels (32, 64, 128, 256)."""

    def __init__(self, num_classes: int = NUM_DOC, dropout: float = 0.3):
        super().__init__()
        self.b1 = _ConvBlock(1, 32)
        self.b2 = _ConvBlock(32, 64)
        self.b3 = _ConvBlock(64, 128)
        self.b4 = _ConvBlock(128, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.gap(x).flatten(1)
        return self.fc(self.drop(x))


class DocImageCNNWide(nn.Module):
    """4 blocks at (64, 128, 256, 384). ~1.3M params."""

    def __init__(self, num_classes: int = NUM_DOC, dropout: float = 0.3):
        super().__init__()
        self.b1 = _ConvBlock(1, 64)
        self.b2 = _ConvBlock(64, 128)
        self.b3 = _ConvBlock(128, 256)
        self.b4 = _ConvBlock(256, 384)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.gap(x).flatten(1)
        return self.fc(self.drop(x))


class DocImageCNNDeep(nn.Module):
    """5 blocks at (64, 128, 256, 384, 384). ~2.6M params (selected)."""

    def __init__(self, num_classes: int = NUM_DOC, dropout: float = 0.4):
        super().__init__()
        self.b1 = _ConvBlock(1, 64)
        self.b2 = _ConvBlock(64, 128)
        self.b3 = _ConvBlock(128, 256)
        self.b4 = _ConvBlock(256, 384)
        self.b5 = _ConvBlock(384, 384)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.gap(x).flatten(1)
        return self.fc(self.drop(x))


ARCH_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline": DocImageCNN,
    "wide": DocImageCNNWide,
    "deep": DocImageCNNDeep,
}


def build_image_model(arch: str = "deep", **kwargs) -> nn.Module:
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"Unknown image arch '{arch}'. Choose from {list(ARCH_REGISTRY)}")
    return ARCH_REGISTRY[arch](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _infer_arch_from_state(state: dict) -> str:
    b1_out = state["b1.conv.weight"].shape[0]
    if b1_out == 32:
        return "baseline"
    if "b5.conv.weight" in state:
        return "deep"
    return "wide"


def save_image_model(model: nn.Module, out_dir: str | Path, arch: str = "deep") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "image_cnn.pt")
    (out / "arch.txt").write_text(arch)


def load_image_model(out_dir: str | Path, device: str = "cpu") -> nn.Module:
    out = Path(out_dir)
    arch_txt = out / "arch.txt"
    state = torch.load(out / "image_cnn.pt", map_location=device, weights_only=True)
    arch = arch_txt.read_text().strip() if arch_txt.exists() else _infer_arch_from_state(state)
    model = build_image_model(arch)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
