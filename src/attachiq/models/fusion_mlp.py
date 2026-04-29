"""Fusion MLP: 26-dim feature vector → 4 triage classes.

Two architecture variants are supported and selected by validation
macro F1 in the fusion training script:

  small : 26 → 64 → 32 → 4
  big   : 26 → 128 → 64 → 4

The big variant is the production default.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import torch
from torch import nn

from attachiq.config import FUSION_INPUT_DIM, NUM_TRIAGE


class FusionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = FUSION_INPUT_DIM,
        hidden_1: int = 128,
        hidden_2: int = 64,
        output_dim: int = NUM_TRIAGE,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_fusion_model(model: FusionMLP, out_dir: str | Path, variant: str = "big") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "fusion_mlp.pt")
    (out / "variant.txt").write_text(variant)
    (out / "arch.json").write_text(
        _json.dumps(
            {
                "input_dim": model.input_dim,
                "hidden_1": model.hidden_1,
                "hidden_2": model.hidden_2,
                "output_dim": model.output_dim,
            }
        )
    )


def load_fusion_model(out_dir: str | Path, device: str = "cpu") -> FusionMLP:
    out = Path(out_dir)
    arch_path = out / "arch.json"
    cfg = _json.loads(arch_path.read_text()) if arch_path.exists() else {}
    model = FusionMLP(
        input_dim=cfg.get("input_dim", FUSION_INPUT_DIM),
        hidden_1=cfg.get("hidden_1", 128),
        hidden_2=cfg.get("hidden_2", 64),
        output_dim=cfg.get("output_dim", NUM_TRIAGE),
    )
    state = torch.load(out / "fusion_mlp.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
