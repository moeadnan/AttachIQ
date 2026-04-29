"""Fine-tuned DistilBERT text request classifier (10 classes).

All transformer layers are unfrozen. A small linear head with dropout is
added on top of the [CLS] hidden state.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from attachiq.config import NUM_REQUEST


class DistilBertRequestClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = NUM_REQUEST,
        dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import DistilBertModel  # noqa: WPS433
        self.backbone = DistilBertModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


def save_text_model(model: DistilBertRequestClassifier, tokenizer, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "text_model.pt")
    tokenizer.save_pretrained(out / "tokenizer")


def load_text_model(out_dir: str | Path, device: str = "cpu") -> tuple[DistilBertRequestClassifier, object]:
    from transformers import DistilBertTokenizerFast  # noqa: WPS433
    out = Path(out_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(out / "tokenizer")
    model = DistilBertRequestClassifier()
    state = torch.load(out / "text_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer
