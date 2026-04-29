"""Evaluate the trained image CNN on the held-out test split."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from attachiq.config import (
    CONFUSION_DIR,
    DOCUMENT_CLASSES,
    IMAGE_CFG,
    IMAGE_MODEL_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    ensure_dirs,
)
from attachiq.data.image_dataset import DocImageDataset
from attachiq.evaluation.metrics import compute_classification_metrics, save_confusion_matrix
from attachiq.logging import get_logger
from attachiq.models.image_cnn import count_parameters, load_image_model

log = get_logger("eval.image")


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    ensure_dirs()
    device = _device()
    model = load_image_model(IMAGE_MODEL_DIR, device=str(device))
    n_params = count_parameters(model)
    test_ds = DocImageDataset(SPLITS_DIR / "image_test.csv", train=False)
    loader = DataLoader(test_ds, batch_size=IMAGE_CFG.batch_size)
    yt: list[int] = []
    yp: list[int] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["labels"].to(device)
            logits = model(x)
            yt.extend(y.cpu().tolist())
            yp.extend(logits.argmax(dim=-1).cpu().tolist())
    metrics = compute_classification_metrics(yt, yp, DOCUMENT_CLASSES)
    metrics["parameters"] = int(n_params)
    Path(REPORTS_DIR / "image_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_confusion_matrix(
        yt, yp, DOCUMENT_CLASSES,
        CONFUSION_DIR / "image_confusion_matrix.png", title="Image classifier (test)",
    )
    log.info(
        f"Image test acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}; "
        f"params={n_params:,}"
    )


if __name__ == "__main__":
    main()
