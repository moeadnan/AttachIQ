"""Train the scratch CNN on document images.

Architecture is selectable: ``--arch baseline | wide | deep``. The default
production checkpoint is the ``deep`` 5-block variant.

Outputs:
    models/image/image_cnn.pt
    models/image/arch.txt
    models/image/label_map.json
    reports/image_metrics.json
    reports/confusion_matrices/image_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
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
from attachiq.models.image_cnn import build_image_model, count_parameters, save_image_model

log = get_logger("train.image")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _evaluate(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    yt: list[int] = []
    yp: list[int] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["labels"].to(device)
            logits = model(x)
            yt.extend(y.cpu().tolist())
            yp.extend(logits.argmax(dim=-1).cpu().tolist())
    return yt, yp


def main(arch: str = "deep", epochs: int | None = None, save: bool = True) -> dict:
    ensure_dirs()
    _seed(IMAGE_CFG.seed)
    device = _device()
    n_epochs = epochs if epochs is not None else IMAGE_CFG.epochs
    log.info(f"Training image CNN arch={arch} epochs={n_epochs} on {device}")

    train_csv = SPLITS_DIR / "image_train.csv"
    val_csv = SPLITS_DIR / "image_val.csv"
    test_csv = SPLITS_DIR / "image_test.csv"
    if not train_csv.exists():
        raise FileNotFoundError("Run image dataset build first.")

    train_ds = DocImageDataset(train_csv, train=True)
    val_ds = DocImageDataset(val_csv, train=False)
    test_ds = DocImageDataset(test_csv, train=False)
    log.info(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=IMAGE_CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=IMAGE_CFG.batch_size)
    test_loader = DataLoader(test_ds, batch_size=IMAGE_CFG.batch_size)

    model = build_image_model(arch).to(device)
    n_params = count_parameters(model)
    log.info(f"Image CNN '{arch}' trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=IMAGE_CFG.learning_rate, weight_decay=IMAGE_CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    best_macro = -1.0
    best_state: dict | None = None
    best_epoch = 0
    patience = 5
    bad = 0

    t_start = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
        avg_loss = running / len(train_ds)
        yt, yp = _evaluate(model, val_loader, device)
        m = compute_classification_metrics(yt, yp, DOCUMENT_CLASSES)
        scheduler.step(m["macro_f1"])
        cur_lr = optimizer.param_groups[0]["lr"]
        log.info(
            f"epoch {epoch}/{n_epochs} train_loss={avg_loss:.4f} "
            f"val_acc={m['accuracy']:.4f} val_macro_f1={m['macro_f1']:.4f} lr={cur_lr:.2e}"
        )
        if m["macro_f1"] > best_macro:
            best_macro = m["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info(f"Early stopping at epoch {epoch}")
                break
    train_secs = time.perf_counter() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)
    if save:
        save_image_model(model, IMAGE_MODEL_DIR, arch=arch)
        Path(IMAGE_MODEL_DIR / "label_map.json").write_text(
            json.dumps({c: i for i, c in enumerate(DOCUMENT_CLASSES)}, indent=2)
        )

    yt, yp = _evaluate(model, test_loader, device)
    test_metrics = compute_classification_metrics(yt, yp, DOCUMENT_CLASSES)
    test_metrics["arch"] = arch
    test_metrics["parameters"] = int(n_params)
    test_metrics["best_val_macro_f1"] = float(best_macro)
    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["train_seconds"] = float(train_secs)
    Path(REPORTS_DIR / "image_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    save_confusion_matrix(
        yt, yp, DOCUMENT_CLASSES,
        CONFUSION_DIR / "image_confusion_matrix.png",
        title=f"Image classifier ({arch}, test)",
    )
    log.info(
        f"[{arch}] test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f} "
        f"best_val={best_macro:.4f} params={n_params:,} train_secs={train_secs:.1f}"
    )
    return test_metrics


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="deep", choices=["baseline", "wide", "deep"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    args = parser.parse_args()
    main(arch=args.arch, epochs=args.epochs, save=not args.no_save_checkpoint)


if __name__ == "__main__":
    cli()
