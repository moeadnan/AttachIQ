"""Fine-tune DistilBERT on the 10-class request taxonomy.

Outputs:
    models/text/text_model.pt
    models/text/tokenizer/
    models/text/label_map.json
    reports/text_metrics.json
    reports/confusion_matrices/text_confusion_matrix.png
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

from attachiq.config import (
    CONFUSION_DIR,
    NUM_REQUEST,
    REPORTS_DIR,
    REQUEST_CLASSES,
    SPLITS_DIR,
    TEXT_CFG,
    TEXT_MODEL_DIR,
    ensure_dirs,
)
from attachiq.data.text_dataset import TextPromptDataset
from attachiq.evaluation.metrics import compute_classification_metrics, save_confusion_matrix
from attachiq.logging import get_logger
from attachiq.models.text_model import DistilBertRequestClassifier, save_text_model

log = get_logger("train.text")


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
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            logits = model(ids, attn)
            yt.extend(y.cpu().tolist())
            yp.extend(logits.argmax(dim=-1).cpu().tolist())
    return yt, yp


def main() -> None:
    ensure_dirs()
    _seed(TEXT_CFG.seed)
    device = _device()
    log.info(f"Training text model on {device}")

    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(TEXT_CFG.model_name)

    train_csv = SPLITS_DIR / "text_train.csv"
    val_csv = SPLITS_DIR / "text_val.csv"
    test_csv = SPLITS_DIR / "text_test.csv"
    if not train_csv.exists():
        raise FileNotFoundError("Run build_text_dataset first.")

    train_ds = TextPromptDataset(train_csv, tokenizer, TEXT_CFG.max_length)
    val_ds = TextPromptDataset(val_csv, tokenizer, TEXT_CFG.max_length)
    test_ds = TextPromptDataset(test_csv, tokenizer, TEXT_CFG.max_length)
    log.info(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=TEXT_CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TEXT_CFG.batch_size)
    test_loader = DataLoader(test_ds, batch_size=TEXT_CFG.batch_size)

    train_labels = train_ds.df["label"].map(train_ds.label_to_idx).to_numpy()
    classes = np.array(sorted(np.unique(train_labels)))
    weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    log.info(
        f"class weights: {dict(zip(REQUEST_CLASSES, [round(w, 3) for w in weights.tolist()]))}"
    )

    model = DistilBertRequestClassifier(
        model_name=TEXT_CFG.model_name, dropout=TEXT_CFG.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TEXT_CFG.learning_rate, weight_decay=TEXT_CFG.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights_t)

    best_macro = -1.0
    best_state = None
    patience = 2
    bad = 0

    for epoch in range(1, TEXT_CFG.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(ids, attn), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * y.size(0)
        avg = running / len(train_ds)
        yt, yp = _evaluate(model, val_loader, device)
        m = compute_classification_metrics(yt, yp, REQUEST_CLASSES)
        log.info(
            f"epoch {epoch}/{TEXT_CFG.epochs} train_loss={avg:.4f} "
            f"val_acc={m['accuracy']:.4f} val_macro_f1={m['macro_f1']:.4f}"
        )
        if m["macro_f1"] > best_macro:
            best_macro = m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info(f"early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_text_model(model, tokenizer, TEXT_MODEL_DIR)
    Path(TEXT_MODEL_DIR / "label_map.json").write_text(
        json.dumps({c: i for i, c in enumerate(REQUEST_CLASSES)}, indent=2)
    )

    yt, yp = _evaluate(model, test_loader, device)
    test_metrics = compute_classification_metrics(yt, yp, REQUEST_CLASSES)
    test_metrics["best_val_macro_f1"] = float(best_macro)
    Path(REPORTS_DIR / "text_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    save_confusion_matrix(
        yt, yp, REQUEST_CLASSES,
        CONFUSION_DIR / "text_confusion_matrix.png",
        title="Text classifier (test)",
    )
    log.info(
        f"text test acc={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
