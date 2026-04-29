"""Train the fusion MLP.

Two architecture variants (small=64/32 and big=128/64) are trained on the
union of the standard policy-derived fusion training set and the hard
rubric-labelled fusion training set; the variant with the best validation
macro F1 is saved as the production checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from attachiq.config import (
    CONFUSION_DIR,
    FUSION_CFG,
    FUSION_MODEL_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    TRIAGE_CLASSES,
    ensure_dirs,
)
from attachiq.data.build_fusion_dataset import FEATURE_COLS as STANDARD_FEATURE_COLS
from attachiq.evaluation.metrics import compute_classification_metrics, save_confusion_matrix
from attachiq.inference.features import materialise_features_for_dataframe
from attachiq.logging import get_logger
from attachiq.models.fusion_mlp import FusionMLP, save_fusion_model

log = get_logger("train.fusion")


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


def _load_standard(split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(SPLITS_DIR / f"fusion_{split}.csv")
    X = df[STANDARD_FEATURE_COLS].to_numpy(dtype=np.float32).copy()
    label_to_idx = {c: i for i, c in enumerate(TRIAGE_CLASSES)}
    y = df["triage_label"].map(label_to_idx).to_numpy(dtype=np.int64)
    return X, y, df


def _load_hard(split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(SPLITS_DIR / f"hard_fusion_{split}.csv")
    log.info(f"materialising 26-dim features for hard split={split} (n={len(df)})...")
    X = materialise_features_for_dataframe(df).copy()
    label_to_idx = {c: i for i, c in enumerate(TRIAGE_CLASSES)}
    y = df["human_triage_label"].map(label_to_idx).to_numpy(dtype=np.int64)
    return X, y, df


def _evaluate(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    yt: list[int] = []
    yp: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            yt.extend(y.cpu().tolist())
            yp.extend(model(x).argmax(dim=-1).cpu().tolist())
    return yt, yp


def _train_variant(
    name: str, hidden_1: int, hidden_2: int,
    X_tr, y_tr, X_v, y_v, device, class_weights=None,
) -> tuple[FusionMLP, float, dict]:
    _seed(FUSION_CFG.seed)
    model = FusionMLP(
        input_dim=X_tr.shape[1], hidden_1=hidden_1, hidden_2=hidden_2,
        output_dim=len(TRIAGE_CLASSES), dropout=FUSION_CFG.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"variant={name} params={n_params:,} hidden=({hidden_1},{hidden_2})")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=FUSION_CFG.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_v), torch.from_numpy(y_v)),
        batch_size=FUSION_CFG.batch_size,
    )
    optim = torch.optim.Adam(
        model.parameters(), lr=FUSION_CFG.learning_rate,
        weight_decay=FUSION_CFG.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device)) if class_weights is not None else nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    bad = 0
    patience = 6
    history: list[dict] = []

    for epoch in range(1, FUSION_CFG.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        avg = running / max(n, 1)
        yt, yp = _evaluate(model, val_loader, device)
        m = compute_classification_metrics(yt, yp, TRIAGE_CLASSES)
        history.append({
            "epoch": epoch, "train_loss": float(avg),
            "val_acc": float(m["accuracy"]), "val_macro_f1": float(m["macro_f1"]),
        })
        log.info(
            f"  [{name}] epoch {epoch:>2d} loss={avg:.4f} "
            f"val_acc={m['accuracy']:.4f} val_f1={m['macro_f1']:.4f}"
        )
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info(f"  [{name}] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1, {
        "variant": name, "hidden_1": hidden_1, "hidden_2": hidden_2,
        "params": int(n_params), "best_val_macro_f1": float(best_f1),
        "epochs_run": len(history), "history": history,
    }


def main(source: str = "union", balance: bool = True) -> None:
    ensure_dirs()
    device = _device()
    log.info(f"Training fusion on source={source} on {device}")

    if source == "union":
        std_train = _load_standard("train")
        std_val = _load_standard("val")
        hrd_train = _load_hard("train")
        hrd_val = _load_hard("val")
        hrd_test = _load_hard("test")
        # Cache materialised hard features for downstream evaluation reuse.
        for split, parts in [("train", hrd_train), ("val", hrd_val), ("test", hrd_test)]:
            np.save(SPLITS_DIR / f"hard_fusion_{split}_features.npy", parts[0])
        X_tr = np.concatenate([std_train[0], hrd_train[0]], axis=0)
        y_tr = np.concatenate([std_train[1], hrd_train[1]], axis=0)
        X_v = np.concatenate([std_val[0], hrd_val[0]], axis=0)
        y_v = np.concatenate([std_val[1], hrd_val[1]], axis=0)
        X_te, y_te = hrd_test[0], hrd_test[1]
    elif source == "hard":
        hrd_train = _load_hard("train")
        hrd_val = _load_hard("val")
        hrd_test = _load_hard("test")
        for split, parts in [("train", hrd_train), ("val", hrd_val), ("test", hrd_test)]:
            np.save(SPLITS_DIR / f"hard_fusion_{split}_features.npy", parts[0])
        X_tr, y_tr = hrd_train[0], hrd_train[1]
        X_v, y_v = hrd_val[0], hrd_val[1]
        X_te, y_te = hrd_test[0], hrd_test[1]
    else:
        std = _load_standard
        X_tr, y_tr, _ = std("train")
        X_v, y_v, _ = std("val")
        X_te, y_te, _ = std("test")
    log.info(f"shapes: train={X_tr.shape} val={X_v.shape} test={X_te.shape}")

    class_weights = None
    if balance:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.array(sorted(np.unique(y_tr)))
        weights = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        log.info(
            f"class weights: {dict(zip(TRIAGE_CLASSES, [round(w, 3) for w in weights.tolist()]))}"
        )

    candidates = [("small", 64, 32), ("big", 128, 64)]
    results = []
    for name, h1, h2 in candidates:
        model, val_f1, info = _train_variant(
            name, h1, h2, X_tr, y_tr, X_v, y_v, device, class_weights
        )
        results.append((model, val_f1, info))

    best_model, best_val_f1, best_info = max(results, key=lambda r: r[1])
    log.info(f"selected variant={best_info['variant']} val_macro_f1={best_val_f1:.4f}")

    save_fusion_model(best_model, FUSION_MODEL_DIR, variant=best_info["variant"])
    Path(FUSION_MODEL_DIR / "label_map.json").write_text(
        json.dumps({c: i for i, c in enumerate(TRIAGE_CLASSES)}, indent=2)
    )

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=FUSION_CFG.batch_size,
    )
    yt, yp = _evaluate(best_model.to(device), test_loader, device)
    test_metrics = compute_classification_metrics(yt, yp, TRIAGE_CLASSES)
    test_metrics["selected_variant"] = best_info["variant"]
    test_metrics["candidates"] = [r[2] for r in results]
    test_metrics["source"] = source
    test_metrics["feature_dim"] = int(X_te.shape[1])
    test_metrics["params"] = int(best_info["params"])
    Path(REPORTS_DIR / "fusion_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    save_confusion_matrix(
        yt, yp, TRIAGE_CLASSES,
        CONFUSION_DIR / "fusion_confusion_matrix.png",
        title=f"Fusion MLP ({source}, {best_info['variant']})",
    )
    log.info(
        f"fusion test acc={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["standard", "hard", "union"], default="union")
    p.add_argument("--no-balance", action="store_true")
    args = p.parse_args()
    main(source=args.source, balance=not args.no_balance)


if __name__ == "__main__":
    cli()
