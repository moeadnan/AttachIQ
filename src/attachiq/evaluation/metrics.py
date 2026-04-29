"""Shared metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    label_idx = list(range(len(labels)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=label_idx, zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_idx, zero_division=0
    )
    per_class = {
        labels[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in label_idx
    }
    report = classification_report(
        y_true, y_pred, labels=label_idx, target_names=labels, zero_division=0, output_dict=True
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "classification_report": report,
        "n": int(len(y_true)),
    }


def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
    out_path: str | Path,
    title: str = "Confusion matrix",
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def latency_summary(times_ms: Sequence[float]) -> dict:
    arr = np.asarray(times_ms, dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "n": int(len(arr)),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }
