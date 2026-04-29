"""Evaluate AttachIQ on the three test sets:
  1. Standard test split (n=1000, policy-derived labels)
  2. Hard challenge set (n=390, human-reviewed, evaluation-only)
  3. Held-out hard rubric fusion test split (n=354)

Writes:
  reports/metrics_summary.json
  reports/evaluation_summary.md
  reports/confusion_matrices/{standard,hard_challenge,hard_fusion_test}_fusion_confusion_matrix.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from attachiq.config import (
    CONFUSION_DIR,
    DOCUMENT_CLASSES,
    FUSION_MODEL_DIR,
    NUM_DOC,
    NUM_REQUEST,
    REPORTS_DIR,
    REQUEST_CLASSES,
    SPLITS_DIR,
    TRIAGE_CLASSES,
)
from attachiq.data.build_fusion_dataset import FEATURE_COLS as STANDARD_FEATURE_COLS
from attachiq.evaluation.metrics import compute_classification_metrics, save_confusion_matrix
from attachiq.inference.features import (
    _device,
    materialise_features_for_dataframe,
)
from attachiq.logging import get_logger
from attachiq.models.fusion_mlp import load_fusion_model
from attachiq.triage.policy import classify_triage

log = get_logger("eval")

LABEL_TO_IDX = {c: i for i, c in enumerate(TRIAGE_CLASSES)}


def _to_idx(labels: Sequence[str]) -> list[int]:
    return [LABEL_TO_IDX[c] for c in labels]


def _argmax_request_from_row(row: pd.Series) -> str | None:
    if int(row.get("has_text", 0)) == 0:
        return None
    cols = [c for c in row.index if c.startswith("req_")]
    probs = row[cols].to_numpy(dtype=float)
    return REQUEST_CLASSES[int(np.argmax(probs))]


def _argmax_document_from_row(row: pd.Series) -> str | None:
    if int(row.get("has_image", 0)) == 0:
        return None
    cols = [c for c in row.index if c.startswith("doc_")]
    probs = row[cols].to_numpy(dtype=float)
    return DOCUMENT_CLASSES[int(np.argmax(probs))]


def _evaluate_method(name: str, y_true: list[str], y_pred: list[str]) -> dict:
    yt = _to_idx(y_true)
    yp = _to_idx(y_pred)
    m = compute_classification_metrics(yt, yp, TRIAGE_CLASSES)
    log.info(f"  {name:24s} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "per_class": m["per_class"]}


# --- Set 1: standard test --------------------------------------------------- #


def evaluate_standard(fusion_model, device) -> dict:
    df = pd.read_csv(SPLITS_DIR / "fusion_test.csv")
    log.info(f"[standard test] n={len(df)}")
    y_true = df["triage_label"].tolist()

    rule_pred: list[str] = []
    text_only_pred: list[str] = []
    image_only_pred: list[str] = []
    for _, row in df.iterrows():
        req = _argmax_request_from_row(row)
        doc = _argmax_document_from_row(row)
        rule_pred.append(classify_triage(req, doc))
        text_only_pred.append(classify_triage(req, None))
        image_only_pred.append(classify_triage(None, doc))

    X = df[STANDARD_FEATURE_COLS].to_numpy(dtype=np.float32).copy()
    with torch.no_grad():
        logits = fusion_model(torch.from_numpy(X).to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()
    fusion_pred = [TRIAGE_CLASSES[i] for i in preds]

    out = {
        "n": int(len(df)),
        "text_only_baseline": _evaluate_method("text_only", y_true, text_only_pred),
        "image_only_baseline": _evaluate_method("image_only", y_true, image_only_pred),
        "rule_table_baseline": _evaluate_method("rule_table", y_true, rule_pred),
        "learned_fusion": _evaluate_method("learned_fusion", y_true, fusion_pred),
    }
    out["fusion_vs_rule_delta"] = (
        out["learned_fusion"]["macro_f1"] - out["rule_table_baseline"]["macro_f1"]
    )
    save_confusion_matrix(
        _to_idx(y_true), _to_idx(fusion_pred), TRIAGE_CLASSES,
        CONFUSION_DIR / "standard_fusion_confusion_matrix.png",
        title="Fusion — standard test",
    )
    return out


# --- Set 2: 390-row hard challenge set -------------------------------------- #


def evaluate_hard_challenge(fusion_model, device) -> dict:
    csv_path = Path("data/processed/challenge_set.csv")
    if not csv_path.exists():
        log.warning("hard challenge set csv not found.")
        return {}
    df = pd.read_csv(csv_path)
    log.info(f"[hard challenge] n={len(df)}")
    y_true = df["human_triage_label"].tolist()

    feats = materialise_features_for_dataframe(df).copy()
    text_probs_arr = feats[:, :NUM_REQUEST]
    image_probs_arr = feats[:, NUM_REQUEST:NUM_REQUEST + NUM_DOC]

    rule_pred: list[str] = []
    text_only_pred: list[str] = []
    image_only_pred: list[str] = []
    for i in range(len(df)):
        has_text = bool(str(df.iloc[i].get("prompt_text", "") or "").strip())
        has_image = bool(str(df.iloc[i].get("image_path", "") or "").strip())
        req = REQUEST_CLASSES[int(np.argmax(text_probs_arr[i]))] if has_text else None
        doc = DOCUMENT_CLASSES[int(np.argmax(image_probs_arr[i]))] if has_image else None
        rule_pred.append(classify_triage(req, doc))
        text_only_pred.append(classify_triage(req, None))
        image_only_pred.append(classify_triage(None, doc))

    with torch.no_grad():
        logits = fusion_model(torch.from_numpy(feats).to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()
    fusion_pred = [TRIAGE_CLASSES[i] for i in preds]

    per_type: dict[str, dict] = {}
    if "challenge_type" in df.columns:
        for ct in sorted(df["challenge_type"].unique()):
            mask = df["challenge_type"] == ct
            yt = _to_idx([y_true[i] for i in range(len(df)) if mask.iloc[i]])
            yfp = _to_idx([fusion_pred[i] for i in range(len(df)) if mask.iloc[i]])
            yrp = _to_idx([rule_pred[i] for i in range(len(df)) if mask.iloc[i]])
            if not yt:
                continue
            f = compute_classification_metrics(yt, yfp, TRIAGE_CLASSES)
            r = compute_classification_metrics(yt, yrp, TRIAGE_CLASSES)
            per_type[ct] = {
                "n": int(mask.sum()),
                "fusion_macro_f1": f["macro_f1"], "rule_macro_f1": r["macro_f1"],
                "delta_fusion_minus_rule_macro_f1": f["macro_f1"] - r["macro_f1"],
            }

    save_confusion_matrix(
        _to_idx(y_true), _to_idx(fusion_pred), TRIAGE_CLASSES,
        CONFUSION_DIR / "hard_challenge_fusion_confusion_matrix.png",
        title="Fusion — 390-row hard challenge",
    )

    out = {
        "n": int(len(df)),
        "text_only_baseline": _evaluate_method("text_only", y_true, text_only_pred),
        "image_only_baseline": _evaluate_method("image_only", y_true, image_only_pred),
        "rule_table_baseline": _evaluate_method("rule_table", y_true, rule_pred),
        "learned_fusion": _evaluate_method("learned_fusion", y_true, fusion_pred),
        "per_challenge_type": per_type,
    }
    out["fusion_vs_rule_delta"] = (
        out["learned_fusion"]["macro_f1"] - out["rule_table_baseline"]["macro_f1"]
    )
    return out


# --- Set 3: hard rubric fusion test ----------------------------------------- #


def evaluate_hard_fusion_test(fusion_model, device) -> dict:
    csv_path = SPLITS_DIR / "hard_fusion_test.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    log.info(f"[hard fusion test] n={len(df)}")
    y_true = df["human_triage_label"].tolist()

    cache_path = SPLITS_DIR / "hard_fusion_test_features.npy"
    if cache_path.exists():
        feats = np.load(cache_path).astype(np.float32, copy=True)
    else:
        feats = materialise_features_for_dataframe(df).copy()

    text_probs_arr = feats[:, :NUM_REQUEST]
    image_probs_arr = feats[:, NUM_REQUEST:NUM_REQUEST + NUM_DOC]

    rule_pred: list[str] = []
    text_only_pred: list[str] = []
    image_only_pred: list[str] = []
    for i in range(len(df)):
        has_text = bool(str(df.iloc[i].get("prompt_text", "") or "").strip())
        has_image = bool(str(df.iloc[i].get("image_path", "") or "").strip())
        req = REQUEST_CLASSES[int(np.argmax(text_probs_arr[i]))] if has_text else None
        doc = DOCUMENT_CLASSES[int(np.argmax(image_probs_arr[i]))] if has_image else None
        rule_pred.append(classify_triage(req, doc))
        text_only_pred.append(classify_triage(req, None))
        image_only_pred.append(classify_triage(None, doc))

    with torch.no_grad():
        logits = fusion_model(torch.from_numpy(feats).to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()
    fusion_pred = [TRIAGE_CLASSES[i] for i in preds]

    save_confusion_matrix(
        _to_idx(y_true), _to_idx(fusion_pred), TRIAGE_CLASSES,
        CONFUSION_DIR / "hard_fusion_test_confusion_matrix.png",
        title="Fusion — hard fusion test",
    )

    out = {
        "n": int(len(df)),
        "text_only_baseline": _evaluate_method("text_only", y_true, text_only_pred),
        "image_only_baseline": _evaluate_method("image_only", y_true, image_only_pred),
        "rule_table_baseline": _evaluate_method("rule_table", y_true, rule_pred),
        "learned_fusion": _evaluate_method("learned_fusion", y_true, fusion_pred),
    }
    out["fusion_vs_rule_delta"] = (
        out["learned_fusion"]["macro_f1"] - out["rule_table_baseline"]["macro_f1"]
    )
    return out


def main() -> None:
    device = _device()
    fusion_model = load_fusion_model(FUSION_MODEL_DIR, device=str(device))

    log.info("standard test ...")
    standard = evaluate_standard(fusion_model, device)
    log.info("hard challenge (390 rows) ...")
    hard = evaluate_hard_challenge(fusion_model, device)
    log.info("hard fusion test ...")
    hard_rubric = evaluate_hard_fusion_test(fusion_model, device)

    out = {
        "standard_test": standard,
        "hard_challenge": hard,
        "hard_fusion_test": hard_rubric,
    }
    Path(REPORTS_DIR / "metrics_summary.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    log.info(f"Wrote {REPORTS_DIR / 'metrics_summary.json'}")

    md = ["# AttachIQ — Final Evaluation Summary\n"]
    for section, title in [
        ("standard_test", "Standard test (n=1000, policy-derived labels)"),
        ("hard_challenge", "Hard challenge set (n=390, human-reviewed)"),
        ("hard_fusion_test", "Hard rubric fusion test (n=354, held-out rubric)"),
    ]:
        s = out[section]
        if not s:
            continue
        md.append(f"## {title}\n")
        md.append("| Method | Accuracy | Macro F1 |")
        md.append("|---|---:|---:|")
        for key, label in [
            ("text_only_baseline", "text-only"),
            ("image_only_baseline", "image-only"),
            ("rule_table_baseline", "rule"),
            ("learned_fusion", "fusion"),
        ]:
            m = s.get(key)
            if not m:
                continue
            md.append(f"| {label} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")
        if "fusion_vs_rule_delta" in s:
            md.append(f"\nFusion vs rule Δ macro F1 = {s['fusion_vs_rule_delta']:+.4f}")
        md.append("")
    Path(REPORTS_DIR / "evaluation_summary.md").write_text("\n".join(md))
    log.info(f"Wrote {REPORTS_DIR / 'evaluation_summary.md'}")


if __name__ == "__main__":
    main()
