"""Build the standard fusion dataset (26-dim features, policy-derived labels).

Features per row:
    10 request-class probabilities
   + 8 document-class probabilities
   + text_conf, image_conf, has_text, has_image
   + text_entropy, image_entropy, text_margin, image_margin
   = 26

Labels: ``classify_triage(true_request, true_document)``. The labeller
uses the **ground-truth** request and document classes so the rule
baseline (which uses *predicted* classes) and the learned fusion are
evaluated fairly on the same distribution.

Outputs:
    data/processed/fusion_features.csv
    data/splits/fusion_{train,val,test}.csv
    reports/fusion_data_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from attachiq.config import (
    DOCUMENT_CLASSES,
    FUSION_INPUT_DIM,
    GLOBAL_SEED,
    IMAGE_MODEL_DIR,
    NUM_DOC,
    NUM_REQUEST,
    PROCESSED_DIR,
    REPORTS_DIR,
    REQUEST_CLASSES,
    SPLITS_DIR,
    TEXT_CFG,
    TEXT_MODEL_DIR,
    TRIAGE_CLASSES,
    ensure_dirs,
)
from attachiq.data.image_dataset import get_transforms
from attachiq.logging import get_logger
from attachiq.triage.policy import classify_triage

log = get_logger("data.fusion")

PROB_COLS_REQ = [f"req_{c}" for c in REQUEST_CLASSES]
PROB_COLS_DOC = [f"doc_{c}" for c in DOCUMENT_CLASSES]
EXTRA_COLS = [
    "text_conf", "image_conf", "has_text", "has_image",
    "text_entropy", "image_entropy", "text_margin", "image_margin",
]
FEATURE_COLS = PROB_COLS_REQ + PROB_COLS_DOC + EXTRA_COLS


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _margin(probs: np.ndarray) -> float:
    if len(probs) < 2:
        return float(probs.max())
    s = np.sort(probs)[::-1]
    return float(s[0] - s[1])


def _text_probs(texts: list[str], model, tokenizer, device, batch_size: int = 32) -> np.ndarray:
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(
                chunk, truncation=True, padding=True,
                max_length=TEXT_CFG.max_length, return_tensors="pt",
            ).to(device)
            logits = model(enc["input_ids"], enc["attention_mask"])
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out)


def _image_probs(paths: list[str], model, device, batch_size: int = 32) -> np.ndarray:
    out: list[np.ndarray] = []
    model.eval()
    transform = get_transforms(train=False)
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            tensors = [transform(Image.open(p).convert("L")) for p in chunk]
            x = torch.stack(tensors).to(device)
            logits = model(x)
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out)


def _stratified_split(df: pd.DataFrame, seed: int = GLOBAL_SEED) -> dict[str, pd.DataFrame]:
    train, temp = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["triage_label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp["triage_label"])
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def main(n_samples: int = 10000) -> None:
    ensure_dirs()
    device = _device()

    text_csv = PROCESSED_DIR / "text_prompts.csv"
    image_manifest = PROCESSED_DIR / "image_manifest.csv"
    if not text_csv.exists():
        raise FileNotFoundError("Run build_text_dataset first.")
    if not image_manifest.exists():
        raise FileNotFoundError("Run build_image_dataset first.")
    if not (TEXT_MODEL_DIR / "text_model.pt").exists():
        raise FileNotFoundError("Train text model first.")
    if not (IMAGE_MODEL_DIR / "image_cnn.pt").exists():
        raise FileNotFoundError("Image CNN missing.")

    text_df = pd.read_csv(text_csv)
    image_df = pd.read_csv(image_manifest)
    log.info(f"text rows={len(text_df)} image rows={len(image_df)} target={n_samples}")

    from attachiq.models.text_model import load_text_model
    from attachiq.models.image_cnn import load_image_model
    text_model, tokenizer = load_text_model(TEXT_MODEL_DIR, device=str(device))
    image_model = load_image_model(IMAGE_MODEL_DIR, device=str(device))

    rng = random.Random(GLOBAL_SEED)
    modes = ["text_only", "image_only", "text_plus_image"]
    weights = [0.30, 0.30, 0.40]

    plan: list[tuple[str, int, int]] = []
    for _ in range(n_samples):
        mode = rng.choices(modes, weights=weights, k=1)[0]
        ti = rng.randrange(len(text_df)) if mode != "image_only" else -1
        ii = rng.randrange(len(image_df)) if mode != "text_only" else -1
        plan.append((mode, ti, ii))

    text_idx_unique = sorted({t for _, t, _ in plan if t >= 0})
    image_idx_unique = sorted({i for _, _, i in plan if i >= 0})
    log.info(f"unique prompts={len(text_idx_unique)} unique images={len(image_idx_unique)}")

    text_probs_map: dict[int, np.ndarray] = {}
    if text_idx_unique:
        texts = text_df.iloc[text_idx_unique]["text"].tolist()
        arr = _text_probs(texts, text_model, tokenizer, device)
        for k, idx in enumerate(text_idx_unique):
            text_probs_map[idx] = arr[k]

    image_probs_map: dict[int, np.ndarray] = {}
    if image_idx_unique:
        paths = image_df.iloc[image_idx_unique]["image_path"].tolist()
        arr = _image_probs(paths, image_model, device)
        for k, idx in enumerate(image_idx_unique):
            image_probs_map[idx] = arr[k]

    rows: list[dict] = []
    for mode, ti, ii in plan:
        req_probs = np.zeros(NUM_REQUEST, dtype=np.float32)
        doc_probs = np.zeros(NUM_DOC, dtype=np.float32)
        text_conf = image_conf = 0.0
        has_text = has_image = 0
        text_entropy = image_entropy = 0.0
        text_margin = image_margin = 0.0
        true_request = None
        true_document = None

        if ti >= 0:
            req_probs = text_probs_map[ti]
            text_conf = float(req_probs.max())
            has_text = 1
            true_request = str(text_df.iloc[ti]["label"])
            text_entropy = _entropy(req_probs)
            text_margin = _margin(req_probs)

        if ii >= 0:
            doc_probs = image_probs_map[ii]
            image_conf = float(doc_probs.max())
            has_image = 1
            true_document = str(image_df.iloc[ii]["label"])
            image_entropy = _entropy(doc_probs)
            image_margin = _margin(doc_probs)

        triage = classify_triage(true_request, true_document)
        row = {col: 0.0 for col in FEATURE_COLS}
        for k, c in enumerate(REQUEST_CLASSES):
            row[f"req_{c}"] = float(req_probs[k])
        for k, c in enumerate(DOCUMENT_CLASSES):
            row[f"doc_{c}"] = float(doc_probs[k])
        row.update({
            "text_conf": float(text_conf),
            "image_conf": float(image_conf),
            "has_text": int(has_text),
            "has_image": int(has_image),
            "text_entropy": float(text_entropy),
            "image_entropy": float(image_entropy),
            "text_margin": float(text_margin),
            "image_margin": float(image_margin),
            "input_mode": mode,
            "true_request": true_request or "",
            "true_document": true_document or "",
            "triage_label": triage,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    feat_dim = sum(1 for c in df.columns if c in FEATURE_COLS)
    if feat_dim != FUSION_INPUT_DIM:
        raise RuntimeError(f"Feature dim mismatch {feat_dim} vs {FUSION_INPUT_DIM}")

    out_csv = PROCESSED_DIR / "fusion_features.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"Wrote {out_csv} (n={len(df)} dim={feat_dim})")

    splits = _stratified_split(df)
    for n, p in splits.items():
        path = SPLITS_DIR / f"fusion_{n}.csv"
        p.to_csv(path, index=False)
        log.info(f"Wrote {path} (n={len(p)})")

    summary = {
        "total": int(len(df)),
        "feature_dim": int(feat_dim),
        "per_triage_class": df["triage_label"].value_counts().to_dict(),
        "per_input_mode": df["input_mode"].value_counts().to_dict(),
        "splits": {n: int(len(p)) for n, p in splits.items()},
        "seed": GLOBAL_SEED,
        "triage_class_order": TRIAGE_CLASSES,
        "feature_cols": FEATURE_COLS,
        "labels_origin": "policy-derived: classify_triage(true_request, true_document)",
    }
    Path(REPORTS_DIR / "fusion_data_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    log.info(f"per-class counts: {summary['per_triage_class']}")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    args = parser.parse_args()
    main(n_samples=args.n)


if __name__ == "__main__":
    cli()
