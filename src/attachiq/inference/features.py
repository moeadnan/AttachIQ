"""Compute the canonical 26-dim fusion feature vector for arbitrary
(prompt, image) inputs.

Used by:
  * the fusion trainer to materialise features over rubric-labelled rows
  * evaluation across the standard test, the 390-row hard challenge set,
    and the held-out hard rubric fusion test split
  * the inference pipeline at demo time
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from attachiq.config import (
    DOCUMENT_CLASSES,
    FUSION_INPUT_DIM,
    IMAGE_MODEL_DIR,
    NUM_DOC,
    NUM_REQUEST,
    REQUEST_CLASSES,
    TEXT_CFG,
    TEXT_MODEL_DIR,
)
from attachiq.data.image_dataset import get_transforms


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _margin(p: np.ndarray) -> float:
    if len(p) < 2:
        return float(p.max())
    s = np.sort(p)[::-1]
    return float(s[0] - s[1])


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def text_probs(texts: Sequence[str], model, tokenizer, device, batch_size: int = 64) -> np.ndarray:
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            enc = tokenizer(
                chunk, truncation=True, padding=True,
                max_length=TEXT_CFG.max_length, return_tensors="pt",
            ).to(device)
            logits = model(enc["input_ids"], enc["attention_mask"])
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, NUM_REQUEST), dtype=np.float32)


def image_probs(paths: Sequence[str], model, device, batch_size: int = 32) -> np.ndarray:
    out: list[np.ndarray] = []
    model.eval()
    transform = get_transforms(train=False)
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = list(paths[i:i + batch_size])
            tensors = []
            for p in chunk:
                img = Image.open(p).convert("L")
                tensors.append(transform(img))
            x = torch.stack(tensors).to(device)
            logits = model(x)
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, NUM_DOC), dtype=np.float32)


def build_feature_row(
    request_probs: np.ndarray | None,
    document_probs: np.ndarray | None,
) -> np.ndarray:
    has_text = int(request_probs is not None)
    has_image = int(document_probs is not None)

    if request_probs is None:
        request_probs = np.zeros(NUM_REQUEST, dtype=np.float32)
        text_conf = 0.0
        text_entropy = 0.0
        text_margin = 0.0
    else:
        text_conf = float(request_probs.max())
        text_entropy = _entropy(request_probs)
        text_margin = _margin(request_probs)

    if document_probs is None:
        document_probs = np.zeros(NUM_DOC, dtype=np.float32)
        image_conf = 0.0
        image_entropy = 0.0
        image_margin = 0.0
    else:
        image_conf = float(document_probs.max())
        image_entropy = _entropy(document_probs)
        image_margin = _margin(document_probs)

    vec = np.concatenate([
        request_probs.astype(np.float32),
        document_probs.astype(np.float32),
        np.array(
            [
                text_conf, image_conf, has_text, has_image,
                text_entropy, image_entropy, text_margin, image_margin,
            ],
            dtype=np.float32,
        ),
    ])
    if vec.shape[0] != FUSION_INPUT_DIM:
        raise ValueError(f"Feature dim mismatch {vec.shape[0]} vs {FUSION_INPUT_DIM}")
    return vec


def materialise_features_for_dataframe(df) -> np.ndarray:
    """Materialise 26-dim features for a dataframe with optional
    ``prompt_text`` and ``image_path`` columns."""
    device = _device()

    from attachiq.models.text_model import load_text_model
    from attachiq.models.image_cnn import load_image_model

    text_model, tokenizer = load_text_model(TEXT_MODEL_DIR, device=str(device))
    image_model = load_image_model(IMAGE_MODEL_DIR, device=str(device))

    has_prompt = "prompt_text" in df.columns
    has_image_col = "image_path" in df.columns

    if has_prompt:
        texts = df["prompt_text"].fillna("").astype(str).tolist()
        text_present = [bool(t.strip()) for t in texts]
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if non_empty:
            probs_arr = text_probs([t for _, t in non_empty], text_model, tokenizer, device)
            text_probs_full = np.zeros((len(df), NUM_REQUEST), dtype=np.float32)
            for (idx, _), prob in zip(non_empty, probs_arr):
                text_probs_full[idx] = prob
        else:
            text_probs_full = np.zeros((len(df), NUM_REQUEST), dtype=np.float32)
    else:
        text_present = [False] * len(df)
        text_probs_full = np.zeros((len(df), NUM_REQUEST), dtype=np.float32)

    if has_image_col:
        paths = df["image_path"].fillna("").astype(str).tolist()
        image_present = [bool(p.strip()) and Path(p).exists() for p in paths]
        non_empty = [(i, p) for i, p in enumerate(paths) if image_present[i]]
        if non_empty:
            probs_arr = image_probs([p for _, p in non_empty], image_model, device)
            image_probs_full = np.zeros((len(df), NUM_DOC), dtype=np.float32)
            for (idx, _), prob in zip(non_empty, probs_arr):
                image_probs_full[idx] = prob
        else:
            image_probs_full = np.zeros((len(df), NUM_DOC), dtype=np.float32)
    else:
        image_present = [False] * len(df)
        image_probs_full = np.zeros((len(df), NUM_DOC), dtype=np.float32)

    feats = np.zeros((len(df), FUSION_INPUT_DIM), dtype=np.float32)
    for i in range(len(df)):
        rp = text_probs_full[i] if text_present[i] else None
        dp = image_probs_full[i] if image_present[i] else None
        feats[i] = build_feature_row(rp, dp)
    return feats


__all__ = [
    "text_probs",
    "image_probs",
    "build_feature_row",
    "materialise_features_for_dataframe",
]
