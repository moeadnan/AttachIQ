"""Canonical AttachIQ inference pipeline.

Used by the Streamlit UI, the CLI, and the tests. Loads the three trained
models lazily on first call: a fine-tuned DistilBERT request classifier
(10 classes), the deep scratch document image CNN (8 classes), and the
fusion MLP that combines their probability outputs and uncertainty
signals into a 4-class triage decision.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from attachiq.config import (
    DECISION_MAP,
    DOCUMENT_CLASSES,
    FUSION_INPUT_DIM,
    FUSION_MODEL_DIR,
    IMAGE_MODEL_DIR,
    NUM_DOC,
    NUM_REQUEST,
    REQUEST_CLASSES,
    TEXT_CFG,
    TEXT_MODEL_DIR,
    TRIAGE_CLASSES,
)
from attachiq.data.image_dataset import get_transforms
from attachiq.inference.explanations import explain
from attachiq.logging import get_logger
from attachiq.schemas import InferenceRequest, InferenceResponse

log = get_logger("inference")


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _margin(p: np.ndarray) -> float:
    if len(p) < 2:
        return float(p.max())
    s = np.sort(p)[::-1]
    return float(s[0] - s[1])


class TriagePipeline:
    """Lazy-loaded multi-modal triage pipeline."""

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device) if device else _select_device()
        self._text_model = None
        self._tokenizer = None
        self._image_model = None
        self._image_transform = None
        self._fusion_model = None

    # --- Lazy loading ----------------------------------------------------- #
    def _ensure_text(self) -> None:
        if self._text_model is None:
            from attachiq.models.text_model import load_text_model
            model, tok = load_text_model(TEXT_MODEL_DIR, device=str(self.device))
            self._text_model = model
            self._tokenizer = tok
            log.info("Loaded text model.")

    def _ensure_image(self) -> None:
        if self._image_model is None:
            from attachiq.models.image_cnn import load_image_model
            self._image_model = load_image_model(IMAGE_MODEL_DIR, device=str(self.device))
            self._image_transform = get_transforms(train=False)
            log.info("Loaded image model.")

    def _ensure_fusion(self) -> None:
        if self._fusion_model is None:
            from attachiq.models.fusion_mlp import load_fusion_model
            self._fusion_model = load_fusion_model(FUSION_MODEL_DIR, device=str(self.device))
            log.info("Loaded fusion model.")

    # --- Branch inference ------------------------------------------------- #
    def _text_branch(self, prompt_text: str) -> tuple[np.ndarray, str, float]:
        self._ensure_text()
        enc = self._tokenizer(
            prompt_text,
            truncation=True,
            padding=True,
            max_length=TEXT_CFG.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self._text_model(enc["input_ids"], enc["attention_mask"])
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return probs.astype(np.float32), REQUEST_CLASSES[idx], float(probs[idx])

    def _image_branch(self, image_path: str) -> tuple[np.ndarray, str, float]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        self._ensure_image()
        image = Image.open(image_path).convert("L")
        x = self._image_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._image_model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return probs.astype(np.float32), DOCUMENT_CLASSES[idx], float(probs[idx])

    @staticmethod
    def _resolve_input_mode(request: InferenceRequest) -> str:
        if request.input_mode is not None:
            return request.input_mode
        has_text = bool(request.prompt_text)
        has_image = bool(request.image_path)
        if has_text and has_image:
            return "text_plus_image"
        if has_text:
            return "text_only"
        return "image_only"

    @staticmethod
    def build_feature_vector(
        request_probs: np.ndarray,
        document_probs: np.ndarray,
        text_conf: float,
        image_conf: float,
        has_text: int,
        has_image: int,
        text_entropy: float,
        image_entropy: float,
        text_margin: float,
        image_margin: float,
    ) -> np.ndarray:
        """Assemble the canonical 26-dim fusion feature vector."""
        if request_probs.shape[0] != NUM_REQUEST or document_probs.shape[0] != NUM_DOC:
            raise ValueError(
                f"Bad probability shapes: request={request_probs.shape}, document={document_probs.shape}"
            )
        vec = np.concatenate(
            [
                request_probs.astype(np.float32),
                document_probs.astype(np.float32),
                np.array(
                    [
                        text_conf, image_conf, has_text, has_image,
                        text_entropy, image_entropy, text_margin, image_margin,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        if vec.shape[0] != FUSION_INPUT_DIM:
            raise ValueError(f"Fusion vector must be {FUSION_INPUT_DIM}-dim, got {vec.shape[0]}")
        return vec

    def _run_fusion(self, vec: np.ndarray) -> tuple[str, float]:
        self._ensure_fusion()
        with torch.no_grad():
            t = torch.from_numpy(vec).unsqueeze(0).to(self.device)
            logits = self._fusion_model(t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return TRIAGE_CLASSES[idx], float(probs[idx])

    # --- Public API ------------------------------------------------------- #
    def predict(self, request: InferenceRequest) -> InferenceResponse:
        t0 = time.perf_counter()
        mode = self._resolve_input_mode(request)

        request_probs = np.zeros(NUM_REQUEST, dtype=np.float32)
        document_probs = np.zeros(NUM_DOC, dtype=np.float32)
        request_type: str | None = None
        document_type: str | None = None
        text_conf = 0.0
        image_conf = 0.0
        has_text = 0
        has_image = 0
        text_entropy = 0.0
        image_entropy = 0.0
        text_margin = 0.0
        image_margin = 0.0

        if mode in ("text_only", "text_plus_image"):
            request_probs, request_type, text_conf = self._text_branch(request.prompt_text or "")
            has_text = 1
            text_entropy = _entropy(request_probs)
            text_margin = _margin(request_probs)

        if mode in ("image_only", "text_plus_image"):
            if not request.image_path:
                raise ValueError("image_path required for image-bearing modes")
            document_probs, document_type, image_conf = self._image_branch(request.image_path)
            has_image = 1
            image_entropy = _entropy(document_probs)
            image_margin = _margin(document_probs)

        vec = self.build_feature_vector(
            request_probs, document_probs,
            text_conf, image_conf, has_text, has_image,
            text_entropy, image_entropy, text_margin, image_margin,
        )
        triage_label, triage_conf = self._run_fusion(vec)
        decision = DECISION_MAP[triage_label]
        explanation = explain(triage_label)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return InferenceResponse(
            input_mode=mode,
            request_type=request_type,
            document_type=document_type,
            compatibility_label=triage_label,
            decision=decision,
            confidence=float(triage_conf),
            explanation=explanation,
            inference_time_ms=float(elapsed_ms),
        )


_singleton: TriagePipeline | None = None


def get_pipeline() -> TriagePipeline:
    global _singleton
    if _singleton is None:
        _singleton = TriagePipeline()
    return _singleton


def predict(request: InferenceRequest) -> InferenceResponse:
    return get_pipeline().predict(request)
