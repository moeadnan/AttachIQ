"""End-to-end inference tests with smoke/mock models.

We don't depend on having trained models here. Instead we build a minimal
mock pipeline by injecting deterministic prob vectors and use the real fusion
logic + policy.
"""

from __future__ import annotations

import numpy as np

from attachiq.config import (
    DECISION_MAP,
    DOCUMENT_CLASSES,
    FUSION_INPUT_DIM,
    NUM_DOC,
    NUM_REQUEST,
    REQUEST_CLASSES,
    TRIAGE_CLASSES,
)
from attachiq.inference.pipeline import TriagePipeline
from attachiq.schemas import InferenceRequest, InferenceResponse
from attachiq.triage.policy import classify_triage


class _FakeFusion:
    """Returns logits that argmax to a label chosen by the rule policy."""

    def __init__(self, request_probs: np.ndarray, document_probs: np.ndarray, has_text: int, has_image: int):
        req = REQUEST_CLASSES[int(np.argmax(request_probs))] if has_text else None
        doc = DOCUMENT_CLASSES[int(np.argmax(document_probs))] if has_image else None
        self._target = classify_triage(req, doc)

    def __call__(self, x):  # noqa: D401
        import torch
        idx = TRIAGE_CLASSES.index(self._target)
        logits = np.full(len(TRIAGE_CLASSES), -5.0, dtype=np.float32)
        logits[idx] = 5.0
        return torch.from_numpy(logits).unsqueeze(0)


class MockPipeline(TriagePipeline):
    """Pipeline subclass with mocked branches and fusion model."""

    def __init__(self, request_label: str | None = "summarization", document_label: str | None = "presentation"):
        super().__init__(device="cpu")
        self.request_label = request_label
        self.document_label = document_label

    def _ensure_text(self) -> None:
        return

    def _ensure_image(self) -> None:
        return

    def _ensure_fusion(self) -> None:
        return

    def _text_branch(self, prompt_text: str):
        probs = np.full(NUM_REQUEST, 0.01, dtype=np.float32)
        if self.request_label is not None:
            probs[REQUEST_CLASSES.index(self.request_label)] = 0.9
        idx = int(np.argmax(probs))
        return probs, REQUEST_CLASSES[idx], float(probs[idx])

    def _image_branch(self, image_path: str):
        probs = np.full(NUM_DOC, 0.01, dtype=np.float32)
        if self.document_label is not None:
            probs[DOCUMENT_CLASSES.index(self.document_label)] = 0.9
        idx = int(np.argmax(probs))
        return probs, DOCUMENT_CLASSES[idx], float(probs[idx])

    def _run_fusion(self, vec: np.ndarray):
        assert vec.shape[0] == FUSION_INPUT_DIM
        # Use the rule policy as the fake fusion oracle for tests.
        # Trailing 8 scalars: text_conf, image_conf, has_text, has_image,
        # text_entropy, image_entropy, text_margin, image_margin.
        has_text = int(vec[-6])
        has_image = int(vec[-5])
        rp = vec[:NUM_REQUEST]
        dp = vec[NUM_REQUEST : NUM_REQUEST + NUM_DOC]
        req = REQUEST_CLASSES[int(np.argmax(rp))] if has_text else None
        doc = DOCUMENT_CLASSES[int(np.argmax(dp))] if has_image else None
        target = classify_triage(req, doc)
        return target, 0.95


def test_text_plus_image_low_risk() -> None:
    pipe = MockPipeline("summarization", "presentation")
    resp = pipe.predict(InferenceRequest(prompt_text="summarize", image_path="/tmp/fake.png"))
    assert isinstance(resp, InferenceResponse)
    assert resp.input_mode == "text_plus_image"
    assert resp.compatibility_label == "compatible_low_risk"
    assert resp.decision == "ALLOW"
    assert resp.request_type == "summarization"
    assert resp.document_type == "presentation"


def test_text_plus_image_block() -> None:
    pipe = MockPipeline("public_sharing", "resume")
    resp = pipe.predict(InferenceRequest(prompt_text="post this publicly", image_path="/tmp/fake.png"))
    assert resp.compatibility_label == "unsafe_external_action"
    assert resp.decision == "BLOCK"


def test_text_plus_image_review() -> None:
    pipe = MockPipeline("financial_extraction", "invoice")
    resp = pipe.predict(InferenceRequest(prompt_text="extract total", image_path="/tmp/fake.png"))
    assert resp.compatibility_label == "compatible_sensitive"
    assert resp.decision == "REVIEW"


def test_text_only_mode() -> None:
    pipe = MockPipeline("summarization", None)
    resp = pipe.predict(InferenceRequest(prompt_text="summarize"))
    assert resp.input_mode == "text_only"
    assert resp.document_type is None
    assert resp.decision in {"ALLOW", "REVIEW", "BLOCK"}


def test_image_only_mode() -> None:
    pipe = MockPipeline(None, "invoice")
    resp = pipe.predict(InferenceRequest(image_path="/tmp/fake.png"))
    assert resp.input_mode == "image_only"
    assert resp.request_type is None


def test_decision_in_map() -> None:
    pipe = MockPipeline("financial_extraction", "presentation")
    resp = pipe.predict(InferenceRequest(prompt_text="extract total", image_path="/tmp/fake.png"))
    assert resp.decision == DECISION_MAP[resp.compatibility_label]
