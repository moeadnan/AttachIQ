"""Tests for Pydantic contracts."""

import pytest
from pydantic import ValidationError

from attachiq.schemas import InferenceRequest, InferenceResponse


def test_request_valid_text_only() -> None:
    req = InferenceRequest(prompt_text="Summarize this", input_mode="text_only")
    assert req.prompt_text == "Summarize this"
    assert req.input_mode == "text_only"


def test_request_valid_image_only() -> None:
    req = InferenceRequest(image_path="/tmp/x.png", input_mode="image_only")
    assert req.image_path.endswith(".png")


def test_request_valid_both() -> None:
    req = InferenceRequest(prompt_text="Hello", image_path="/tmp/x.png")
    assert req.prompt_text == "Hello"
    assert req.image_path.endswith(".png")


def test_request_empty_fails() -> None:
    with pytest.raises(ValidationError):
        InferenceRequest()


def test_request_blank_fails() -> None:
    with pytest.raises(ValidationError):
        InferenceRequest(prompt_text="   ")


def test_request_mode_mismatch_fails() -> None:
    with pytest.raises(ValidationError):
        InferenceRequest(prompt_text="hi", input_mode="image_only")
    with pytest.raises(ValidationError):
        InferenceRequest(image_path="/tmp/x.png", input_mode="text_only")
    with pytest.raises(ValidationError):
        InferenceRequest(prompt_text="hi", input_mode="text_plus_image")


def test_response_valid() -> None:
    resp = InferenceResponse(
        input_mode="text_plus_image",
        request_type="summarization",
        document_type="invoice",
        compatibility_label="compatible_sensitive",
        decision="REVIEW",
        confidence=0.81,
        explanation="ok",
        inference_time_ms=12.3,
    )
    assert resp.decision == "REVIEW"
    assert 0.0 <= resp.confidence <= 1.0


def test_response_invalid_decision() -> None:
    with pytest.raises(ValidationError):
        InferenceResponse(
            input_mode="text_only",
            compatibility_label="compatible_low_risk",
            decision="MAYBE",  # type: ignore[arg-type]
            confidence=0.5,
            explanation="x",
            inference_time_ms=1.0,
        )


def test_response_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        InferenceResponse(
            input_mode="text_only",
            compatibility_label="compatible_low_risk",
            decision="ALLOW",
            confidence=1.7,
            explanation="x",
            inference_time_ms=1.0,
        )
