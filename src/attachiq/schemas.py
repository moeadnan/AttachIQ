"""Pydantic v2 contracts for AttachIQ requests and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

InputMode = Literal["text_only", "image_only", "text_plus_image"]
Decision = Literal["ALLOW", "REVIEW", "BLOCK"]


class InferenceRequest(BaseModel):
    """User input to the AttachIQ inference pipeline."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt_text: str | None = Field(default=None, description="Optional user prompt.")
    image_path: str | None = Field(default=None, description="Optional path to a document image.")
    input_mode: InputMode | None = Field(
        default=None,
        description="Optional explicit mode. The pipeline resolves it from inputs if missing.",
    )

    @model_validator(mode="after")
    def _check_at_least_one_input(self) -> "InferenceRequest":
        has_text = bool(self.prompt_text) and self.prompt_text.strip() != ""
        has_image = bool(self.image_path) and self.image_path.strip() != ""
        if not has_text and not has_image:
            raise ValueError("InferenceRequest requires at least one of prompt_text or image_path.")

        if self.input_mode == "text_only" and not has_text:
            raise ValueError("input_mode=text_only requires prompt_text.")
        if self.input_mode == "image_only" and not has_image:
            raise ValueError("input_mode=image_only requires image_path.")
        if self.input_mode == "text_plus_image" and not (has_text and has_image):
            raise ValueError("input_mode=text_plus_image requires both prompt_text and image_path.")
        return self


class InferenceResponse(BaseModel):
    """Structured triage result returned by the pipeline."""

    model_config = ConfigDict(extra="forbid")

    input_mode: InputMode
    request_type: str | None = None
    document_type: str | None = None
    compatibility_label: str
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    inference_time_ms: float = Field(ge=0.0)
