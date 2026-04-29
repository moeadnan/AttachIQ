"""Deterministic explanation templates for triage decisions."""

from __future__ import annotations

EXPLANATIONS: dict[str, str] = {
    "compatible_low_risk": "The request matches the attachment and appears low risk.",
    "compatible_sensitive": (
        "The request matches the attachment, but it involves financial, personal, "
        "or structured information."
    ),
    "mismatch_unclear": "The request and attachment do not clearly fit the expected pattern.",
    "unsafe_external_action": (
        "The request asks to share, publish, or delete material that may be sensitive."
    ),
}


def explain(triage_label: str) -> str:
    return EXPLANATIONS.get(triage_label, "No explanation available.")
