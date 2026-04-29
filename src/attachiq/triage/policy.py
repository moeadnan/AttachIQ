"""Deterministic triage policy.

Operates over the 10 request classes and the 8 document classes and returns
one of the 4 triage classes used by the system: ``compatible_low_risk``,
``compatible_sensitive``, ``mismatch_unclear``, or ``unsafe_external_action``.

Used both as the rule-table baseline and as the labeller for the standard
fusion training dataset. NOT used to label the hard rubric-labelled fusion
dataset, which is human-curated.
"""

from __future__ import annotations

from attachiq.config import (
    DECISION_MAP,
    DOCUMENT_CLASSES,
    REQUEST_CLASSES,
    TRIAGE_CLASSES,
)

LOW_RISK_DOCS = {"presentation", "letter"}
SENSITIVE_DOCS = {"invoice", "form", "resume", "email", "handwritten", "report"}
EXTRACTABLE_DOCS = {"invoice", "form", "resume", "email", "report"}
PUBLIC_RISKY_DOCS = {"invoice", "form", "resume", "email", "handwritten", "report"}


def _validate(request: str | None, document: str | None) -> None:
    if request is not None and request not in REQUEST_CLASSES:
        raise ValueError(f"Unknown request class: {request}")
    if document is not None and document not in DOCUMENT_CLASSES:
        raise ValueError(f"Unknown document class: {document}")


def classify_triage(request: str | None, document: str | None) -> str:
    """Map a (request, document) pair to one of the 4 triage classes."""
    _validate(request, document)

    if request is None and document is None:
        return "mismatch_unclear"

    # Ambiguity dominates: anything explicitly ambiguous routes to REVIEW.
    if request == "ambiguous_or_unclear":
        return "mismatch_unclear"

    # Permanent deletion is always unsafe regardless of the document.
    if request == "delete_permanent":
        return "unsafe_external_action"

    # Public sharing nuance.
    if request == "public_sharing":
        if document is None:
            return "unsafe_external_action"
        if document in PUBLIC_RISKY_DOCS:
            return "unsafe_external_action"
        return "compatible_low_risk"

    # Image-only path with a known document.
    if request is None:
        if document in SENSITIVE_DOCS:
            return "compatible_sensitive"
        return "compatible_low_risk"

    # Text-only path with a known request.
    if document is None:
        if request in {
            "summarization", "document_classification", "archive_retain",
            "redaction_or_safe_transform",
        }:
            return "compatible_low_risk"
        if request in {"information_extraction", "financial_extraction", "internal_sharing"}:
            return "mismatch_unclear"
        return "mismatch_unclear"

    # Both modalities present.
    if request == "archive_retain":
        if document in LOW_RISK_DOCS:
            return "compatible_low_risk"
        return "compatible_sensitive"

    if request == "redaction_or_safe_transform":
        if document in SENSITIVE_DOCS:
            return "compatible_sensitive"
        return "compatible_low_risk"

    if request == "summarization":
        if document in LOW_RISK_DOCS or document == "report":
            return "compatible_low_risk"
        if document in SENSITIVE_DOCS:
            return "compatible_sensitive"
        return "compatible_low_risk"

    if request == "document_classification":
        return "compatible_low_risk"

    if request == "financial_extraction":
        if document == "invoice":
            return "compatible_sensitive"
        if document == "form":
            return "compatible_sensitive"
        return "mismatch_unclear"

    if request == "information_extraction":
        if document in EXTRACTABLE_DOCS:
            return "compatible_sensitive"
        return "mismatch_unclear"

    if request == "internal_sharing":
        if document in SENSITIVE_DOCS:
            return "compatible_sensitive"
        return "compatible_low_risk"

    return "mismatch_unclear"


def decision_for_triage(triage_label: str) -> str:
    if triage_label not in TRIAGE_CLASSES:
        raise ValueError(f"Unknown triage class: {triage_label}")
    return DECISION_MAP[triage_label]
