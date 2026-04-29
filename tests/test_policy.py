"""Tests for the rule-based triage policy."""

import pytest

from attachiq.config import DOCUMENT_CLASSES, REQUEST_CLASSES, TRIAGE_CLASSES
from attachiq.triage.policy import classify_triage, decision_for_triage


def test_classify_returns_known_label_for_all_pairs() -> None:
    for r in REQUEST_CLASSES:
        for d in DOCUMENT_CLASSES:
            label = classify_triage(r, d)
            assert label in TRIAGE_CLASSES


def test_summarization_presentation_low_risk() -> None:
    assert classify_triage("summarization", "presentation") == "compatible_low_risk"


def test_financial_extraction_invoice_sensitive() -> None:
    assert classify_triage("financial_extraction", "invoice") == "compatible_sensitive"


def test_public_sharing_resume_unsafe() -> None:
    assert classify_triage("public_sharing", "resume") == "unsafe_external_action"


def test_delete_permanent_unsafe() -> None:
    # Permanent deletion is unsafe regardless of the document.
    assert classify_triage("delete_permanent", "invoice") == "unsafe_external_action"
    assert classify_triage("delete_permanent", "letter") == "unsafe_external_action"
    assert classify_triage("delete_permanent", None) == "unsafe_external_action"


def test_archive_retain_low_risk_for_safe_docs() -> None:
    assert classify_triage("archive_retain", "presentation") == "compatible_low_risk"
    assert classify_triage("archive_retain", "letter") == "compatible_low_risk"


def test_archive_retain_sensitive_for_personal_docs() -> None:
    assert classify_triage("archive_retain", "resume") == "compatible_sensitive"
    assert classify_triage("archive_retain", "invoice") == "compatible_sensitive"


def test_ambiguous_or_unclear_routes_to_review() -> None:
    for d in DOCUMENT_CLASSES:
        assert classify_triage("ambiguous_or_unclear", d) == "mismatch_unclear"


def test_redaction_or_safe_transform() -> None:
    # Sensitive document → review even with redaction modifier.
    assert classify_triage("redaction_or_safe_transform", "resume") == "compatible_sensitive"
    # Low-risk document → allow.
    assert classify_triage("redaction_or_safe_transform", "presentation") == "compatible_low_risk"


def test_extract_total_from_presentation_unclear() -> None:
    assert classify_triage("financial_extraction", "presentation") == "mismatch_unclear"


def test_text_only_modes() -> None:
    assert classify_triage("summarization", None) == "compatible_low_risk"
    assert classify_triage("public_sharing", None) == "unsafe_external_action"


def test_image_only_sensitive_doc() -> None:
    assert classify_triage(None, "invoice") == "compatible_sensitive"
    assert classify_triage(None, "presentation") == "compatible_low_risk"


def test_decision_mapping() -> None:
    assert decision_for_triage("compatible_low_risk") == "ALLOW"
    assert decision_for_triage("compatible_sensitive") == "REVIEW"
    assert decision_for_triage("mismatch_unclear") == "REVIEW"
    assert decision_for_triage("unsafe_external_action") == "BLOCK"


def test_decision_mapping_invalid() -> None:
    with pytest.raises(ValueError):
        decision_for_triage("nope")
