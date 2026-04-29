"""Build the hard rubric-labelled fusion training dataset.

Approximately 2,360 (prompt, image) pairs labelled by hand against the
spec rubric. Used as additional supervision for the fusion MLP that does
not rely on the policy-derived label distribution.

Constraints:
  * Real RVL-CDIP images only, drawn from
    ``data/raw/images/<class>/`` with image indices >= 700 so the set
    is disjoint from the 390-row evaluation-only hard challenge set
    (which uses indices 300..~700).
  * Triage labels are rubric-aligned and never produced by
    ``classify_triage`` or by any model output.
  * 70/15/15 stratified split by ``human_triage_label``.

Outputs:
  data/processed/hard_fusion_dataset.csv
  data/splits/hard_fusion_{train,val,test}.csv
  reports/hard_fusion_data_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from attachiq.config import (
    DOCUMENT_CLASSES,
    GLOBAL_SEED,
    PROCESSED_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    TRIAGE_CLASSES,
    ensure_dirs,
)
from attachiq.logging import get_logger

log = get_logger("data.hard_fusion")

CHALLENGE_TYPES = (
    "vague_prompt", "ambiguous_action", "archive_vs_delete",
    "redaction_modifier", "public_vs_internal_sharing",
    "partial_extraction", "misleading_request_document_pair",
    "visually_similar_document", "sensitivity_modifier",
    "conflicting_instruction",
)
DIFFICULTY_LEVELS = ("hard", "very_hard", "adversarial_but_realistic")
REVIEW_STATUSES = ("rubric_assigned", "borderline_reviewed", "revised_after_review")

IMAGES_ROOT = Path("data/raw/images")
BASE_IDX = 700
MAX_PER_CLASS = 999

TUPLES: list[tuple[str, str, str, str, str, str, str, str]] = []


def _add_for_pairs(prompt: str, mapping: dict[str, str], ct: str, intent: str,
                   diff: str = "hard", review: str = "rubric_assigned",
                   note: str = "") -> None:
    for cls, label in mapping.items():
        TUPLES.append((prompt, cls, intent, label, ct, diff, review, note))


# --- Rubric tuples ---------------------------------------------------------- #
VAGUE_PROMPTS = [
    "Can you handle this?", "Take care of this.", "Process this.",
    "Do the necessary.", "Check this and tell me what to do.",
    "Use this where needed.", "Can you help with this file?",
    "Please deal with this.", "Review this and proceed.",
    "Do whatever is appropriate.",
]
for p in VAGUE_PROMPTS:
    for cls in DOCUMENT_CLASSES:
        TUPLES.append((p, cls, "ambiguous_or_unclear", "mismatch_unclear",
                       "vague_prompt", "hard", "rubric_assigned",
                       "Vague intent; review."))

AMBIGUOUS_ACTIONS = [
    "Prepare this.", "Use this.", "Move forward with this.",
    "Handle as discussed.", "Do the next step.",
    "Can you make this ready?", "Complete this.",
    "Follow up on this.", "Take action on this.", "Proceed with this.",
]
for p in AMBIGUOUS_ACTIONS:
    for cls in DOCUMENT_CLASSES:
        TUPLES.append((p, cls, "ambiguous_or_unclear", "mismatch_unclear",
                       "ambiguous_action", "hard", "rubric_assigned",
                       "Ambiguous action verb; review."))

ARCHIVE_PROMPTS = [
    "Archive this, do not delete it.", "Move this to long-term storage.",
    "Keep this for records.", "Store this for audit.",
    "Retain this but remove it from active view.",
    "Hold this for retention.", "Keep this on file.",
    "Save this for our audit trail.",
]
for p in ARCHIVE_PROMPTS:
    _add_for_pairs(
        p,
        {"letter": "compatible_low_risk", "presentation": "compatible_low_risk",
         "report": "compatible_low_risk",
         "invoice": "compatible_sensitive", "form": "compatible_sensitive",
         "resume": "compatible_sensitive", "email": "compatible_sensitive",
         "handwritten": "compatible_sensitive"},
        "archive_vs_delete", "archive_retain",
        diff="very_hard", review="borderline_reviewed",
        note="Archive of sensitive doc -> review; non-sensitive -> allow.",
    )

DELETE_PROMPTS = [
    "Delete this permanently.", "Permanently remove this from the system.",
    "Destroy this file after checking.", "Discard the outdated version only.",
    "Remove only the duplicate copy.", "Wipe this from the system.",
]
for p in DELETE_PROMPTS:
    for cls in DOCUMENT_CLASSES:
        TUPLES.append((p, cls, "delete_permanent", "unsafe_external_action",
                       "archive_vs_delete", "hard", "rubric_assigned",
                       "Permanent deletion -> block."))

REDACTION = [
    "Summarize this without personal details.",
    "Share this after removing sensitive information.",
    "Extract only non-sensitive fields.",
    "Make a public-safe summary.",
    "Remove names before summarizing.",
    "Summarize but hide financial figures.",
    "Prepare a version without private data.",
    "Anonymize personal information and pass it on.",
]
for p in REDACTION:
    _add_for_pairs(
        p,
        {"resume": "compatible_sensitive", "invoice": "compatible_sensitive",
         "form": "compatible_sensitive", "email": "compatible_sensitive",
         "handwritten": "compatible_sensitive", "report": "compatible_sensitive",
         "letter": "compatible_low_risk", "presentation": "compatible_low_risk"},
        "redaction_modifier", "redaction_or_safe_transform",
        diff="very_hard", review="borderline_reviewed",
        note="Modifier softens but does not erase sensitivity.",
    )

PUBLIC_REDACT = [
    "Use this publicly after anonymizing it.",
    "Make this public but keep the names hidden.",
    "Share publicly after removing PII.",
]
for p in PUBLIC_REDACT:
    _add_for_pairs(
        p,
        {"resume": "compatible_sensitive", "invoice": "compatible_sensitive",
         "form": "compatible_sensitive", "report": "compatible_sensitive",
         "letter": "compatible_low_risk", "presentation": "compatible_low_risk"},
        "redaction_modifier", "redaction_or_safe_transform",
        diff="adversarial_but_realistic", review="borderline_reviewed",
        note="Public + redaction softens but does not erase exposure risk.",
    )

INTERNAL = [
    "Share this with my finance team.", "Forward this internally only.",
    "Send this to my manager.", "Share this with HR.",
    "Send this to my coworker.", "Loop in my supervisor on this.",
    "Distribute this to my colleagues.",
]
for p in INTERNAL:
    _add_for_pairs(
        p,
        {"invoice": "compatible_sensitive", "form": "compatible_sensitive",
         "resume": "compatible_sensitive", "email": "compatible_sensitive",
         "report": "compatible_sensitive", "handwritten": "compatible_sensitive",
         "letter": "compatible_low_risk", "presentation": "compatible_low_risk"},
        "public_vs_internal_sharing", "internal_sharing",
        note="Internal sharing of sensitive doc -> review.",
    )

PUBLIC = [
    "Post this publicly.", "Use this in a public announcement.",
    "Upload this where everyone can see it.",
    "Send this to the public mailing list.",
    "Put this on the website.", "Tweet this.",
    "Publish this externally.",
]
for p in PUBLIC:
    _add_for_pairs(
        p,
        {"invoice": "unsafe_external_action", "form": "unsafe_external_action",
         "resume": "unsafe_external_action", "email": "unsafe_external_action",
         "report": "unsafe_external_action", "handwritten": "unsafe_external_action",
         "letter": "compatible_low_risk", "presentation": "compatible_low_risk"},
        "public_vs_internal_sharing", "public_sharing",
        note="Public sharing of sensitive doc -> block.",
    )

PARTIAL = [
    ("Extract only the invoice number.", "information_extraction",
     {"invoice": "compatible_sensitive", "form": "mismatch_unclear",
      "letter": "mismatch_unclear", "report": "mismatch_unclear"}),
    ("Find the date, not the amount.", "financial_extraction",
     {"invoice": "compatible_sensitive", "letter": "compatible_low_risk",
      "form": "compatible_sensitive", "email": "compatible_sensitive"}),
    ("Pull only contact details.", "information_extraction",
     {"resume": "compatible_sensitive", "email": "compatible_sensitive",
      "letter": "compatible_low_risk", "form": "compatible_sensitive"}),
    ("Extract the name and email.", "information_extraction",
     {"resume": "compatible_sensitive", "email": "compatible_sensitive",
      "form": "compatible_sensitive"}),
    ("Find the approval date.", "information_extraction",
     {"form": "compatible_sensitive", "report": "compatible_low_risk"}),
    ("Extract the project title only.", "information_extraction",
     {"presentation": "compatible_low_risk", "report": "compatible_low_risk"}),
    ("Find the sender and subject.", "information_extraction",
     {"email": "compatible_sensitive", "letter": "compatible_low_risk",
      "resume": "mismatch_unclear", "presentation": "mismatch_unclear"}),
    ("Extract the total amount if available.", "financial_extraction",
     {"invoice": "compatible_sensitive", "report": "compatible_sensitive",
      "presentation": "mismatch_unclear", "form": "compatible_sensitive"}),
]
for p, intent, mapping in PARTIAL:
    _add_for_pairs(p, mapping, "partial_extraction", intent,
                   note="Partial extraction; sensitive if doc is sensitive.")

MISLEADING_PAIRS = [
    ("Extract the invoice total.", "financial_extraction",
     {"presentation": "mismatch_unclear", "letter": "mismatch_unclear",
      "report": "mismatch_unclear", "email": "mismatch_unclear",
      "handwritten": "mismatch_unclear", "resume": "mismatch_unclear"}),
    ("Summarize this slide.", "summarization",
     {"invoice": "compatible_sensitive", "form": "compatible_sensitive",
      "resume": "compatible_sensitive", "email": "compatible_sensitive"}),
    ("Find the sender and subject.", "information_extraction",
     {"resume": "mismatch_unclear", "form": "mismatch_unclear",
      "invoice": "mismatch_unclear", "presentation": "mismatch_unclear"}),
    ("Extract form fields.", "information_extraction",
     {"letter": "mismatch_unclear", "invoice": "mismatch_unclear",
      "presentation": "mismatch_unclear"}),
    ("Use this in my application.", "summarization",
     {"resume": "compatible_low_risk", "letter": "compatible_low_risk",
      "form": "compatible_sensitive", "invoice": "mismatch_unclear"}),
    ("Read the line items.", "information_extraction",
     {"presentation": "mismatch_unclear", "letter": "mismatch_unclear",
      "report": "mismatch_unclear"}),
]
for p, intent, mapping in MISLEADING_PAIRS:
    _add_for_pairs(p, mapping, "misleading_request_document_pair", intent,
                   note="Request and document do not fit; review/allow per rubric.")

VISUAL_SIMILAR = [
    ("Pull invoice line items.", "information_extraction",
     {"form": "mismatch_unclear", "letter": "mismatch_unclear"}),
    ("Extract the work history.", "information_extraction",
     {"form": "mismatch_unclear", "letter": "mismatch_unclear"}),
    ("Read the table values.", "information_extraction",
     {"report": "compatible_sensitive", "presentation": "mismatch_unclear",
      "invoice": "compatible_sensitive", "form": "compatible_sensitive"}),
    ("Extract sender and recipient.", "information_extraction",
     {"letter": "compatible_sensitive", "email": "compatible_sensitive",
      "report": "mismatch_unclear"}),
    ("Identify form sections.", "information_extraction",
     {"form": "compatible_sensitive", "invoice": "mismatch_unclear"}),
    ("Just classify the kind of document this is.", "document_classification",
     {c: "compatible_low_risk" for c in DOCUMENT_CLASSES}),
]
for p, intent, mapping in VISUAL_SIMILAR:
    _add_for_pairs(p, mapping, "visually_similar_document", intent,
                   note="Label uses TRUE document class.")

SENS_MOD = [
    ("This is internal, summarize it.", "summarization",
     {"invoice": "compatible_sensitive", "resume": "compatible_sensitive",
      "presentation": "compatible_low_risk", "report": "compatible_low_risk",
      "letter": "compatible_low_risk"}),
    ("This is public already, summarize it.", "summarization",
     {"report": "compatible_low_risk", "presentation": "compatible_low_risk",
      "letter": "compatible_low_risk"}),
    ("This contains personal details, extract only the safe fields.",
     "information_extraction",
     {"resume": "compatible_sensitive", "form": "compatible_sensitive",
      "email": "compatible_sensitive"}),
    ("This has financial data, summarize without amounts.", "summarization",
     {"invoice": "compatible_sensitive", "report": "compatible_sensitive"}),
    ("This is for audit records only.", "archive_retain",
     {"invoice": "compatible_sensitive", "form": "compatible_sensitive",
      "report": "compatible_sensitive"}),
    ("This is for my private use only.", "summarization",
     {"resume": "compatible_low_risk", "invoice": "compatible_sensitive",
      "letter": "compatible_low_risk"}),
    ("This is for a public post.", "public_sharing",
     {"presentation": "compatible_low_risk", "letter": "compatible_low_risk",
      "invoice": "unsafe_external_action", "resume": "unsafe_external_action",
      "report": "unsafe_external_action"}),
    ("This is a draft, do not share externally.", "internal_sharing",
     {"report": "compatible_sensitive", "presentation": "compatible_low_risk",
      "letter": "compatible_low_risk"}),
]
for p, intent, mapping in SENS_MOD:
    _add_for_pairs(p, mapping, "sensitivity_modifier", intent,
                   note="Modifier shifts label per rubric.")

CONFLICTS = [
    ("Post this publicly but remove all sensitive details first.",
     "redaction_or_safe_transform",
     {"resume": "compatible_sensitive", "invoice": "compatible_sensitive",
      "report": "compatible_sensitive", "presentation": "compatible_low_risk"}),
    ("Make this public but keep the names hidden.",
     "redaction_or_safe_transform",
     {"resume": "compatible_sensitive", "form": "compatible_sensitive",
      "presentation": "compatible_low_risk"}),
    ("Delete this but keep it for records.",
     "ambiguous_or_unclear",
     {"report": "mismatch_unclear", "invoice": "mismatch_unclear",
      "form": "mismatch_unclear"}),
    ("Archive this permanently delete it.", "ambiguous_or_unclear",
     {"invoice": "mismatch_unclear", "report": "mismatch_unclear"}),
    ("Share this externally but only with the finance team.", "internal_sharing",
     {"invoice": "compatible_sensitive", "report": "compatible_sensitive"}),
    ("Extract the total but do not read the financial details.",
     "ambiguous_or_unclear",
     {"invoice": "mismatch_unclear", "report": "mismatch_unclear"}),
    ("Forward it internally to the public website team.", "public_sharing",
     {"report": "unsafe_external_action", "presentation": "compatible_sensitive"}),
    ("Summarize for LinkedIn but keep it confidential.", "ambiguous_or_unclear",
     {"resume": "mismatch_unclear", "report": "mismatch_unclear"}),
    ("Publish this privately.", "ambiguous_or_unclear",
     {"report": "mismatch_unclear", "invoice": "mismatch_unclear"}),
    ("Delete the duplicate but preserve the original.", "delete_permanent",
     {"invoice": "unsafe_external_action", "report": "unsafe_external_action",
      "letter": "unsafe_external_action"}),
]
for p, intent, mapping in CONFLICTS:
    diff = "adversarial_but_realistic" if "but" in p else "very_hard"
    _add_for_pairs(p, mapping, "conflicting_instruction", intent,
                   diff=diff, review="borderline_reviewed",
                   note="Conflicting instruction; rubric resolution.")


def _img(cls: str, idx: int) -> str:
    return str(IMAGES_ROOT / cls / f"{cls}_{idx:06d}.png")


def main(images_per_pair: int = 4) -> None:
    ensure_dirs()
    rng = random.Random(GLOBAL_SEED)
    seen: set[tuple[str, str, str]] = set()
    for tup in TUPLES:
        prompt, cls, _intent, _label, ct, _diff, _review, _note = tup
        key = (prompt, cls, ct)
        if key in seen:
            raise ValueError(f"duplicate rubric tuple: {key}")
        seen.add(key)
    log.info(f"Total unique rubric tuples: {len(TUPLES)}")

    rows: list[dict] = []
    per_class_offset: dict[str, int] = {c: 0 for c in DOCUMENT_CLASSES}
    for prompt, cls, intent, label, ct, diff, review, note in TUPLES:
        for k in range(images_per_pair):
            idx = BASE_IDX + per_class_offset[cls]
            per_class_offset[cls] += 1
            if idx > MAX_PER_CLASS:
                idx = BASE_IDX + (per_class_offset[cls] - 1) % (MAX_PER_CLASS - BASE_IDX + 1)
            image_path = _img(cls, idx)
            if not Path(image_path).exists():
                idx = BASE_IDX + (k * 17) % (MAX_PER_CLASS - BASE_IDX + 1)
                image_path = _img(cls, idx)
                if not Path(image_path).exists():
                    raise FileNotFoundError(f"missing image for {cls}: {image_path}")
            rows.append({
                "prompt_text": prompt,
                "image_path": image_path,
                "true_request_intent_manual": intent,
                "true_document_class": cls,
                "human_triage_label": label,
                "rationale": note,
                "challenge_type": ct,
                "difficulty_level": diff,
                "label_review_status": review,
                "label_review_note": note,
            })
    log.info(f"Materialised {len(rows)} rows ({images_per_pair} images per rubric tuple).")

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=GLOBAL_SEED).reset_index(drop=True)

    out_csv = PROCESSED_DIR / "hard_fusion_dataset.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"Wrote {out_csv}")

    train, temp = train_test_split(
        df, test_size=0.30, random_state=GLOBAL_SEED, stratify=df["human_triage_label"]
    )
    val, test = train_test_split(
        temp, test_size=0.5, random_state=GLOBAL_SEED,
        stratify=temp["human_triage_label"],
    )
    parts = {"train": train.reset_index(drop=True),
             "val": val.reset_index(drop=True),
             "test": test.reset_index(drop=True)}
    for name, p in parts.items():
        path = SPLITS_DIR / f"hard_fusion_{name}.csv"
        p.to_csv(path, index=False)
        log.info(f"Wrote {path} (n={len(p)})")

    summary = {
        "total": int(len(df)),
        "unique_rubric_tuples": len(TUPLES),
        "images_per_pair": int(images_per_pair),
        "challenge_types": df["challenge_type"].value_counts().to_dict(),
        "per_human_triage_label": df["human_triage_label"].value_counts().to_dict(),
        "per_document_class": df["true_document_class"].value_counts().to_dict(),
        "difficulty_distribution": df["difficulty_level"].value_counts().to_dict(),
        "label_review_status_distribution": df["label_review_status"].value_counts().to_dict(),
        "splits": {n: int(len(p)) for n, p in parts.items()},
        "labels_origin": (
            "manual / rubric-aligned; never produced by classify_triage or by any "
            "model output. Disjoint image-index range from the 390-row evaluation-only "
            "hard challenge set (BASE_IDX=700)."
        ),
        "intended_use": "training and validation of fusion only; the test split is held out.",
    }
    Path(REPORTS_DIR / "hard_fusion_data_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-per-pair", type=int, default=4)
    args = parser.parse_args()
    main(images_per_pair=args.images_per_pair)


if __name__ == "__main__":
    cli()
