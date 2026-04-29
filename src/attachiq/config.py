"""Project-wide configuration: paths, label vocabularies, hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
DEMO_DIR = DATA_DIR / "demo_samples"

MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_MODEL_DIR = MODELS_DIR / "image"
TEXT_MODEL_DIR = MODELS_DIR / "text"
FUSION_MODEL_DIR = MODELS_DIR / "fusion"

REPORTS_DIR = PROJECT_ROOT / "reports"
CONFUSION_DIR = REPORTS_DIR / "confusion_matrices"
FIGURES_DIR = REPORTS_DIR / "figures"

# 10 request classes covering summarization, extraction, sharing,
# permanent deletion, archive/retention, ambiguity, and redaction-safe
# transformation.
REQUEST_CLASSES: list[str] = [
    "summarization",
    "information_extraction",
    "financial_extraction",
    "document_classification",
    "internal_sharing",
    "public_sharing",
    "delete_permanent",
    "archive_retain",
    "ambiguous_or_unclear",
    "redaction_or_safe_transform",
]

# 8 document classes drawn from the RVL-CDIP taxonomy.
DOCUMENT_CLASSES: list[str] = [
    "invoice",
    "form",
    "letter",
    "report",
    "email",
    "resume",
    "presentation",
    "handwritten",
]

TRIAGE_CLASSES: list[str] = [
    "compatible_low_risk",
    "compatible_sensitive",
    "mismatch_unclear",
    "unsafe_external_action",
]

DECISION_MAP: dict[str, str] = {
    "compatible_low_risk": "ALLOW",
    "compatible_sensitive": "REVIEW",
    "mismatch_unclear": "REVIEW",
    "unsafe_external_action": "BLOCK",
}

NUM_REQUEST = len(REQUEST_CLASSES)
NUM_DOC = len(DOCUMENT_CLASSES)
NUM_TRIAGE = len(TRIAGE_CLASSES)
# 26-dim fusion vector: 10 request probabilities + 8 document
# probabilities + text_conf + image_conf + has_text + has_image
# + text_entropy + image_entropy + text_margin + image_margin.
FUSION_INPUT_DIM = NUM_REQUEST + NUM_DOC + 8  # 10 + 8 + 8 = 26


@dataclass(frozen=True)
class TextConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    dropout: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class ImageConfig:
    image_size: int = 224
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    seed: int = 42


@dataclass(frozen=True)
class FusionConfig:
    input_dim: int = FUSION_INPUT_DIM
    hidden_1: int = 128
    hidden_2: int = 64
    output_dim: int = NUM_TRIAGE
    dropout: float = 0.2
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42


TEXT_CFG = TextConfig()
IMAGE_CFG = ImageConfig()
FUSION_CFG = FusionConfig()

GLOBAL_SEED = 42


def ensure_dirs() -> None:
    """Create all expected project directories if missing."""
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        DEMO_DIR,
        TEXT_MODEL_DIR,
        IMAGE_MODEL_DIR,
        FUSION_MODEL_DIR,
        REPORTS_DIR,
        CONFUSION_DIR,
        FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
