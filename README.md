# AttachIQ

**Multimodal Request-Attachment Triage for AI Assistants**

AttachIQ takes a user prompt and/or an uploaded document image and
classifies the combined request-attachment situation into one of four
operational classes:

| Triage class            | UI decision |
|-------------------------|-------------|
| `compatible_low_risk`   | ALLOW       |
| `compatible_sensitive`  | REVIEW      |
| `mismatch_unclear`      | REVIEW      |
| `unsafe_external_action`| BLOCK       |

The system is a deep-learning project (MAAI7103) that runs **fully
locally**: no LLM API calls, no OCR, no AutoML, no pretrained vision
backbone, no external services at runtime.

## Architecture

Three PyTorch models, one structured pipeline.

1. **Text classifier** — fine-tuned `distilbert-base-uncased`,
   **10 request classes**: `summarization`, `information_extraction`,
   `financial_extraction`, `document_classification`, `internal_sharing`,
   `public_sharing`, `delete_permanent`, `archive_retain`,
   `ambiguous_or_unclear`, `redaction_or_safe_transform`.
2. **Image classifier** — CNN trained **from scratch** (no pretrained
   backbone): five `Conv-BN-ReLU-MaxPool` blocks at channels
   64/128/256/384/384 + AdaptiveAvgPool + Dropout + Linear,
   2,586,568 parameters; **8 document classes**: `invoice`, `form`,
   `letter`, `report`, `email`, `resume`, `presentation`, `handwritten`.
3. **Fusion MLP** — trained from scratch on a **26-dim** feature vector
   (10 request probabilities + 8 document probabilities + text and
   image confidence + has_text and has_image flags + text and image
   entropy + text and image top-1−top-2 margin). Architecture
   `26 → 128 → 64 → 4`, 11,972 parameters, → 4 triage classes.

All three models are loaded locally at inference time. Pydantic v2
contracts (`InferenceRequest`, `InferenceResponse`) enforce structured
input and JSON output. A single canonical `TriagePipeline` is shared by
the CLI, the Streamlit demo, and the tests.

## Setup (Python 3.13 + uv)

```bash
uv sync --extra dev
```

## Just recipes

| Recipe                | Purpose                                                |
|-----------------------|--------------------------------------------------------|
| `just install`        | `uv sync --extra dev`                                  |
| `just data`           | Build text + image + standard fusion + hard rubric fusion datasets |
| `just train`          | Train the image CNN, the text classifier, and the fusion MLP |
| `just train-text`     | Fine-tune DistilBERT (10 classes)                      |
| `just train-image`    | Train the deep scratch CNN (8 classes)                 |
| `just train-fusion`   | Train the fusion MLP and select the variant by val macro F1 |
| `just evaluate`       | Evaluate on standard test, hard challenge, hard rubric fusion test |
| `just evaluate-image` | Evaluate the image CNN on its held-out test split      |
| `just demo`           | Launch Streamlit demo                                  |
| `just cli ARGS`       | Run inference CLI                                      |
| `just test`           | Run pytest                                             |
| `just clean`          | Remove caches                                          |

## Datasets

### Text request prompts

`just data` (or `python -m attachiq.data.build_text_dataset`) generates
**7,200** labelled prompts across the 10 request classes (720 per
class) using controlled templates plus paraphrases plus messy realistic
phrasings plus light typo variants. Stratified 80/10/10 split: 5,760 /
720 / 720. Outputs:

- `data/processed/text_prompts.csv`
- `data/splits/text_{train,val,test}.csv`
- `reports/text_data_summary.json`

The text dataset is templated; the text classifier saturates on its
own held-out test split. The substantive comparison happens at the
multimodal fusion level.

### Document images (real RVL-CDIP)

`python -m attachiq.data.build_image_dataset` tries real RVL-CDIP
sources in this priority:

1. **Local FULL chainyo-format snapshot** at
   `~/datasets/attachiq/rvl-cdip/` (parquet, 16 classes). The builder
   reads only the parquet shards needed for the 8 mapped classes and
   samples deterministically up to `--per-class N` images per class
   (default 1000 → 8000 images total, split 80/10/10 = 6400 / 800 / 800).
2. **Local SMALL real RVL-CDIP subset** in ImageFolder layout at
   `~/datasets/attachiq/rvl_cdip-small-200/`.
3. **HF parquet mirror**: `vaclavpechtor/rvl_cdip-small-200`.

The original `aharley/rvl_cdip` repo on the Hugging Face Hub ships an
`rvl_cdip.py` dataset script. `datasets >= 4.0` removed support for
dataset scripts, so that loader fails. AttachIQ does **not** call it.

Preprocessing: grayscale, resize 224×224, normalise. Train-only
augmentations: small rotation (±4°), small translation (±3%), small
scale (0.95–1.05), brightness/contrast jitter (0.15), Gaussian noise
(σ = 0.015). **No horizontal flip** (document text is not flip-invariant).

### Standard fusion features

10,000 (prompt, document) pairs across the three input modes
(`text_only`, `image_only`, `text_plus_image`) at 26 dimensions per
row. Triage labels assigned by
`attachiq.triage.policy.classify_triage` over **ground-truth** request
and document classes, so the rule baseline (which uses *predicted*
classes) and the learned fusion are evaluated fairly on the same
distribution. Stratified 80/10/10 = 8,000 / 1,000 / 1,000.

### Hard rubric fusion training set

2,360 hand-labelled (prompt, image) pairs across 10 challenge
categories (vague_prompt, ambiguous_action, archive_vs_delete,
redaction_modifier, public_vs_internal_sharing, partial_extraction,
misleading_request_document_pair, visually_similar_document,
sensitivity_modifier, conflicting_instruction). Image indices begin at
`BASE_IDX = 700` so they are disjoint from the 390-row evaluation-only
hard challenge set (which uses `BASE_IDX = 300`). Stratified 70/15/15 =
1,652 / 354 / 354. Used as additional supervision for the fusion MLP.

### Hard challenge set (evaluation-only)

390 hand-curated (prompt, image) pairs across 10 challenge categories.
Triage labels assigned by hand against the rubric in the spec. Used
purely to evaluate robustness on realistic ambiguity; never used for
training, hyperparameter tuning, model selection, threshold tuning, or
early stopping.

## Training

```bash
just train
```

Each step writes its checkpoint and metrics:

- `models/text/`, `reports/text_metrics.json`
- `models/image/image_cnn.pt`, `models/image/arch.txt`,
  `reports/image_metrics.json`
- `models/fusion/fusion_mlp.pt`, `models/fusion/arch.json`,
  `reports/fusion_metrics.json`

Confusion matrices are written to `reports/confusion_matrices/`.

## Evaluation

```bash
just evaluate
```

Writes `reports/metrics_summary.json` and `reports/evaluation_summary.md`
covering all three test sets.

### Final metrics

| Evaluation                          | Method               | Accuracy | Macro F1 |
|-------------------------------------|----------------------|---:|---:|
| Standard test (n = 1000)            | text-only baseline   | 0.6020 | 0.5424 |
| Standard test (n = 1000)            | image-only baseline  | 0.6020 | 0.4497 |
| Standard test (n = 1000)            | rule-table baseline  | 0.9610 | **0.9664** |
| Standard test (n = 1000)            | learned fusion       | 0.9603 | 0.9638 |
| Hard challenge (n = 390)            | text-only baseline   | 0.5641 | 0.5089 |
| Hard challenge (n = 390)            | image-only baseline  | 0.3795 | 0.2424 |
| Hard challenge (n = 390)            | rule-table baseline  | 0.7821 | 0.7640 |
| Hard challenge (n = 390)            | **learned fusion**   | **0.8256** | **0.8207** |
| Hard rubric fusion test (n = 354)   | text-only baseline   | 0.5593 | 0.4876 |
| Hard rubric fusion test (n = 354)   | image-only baseline  | 0.4237 | 0.2851 |
| Hard rubric fusion test (n = 354)   | rule-table baseline  | 0.8729 | 0.8628 |
| Hard rubric fusion test (n = 354)   | **learned fusion**   | **0.9040** | **0.8963** |

- On the standard test the rule-table baseline narrowly leads fusion
  (Δ = −0.0026 macro F1, effectively tied). The rule is the labelling
  function for that distribution, so as the unimodal models become
  accurate the two methods converge.
- On the human-reviewed hard challenge set fusion clearly leads the
  rule (Δ = +0.0567 macro F1).
- On the held-out hard rubric fusion test split fusion clearly leads
  the rule (Δ = +0.0335 macro F1).
- Fusion strongly outperforms both unimodal baselines on every
  evaluation.

Total project parameters ≈ 69 M (DistilBERT 66.4 M, image CNN 2.59 M,
fusion MLP 12 K). All under the rubric caps.

Latency: mean 5–6 ms per pipeline call after warm-up, p95 ≈ 9–12 ms.

## Demo

```bash
just demo
```

> **Warm-up note.** The first inference inside a fresh Streamlit
> session includes ~2 seconds of model loading. Click "Assess Request"
> once with any sample before presenting so the live cases come back
> in <30 ms.

The Streamlit app at `src/attachiq/ui/streamlit_app.py` exposes a
prompt text area, an image uploader, and an "Assess Request" button.
Output: input mode, request type + confidence, document type +
confidence, compatibility label, decision badge, deterministic
explanation, inference time, and the full Pydantic JSON response.

## CLI

```bash
just cli --text "Summarize this slide" --image data/demo_samples/presentation_demo.png
```

## Limitations

- The text dataset is templated; in production it would be supplemented
  with real prompt-log data.
- The image classifier reads layout, not content. There is no OCR.
  AttachIQ does not understand legal confidentiality.
- Hard rubric fusion training and the 390-row hard challenge set were
  labelled by a single reviewer against the rubric; there is no
  inter-rater agreement check.
- Hard rubric fusion training shares prompt patterns with the 390-row
  evaluation set (different image instances). The held-out
  `hard_fusion_test` split (n = 354) is the cleaner held-out evaluation
  for the rubric distribution.
- The image CNN is intentionally small (2.59 M parameters); the
  weakest class on the 8-class document task is `report` at per-class
  F1 ≈ 0.57.

## Reproducibility

- Python 3.13.
- All seeds fixed (`GLOBAL_SEED = 42`).
- Pinned dependencies via `uv.lock`.
- All metrics under `reports/`.
- All checkpoints under `models/`.
- Tests: 37 / 37 passing.
