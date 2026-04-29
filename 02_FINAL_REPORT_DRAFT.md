# AttachIQ — Final Implementation Report

**Author:** Mohammad Abu Jafar
**Course context:** MAAI7103 — Deep Learning: From Foundations to Application

## Abstract

AttachIQ is a fully local PyTorch system for multimodal request-attachment
triage. Given an optional user prompt and an optional document image, it
classifies the (request, document) situation into one of four operational
classes — `compatible_low_risk` (ALLOW), `compatible_sensitive` (REVIEW),
`mismatch_unclear` (REVIEW), `unsafe_external_action` (BLOCK) — using a
fine-tuned DistilBERT request classifier (10 classes), a from-scratch
CNN document classifier (8 RVL-CDIP-style classes, 2,586,568 parameters),
and a small fusion MLP that combines their probability outputs together
with explicit uncertainty signals (entropy and top-1−top-2 margin) into a
26-dimensional feature vector. The system is evaluated on a
policy-derived held-out test split (n = 1000), on a 390-example
human-reviewed hard challenge set, and on a held-out hard rubric-labelled
fusion test split (n = 354). On the standard test the learned fusion and
the rule-table baseline are effectively tied (macro F1 0.9638 vs 0.9664).
On the hard challenge set the learned fusion reaches macro F1 0.8207
versus a rule-table baseline of 0.7640. On the hard rubric fusion test
fusion reaches 0.8963 versus 0.8628. All metrics are reproducible from a
single command sequence; no OCR, no LLM API, no AutoML, and no
pretrained vision backbone are used at any stage.

## 1. Problem

Modern AI assistants routinely receive a user prompt together with an
uploaded document image. The right operational handling — allow,
review, or block — depends jointly on **what is being asked** and
**what is attached**: the same prompt ("summarise this") is low-risk
over a slide but sensitive over an invoice; an "extract the total"
request over a presentation is a mismatch; "post this publicly" over a
resume is unsafe. A text-only or image-only system cannot resolve
this; the decision lives in the interaction. AttachIQ asks one
empirical question:

> *Does a learned multimodal fusion add measurable value beyond a
> hand-coded rule table on this triage task?*

## 2. Methodology

AttachIQ is a fully local PyTorch pipeline with **three trained models**:

1. A fine-tuned `distilbert-base-uncased` text request classifier
   (10 classes: summarization, information_extraction,
   financial_extraction, document_classification, internal_sharing,
   public_sharing, delete_permanent, archive_retain,
   ambiguous_or_unclear, redaction_or_safe_transform), all transformer
   layers unfrozen, dropout + linear head.
2. A CNN trained **from scratch** for document-image classification
   (8 RVL-CDIP-style classes; selected architecture: five
   Conv-BN-ReLU-MaxPool blocks at channels 64/128/256/384/384 +
   AdaptiveAvgPool + Dropout + Linear; 2,586,568 parameters).
3. A **fusion MLP** trained from scratch on a **26-dimensional**
   feature vector (10 request probabilities + 8 document probabilities
   + text/image confidence + has_text/has_image flags + text/image
   entropy + text/image top-1−top-2 margin) → 4 triage classes.
   Selected architecture: 26 → 128 → 64 → 4 (~12K parameters).

All three models are loaded locally at inference time; there are no
LLM APIs, no OCR, no pretrained vision backbone, no AutoML, and no
external services at runtime. Pydantic v2 contracts (`InferenceRequest`,
`InferenceResponse`) enforce structured input and JSON output. A single
canonical `TriagePipeline` is shared by the CLI, the Streamlit demo,
and the tests.

## 3. Datasets

### 3.1 Text request prompts (synthetic-but-realistic)

7,200 prompts across 10 request classes (720 per class), built from
controlled templates, paraphrase variants ("Hey, …please…", "…ASAP."),
messy short forms ("can u summarize this quickly", "fwd to manager",
"nuke this attachment"), and light typo variants. Fixed seed (42),
80/10/10 stratified split: 5,760 / 720 / 720. Outputs in
`data/processed/text_prompts.csv` and `data/splits/text_*.csv`.
Disclosure: the dataset is templated, so the text classifier
achieves near-saturated unimodal accuracy on the test split; the
substantive comparison happens at the multimodal fusion level.

### 3.2 Document images (real RVL-CDIP)

Real RVL-CDIP via local parquet snapshot, no dataset scripts. The
builder tries the local FULL chainyo-format parquet snapshot first,
then a local SMALL ImageFolder subset, then an HF parquet mirror. The
final image set is 8 classes × 1,000 images = 8,000 images, sampled
deterministically with seed 42, stratified 80/10/10 = 6,400 / 800 /
800. The original `aharley/rvl_cdip` script-based loader is **not**
used because `datasets >= 4.0` removed support for dataset scripts.

### 3.3 Fusion features (built from the trained models)

Two complementary supervision sources are used for the fusion MLP:

1. **Standard fusion features** — 10,000 (prompt, document) pairs,
   ~30 % text-only, ~30 % image-only, ~40 % text+image, drawn
   deterministically from the text dataset and the image manifest.
   For each pair the trained text classifier and the deep CNN
   produce real probability vectors; missing modalities are zeroed
   and the **26-dim** feature vector is assembled (10 request probs
   + 8 document probs + text/image confidence + has_text/has_image
   flags + text/image entropy + text/image top-1−top-2 margin).
   Triage labels come from `attachiq.triage.policy.classify_triage`
   applied to the **true** underlying request and document classes,
   so labels reflect ground truth rather than noisy predictions.
   Stratified 80/10/10 → 8,000 / 1,000 / 1,000.
2. **Hard rubric fusion** — 2,360 prompt + image pairs across ten
   challenge categories, with rubric-aligned triage labels assigned
   by hand (never by `classify_triage` and never by any model
   output). Image indices begin at `BASE_IDX = 700` to keep this
   dataset disjoint from the 390-row evaluation-only hard challenge
   set (which uses `BASE_IDX = 300`). Stratified 70/15/15 →
   1,652 / 354 / 354.

The fusion MLP is trained on the union of both training splits and
selected on the union of both validation splits. The held-out
`hard_fusion_test` split is used as a held-out rubric evaluation in
addition to the standard test and the 390-row hard challenge.

## 4. Architecture

End-to-end pipeline:

1. Validate `InferenceRequest` (Pydantic v2).
2. Resolve `input_mode` from the inputs.
3. Run the text branch if `prompt_text` is present (probability vector,
   argmax → request class, max prob → text confidence, plus entropy
   and top-1−top-2 margin).
4. Run the image branch if `image_path` is present (probability vector,
   argmax → document class, max prob → image confidence, plus entropy
   and top-1−top-2 margin).
5. Build the **exact 26-dim** fusion feature vector
   (10 request probs + 8 document probs + text_conf + image_conf +
   has_text + has_image + text_entropy + image_entropy +
   text_margin + image_margin), zeroing the missing modality.
6. Run the fusion MLP → triage class + confidence.
7. Map triage → decision (`ALLOW` / `REVIEW` / `BLOCK`).
8. Generate a deterministic-template explanation.
9. Measure `inference_time_ms` and return the `InferenceResponse`.

Model checkpoints live under `models/text/`, `models/image/`,
`models/fusion/` with `label_map.json` and confusion matrices in
`reports/confusion_matrices/`.

## 5. Implementation

- Python 3.13, `uv` project, pinned dependencies via `pyproject.toml`
  and `uv.lock`.
- `justfile` with `install`, `data`, `train`, `train-text`,
  `train-image`, `train-fusion`, `evaluate`, `evaluate-image`, `demo`,
  `cli`, `test`, `clean`.
- PyTorch on Apple MPS (auto-falls back to CUDA / CPU).
- Loguru everywhere, no `print` calls in `src/`.
- Tests: 37 passing — schemas, policy, fusion-feature construction,
  metric utilities, end-to-end inference with mock models.

## 6. Evaluation

Per-model held-out test metrics on real RVL-CDIP image data:

| Model        | Accuracy | Macro F1 | Params  |
|--------------|----------|----------|---------|
| Text classifier (10 classes) | 1.0000 | 1.0000 | 66.4 M |
| Image CNN (deep, 5 blocks @ 64/128/256/384/384) | 0.7888 | 0.7864 | 2,586,568 |
| Fusion MLP (26 → 128 → 64 → 4) | 0.9580 | 0.9638 | 11,972 |

(Text near-saturation reflects the templated text dataset, not a real
cap on the model.)

## 7. Baselines

Three baselines are evaluated alongside the learned fusion:

1. **Text-only** — `classify_triage(predicted_request, None)`.
2. **Image-only** — `classify_triage(None, predicted_document)`.
3. **Rule-table** — `classify_triage(predicted_request, predicted_document)`
   (argmax of the trained models' probability outputs).

The text-only and image-only baselines drop the missing modality
entirely before applying the rule. The rule-table baseline is the
labelling function applied to noisy predictions, which is exactly the
comparison fusion has to beat.

## 8. Results

### Standard test (n = 1000, policy-derived labels)

| Method               | Accuracy | Macro F1 |
|----------------------|---:|---:|
| text-only baseline   | 0.5330 | 0.5424 |
| image-only baseline  | 0.6020 | 0.4497 |
| **rule-table baseline** | **0.9610** | **0.9664** |
| learned fusion       | 0.9580 | 0.9638 |

- Best method by macro F1: rule-table baseline (0.9664), fusion
  0.0026 behind. The two are effectively tied within noise; the rule
  is the labelling function for this distribution, so as the unimodal
  models become accurate the two methods converge.
- Fusion outperforms the unimodal baselines by +0.421 macro F1 over
  text-only and +0.514 over image-only.

### 390-row hard challenge set (human-reviewed, evaluation-only)

| Method               | Accuracy | Macro F1 |
|----------------------|---:|---:|
| text-only baseline   | 0.5641 | 0.5089 |
| image-only baseline  | 0.3795 | 0.2424 |
| rule-table baseline  | 0.7718 | 0.7640 |
| **learned fusion**   | **0.8256** | **0.8207** |

- Fusion clearly beats the rule on this distribution (Δ = +0.057
  macro F1).
- All four explicitly probed weak categories — vague_prompt,
  ambiguous_action, archive_vs_delete, redaction_modifier — are at
  non-zero macro F1 under fusion.

### Hard rubric fusion test (n = 354, held out)

| Method               | Accuracy | Macro F1 |
|----------------------|---:|---:|
| text-only baseline   | 0.5593 | 0.4876 |
| image-only baseline  | 0.4237 | 0.2851 |
| rule-table baseline  | 0.8729 | 0.8628 |
| **learned fusion**   | **0.9040** | **0.8963** |

- Fusion clearly beats the rule (Δ = +0.034 macro F1).
- Both fusion and rule strongly outperform unimodal baselines.

### Latency

Mean 5–6 ms per pipeline call after warm-up, p95 ≈ 9–12 ms over 20
probe runs (CPU / MPS), well under the 2-second target. First-call
inference includes ~2 s of model loading.

## 9. Insights

1. The fusion network is small (11,972 parameters) and combines the
   unimodal probability vectors with explicit uncertainty signals
   (entropy and top-1−top-2 margin) into a 26-dim feature vector. On
   the policy-derived standard test it effectively ties the rule
   baseline (the rule is the labelling function for that
   distribution); on rubric-labelled evaluations it cleanly
   outperforms the rule (+0.057 macro F1 on the 390-row hard
   challenge, +0.034 on the held-out hard rubric fusion test).
2. The unimodal-rule baselines collapse where expected: without the
   document, the system cannot discriminate `compatible_low_risk`
   from `compatible_sensitive`; without the prompt, it cannot
   discriminate `summarization` from `public_sharing` or
   `delete_permanent`.
3. The dominant engineering effort is data and contract plumbing
   rather than model design: maintaining the 26-dim feature contract
   end-to-end, assigning standard fusion labels from ground truth
   rather than predictions, hand-curating a 2,360-row rubric-aligned
   fusion supervision set with disjoint image indices from the
   evaluation-only hard challenge, and migrating the image dataset
   from the deprecated `aharley/rvl_cdip` script-based loader to a
   parquet snapshot of `chainyo/rvl-cdip`.

## 10. Limitations

- The text dataset is templated; a follow-up could seed it from real
  user logs to stress the text classifier and the fusion together.
- The selected scratch image CNN (2,586,568 parameters) reaches macro
  F1 0.7864 on the held-out test split. A larger from-scratch
  backbone could lift image macro F1 further and therefore lift
  fusion's ceiling, but is excluded from this scope to keep the
  project local, fast, and well under the scratch-model parameter cap.
- The hard rubric fusion training set shares prompt patterns with
  the 390-row evaluation set (different image instances). The
  held-out `hard_fusion_test` split (n = 354) is the cleaner held-out
  evaluation for the rubric distribution.
- The 390-row hard challenge set and the 2,360-row hard rubric fusion
  set were labelled by a single reviewer against the rubric. No
  inter-rater agreement check.
- AttachIQ classifies layout and prompt intent. It does **not** read
  document content (no OCR) and does not understand legal
  confidentiality. The output is a triage signal, not a compliance
  decision.
- No real institutional deployment is included.

## 11. Reproducibility

```bash
uv sync --extra dev          # or: just install
just data                    # text + image (real RVL-CDIP) + standard + hard rubric fusion
just train                   # image -> text -> fusion
just evaluate                # writes reports/metrics_summary.json + evaluation_summary.md
just test                    # 37/37
just demo                    # Streamlit at http://localhost:8501
```

Fixed seeds (`GLOBAL_SEED = 42`), pinned deps via `uv.lock`, all
metrics under `reports/`, all checkpoints under `models/`.
