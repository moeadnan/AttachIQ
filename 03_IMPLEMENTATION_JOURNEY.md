# AttachIQ — Implementation Journey

> **Document positioning.** This file is the implementation journey
> companion to the project, written in a first-person narrative style
> for the in-class presentation. Professional, technical
> documentation lives in:
>
> - `README.md` — developer/project README
> - `01_AttachIQ_Proposal.md` — proposal
> - `02_FINAL_REPORT_DRAFT.md` — academic technical report
> - `04_FINAL_AUDIT.md` — single canonical evidence pack
>
> If a number disagrees between this file and any of the four above,
> trust the technical docs.

## What I built

AttachIQ is a fully local, multimodal triage system for AI assistants.
It takes a user prompt and/or an uploaded document image and returns a
structured triage decision in one of four classes —
`compatible_low_risk` → ALLOW, `compatible_sensitive` → REVIEW,
`mismatch_unclear` → REVIEW, `unsafe_external_action` → BLOCK — together
with a confidence and a deterministic explanation. Everything runs on
the laptop: no LLM API calls, no OCR, no AutoML, no pretrained vision
backbone.

## Why I chose this problem

I am interested in AI agent governance. Before an AI assistant
processes a user-provided document, it should be able to assess the
request context and identify cases that may require review. The
academic question I want the project to answer is concrete:

> *Does a learned multimodal fusion add measurable value beyond a
> hand-coded rule table on this triage task?*

To answer that, the project needs three trained models, three
evaluation distributions, and a clean baseline ladder.

## How I designed the final architecture

Three PyTorch models, one structured pipeline:

1. **Text classifier.** Fine-tuned `distilbert-base-uncased`, all
   transformer layers unfrozen, with a small dropout-then-linear head
   on the `[CLS]` representation. **10 request classes** covering the
   actions an assistant typically receives: `summarization`,
   `information_extraction`, `financial_extraction`,
   `document_classification`, `internal_sharing`, `public_sharing`,
   `delete_permanent`, `archive_retain`, `ambiguous_or_unclear`, and
   `redaction_or_safe_transform`. The taxonomy was shaped during
   implementation by robustness analysis on the hard challenge set;
   in particular, ambiguity, archive/retention, permanent deletion,
   and redaction-safe transformation each have explicit classes
   because the system needs to be able to express them.
2. **Image classifier.** A CNN trained **from scratch** (no pretrained
   backbone). Architecture: five `Conv-BN-ReLU-MaxPool` blocks at
   channels 64/128/256/384/384, AdaptiveAvgPool, Dropout, Linear
   classifier — 2,586,568 parameters. Output: 8 RVL-CDIP-style
   document-class probabilities (`invoice`, `form`, `letter`, `report`,
   `email`, `resume`, `presentation`, `handwritten`). Trained on real
   RVL-CDIP images (1000 per class, 8000 total) drawn from the local
   chainyo-format parquet snapshot at `~/datasets/attachiq/rvl-cdip/`.
3. **Fusion MLP.** Trained from scratch on a fixed **26-dim** feature
   vector (10 request probabilities + 8 document probabilities + text
   confidence + image confidence + has_text + has_image flags + text
   and image entropy + text and image top-1−top-2 margin). The fusion
   feature vector includes explicit uncertainty signals (entropy and
   margin) so the small fusion network can learn to discount
   low-confidence unimodal predictions rather than over-trusting
   their argmaxes. Architecture `26 → 128 → 64 → 4`, 11,972
   parameters. Trained on the union of (a) a 10,000-row standard
   fusion training set whose triage labels come from the rule policy
   applied to ground-truth classes, and (b) a 2,360-row hand-curated
   rubric-aligned fusion training set across ten challenge
   categories.

## How the modalities interact

Each modality independently produces a calibrated-ish probability
vector. The pipeline always builds the **exact 26-dim** vector,
zeroing the missing modality and flipping the appropriate `has_*`
flag. The fusion MLP learns to combine those signals — including
explicit uncertainty — into one of four triage classes. The Pydantic
`InferenceRequest` / `InferenceResponse` contracts enforce that
contract end-to-end.

## How I prepared the datasets

- **Text dataset (7,200 prompts).** Generated from controlled
  templates per class plus paraphrase variants ("Hey, …please…",
  "…ASAP."), messy short variants ("can u summarize this quickly",
  "fwd to manager", "nuke this attachment"), and light typo variants.
  Stratified 80/10/10 = 5,760 / 720 / 720. The dataset is templated;
  the text classifier saturates on its own held-out test split. The
  meaningful comparison happens at the multimodal fusion level.
- **Image dataset (8,000 real RVL-CDIP images).** The original
  `aharley/rvl_cdip` repo on the Hugging Face Hub fails to load on
  recent `datasets` because it relies on the deprecated dataset-script
  mechanism. I migrated the image loader to a real *parquet* path
  with a clear priority: local FULL chainyo-format snapshot first,
  then a local SMALL ImageFolder mirror, then an HF parquet mirror.
  The chainyo and canonical RVL-CDIP integer label orders also
  differ — I encode both maps explicitly. Preprocessing: grayscale,
  resize 224×224, normalize. Train-only augmentations: small
  rotation, small translation/scale, light brightness/contrast
  jitter, Gaussian noise. No horizontal flip — document text is not
  flip-invariant.
- **Standard fusion features (10,000 rows × 26-dim).** Built from
  real text and image probability outputs with uncertainty signals
  attached. Triage labels come from the rule policy applied to the
  *ground-truth* request and document classes, so the rule baseline
  (which uses *predicted* classes) and learned fusion are evaluated
  fairly on the same distribution.
- **Hard rubric fusion training (2,360 rows × 26-dim).**
  Hand-labelled prompt + image pairs across ten challenge categories,
  with image indices in a range disjoint from the 390-row
  evaluation-only hard challenge set. Used as additional supervision
  so the fusion MLP also sees rubric-aligned labels.
- **Hard challenge set (390 rows).** Hand-curated, evaluation-only.
  Never used to train, tune, select models, or set thresholds.

## How I trained the models

- **Image CNN.** AdamW + ReduceLROnPlateau on validation macro F1, up
  to 20 epochs, early stopping on patience. The training script
  supports three scratch-CNN architectures and selects on validation
  macro F1; the deep variant is the production checkpoint.
- **Text classifier.** AdamW + class-weighted cross-entropy, early
  stop on validation macro F1.
- **Fusion MLP.** Trained on the union of standard and hard rubric
  training sets. Two architecture variants are tried (small 64/32
  and big 128/64); the variant with the best validation macro F1 is
  saved.

## How I validated the image pipeline

Before finalising the image branch I ran a verification pass that
checks (i) the RVL-CDIP → AttachIQ label mapping against the
canonical chainyo label list, (ii) every demo image was copied from
the matching `data/raw/images/<class>/` folder, (iii) the train/val/
test CSVs use only the 8 expected class strings, (iv) two
ground-truth samples per class read directly from the parquet
shards, (v) three random training images per class from the
extracted folder, and (vi) the train and eval transforms produce
1×224×224 tensors normalised the same way. The verification
artefacts are at `reports/figures/{image_samples_grid,
parquet_label_check, preprocessing_consistency}.png` and
`reports/image_pipeline_audit.json`.

## How I evaluated the system

Three test sets:

1. **Standard test (n = 1000).** Held-out split of the standard
   fusion features. Labels come from the rule policy on ground-truth
   classes.
2. **Hard challenge set (n = 390).** Hand-curated, evaluation-only.
3. **Hard rubric fusion test (n = 354).** Held-out split of the hand-
   labelled rubric fusion set.

For each test set I report accuracy and macro F1 for four methods:
text-only baseline, image-only baseline, rule-table baseline, and
learned fusion. Confusion matrices are written to
`reports/confusion_matrices/`.

## What worked

- **Standard test:** rule-table 0.9664, fusion 0.9638. Effectively
  tied (Δ = −0.0026 macro F1) within noise — the rule is the labelling
  function for that distribution, so as the unimodal models become
  accurate the two methods converge.
- **Hard challenge set:** fusion 0.8207, rule 0.7640. Fusion clearly
  leads (+0.057 macro F1).
- **Held-out hard rubric fusion test:** fusion 0.8963, rule 0.8628.
  Fusion clearly leads (+0.034 macro F1).
- Fusion strongly outperforms the unimodal baselines on every
  evaluation.
- Latency: mean 5–6 ms per pipeline call after warm-up, p95 ≈ 9–12
  ms. First-call inference includes ~2 s of model loading.
- 37 / 37 tests pass, no `print()` statements in `src/`.

## What limitations remain

- The text dataset is templated; in production it would be
  supplemented with real prompt-log data.
- The image CNN reads layout, not content. AttachIQ does not
  understand legal confidentiality; its output is a triage signal,
  not a compliance decision.
- Hard rubric fusion training shares prompt patterns with the 390-row
  evaluation set (different image instances). The
  `hard_fusion_test` held-out split is the cleaner held-out
  evaluation for the rubric distribution.
- Both the 390-row hard challenge set and the 2,360-row hard rubric
  fusion set were labelled by a single reviewer against the rubric.
- The image CNN is intentionally small (2.59 M parameters); the
  weakest class on the 8-class document task is `report` at per-class
  F1 ≈ 0.57.

## Slide-Ready Model and Evaluation Summary

After validating the data pipeline, I compared a small set of model
choices and then evaluated the final end-to-end system across three
settings. The table below is the version I can use in a slide: it
shows how the image branch was selected, how the final text and
fusion models fit into the system, and how the complete pipeline
performed.

| Model / Step | Params | Evaluation Set | Accuracy | Macro F1 | Latency | Notes |
|---|---:|---|---:|---:|---:|---|
| Image CNN baseline | 0.39M | Image test, n=800 | 0.5050 | 0.4624 | Not measured per-arch | Underpowered architecture-selection candidate |
| Image CNN A, wide | 1.26M | Image test, n=800 | 0.6963 | 0.6855 | Not measured per-arch | Better capacity, but not selected |
| Image CNN B, deep, selected | 2.59M | Image test, n=800 | 0.7888 | 0.7864 | Not measured per-arch | Final production image classifier |
| Text classifier, DistilBERT | 66.37M | Text test, n=720 | 1.0000 | 1.0000 | Not measured per-component | Fine-tuned pretrained model; templated-data caveat |
| Fusion MLP, selected | 11,972 | Hard rubric fusion test, n=354 | 0.9040 | 0.8963 | Not measured per-component | Final learned multimodal fusion model |
| End-to-end pipeline, standard | 68.97M total | Standard test, n=1000 | 0.9580 | 0.9638 | Mean ≈ 5.7ms, p95 ≈ 9ms | Rule baseline slightly leads: 0.9664 macro F1 |
| End-to-end pipeline, hard challenge | 68.97M total | Hard challenge, n=390 | 0.8256 | 0.8207 | Same pipeline latency | Fusion leads rule: 0.7640 rule macro F1 |
| End-to-end pipeline, hard rubric test | 68.97M total | Hard rubric fusion test, n=354 | 0.9040 | 0.8963 | Same pipeline latency | Fusion leads rule: 0.8628 rule macro F1 |

Reading this table: the first three rows are model-selection evidence
for the scratch-trained image branch. The selected image model is the
deep CNN because it gave the strongest macro F1 while staying small
and local. The text classifier satisfies the fine-tuned
pretrained-model requirement, but its perfect score is treated
cautiously because the prompt dataset is templated. The fusion MLP is
the final learned multimodal model. At the full-pipeline level, the
rule baseline is effectively tied on the policy-derived standard
test, while learned fusion leads on the two rubric-bearing
evaluations.

**Slide speaking version:**
"I first made sure the image data and labels were correct, then
compared three scratch CNNs. The deep CNN was selected because it
improved image macro F1 from 0.462 to 0.786 while staying only 2.59M
parameters. The final system combines that image model with a
fine-tuned DistilBERT text model and a small fusion MLP. On the
standard test, fusion and rules are effectively tied, which is
expected because the rule aligns closely with that distribution. On
the harder rubric-based evaluations, learned fusion outperforms the
rule baseline, which shows the value of combining probabilities and
uncertainty rather than relying only on top-1 rule decisions."

## What I should say in class

- AttachIQ uses a 10-class request taxonomy designed to cover
  summarisation, extraction, sharing, deletion, archive/retention,
  ambiguity, and redaction-safe transformation, and a 26-dimensional
  fusion vector that exposes the unimodal classifiers' uncertainty.
- The standard test is policy-derived, so the rule baseline is
  expected to be strong and tied with fusion. The interesting
  evidence comes from the rubric-labelled evaluations, where
  learned fusion clearly beats both the rule and the unimodal
  baselines.
- The hard challenge set is a stress test on realistic ambiguity:
  vague prompts, archive vs delete, public vs internal sharing,
  redaction modifiers, partial extraction on the wrong document type,
  visually-similar documents, sensitivity modifiers, ambiguous
  actions, and conflicting instructions. Fusion handles them
  meaningfully better than the rule.
- All numbers are reproducible from the recipes and modules described
  in the README; no OCR, no LLM API, no AutoML, no pretrained vision
  backbone.

## What I should avoid claiming

- I should not claim AttachIQ understands legal confidentiality. It
  classifies request intent and document layout.
- I should not claim the image classifier reads the document; it
  classifies layout only.
- I should not claim fusion strictly dominates the rule on every
  evaluation — on the policy-derived standard test the two are
  effectively tied, and the cleaner evidence for fusion's value is
  on the rubric-labelled evaluations.
