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

---

## Table of Contents

1. [What I built](#1-what-i-built)
2. [Why I chose this problem](#2-why-i-chose-this-problem)
3. [How I designed the final architecture](#3-how-i-designed-the-final-architecture)
   - 3.1 [Text classifier (DistilBERT, fine-tuned)](#31-text-classifier-distilbert-fine-tuned)
   - 3.2 [Image classifier (CNN from scratch)](#32-image-classifier-cnn-from-scratch)
   - 3.3 [Fusion MLP (trained from scratch)](#33-fusion-mlp-trained-from-scratch)
4. [How the modalities interact](#4-how-the-modalities-interact)
5. [How I prepared the datasets](#5-how-i-prepared-the-datasets)
   - 5.1 [Text dataset (7,200 prompts)](#51-text-dataset-7200-prompts)
   - 5.2 [Image dataset (8,000 real RVL-CDIP images)](#52-image-dataset-8000-real-rvl-cdip-images)
   - 5.3 [Standard fusion features (10,000 rows × 26-dim)](#53-standard-fusion-features-10000-rows--26-dim)
   - 5.4 [Hard rubric fusion training (2,360 rows × 26-dim)](#54-hard-rubric-fusion-training-2360-rows--26-dim)
   - 5.5 [Hard challenge set (390 rows)](#55-hard-challenge-set-390-rows)
6. [How I trained the models](#6-how-i-trained-the-models)
7. [How I validated the image pipeline](#7-how-i-validated-the-image-pipeline)
8. [How I evaluated the system](#8-how-i-evaluated-the-system)
9. [What worked](#9-what-worked)
10. [What limitations remain](#10-what-limitations-remain)
11. [Slide-ready model and evaluation summary](#11-slide-ready-model-and-evaluation-summary)
12. [What I should say in class](#12-what-i-should-say-in-class)
13. [What I should avoid claiming](#13-what-i-should-avoid-claiming)

---

## 1. What I built

AttachIQ is a fully local, multimodal triage system for AI assistants.
It takes a user prompt and/or an uploaded document image and returns a
structured triage decision in one of four classes:

- `compatible_low_risk` → **ALLOW**
- `compatible_sensitive` → **REVIEW**
- `mismatch_unclear` → **REVIEW**
- `unsafe_external_action` → **BLOCK**

…together with a confidence score and a deterministic explanation.

**What "fully local" means in practice:**

- No LLM API calls (no OpenAI / Anthropic / etc.).
- No OCR (I never read text *out of* the image).
- No AutoML and no pretrained vision backbone — the image CNN is
  trained from scratch.
- The whole pipeline runs on a laptop CPU/GPU; the only pretrained
  weights I use are DistilBERT for the text branch.

---

## 2. Why I chose this problem

I am interested in AI agent governance. Before an AI assistant
processes a user-provided document, it should be able to assess the
request context and identify cases that may require review. The
academic question I wanted the project to answer is concrete:

> *Does a learned multimodal fusion add measurable value beyond a
> hand-coded rule table on this triage task?*

To answer that fairly, the project needs:

- **Three trained models** — one per modality plus a fusion model.
- **Three evaluation distributions** — a standard policy-derived
  test, a hand-curated hard challenge set, and a held-out hard rubric
  test.
- **A clean baseline ladder** — text-only, image-only, rule-table,
  and learned fusion — all evaluated on the same inputs.

---

## 3. How I designed the final architecture

Three PyTorch models wired into one structured pipeline.

### 3.1 Text classifier (DistilBERT, fine-tuned)

- **Backbone:** `distilbert-base-uncased` from Hugging Face.
- **Fine-tuning strategy:** all transformer layers unfrozen (full
  fine-tune, not just the head).
- **Head:** a small `Dropout → Linear` on top of the `[CLS]`
  representation.
- **Output:** **10 request classes** that cover the actions an
  assistant typically receives:
  - `summarization`
  - `information_extraction`
  - `financial_extraction`
  - `document_classification`
  - `internal_sharing`
  - `public_sharing`
  - `delete_permanent`
  - `archive_retain`
  - `ambiguous_or_unclear`
  - `redaction_or_safe_transform`
- **How the taxonomy was chosen:** it was shaped *during*
  implementation by robustness analysis on the hard challenge set.
  In particular, ambiguity, archive/retention, permanent deletion,
  and redaction-safe transformation each got their own class because
  the system needs to be able to *express* them, not collapse them
  into a generic "other".

### 3.2 Image classifier (CNN from scratch)

- **From scratch on purpose:** no pretrained vision backbone — this
  was a project requirement and an honest test of whether a small CNN
  can learn document layout on its own.
- **Architecture:** five `Conv → BatchNorm → ReLU → MaxPool` blocks
  with channel widths `64 → 128 → 256 → 384 → 384`, followed by
  `AdaptiveAvgPool → Dropout → Linear` classifier.
- **Size:** **2,586,568 parameters** — intentionally small.
- **Output:** 8 RVL-CDIP-style document-class probabilities:
  - `invoice`, `form`, `letter`, `report`, `email`, `resume`,
    `presentation`, `handwritten`.
- **Training data:** real RVL-CDIP images, 1,000 per class
  (8,000 total), drawn from the local chainyo-format parquet
  snapshot at `~/datasets/attachiq/rvl-cdip/`.

### 3.3 Fusion MLP (trained from scratch)

- **Input:** a fixed **26-dim** feature vector built deterministically
  per request:
  - 10 request-class probabilities (from the text branch).
  - 8 document-class probabilities (from the image branch).
  - text confidence + image confidence.
  - `has_text` + `has_image` flags (so the model can tell which
    modalities are actually present).
  - text entropy + image entropy.
  - text top-1 − top-2 margin + image top-1 − top-2 margin.
- **Why entropy and margin are in the vector:** they are explicit
  *uncertainty signals*. The fusion network can learn to discount
  low-confidence unimodal predictions instead of blindly trusting
  their argmaxes.
- **Architecture:** `26 → 128 → 64 → 4`, **11,972 parameters**.
- **Training data:** the union of two sets:
  - (a) a 10,000-row standard fusion training set whose triage
    labels come from the rule policy applied to ground-truth
    classes.
  - (b) a 2,360-row hand-curated rubric-aligned fusion training set
    spanning ten challenge categories.

---

## 4. How the modalities interact

- Each modality independently produces a probability vector
  (10-dim for text, 8-dim for image).
- The pipeline always builds the **exact same 26-dim** vector,
  zeroing the missing modality and flipping the corresponding
  `has_*` flag if a modality is absent.
- The fusion MLP consumes that vector and outputs one of four triage
  classes plus a confidence score.
- Pydantic `InferenceRequest` / `InferenceResponse` contracts enforce
  this end-to-end so I cannot accidentally feed a malformed vector
  into the model.

---

## 5. How I prepared the datasets

### 5.1 Text dataset (7,200 prompts)

- **How it was generated:** controlled templates per class, plus
  several variant generators stacked on top:
  - paraphrase variants — e.g. `"Hey, …please…"`, `"…ASAP."`.
  - messy short variants — e.g. `"can u summarize this quickly"`,
    `"fwd to manager"`, `"nuke this attachment"`.
  - light typo variants.
- **Split:** stratified 80 / 10 / 10 = **5,760 / 720 / 720**.
- **Honesty caveat:** the dataset is templated; the text classifier
  saturates on its own held-out test split, which is why the
  meaningful comparison happens at the multimodal fusion level
  rather than text-only.

### 5.2 Image dataset (8,000 real RVL-CDIP images)

- **The HF problem I had to fix:** the original `aharley/rvl_cdip`
  repo on the Hub fails to load on recent `datasets` because it
  relies on the deprecated dataset-script mechanism.
- **What I did about it:** I migrated the image loader to a real
  parquet path with a clear priority:
  1. local FULL chainyo-format snapshot first,
  2. then a local SMALL ImageFolder mirror,
  3. then an HF parquet mirror as a last resort.
- **Label-order gotcha:** the chainyo and the canonical RVL-CDIP
  integer label orders differ. I encoded **both maps explicitly** so
  the wrong label space can never silently leak in.
- **Preprocessing:** grayscale → resize 224×224 → normalize.
- **Train-only augmentations:** small rotation, small
  translation/scale, light brightness/contrast jitter, Gaussian
  noise.
- **Augmentation I deliberately did *not* use:** horizontal flip —
  document text is not flip-invariant.

### 5.3 Standard fusion features (10,000 rows × 26-dim)

- Built from **real text and image probability outputs** with the
  uncertainty signals attached.
- **Triage labels** come from the rule policy applied to the
  *ground-truth* request and document classes. This matters for
  fairness:
  - the rule baseline uses *predicted* classes,
  - the learned fusion uses the same predicted-class features,
  - so both are evaluated on the same distribution and the rule
    baseline is not given an unfair head start.

### 5.4 Hard rubric fusion training (2,360 rows × 26-dim)

- Hand-labelled prompt + image pairs across **ten challenge
  categories**.
- **Critical guardrail:** image indices are drawn from a range
  *disjoint* from the 390-row evaluation-only hard challenge set, so
  no image leaks from training into evaluation.
- **Why it exists:** to give the fusion MLP rubric-aligned
  supervision in addition to the policy-derived standard training
  set.

### 5.5 Hard challenge set (390 rows)

- Hand-curated, **evaluation-only**.
- **Never** used to train, tune, select models, or set thresholds.

---

## 6. How I trained the models

- **Image CNN.**
  - Optimizer: AdamW.
  - LR schedule: `ReduceLROnPlateau` on validation macro F1.
  - Epochs: up to 20, with early stopping on patience.
  - Architecture selection: the training script supports three
    scratch-CNN architectures and picks the one with the best
    validation macro F1. The deep variant won and is the
    production checkpoint.
- **Text classifier.**
  - Optimizer: AdamW.
  - Loss: class-weighted cross-entropy.
  - Stopping: early stop on validation macro F1.
- **Fusion MLP.**
  - Trained on the union of standard and hard rubric training
    sets.
  - Architecture sweep: two variants (small `64/32` and big
    `128/64`).
  - The variant with the best validation macro F1 is saved.

---

## 7. How I validated the image pipeline

Before finalising the image branch I ran a verification pass that
checks, in order:

1. The RVL-CDIP → AttachIQ label mapping against the canonical
   chainyo label list.
2. That every demo image was actually copied from the matching
   `data/raw/images/<class>/` folder.
3. That the train / val / test CSVs use only the 8 expected class
   strings.
4. Two ground-truth samples per class read directly from the
   parquet shards.
5. Three random training images per class from the extracted
   folder.
6. That the train and eval transforms produce `1×224×224` tensors
   normalised the same way.

**Where to find the artefacts:**

- `reports/figures/image_samples_grid.png`
- `reports/figures/parquet_label_check.png`
- `reports/figures/preprocessing_consistency.png`
- `reports/image_pipeline_audit.json`

---

## 8. How I evaluated the system

I report results on **three** test sets:

1. **Standard test (n = 1,000).** Held-out split of the standard
   fusion features. Labels come from the rule policy on
   ground-truth classes.
2. **Hard challenge set (n = 390).** Hand-curated, evaluation-only.
3. **Hard rubric fusion test (n = 354).** Held-out split of the
   hand-labelled rubric fusion set.

For each test set I report **accuracy** and **macro F1** for four
methods:

- text-only baseline,
- image-only baseline,
- rule-table baseline,
- learned fusion.

Confusion matrices are written to `reports/confusion_matrices/`.

---

## 9. What worked

- **Standard test:** rule-table 0.9664, fusion 0.9638.
  - Effectively tied (Δ = −0.0026 macro F1) within noise.
  - Expected: the rule *is* the labelling function for that
    distribution, so as the unimodal models become accurate the
    two methods converge.
- **Hard challenge set:** fusion **0.8207**, rule 0.7640.
  - Fusion clearly leads (**+0.057 macro F1**).
- **Held-out hard rubric fusion test:** fusion **0.8963**,
  rule 0.8628.
  - Fusion clearly leads (**+0.034 macro F1**).
- Fusion strongly outperforms the unimodal baselines on every
  evaluation.
- **Latency:** mean **5–6 ms** per pipeline call after warm-up,
  p95 ≈ **9–12 ms**. First-call inference includes ~2 s of model
  loading.
- **Tests:** 37 / 37 pass; no `print()` statements in `src/`.

---

## 10. What limitations remain

- The text dataset is templated; in production it would be
  supplemented with real prompt-log data.
- The image CNN reads **layout, not content**. AttachIQ does not
  understand legal confidentiality; its output is a *triage signal*,
  not a compliance decision.
- The hard rubric fusion training shares prompt patterns with the
  390-row evaluation set (different image instances). The
  `hard_fusion_test` held-out split is therefore the cleaner
  held-out evaluation for the rubric distribution.
- Both the 390-row hard challenge set and the 2,360-row hard rubric
  fusion set were labelled by a single reviewer against the rubric.
- The image CNN is intentionally small (2.59 M parameters); the
  weakest class on the 8-class document task is `report` at
  per-class F1 ≈ 0.57.

---

## 11. Slide-ready model and evaluation summary

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
| Image CNN B, deep, **selected** | 2.59M | Image test, n=800 | 0.7888 | 0.7864 | Not measured per-arch | Final production image classifier |
| Text classifier, DistilBERT | 66.37M | Text test, n=720 | 1.0000 | 1.0000 | Not measured per-component | Fine-tuned pretrained model; templated-data caveat |
| Fusion MLP, **selected** | 11,972 | Hard rubric fusion test, n=354 | 0.9040 | 0.8963 | Not measured per-component | Final learned multimodal fusion model |
| End-to-end pipeline, standard | 68.97M total | Standard test, n=1000 | 0.9580 | 0.9638 | Mean ≈ 5.7ms, p95 ≈ 9ms | Rule baseline slightly leads: 0.9664 macro F1 |
| End-to-end pipeline, hard challenge | 68.97M total | Hard challenge, n=390 | 0.8256 | 0.8207 | Same pipeline latency | Fusion leads rule: 0.7640 rule macro F1 |
| End-to-end pipeline, hard rubric test | 68.97M total | Hard rubric fusion test, n=354 | 0.9040 | 0.8963 | Same pipeline latency | Fusion leads rule: 0.8628 rule macro F1 |

**How to read this table:**

- The first three rows are **model-selection evidence** for the
  scratch-trained image branch.
- The selected image model is the deep CNN because it gave the
  strongest macro F1 while staying small and local.
- The text classifier satisfies the fine-tuned pretrained-model
  requirement, but its perfect score is treated *cautiously*
  because the prompt dataset is templated.
- The fusion MLP is the final learned multimodal model.
- At the full-pipeline level:
  - The rule baseline is effectively tied on the policy-derived
    standard test.
  - Learned fusion **leads** on the two rubric-bearing
    evaluations.

**Slide speaking version:**

> "I first made sure the image data and labels were correct, then
> compared three scratch CNNs. The deep CNN was selected because it
> improved image macro F1 from 0.462 to 0.786 while staying only
> 2.59M parameters. The final system combines that image model with
> a fine-tuned DistilBERT text model and a small fusion MLP. On the
> standard test, fusion and rules are effectively tied, which is
> expected because the rule aligns closely with that distribution.
> On the harder rubric-based evaluations, learned fusion outperforms
> the rule baseline, which shows the value of combining
> probabilities and uncertainty rather than relying only on top-1
> rule decisions."

---

## 12. What I should say in class

- AttachIQ uses a **10-class request taxonomy** designed to cover
  summarisation, extraction, sharing, deletion, archive/retention,
  ambiguity, and redaction-safe transformation, and a
  **26-dimensional fusion vector** that exposes the unimodal
  classifiers' uncertainty.
- The standard test is **policy-derived**, so the rule baseline is
  expected to be strong and tied with fusion. The interesting
  evidence comes from the **rubric-labelled evaluations**, where
  learned fusion clearly beats both the rule and the unimodal
  baselines.
- The hard challenge set is a stress test on realistic ambiguity:
  - vague prompts,
  - archive vs delete,
  - public vs internal sharing,
  - redaction modifiers,
  - partial extraction on the wrong document type,
  - visually-similar documents,
  - sensitivity modifiers,
  - ambiguous actions,
  - conflicting instructions.

  Fusion handles them meaningfully better than the rule.
- All numbers are reproducible from the recipes and modules
  described in the README; no OCR, no LLM API, no AutoML, no
  pretrained vision backbone.

---

## 13. What I should avoid claiming

- I should **not** claim AttachIQ understands legal
  confidentiality. It classifies request intent and document
  layout.
- I should **not** claim the image classifier reads the document;
  it classifies layout only.
- I should **not** claim fusion strictly dominates the rule on
  every evaluation — on the policy-derived standard test the two
  are effectively tied, and the cleaner evidence for fusion's
  value is on the rubric-labelled evaluations.
