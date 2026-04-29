# AttachIQ

## Multimodal Request-Attachment Triage for AI Assistants

**Author:** Mohammad Abu Jafar
**Course:** MAAI7103 — Deep Learning: From Foundations to Application
**Instructor:** Prof. Saravanan Thirumuruganathan

---

## 1. Problem Statement and Motivation

Modern AI assistants are routinely used as document helpers. Users
upload screenshots, forms, invoices, slides, and personal records,
then ask the assistant to do something with them. The right
operational handling of such a request — allow, review, or block —
depends jointly on **what is being asked** and **what is attached**:
neither modality alone determines the correct response.

The same prompt can fall into different operational categories
depending on the attachment:

| Prompt | Attachment | Triage |
|---|---|---|
| "Summarize this." | Presentation slide | `compatible_low_risk` |
| "Summarize this." | Invoice | `compatible_sensitive` |
| "Extract the total amount." | Presentation slide | `mismatch_unclear` |
| "Post this publicly." | Resume | `unsafe_external_action` |

A text-only system reading "summarise this" would label every row in
this table identically. An image-only system seeing an invoice cannot
know whether the user wants to summarise it, extract from it, share
it, or delete it. The triage decision lives in the **interaction**
between the two modalities.

The academic question this project answers is concrete:

> *Does a learned multimodal fusion add measurable value beyond a
> hand-coded rule table on this triage task?*

### 1.1 GenAI inspiration — from AgentCTRL to AttachIQ

AttachIQ grew out of the same governance question behind my broader GenAI project, AgentCTRL: when an AI system is about to take an action, should that action be allowed, reviewed, or blocked?

For this deep learning course, I narrowed that idea into a measurable supervised learning problem. Instead of governing enterprise AI-agent tool execution, AttachIQ focuses on a specific multimodal scenario: a user prompt paired with a document image. The model’s task is to understand both signals together and triage the request-document situation into ALLOW, REVIEW, or BLOCK.

This keeps the conceptual link to GenAI governance, but implements it through trained, local, non-generative deep learning models: a fine-tuned text classifier, a scratch-trained document-image CNN, and a learned fusion model.

## 2. System Architecture

AttachIQ is a fully local PyTorch pipeline with three trained
components:

1. A fine-tuned `distilbert-base-uncased` text request classifier
   (10 request classes).
2. A CNN trained from scratch for document-image classification
   (8 RVL-CDIP-style classes).
3. A small fusion MLP that combines the two probability vectors plus
   uncertainty signals into a 4-class triage decision.

### 2.1 End-to-end pipeline

```
        prompt text                         document image
              │                                    │
              ▼                                    ▼
    ┌────────────────────┐               ┌────────────────────┐
    │  DistilBERT (text) │               │    Scratch CNN      │
    │  10 request probs  │               │    8 document probs │
    │  (~66 M params)    │               │   (~2.59 M params)  │
    └──────────┬─────────┘               └──────────┬─────────┘
               │                                    │
               └──────────────┬─────────────────────┘
                              ▼
                  26-dim feature vector
                  10 request probs + 8 document probs
                  + text_conf + image_conf
                  + has_text + has_image
                  + text_entropy + image_entropy
                  + text_margin + image_margin
                              │
                              ▼
                ┌──────────────────────────┐
                │  Fusion MLP              │
                │  26 → 128 → 64 → 4       │
                │  (~12 K params)          │
                └──────────┬───────────────┘
                           ▼
                ┌──────────────────────────┐
                │  Structured output        │
                │  (Pydantic JSON)          │
                │  request_type             │
                │  document_type            │
                │  compatibility_label      │
                │  decision: ALLOW / REVIEW │
                │  / BLOCK                  │
                │  confidence, explanation  │
                │  inference_time_ms        │
                └──────────────────────────┘
```

### 2.2 Input modes

| Mode | Description |
|---|---|
| `text_only` | Prompt only; image branch zeroed; `has_image = 0` |
| `image_only` | Image only; text branch zeroed; `has_text = 0` |
| `text_plus_image` | Both branches active; `has_text = 1` and `has_image = 1` |

When a modality is absent, its probability vector and confidence are
zeroed and the corresponding `has_*` flag is set to 0. The fusion
model sees the presence flags and discounts missing inputs rather than
treating them as confident-zero predictions.

### 2.3 Component 1 — Text Request Classifier (fine-tuned)

**Task:** map a free-form prompt to one of ten request classes.

| Class | Example prompt |
|---|---|
| `summarization` | "Summarize this document." |
| `information_extraction` | "Pull out the key fields." |
| `financial_extraction` | "Find the total amount." |
| `document_classification` | "What type of file is this?" |
| `internal_sharing` | "Send this to my manager." |
| `public_sharing` | "Post this online." |
| `delete_permanent` | "Delete this file permanently." |
| `archive_retain` | "Archive this, do not delete it." |
| `ambiguous_or_unclear` | "Can you handle this?" |
| `redaction_or_safe_transform` | "Summarize without personal details." |

**Model:** `distilbert-base-uncased` (~66 M parameters) with all
transformer layers unfrozen and a dropout + linear head over the
`[CLS]` representation. Trained with AdamW and class-weighted
cross-entropy. **Augmentations:** light typo variants and paraphrase
expansions are baked into the dataset; no test-time augmentation.

### 2.4 Component 2 — Document Image Classifier (from scratch)

**Task:** map a document image to one of eight RVL-CDIP classes:
`invoice`, `form`, `letter`, `report`, `email`, `resume`,
`presentation`, `handwritten`.

**Architecture:** five `Conv-BN-ReLU-MaxPool` blocks at channels
64/128/256/384/384, AdaptiveAvgPool, Dropout, Linear classifier —
2,586,568 parameters. No pretrained backbone.

**Preprocessing:** grayscale, resize 224×224, normalise. Train-only
augmentations: small rotation (±4°), small translation (±3%), small
scale (0.95–1.05), light brightness/contrast jitter, Gaussian noise
(σ = 0.015). **No horizontal flip** (document text is not
flip-invariant).

### 2.5 Component 3 — Fusion Classifier (from scratch)

**Inputs (26-dim feature vector):** request-class probabilities (10),
document-class probabilities (8), top-1 text confidence (1), top-1
image confidence (1), `has_text` flag (1), `has_image` flag (1),
text entropy (1), image entropy (1), text top-1−top-2 margin (1),
image top-1−top-2 margin (1).

**Architecture:** `26 → 128 → 64 → 4`, ~12 K parameters, dropout 0.2.

**Training:** trained on the union of (a) a 10,000-row standard
fusion training set whose triage labels come from the rule policy
applied to ground-truth classes, and (b) a 2,360-row hand-curated
rubric-aligned fusion training set across ten challenge categories.
Variant selection (small 64/32 vs big 128/64) by validation macro F1
only.

### 2.6 Pydantic I/O contracts

```json
{
  "input_mode": "text_plus_image",
  "request_type": "financial_extraction",
  "document_type": "invoice",
  "compatibility_label": "compatible_sensitive",
  "decision": "REVIEW",
  "confidence": 0.87,
  "explanation": "The request matches the invoice, but it involves extracting financial information.",
  "inference_time_ms": 24.0
}
```

## 3. Project Scope and Deliverables

- A reproducible PyTorch repository.
- Three trained models (text DistilBERT, scratch CNN, fusion MLP).
- A `justfile` covering data preparation, model training, evaluation,
  demo, CLI inference, testing, and cleanup.
- Pydantic v2 contracts and structured JSON output.
- A Streamlit demo.
- A held-out test evaluation, a 390-row human-reviewed hard challenge
  set, and a held-out hard rubric fusion test split.
- A baseline ladder (text-only, image-only, rule-table, learned
  fusion).
- All metrics under `reports/` and all checkpoints under `models/`.

## 4. Dataset and Information Sources

- **Text request prompts (synthetic-but-realistic).** 7,200 labelled
  prompts across the 10 request classes (720 per class), generated
  programmatically from controlled templates plus paraphrase variants
  plus messy short forms plus light typo variants. Stratified
  80/10/10 split = 5,760 / 720 / 720. A follow-up could supplement
  this set with a small manually reviewed sample drawn from public
  LLM interaction corpora such as
  [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
  or [WildChat](https://huggingface.co/datasets/allenai/WildChat),
  filtered to document-bearing requests, to reduce the over-clean
  trap of pure synthetic data.
- **Document images (real RVL-CDIP).** Real RVL-CDIP via local
  parquet snapshot, 8 mapped classes × 1000 images = 8,000 images,
  stratified 80/10/10 = 6,400 / 800 / 800.
- **Standard fusion features.** 10,000 rows × 26-dim built from real
  text and image probability outputs with uncertainty signals
  attached. Triage labels from the deterministic policy applied to
  ground-truth classes.
- **Hard rubric fusion training.** 2,360 rows × 26-dim,
  hand-labelled across 10 challenge categories. Disjoint image
  indices from the 390-row evaluation-only hard challenge set.
- **Hard challenge set.** 390 hand-curated (prompt, image) pairs;
  evaluation-only.

## 5. Evaluation and Success Criteria

The four-class triage task is compared across:

1. **Text-only DL** — DistilBERT mapped via the rule policy with no
   document.
2. **Image-only DL** — CNN mapped via the rule policy with no
   request.
3. **Rule table** — `(predicted_request, predicted_document) →
   triage_class` policy; invoked via the rule baseline path of the
   evaluator.
4. **Learned fusion** — the trained MLP on the 26-dim feature vector.

Three held-out evaluations:

- Standard test (n = 1000, policy-derived labels).
- Hard challenge set (n = 390, human-reviewed).
- Hard rubric fusion test (n = 354, held-out rubric).

**Primary metrics:** accuracy and macro F1, with per-class precision,
recall, and F1 in the metrics JSON. **Practical metric:** end-to-end
triage latency on a held-out paired reference set, including the
390-row human-reviewed hard challenge set and the held-out hard
rubric fusion test split.

## 6. Risks, Limitations, and Ethical Considerations

- The text dataset is templated; the text classifier saturates on
  its own held-out test split. The substantive comparison happens at
  the multimodal fusion level.
- The image classifier reads document layout, not content. There is
  no OCR. AttachIQ does not understand legal confidentiality.
- The hard rubric fusion training set shares prompt patterns with
  the 390-row evaluation set (image instances are disjoint). The
  held-out hard rubric fusion test split is the cleaner held-out
  evaluation for the rubric distribution.
- Both hand-labelled sets were labelled by a single reviewer against
  the rubric.
- The output is a triage signal, not a compliance decision.

## 7. Constraints

- No OCR.
- No LLM API calls at runtime.
- No AutoML.
- No pretrained vision backbone.
- Local inference only.
- Pydantic v2 structured output.
- `uv` project + `uv.lock`; Python 3.13.
- Loguru, no `print()` statements in `src/`.

## 8. Final Deliverable and Live Demo

A Streamlit demo at `src/attachiq/ui/streamlit_app.py`, launched via
`just demo`, exposes a prompt text area, an image uploader, and an
"Assess Request" button. The demo returns the input mode, request and
document types with confidence, the compatibility label, an
ALLOW / REVIEW / BLOCK decision badge, a deterministic explanation,
the inference time, and the full Pydantic JSON response.
