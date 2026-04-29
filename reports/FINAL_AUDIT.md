# AttachIQ — Final Audit

## 1. Files created

```
attachiq/
├── pyproject.toml
├── uv.lock
├── justfile
├── README.md
├── AttachIQ_Proposal.md
├── data/
│   ├── raw/images/<8 classes>/*.png                         (8,000 real RVL-CDIP)
│   ├── processed/text_prompts.csv                           (7,200 prompts, 10 classes)
│   ├── processed/image_manifest.csv                         (8,000 rows)
│   ├── processed/fusion_features.csv                        (10,000 rows × 26-dim)
│   ├── processed/hard_fusion_dataset.csv                    (2,360 hand-labelled pairs)
│   ├── processed/challenge_set.csv                          (390 hard challenge pairs)
│   ├── splits/text_{train,val,test}.csv                     (5,760 / 720 / 720)
│   ├── splits/image_{train,val,test}.csv                    (6,400 / 800 / 800)
│   ├── splits/fusion_{train,val,test}.csv                   (8,000 / 1,000 / 1,000)
│   ├── splits/hard_fusion_{train,val,test}.csv              (1,652 / 354 / 354)
│   ├── splits/hard_fusion_{train,val,test}_features.npy     (cached 26-dim features)
│   └── demo_samples/{invoice,letter,presentation,resume}_demo.png
├── models/
│   ├── image/                                               (deep CNN, 2,586,568 params)
│   │   └── image_cnn.pt + label_map.json + arch.txt
│   ├── text/                                                (DistilBERT, 10 classes)
│   │   └── text_model.pt + tokenizer/ + label_map.json
│   └── fusion/                                              (Fusion MLP, 11,972 params)
│       └── fusion_mlp.pt + arch.json + variant.txt + label_map.json
├── reports/
│   ├── FINAL_AUDIT.md                                       (this file)
│   ├── FINAL_REPORT_DRAFT.md
│   ├── IMPLEMENTATION_JOURNEY.md
│   ├── metrics_summary.json
│   ├── evaluation_summary.md
│   ├── text_metrics.json + text_data_summary.json
│   ├── image_metrics.json + image_data_summary.json
│   ├── fusion_metrics.json + fusion_data_summary.json
│   ├── hard_fusion_data_summary.json
│   ├── challenge_set_summary.json
│   ├── latency_metrics.json
│   ├── image_pipeline_audit.json
│   ├── confusion_matrices/{text,image,fusion,
│   │     standard_fusion,hard_challenge_fusion,hard_fusion_test}_confusion_matrix.png
│   └── figures/{image_samples_grid,parquet_label_check,preprocessing_consistency}.png
├── src/attachiq/
│   ├── __init__.py, config.py, schemas.py, logging.py
│   ├── data/{__init__, build_text_dataset, build_image_dataset,
│   │         build_fusion_dataset, build_hard_fusion_dataset,
│   │         text_dataset, image_dataset}.py
│   ├── models/{__init__, text_model, image_cnn, fusion_mlp}.py
│   ├── training/{__init__, train_text, train_image, train_fusion}.py
│   ├── evaluation/{__init__, metrics, evaluate, evaluate_image}.py
│   ├── triage/{__init__, policy}.py
│   ├── inference/{__init__, pipeline, features, explanations, cli}.py
│   └── ui/{__init__, streamlit_app}.py
├── scripts/
│   └── verify_image_pipeline.py
└── tests/
    ├── test_schemas.py        (8 tests)
    ├── test_policy.py         (12 tests)
    ├── test_fusion_features.py (4 tests)
    ├── test_inference.py      (6 tests)
    └── test_metrics.py        (4 tests)
```

## 2. Commands run (in order)

| # | Command | Result |
|---|---------|--------|
| 1 | `uv sync --extra dev` | OK (Python 3.13.7) |
| 2 | `python -m attachiq.data.build_image_dataset --per-class 1000` | 8,000 real RVL-CDIP images, 8 classes |
| 3 | `python -m attachiq.training.train_image --arch deep --epochs 20` | Test acc 0.7888, macro F1 0.7864, 2,586,568 params |
| 4 | `python -m attachiq.data.build_text_dataset` | 7,200 prompts, 10 classes, 5,760 / 720 / 720 |
| 5 | `python -m attachiq.training.train_text` | Test acc 1.0000, macro F1 1.0000 (templated data) |
| 6 | `python -m attachiq.data.build_fusion_dataset` | 10,000 rows × 26-dim features |
| 7 | `python -m attachiq.data.build_hard_fusion_dataset` | 2,360 hand-labelled rows × 26-dim |
| 8 | `python -m attachiq.training.train_fusion --source union` | Selected `big` variant (26 → 128 → 64 → 4, 11,972 params) |
| 9 | `python -m attachiq.evaluation.evaluate` | wrote `reports/metrics_summary.json` and `reports/evaluation_summary.md` |
| 10| `python -m attachiq.inference.cli --text … --image …` | End-to-end OK, JSON output |
| 11| `just challenge`, `just run-baseline`, `just evaluate`, `just data`, `just test` | All passed |
| 12| `just demo` | Streamlit launches at http://localhost:8501 (verified) |

## 3. Pass/fail checklist

| Requirement | Status | Notes |
|---|---|---|
| Python 3.13 | ✅ | 3.13.7 |
| uv project + pyproject.toml + uv.lock | ✅ | Pinned via `>=` ranges + uv.lock |
| justfile with required recipes | ✅ | install, data, train, train-text/image/fusion, evaluate, evaluate-image, demo, cli, test, clean |
| PyTorch | ✅ | torch 2.9.1 |
| Pydantic v2 contracts | ✅ | InferenceRequest, InferenceResponse |
| Loguru, no print | ✅ | grep `^\s*print\(` in `src/` returns no matches; CLI uses `sys.stdout.write` for JSON |
| No AutoML | ✅ | hand-defined architectures and hyperparams |
| No OCR | ✅ | none in src |
| No external LLM API at runtime | ✅ | none in src |
| No pretrained vision backbone | ✅ | scratch CNN only |
| Local inference only | ✅ | end-to-end on Apple MPS / CPU |
| ≥ 1 fine-tuned pretrained model | ✅ | DistilBERT, all transformer layers unfrozen |
| ≥ 1 model trained from scratch | ✅ | image CNN + fusion MLP both from scratch |
| > 1 DL model | ✅ | 3 trained models |
| Multimodal end-to-end pipeline | ✅ | text_only / image_only / text_plus_image |
| Structured JSON output | ✅ | Pydantic-validated |
| Held-out test evaluation | ✅ | standard test + 390-row hard challenge + held-out hard rubric fusion test |
| Baseline comparison | ✅ | text-only, image-only, rule-table, learned fusion |
| Near real-time inference | ✅ | mean ≈ 5–6 ms / p95 ≈ 9–12 ms |
| Streamlit demo | ✅ | `just demo` |
| Tests passing | ✅ | 37 / 37 |

## 4. Final metrics

### Standard test split (n = 1000)

| Method                | Accuracy | Macro F1 |
|---|---:|---:|
| Text classifier (10 classes) | 1.0000 | 1.0000 |
| Image CNN (deep, unchanged from training) | 0.7888 | 0.7864 |
| Fusion MLP                | 0.9603 | 0.9638 |
| **Rule-table baseline**   | **0.9610** | **0.9664** |
| Text-only baseline        | 0.6020 | 0.5424 |
| Image-only baseline       | 0.6020 | 0.4497 |

### Human-reviewed hard challenge set (n = 390)

| Method                | Accuracy | Macro F1 |
|---|---:|---:|
| **Learned fusion**    | **0.8256** | **0.8207** |
| Rule-table baseline   | 0.7821 | 0.7640 |
| Text-only baseline    | 0.5641 | 0.5089 |
| Image-only baseline   | 0.3795 | 0.2424 |

Fusion vs Rule on the hard challenge: Δ = +0.0567 macro F1.

### Held-out hard rubric fusion test (n = 354)

| Method                | Accuracy | Macro F1 |
|---|---:|---:|
| **Learned fusion**    | **0.9040** | **0.8963** |
| Rule-table baseline   | 0.8729 | 0.8628 |
| Text-only baseline    | 0.5593 | 0.4876 |
| Image-only baseline   | 0.4237 | 0.2851 |

Fusion vs Rule on the hard fusion test: Δ = +0.0335 macro F1.

## 5. Model parameter counts

| Model      | Trainable params |
|-----------|------------------|
| Text classifier (DistilBERT + 10-class head) | 66,368,266 |
| Image CNN (deep, 5 blocks @ 64/128/256/384/384) | 2,586,568 |
| Fusion MLP (26 → 128 → 64 → 4) | 11,972 |
| **Total** | **≈ 68.97 M** (well below the 500 M project cap) |

## 6. Data sources used

- **Text:** synthetic-but-realistic (templates + paraphrases + messy
  variants + light typo variants), **7,200 prompts across 10 classes**,
  fixed seed. Splits 5,760 / 720 / 720. Recorded in
  `reports/text_data_summary.json`.
- **Image:** real RVL-CDIP, local FULL chainyo-format snapshot at
  `~/datasets/attachiq/rvl-cdip`, sampled 1,000 per class
  deterministically with seed 42. Source recorded in
  `reports/image_data_summary.json`.
- **Standard fusion:** 10,000 rows × **26-dim** features built from
  real text and image probability outputs with uncertainty signals;
  labels from `attachiq.triage.policy.classify_triage` on
  ground-truth request/document classes. Recorded in
  `reports/fusion_data_summary.json`.
- **Hard rubric fusion:** 2,360 rows × 26-dim features, hand-labelled
  against the rubric across 10 challenge categories. Image indices
  ≥ 700 to keep them disjoint from the 390-row evaluation-only hard
  challenge set. Recorded in `reports/hard_fusion_data_summary.json`.
- **Hard challenge set:** 390 hand-curated (prompt, image) pairs
  across 10 challenge categories. Evaluation-only.

## 7. Image-CNN architecture selection (training-time evidence)

The image CNN is trained from scratch in three architecture variants
and the variant with the best validation macro F1 is saved:

| Variant | Params | Best val macro F1 | Test acc | Test macro F1 |
|---|---:|---:|---:|---:|
| baseline (4 blocks @ 32/64/128/256) | 390,376 | 0.491 | 0.505 | 0.462 |
| wide (4 blocks @ 64/128/256/384) | 1,258,696 | 0.700 | 0.696 | 0.686 |
| **deep (5 blocks @ 64/128/256/384/384) — selected** | **2,586,568** | **0.806** | **0.789** | **0.786** |

This is model-selection evidence, not a project history; only the deep
checkpoint is shipped.

## 8. Demo cases

Four scripted demo cases run through the production pipeline (warm
state):

| # | Prompt | Image | Predicted doc | Decision |
|---|---|---|---|---|
| 1 | "Summarize this slide"        | presentation_demo.png | presentation | ALLOW (compatible_low_risk) |
| 2 | "Extract the total amount"    | invoice_demo.png      | invoice      | REVIEW (compatible_sensitive) |
| 3 | "Post this publicly"          | resume_demo.png       | resume       | BLOCK (unsafe_external_action) |
| 4 | "Archive this, do not delete it." | letter_demo.png   | letter       | ALLOW (compatible_low_risk) |

4 / 4 image classifications correct, 4 / 4 final triage decisions correct.

## 9. Remaining caveats

- Fusion does not measurably beat the rule on the standard test split
  (Δ = −0.0026 macro F1). Both methods outperform unimodal baselines
  by large margins on every evaluation. Fusion clearly leads on the
  rubric-bearing evaluations (+0.057 hard challenge, +0.034 hard
  fusion test).
- Text classifier saturates at 1.0000 macro F1 on its templated test
  split. Real prompt-log evaluation is future work.
- `report` is the weakest image class (per-class F1 ≈ 0.57); all
  other image classes are 0.66–0.93.
- Hard rubric fusion training and the 390-row hard challenge set
  share prompt patterns; image instances are disjoint. The held-out
  `hard_fusion_test` split (n = 354) is the cleaner held-out
  evaluation for the rubric distribution.
- Single human reviewer for hard labels.

## 10. Final readiness

- All metrics regenerable from the recipes and modules described
  above.
- 37 / 37 tests pass.
- Streamlit demo verified to launch (HTTP 200 on
  `/_stcore/health`).
- Total project parameters ≈ 68.97 M, all under rubric caps.
- No OCR, no LLM API, no AutoML, no pretrained vision backbone, local
  inference only.
