# AttachIQ — Final Evaluation Summary

## Standard test (n=1000, policy-derived labels)

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| text-only | 0.5330 | 0.5424 |
| image-only | 0.6020 | 0.4497 |
| rule | 0.9610 | 0.9664 |
| fusion | 0.9580 | 0.9638 |

Fusion vs rule Δ macro F1 = -0.0026

## Hard challenge set (n=390, human-reviewed)

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| text-only | 0.5641 | 0.5089 |
| image-only | 0.3795 | 0.2424 |
| rule | 0.7718 | 0.7640 |
| fusion | 0.8256 | 0.8207 |

Fusion vs rule Δ macro F1 = +0.0567

## Hard rubric fusion test (n=354, held-out rubric)

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| text-only | 0.5593 | 0.4876 |
| image-only | 0.4237 | 0.2851 |
| rule | 0.8729 | 0.8628 |
| fusion | 0.9040 | 0.8963 |

Fusion vs rule Δ macro F1 = +0.0334
