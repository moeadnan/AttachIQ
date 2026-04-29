"""Image-pipeline verification (read-only).

Outputs:
  reports/figures/image_samples_grid.png      (3 samples per class from extracted data)
  reports/figures/parquet_label_check.png     (parquet ground-truth grid)
  reports/figures/preprocessing_consistency.png  (train vs eval transform on same image)
  reports/image_pipeline_audit.json           (machine-readable summary)
"""

from __future__ import annotations

import json
import random
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

from attachiq.config import DOCUMENT_CLASSES, FIGURES_DIR, REPORTS_DIR, SPLITS_DIR  # noqa: E402
from attachiq.data.build_image_dataset import (  # noqa: E402
    LOCAL_FULL_RVL,
    RVL_LABEL_INT_CHAINYO,
)
from attachiq.data.image_dataset import get_transforms  # noqa: E402
from attachiq.logging import get_logger  # noqa: E402

log = get_logger("verify.image")

# Canonical chainyo names exactly as listed in dataset_infos.json:
CHAINYO_NAMES = [
    "advertisement",
    "budget",
    "email",
    "file_folder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "news_article",
    "presentation",
    "questionnaire",
    "resume",
    "scientific_publication",
    "scientific_report",
    "specification",
]


def _check_mapping() -> dict:
    """Verify the chainyo->AttachIQ mapping against the canonical chainyo names."""
    expected = {
        "email": 2,
        "form": 4,
        "handwritten": 5,
        "invoice": 6,
        "letter": 7,
        "presentation": 10,
        "resume": 12,
        "report": 14,  # scientific_report
    }
    chainyo_label_for = {
        "email": "email",
        "form": "form",
        "handwritten": "handwritten",
        "invoice": "invoice",
        "letter": "letter",
        "presentation": "presentation",
        "resume": "resume",
        "report": "scientific_report",
    }
    issues = []
    for ours, idx in RVL_LABEL_INT_CHAINYO.items():
        if expected[ours] != idx:
            issues.append(f"{ours}: code says {idx}, expected {expected[ours]}")
        chainyo_name = CHAINYO_NAMES[idx]
        if chainyo_name != chainyo_label_for[ours]:
            issues.append(
                f"{ours}: code maps to chainyo idx {idx} == '{chainyo_name}' "
                f"but should be '{chainyo_label_for[ours]}'"
            )
    return {
        "code_mapping": RVL_LABEL_INT_CHAINYO,
        "expected_mapping": expected,
        "chainyo_idx_to_name": dict(enumerate(CHAINYO_NAMES)),
        "issues": issues,
        "ok": len(issues) == 0,
    }


def _scan_shards_by_label(data_dir: Path) -> dict[int, list[Path]]:
    """For each chainyo label int we care about, list parquet shards that contain it."""
    result: dict[int, list[Path]] = {lbl: [] for lbl in RVL_LABEL_INT_CHAINYO.values()}
    for path in sorted(data_dir.glob("*.parquet")):
        try:
            tbl = pq.read_table(path, columns=["label"])
            uniq = set(tbl.column("label").to_pylist())
        except Exception as exc:  # noqa: BLE001
            log.debug(f"skip {path}: {exc!r}")
            continue
        for lbl in result:
            if lbl in uniq:
                result[lbl].append(path)
    return result


def _draw_grid(images_with_titles: list[tuple[np.ndarray, str]], cols: int, out_path: Path, suptitle: str) -> None:
    n = len(images_with_titles)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.0))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.array(axes).reshape(rows, cols)
    for k in range(rows * cols):
        r, c = k // cols, k % cols
        ax = axes[r, c]
        ax.axis("off")
        if k < n:
            img, title = images_with_titles[k]
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=8)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _extracted_grid() -> Path:
    """Grid of 3 sample images per AttachIQ class from data/raw/images/<class>/."""
    rng = random.Random(42)
    images_dir = Path("data/raw/images")
    panels: list[tuple[np.ndarray, str]] = []
    for cls in DOCUMENT_CLASSES:
        cls_dir = images_dir / cls
        files = sorted(cls_dir.glob("*.png"))
        if not files:
            for _ in range(3):
                panels.append((np.zeros((64, 64), dtype=np.uint8), f"{cls}\n(MISSING)"))
            continue
        chosen = rng.sample(files, min(3, len(files)))
        for f in chosen:
            img = Image.open(f).convert("L")
            arr = np.array(img.resize((256, 256)))
            panels.append((arr, f"{cls}\n{f.name}"))
    out = FIGURES_DIR / "image_samples_grid.png"
    _draw_grid(
        panels,
        cols=3,
        out_path=out,
        suptitle="3 random samples per AttachIQ class (from data/raw/images/)",
    )
    return out


def _parquet_grid(shard_index: dict[int, list[Path]]) -> tuple[Path, dict]:
    """Grid of 2 samples per chainyo target label, pulled directly from parquet."""
    rng = random.Random(42)
    panels: list[tuple[np.ndarray, str]] = []
    sampled_records: list[dict] = []
    inverted = {v: k for k, v in RVL_LABEL_INT_CHAINYO.items()}
    for ours in DOCUMENT_CLASSES:
        chainyo_int = RVL_LABEL_INT_CHAINYO[ours]
        shards = shard_index.get(chainyo_int, [])
        if not shards:
            for _ in range(2):
                panels.append((np.zeros((64, 64), dtype=np.uint8), f"{ours}\n(no shard)"))
            continue
        shard = shards[len(shards) // 2]  # pick a middle shard
        try:
            tbl = pq.read_table(shard)
            labels_arr = tbl.column("label").to_pylist()
            images_arr = tbl.column("image").to_pylist()
        except Exception as exc:  # noqa: BLE001
            log.warning(f"skip shard {shard.name}: {exc!r}")
            continue
        idxs = [i for i, lbl in enumerate(labels_arr) if lbl == chainyo_int]
        chosen = rng.sample(idxs, min(2, len(idxs)))
        for i in chosen:
            rec = images_arr[i]
            raw = rec["bytes"] if isinstance(rec, dict) and "bytes" in rec else None
            if not raw:
                continue
            try:
                img = Image.open(BytesIO(raw)).convert("L")
            except Exception as exc:  # noqa: BLE001
                log.debug(f"decode fail: {exc!r}")
                continue
            arr = np.array(img.resize((256, 256)))
            chainyo_name = CHAINYO_NAMES[chainyo_int]
            panels.append(
                (
                    arr,
                    f"AttachIQ='{ours}'\nchainyo[{chainyo_int}]='{chainyo_name}'\n{shard.name}",
                )
            )
            sampled_records.append(
                {
                    "attachiq_class": ours,
                    "chainyo_int": int(chainyo_int),
                    "chainyo_name": chainyo_name,
                    "shard": shard.name,
                    "row": int(i),
                    "image_size": list(img.size),
                }
            )
    out = FIGURES_DIR / "parquet_label_check.png"
    _draw_grid(
        panels,
        cols=4,
        out_path=out,
        suptitle="Parquet ground truth: 2 samples per chainyo label (visual sanity check)",
    )
    return out, {
        "inverted_mapping_for_reference": inverted,
        "sampled_records": sampled_records,
    }


def _preprocessing_consistency_check() -> tuple[Path, dict]:
    """Run train-mode and eval-mode transforms on the same demo image. Compare."""
    candidate = Path("data/demo_samples/invoice_demo.png")
    if not candidate.exists():
        candidate = next(Path("data/raw/images/invoice").glob("*.png"))
    img = Image.open(candidate).convert("L")
    eval_t = get_transforms(train=False)
    train_t = get_transforms(train=True)

    eval_x = eval_t(img)
    train_x_a = train_t(img)
    train_x_b = train_t(img)

    rows = []
    for tag, t in [("original", img), ("eval", eval_x), ("train_run_1", train_x_a), ("train_run_2", train_x_b)]:
        if isinstance(t, Image.Image):
            arr = np.array(t.resize((224, 224)))
        else:
            arr = t.squeeze().numpy()
            arr = (arr * 0.5) + 0.5  # un-normalise (mean=0.5, std=0.5)
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        title = f"{tag}\nshape={arr.shape}"
        if not isinstance(t, Image.Image):
            mean = float(t.mean())
            mn = float(t.min())
            mx = float(t.max())
            title += f"\nmean={mean:.3f} [{mn:.2f},{mx:.2f}]"
        rows.append((arr, title))

    out = FIGURES_DIR / "preprocessing_consistency.png"
    _draw_grid(
        rows,
        cols=4,
        out_path=out,
        suptitle=f"Same image through eval and train transforms (path={candidate.name})",
    )

    eval_stats = {
        "shape": list(eval_x.shape),
        "min": float(eval_x.min()),
        "max": float(eval_x.max()),
        "mean": float(eval_x.mean()),
        "std": float(eval_x.std()),
    }
    train_stats = {
        "shape": list(train_x_a.shape),
        "min": float(train_x_a.min()),
        "max": float(train_x_a.max()),
        "mean": float(train_x_a.mean()),
        "std": float(train_x_a.std()),
    }
    return out, {"eval": eval_stats, "train_run_1": train_stats}


def _split_label_consistency() -> dict:
    """Verify train/val/test CSVs use only the 8 expected AttachIQ class names."""
    out: dict = {}
    for name in ("train", "val", "test"):
        path = SPLITS_DIR / f"image_{name}.csv"
        if not path.exists():
            out[name] = {"missing": True}
            continue
        df = pd.read_csv(path)
        labels = sorted(df["label"].unique())
        unknown = [l for l in labels if l not in DOCUMENT_CLASSES]
        out[name] = {
            "n": int(len(df)),
            "labels": labels,
            "per_class": df["label"].value_counts().to_dict(),
            "unknown_labels": unknown,
            "all_in_DOCUMENT_CLASSES": len(unknown) == 0,
        }
    return out


def _label_map_consistency() -> dict:
    out: dict = {}
    label_map_path = Path("models/image/label_map.json")
    if label_map_path.exists():
        saved = json.loads(label_map_path.read_text())
        expected = {c: i for i, c in enumerate(DOCUMENT_CLASSES)}
        out["saved_label_map"] = saved
        out["expected_label_map"] = expected
        out["match"] = saved == expected
    else:
        out["error"] = "models/image/label_map.json missing"
    return out


def _demo_sample_check() -> dict:
    """For each demo image, record which AttachIQ class folder it was copied from."""
    demo = Path("data/demo_samples")
    out: list[dict] = []
    for f in sorted(demo.glob("*.png")):
        # filename convention: <class>_demo.png, copied from data/raw/images/<class>/<class>_000000.png
        cls_guess = f.stem.replace("_demo", "")
        source = Path("data/raw/images") / cls_guess / f"{cls_guess}_000000.png"
        out.append(
            {
                "demo_file": f.name,
                "claimed_class": cls_guess,
                "source_in_raw": str(source),
                "source_exists": source.exists(),
                "size_bytes": f.stat().st_size,
            }
        )
    return out


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    audit: dict = {}

    # 1. mapping
    audit["mapping_check"] = _check_mapping()
    log.info(
        f"mapping ok={audit['mapping_check']['ok']}, "
        f"issues={audit['mapping_check']['issues']}"
    )

    # 2/4. parquet ground truth
    data_dir = LOCAL_FULL_RVL / "data"
    if data_dir.exists():
        log.info(f"scanning shards in {data_dir} ...")
        shard_index = _scan_shards_by_label(data_dir)
        audit["shard_index_summary"] = {
            CHAINYO_NAMES[k]: len(v) for k, v in shard_index.items()
        }
        parquet_path, parquet_audit = _parquet_grid(shard_index)
        audit["parquet_grid"] = str(parquet_path)
        audit["parquet_samples"] = parquet_audit
    else:
        audit["parquet_grid"] = "skipped: local FULL parquet not found"

    # 3/8. split label + label_map consistency
    audit["split_label_consistency"] = _split_label_consistency()
    audit["label_map_consistency"] = _label_map_consistency()

    # 5. preprocessing consistency
    pp_path, pp_stats = _preprocessing_consistency_check()
    audit["preprocessing_grid"] = str(pp_path)
    audit["preprocessing_stats"] = pp_stats

    # 6. demo samples
    audit["demo_samples"] = _demo_sample_check()

    # 7. extracted contact sheet
    extracted_path = _extracted_grid()
    audit["extracted_grid"] = str(extracted_path)

    out_json = REPORTS_DIR / "image_pipeline_audit.json"
    out_json.write_text(json.dumps(audit, indent=2, default=str))
    log.info(f"Wrote {out_json}")
    log.info(f"Wrote contact sheet {extracted_path}")
    if "parquet_grid" in audit:
        log.info(f"Wrote parquet grid {audit['parquet_grid']}")
    log.info(f"Wrote preprocessing grid {pp_path}")


if __name__ == "__main__":
    main()
