"""Build the document image dataset from real RVL-CDIP.

Tries real RVL-CDIP sources in this priority order and writes a manifest
with per-class subfolders of grayscale 512×512 PNGs:

  A. Local FULL chainyo-format RVL-CDIP parquet snapshot:
       ~/datasets/attachiq/rvl-cdip/  (data/*.parquet, 16 classes)
     Up to 1000 images per AttachIQ class, sampled deterministically.

  B. Local SMALL real RVL-CDIP subset in ImageFolder layout:
       ~/datasets/attachiq/rvl_cdip-small-200/

  C. Hugging Face parquet mirror (no scripts):
       `vaclavpechtor/rvl_cdip-small-200`

The original `aharley/rvl_cdip` repo ships an `rvl_cdip.py` dataset
script which is incompatible with `datasets >= 4.0`. We do NOT call
that loader.

Outputs:
  data/processed/image_manifest.csv
  data/splits/image_{train,val,test}.csv
  reports/image_data_summary.json
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from attachiq.config import (
    DOCUMENT_CLASSES,
    GLOBAL_SEED,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    ensure_dirs,
)
from attachiq.logging import get_logger

log = get_logger("data.image")

IMAGES_DIR = RAW_DIR / "images"
LOCAL_FULL_RVL = Path.home() / "datasets" / "attachiq" / "rvl-cdip"
LOCAL_SMALL_RVL = Path.home() / "datasets" / "attachiq" / "rvl_cdip-small-200"

# chainyo (full) integer-label mapping → AttachIQ class name.
RVL_LABEL_INT_CHAINYO: dict[str, int] = {
    "email": 2, "form": 4, "handwritten": 5, "invoice": 6,
    "letter": 7, "presentation": 10, "resume": 12, "report": 14,
}

# Canonical aharley int order (used by the small HF parquet mirror).
RVL_LABEL_INT_CANONICAL: dict[str, int] = {
    "letter": 0, "form": 1, "email": 2, "handwritten": 3,
    "report": 5, "invoice": 11, "presentation": 12, "resume": 14,
}

SMALL_FOLDER_NAME: dict[str, str] = {
    "letter": "letter", "form": "form", "email": "email",
    "handwritten": "handwritten", "invoice": "invoice", "resume": "resume",
    "presentation": "presentation", "report": "scientific_report",
}

DEFAULT_HF_DATASET = "vaclavpechtor/rvl_cdip-small-200"


def _save_image_bytes(raw: bytes, dest: Path) -> bool:
    try:
        img = Image.open(io.BytesIO(raw))
        if img.mode != "L":
            img = img.convert("L")
        img.thumbnail((512, 512))
        img.save(dest, optimize=True)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug(f"failed to decode image: {exc!r}")
        return False


def _save_image_file(src: Path, dest: Path) -> bool:
    try:
        img = Image.open(src)
        if img.mode != "L":
            img = img.convert("L")
        img.thumbnail((512, 512))
        img.save(dest, optimize=True)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug(f"failed to load {src}: {exc!r}")
        return False


def _try_local_full(per_class: int, seed: int) -> tuple[bool, str | None, dict[str, int]]:
    if not LOCAL_FULL_RVL.exists():
        return False, None, {}
    data_dir = LOCAL_FULL_RVL / "data"
    if not data_dir.exists():
        return False, None, {}
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        log.warning(f"pyarrow unavailable: {exc!r}")
        return False, None, {}

    rng = random.Random(seed)
    int_to_our = {v: k for k, v in RVL_LABEL_INT_CHAINYO.items()}

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        return False, None, {}
    train_files = [p for p in parquet_files if p.name.startswith("train-")]
    val_files = [p for p in parquet_files if p.name.startswith("val-")]
    test_files = [p for p in parquet_files if p.name.startswith("test-")]
    log.info(
        f"Local FULL RVL-CDIP at {LOCAL_FULL_RVL}: "
        f"{len(train_files)} train + {len(val_files)} val + {len(test_files)} test shards."
    )

    # Index shards by which target labels they carry.
    target = set(int_to_our.keys())
    shard_label_count: dict[Path, dict[int, int]] = {}
    for path in train_files + val_files + test_files:
        try:
            tbl = pq.read_table(path, columns=["label"])
            labels = tbl.column("label").to_pylist()
        except Exception:
            continue
        cnt: dict[int, int] = {}
        for lbl in labels:
            if lbl in target:
                cnt[lbl] = cnt.get(lbl, 0) + 1
        if cnt:
            shard_label_count[path] = cnt

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for c in DOCUMENT_CLASSES:
        cls_dir = IMAGES_DIR / c
        cls_dir.mkdir(parents=True, exist_ok=True)
        for old in cls_dir.iterdir():
            if old.is_file():
                old.unlink()

    counts = {c: 0 for c in DOCUMENT_CLASSES}
    targets = {c: per_class for c in DOCUMENT_CLASSES}

    for path, shard_cnt in shard_label_count.items():
        labels_in_shard = set(shard_cnt.keys())
        needed_here = [
            lbl for lbl in labels_in_shard
            if counts[int_to_our[lbl]] < targets[int_to_our[lbl]]
        ]
        if not needed_here:
            continue
        try:
            tbl = pq.read_table(path)
            labels_arr = tbl.column("label").to_pylist()
            images_arr = tbl.column("image").to_pylist()
        except Exception:
            continue
        per_label_indices: dict[int, list[int]] = {}
        for i, lbl in enumerate(labels_arr):
            if lbl in needed_here:
                per_label_indices.setdefault(lbl, []).append(i)
        for lbl, idxs in per_label_indices.items():
            our_name = int_to_our[lbl]
            remaining = targets[our_name] - counts[our_name]
            if remaining <= 0:
                continue
            rng.shuffle(idxs)
            for i in idxs[:remaining]:
                rec = images_arr[i]
                raw = rec["bytes"] if isinstance(rec, dict) and "bytes" in rec else None
                if not raw:
                    continue
                k = counts[our_name]
                dest = IMAGES_DIR / our_name / f"{our_name}_{k:06d}.png"
                if _save_image_bytes(raw, dest):
                    counts[our_name] += 1
        log.info(
            "after shard "
            + path.name + ": "
            + ", ".join(f"{c}={counts[c]}/{targets[c]}" for c in DOCUMENT_CLASSES)
        )
        if all(counts[c] >= targets[c] for c in DOCUMENT_CLASSES):
            break

    if any(counts[c] == 0 for c in DOCUMENT_CLASSES):
        return False, None, counts
    if min(counts.values()) < 50:
        return False, None, counts
    return True, f"local:{LOCAL_FULL_RVL}", counts


def _try_local_small(seed: int) -> tuple[bool, str | None, dict[str, int]]:
    if not LOCAL_SMALL_RVL.exists():
        return False, None, {}
    counts = {c: 0 for c in DOCUMENT_CLASSES}
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for c in DOCUMENT_CLASSES:
        (IMAGES_DIR / c).mkdir(parents=True, exist_ok=True)
    for our_name, folder in SMALL_FOLDER_NAME.items():
        for split in ("train", "validation"):
            src_dir = LOCAL_SMALL_RVL / split / folder
            if not src_dir.exists():
                continue
            for src in sorted(src_dir.iterdir()):
                if src.is_file():
                    k = counts[our_name]
                    dest = IMAGES_DIR / our_name / f"{our_name}_{k:06d}.png"
                    if _save_image_file(src, dest):
                        counts[our_name] += 1
    if any(n == 0 for n in counts.values()):
        return False, None, counts
    return True, f"local:{LOCAL_SMALL_RVL}", counts


def _try_hf_mirror(seed: int, dataset_name: str = DEFAULT_HF_DATASET) -> tuple[bool, str | None, dict[str, int]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        log.warning(f"datasets unavailable: {exc!r}")
        return False, None, {}
    try:
        ds = load_dataset(dataset_name)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"HF mirror load failed: {exc!r}")
        return False, None, {}
    int_to_our = {v: k for k, v in RVL_LABEL_INT_CANONICAL.items()}
    counts = {c: 0 for c in DOCUMENT_CLASSES}
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for c in DOCUMENT_CLASSES:
        (IMAGES_DIR / c).mkdir(parents=True, exist_ok=True)
    for split_name in ds.keys():
        for sample in ds[split_name]:
            label = sample.get("label")
            if label is None or label not in int_to_our:
                continue
            our_name = int_to_our[label]
            try:
                img = sample["image"]
                if img.mode != "L":
                    img = img.convert("L")
                img.thumbnail((512, 512))
                k = counts[our_name]
                img.save(IMAGES_DIR / our_name / f"{our_name}_{k:06d}.png", optimize=True)
                counts[our_name] += 1
            except Exception:
                continue
    if any(n == 0 for n in counts.values()):
        return False, None, counts
    return True, f"hf:{dataset_name}", counts


def _build_manifest(seed: int = GLOBAL_SEED) -> pd.DataFrame:
    rows: list[dict] = []
    for cls in DOCUMENT_CLASSES:
        for path in sorted((IMAGES_DIR / cls).glob("*.png")):
            rows.append({"image_path": str(path), "label": cls})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _stratified_split(df: pd.DataFrame, seed: int = GLOBAL_SEED) -> dict[str, pd.DataFrame]:
    train, temp = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp["label"])
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def main(per_class: int = 1000, dataset_name: str | None = None) -> None:
    ensure_dirs()
    log.info(f"Building image dataset (per_class={per_class}).")

    ok, source, counts = _try_local_full(per_class=per_class, seed=GLOBAL_SEED)
    if not ok:
        log.warning("Local FULL RVL-CDIP unavailable. Trying local SMALL.")
        ok, source, counts = _try_local_small(seed=GLOBAL_SEED)
    if not ok:
        log.warning("Local SMALL unavailable. Trying HF parquet mirror.")
        ok, source, counts = _try_hf_mirror(
            seed=GLOBAL_SEED, dataset_name=dataset_name or DEFAULT_HF_DATASET,
        )
    if not ok:
        raise RuntimeError(
            "Could not build image dataset from any RVL-CDIP source "
            "(local FULL, local SMALL, HF mirror)."
        )

    df = _build_manifest()
    if len(df) == 0:
        raise RuntimeError("Image manifest is empty.")
    out_csv = PROCESSED_DIR / "image_manifest.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"Wrote {out_csv} (n={len(df)})")

    splits = _stratified_split(df)
    for name, part in splits.items():
        path = SPLITS_DIR / f"image_{name}.csv"
        part.to_csv(path, index=False)
        log.info(f"Wrote {path} (n={len(part)})")

    summary = {
        "total": int(len(df)),
        "per_class": df["label"].value_counts().to_dict(),
        "splits": {n: int(len(p)) for n, p in splits.items()},
        "source": source,
        "extracted_counts": counts,
        "per_class_target": int(per_class),
        "seed": GLOBAL_SEED,
    }
    Path(REPORTS_DIR / "image_data_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    log.info(f"Image data summary written. source={source} total={len(df)}")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override the HF parquet mirror as last-resort fallback.")
    args = parser.parse_args()
    main(per_class=args.per_class, dataset_name=args.dataset)


if __name__ == "__main__":
    cli()
