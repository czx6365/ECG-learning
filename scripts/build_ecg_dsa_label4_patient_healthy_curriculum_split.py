from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 0
PATIENT_DIR = Path("datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label3")
HEALTHY_DIR = Path("datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2")
OUT_DIR = Path("datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1")

LABEL_MAP = {
    "OMI": 0,
    "NOMI": 1,
    "对照组": 2,
    "健康": 3,
}
LABEL_NAMES = ["OMI", "NOMI", "对照组", "健康"]


def one_hot(label_ids: np.ndarray, n_classes: int = 4) -> np.ndarray:
    y = np.zeros((len(label_ids), n_classes), dtype=np.int64)
    y[np.arange(len(label_ids)), label_ids.astype(int)] = 1
    return y


def load_patient_selected() -> dict[str, pd.DataFrame]:
    selected: dict[str, pd.DataFrame] = {}
    total = 0
    seen_original_rows: set[int] = set()
    for split in ["train", "val", "test"]:
        meta = pd.read_csv(PATIENT_DIR / f"{split}_metadata.csv")
        meta = meta.reset_index().rename(columns={"index": "source_index"})
        meta["source_split"] = split
        meta["source_dataset"] = str(PATIENT_DIR)
        meta["source_cohort"] = "patient"

        # 一个 original_row_index 可能对应多条 ECG；这里每条病例记录只保留第一条 ECG。
        dedup = meta.drop_duplicates("original_row_index", keep="first").copy()
        overlap = seen_original_rows.intersection(set(dedup["original_row_index"].tolist()))
        if overlap:
            raise RuntimeError(f"original_row_index overlaps across splits: {len(overlap)}")
        seen_original_rows.update(dedup["original_row_index"].tolist())

        dedup["original_label_name"] = dedup["label_name"]
        dedup["original_label_id"] = dedup["label_id"]
        dedup["cohort"] = "patient"
        if dedup["patient_sn"].isna().any():
            raise RuntimeError(f"Missing patient_sn in patient split {split}")
        dedup["entity_id"] = "patient:" + dedup["patient_sn"].astype(str)
        dedup["label_name"] = dedup["label_name"].map(str)
        dedup["label_id"] = dedup["label_name"].map(LABEL_MAP).astype(int)
        dedup["label"] = dedup["label_id"]
        dedup["selected_by"] = "first_ecg_per_original_row_index"
        selected[split] = dedup
        total += len(dedup)

    if total != 4675:
        raise RuntimeError(f"Expected 4675 patient records, got {total}")

    entity_sets = {split: set(frame["entity_id"]) for split, frame in selected.items()}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = entity_sets[left].intersection(entity_sets[right])
        if overlap:
            raise RuntimeError(f"Patient identities overlap between {left} and {right}: {len(overlap)}")
    return selected


def load_healthy_pool() -> pd.DataFrame:
    frames = []
    for split in ["train", "val"]:
        meta = pd.read_csv(HEALTHY_DIR / f"{split}_metadata.csv")
        meta = meta.reset_index().rename(columns={"index": "source_index"})
        meta["source_split"] = split
        meta["source_dataset"] = str(HEALTHY_DIR)
        frames.append(meta)
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["cohort"].astype(str).str.lower().eq("healthy")].copy()
    # 健康数据文件名格式为 <record_id>_<subject_id>.csv。subject_id 才是个体键；
    # 同一个体可有两条记录，并可能出现在旧 train/val 两侧，因此必须先按个体去重。
    stem = pool["csv_name"].astype(str).str.replace(r"\.csv$", "", regex=True)
    pool["healthy_subject_id"] = stem.str.rsplit("_", n=1).str[-1]
    if pool["healthy_subject_id"].isna().any() or pool["healthy_subject_id"].eq("").any():
        raise RuntimeError("Unable to derive healthy subject ID from csv_name")
    unique_pool = (
        pool.sort_values(["healthy_subject_id", "csv_name", "source_split"], kind="stable")
        .drop_duplicates("healthy_subject_id", keep="first")
        .reset_index(drop=True)
    )
    if len(unique_pool) < 4675:
        raise RuntimeError(f"Healthy subject pool too small: {len(unique_pool)}")
    sampled = unique_pool.sample(n=4675, random_state=SEED, replace=False).reset_index(drop=True)
    sampled["source_cohort"] = "healthy"
    sampled["cohort"] = "healthy"
    sampled["entity_id"] = "healthy:" + sampled["healthy_subject_id"].astype(str)
    sampled["label_name"] = "健康"
    sampled["label_id"] = LABEL_MAP["健康"]
    sampled["label"] = LABEL_MAP["健康"]
    sampled["selected_by"] = f"random_sample_seed_{SEED}"
    return sampled


def copy_rows(dst: np.ndarray, start: int, src_dir: Path, split: str, indices: np.ndarray, suffix: str) -> int:
    arr = np.load(src_dir / f"{split}_{suffix}.npy", mmap_mode="r")
    n = len(indices)
    if n:
        dst[start : start + n] = arr[indices]
    return start + n


def write_split(split: str, patient_meta: pd.DataFrame, healthy_meta: pd.DataFrame) -> dict:
    out_meta = pd.concat([patient_meta, healthy_meta], ignore_index=True, sort=False)
    out_meta = out_meta.sample(frac=1.0, random_state=SEED + {"train": 101, "val": 107, "test": 113}[split]).reset_index(drop=True)

    n = len(out_meta)
    labels = one_hot(out_meta["label_id"].to_numpy(dtype=np.int64), n_classes=4)
    np.save(OUT_DIR / f"{split}_labels.npy", labels)

    specs = {
        "data": ((n, 256, 96), np.float32),
        "data_in_chans": ((n, 256), np.int32),
        "data_in_times": ((n, 256), np.int32),
    }

    for suffix, (shape, dtype) in specs.items():
        dst = np.lib.format.open_memmap(OUT_DIR / f"{split}_{suffix}.npy", mode="w+", dtype=dtype, shape=shape)
        indexed = out_meta.reset_index().rename(columns={"index": "out_index"})
        for (src_dataset, src_split), group in indexed.groupby(["source_dataset", "source_split"], sort=False):
            src = np.load(Path(src_dataset) / f"{src_split}_{suffix}.npy", mmap_mode="r")
            dst[group["out_index"].to_numpy(dtype=np.int64)] = src[group["source_index"].to_numpy(dtype=np.int64)]
        del dst

    out_meta["split"] = split
    out_meta.to_csv(OUT_DIR / f"{split}_metadata.csv", index=False)

    return {
        "samples": n,
        "patient_samples": int((out_meta["cohort"] == "patient").sum()),
        "healthy_samples": int((out_meta["cohort"] == "healthy").sum()),
        "label_counts": {name: int((out_meta["label_name"] == name).sum()) for name in LABEL_NAMES},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    patient_by_split = load_patient_selected()
    split_patient_counts = {split: len(df) for split, df in patient_by_split.items()}

    healthy = load_healthy_pool()
    healthy_by_split = {}
    start = 0
    for split in ["train", "val", "test"]:
        count = split_patient_counts[split]
        part = healthy.iloc[start : start + count].copy()
        part["assigned_split"] = split
        healthy_by_split[split] = part
        start += count
    if start != 4675:
        raise RuntimeError(f"Healthy assignment mismatch: {start}")

    healthy_entity_sets = {split: set(frame["entity_id"]) for split, frame in healthy_by_split.items()}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = healthy_entity_sets[left].intersection(healthy_entity_sets[right])
        if overlap:
            raise RuntimeError(f"Healthy identities overlap between {left} and {right}: {len(overlap)}")

    split_summaries = {}
    all_meta = []
    for split in ["train", "val", "test"]:
        split_summaries[split] = write_split(split, patient_by_split[split], healthy_by_split[split])
        all_meta.append(pd.read_csv(OUT_DIR / f"{split}_metadata.csv"))

    matched = pd.concat(all_meta, ignore_index=True)
    matched.to_csv(OUT_DIR / "matched_metadata.csv", index=False)
    pd.DataFrame().to_csv(OUT_DIR / "missing_metadata.csv", index=False)

    entity_sets = {
        split: set(pd.read_csv(OUT_DIR / f"{split}_metadata.csv")["entity_id"].astype(str))
        for split in ["train", "val", "test"]
    }
    overlap_counts = {
        f"{left}_{right}": len(entity_sets[left].intersection(entity_sets[right]))
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]
    }
    if any(overlap_counts.values()):
        raise RuntimeError(f"Final entity overlap detected: {overlap_counts}")

    manifest_columns = [
        "split", "cohort", "entity_id", "label_name", "label_id", "source_dataset",
        "source_split", "source_index", "csv_name", "patient_sn", "healthy_subject_id",
    ]
    split_manifest = matched.reindex(columns=manifest_columns)
    split_manifest.to_csv(OUT_DIR / "split_manifest.csv", index=False)
    manifest_sha256 = hashlib.sha256((OUT_DIR / "split_manifest.csv").read_bytes()).hexdigest()

    summary = {
        "dataset_version": "patient_split_v1",
        "output_dir": str(OUT_DIR),
        "seed": SEED,
        "patient_source": str(PATIENT_DIR),
        "healthy_source": str(HEALTHY_DIR),
        "patient_selection": "deduplicate current label3 qrs dataset by original_row_index, keep first ECG",
        "healthy_selection": "derive subject_id from csv_name suffix, keep one ECG per subject, then sample 4675 subjects with seed 0",
        "identity_keys": {"patient": "patient_sn", "healthy": "csv_name suffix after final underscore"},
        "cross_split_entity_overlap": overlap_counts,
        "split_manifest_sha256": manifest_sha256,
        "label_map": LABEL_MAP,
        "total_samples": int(sum(s["samples"] for s in split_summaries.values())),
        "total_patient_samples": int(sum(s["patient_samples"] for s in split_summaries.values())),
        "total_healthy_samples": int(sum(s["healthy_samples"] for s in split_summaries.values())),
        "splits": split_summaries,
    }
    with open(OUT_DIR / "prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(
            "# ECG_DSA QRS Label4 Patient + Healthy Finetune Dataset\n\n"
            "- Dataset version: `patient_split_v1`.\n"
            "- Classes: OMI, NOMI, 对照组, 健康\n"
            "- Patient samples: 4675, from `data_0521_label3`, deduplicated by `original_row_index`.\n"
            "- Patient grouping key: `patient_sn`; cross-split overlap is asserted to be zero.\n"
            "- Healthy samples: 4675 unique subjects from `patient_healthy_raw_v2`; one ECG per subject.\n"
            "- Healthy grouping key: the suffix after the final underscore in `csv_name`.\n"
            "- Split sizes remain 6640/1344/1366; all three split identity sets are disjoint.\n"
            "- `split_manifest.csv` is the frozen split contract; its SHA-256 is stored in `prepare_summary.json`.\n"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
