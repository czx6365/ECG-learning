"""Build a four-class ECG-level dataset without collapsing multiple ECGs per label row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_MAP = {"OMI": 0, "NOMI": 1, "对照组": 2, "健康": 3}
SPLITS = ("train", "val", "test")
ARRAY_SUFFIXES = ("data", "data_in_chans", "data_in_times")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patient_dir",
        type=Path,
        default=Path("datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label3"),
    )
    parser.add_argument(
        "--healthy_dir",
        type=Path,
        default=Path("datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_ecg8303_healthy8303"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_patient_ecgs(patient_dir: Path) -> dict[str, pd.DataFrame]:
    selected = {}
    all_meta = []
    for split in SPLITS:
        meta = pd.read_csv(patient_dir / f"{split}_metadata.csv")
        meta = meta.reset_index().rename(columns={"index": "source_index"})
        if meta[["csv_name", "file_name", "patient_sn", "label_name"]].isna().any().any():
            raise ValueError(f"Missing ECG file, patient, or label in {split} metadata")
        if meta["file_name"].astype(str).str.contains(";", regex=False).any():
            raise ValueError(f"{split} still has multi-file rows; expected one ECG per row")

        mapped = meta["label_name"].map(LABEL_MAP)
        if mapped.isna().any() or (mapped == LABEL_MAP["健康"]).any():
            raise ValueError(f"Unexpected patient label in {split} metadata")
        labels = np.load(patient_dir / f"{split}_labels.npy", mmap_mode="r")
        if labels.shape != (len(meta), 3) or not np.array_equal(
            labels.argmax(axis=1), mapped.to_numpy(dtype=np.int64)
        ):
            raise ValueError(f"{split} metadata and labels.npy disagree")
        for suffix in ARRAY_SUFFIXES:
            arr = np.load(patient_dir / f"{split}_{suffix}.npy", mmap_mode="r")
            if len(arr) != len(meta):
                raise ValueError(f"{split}_{suffix}.npy length does not match metadata")

        meta["source_dataset"] = str(patient_dir)
        meta["source_split"] = split
        meta["source_cohort"] = "patient"
        meta["original_label_name"] = meta["label_name"]
        meta["original_label_id"] = meta["label_id"]
        meta["cohort"] = "patient"
        meta["label_id"] = mapped.astype(np.int64)
        meta["label"] = meta["label_id"]
        meta["selected_by"] = "all_ecgs_one_label_per_ecg"
        selected[split] = meta
        all_meta.append(meta)

    combined = pd.concat(all_meta, ignore_index=True)
    if combined["csv_name"].duplicated().any():
        raise ValueError("A patient ECG file occurs more than once")
    if combined.groupby("patient_sn", dropna=False)["source_split"].nunique().gt(1).any():
        raise ValueError("Patient-level train/val/test leakage in source data")
    if combined.groupby("original_row_index")["label_id"].nunique().gt(1).any():
        raise ValueError("One original label row maps to conflicting labels")
    return selected


def sample_healthy(
    healthy_dir: Path, patient_ecgs: dict[str, pd.DataFrame], seed: int
) -> dict[str, pd.DataFrame]:
    frames = []
    for split in ("train", "val"):
        meta = pd.read_csv(healthy_dir / f"{split}_metadata.csv")
        meta = meta.reset_index().rename(columns={"index": "source_index"})
        meta["source_dataset"] = str(healthy_dir)
        meta["source_split"] = split
        frames.append(meta)
    pool = pd.concat(frames, ignore_index=True)
    pool = pool.loc[pool["cohort"].astype(str).str.lower().eq("healthy")].copy()
    patient_files = set(pd.concat(patient_ecgs.values())["csv_name"])
    pool = pool.loc[~pool["csv_name"].isin(patient_files)].copy()
    if pool["csv_name"].isna().any() or pool["csv_name"].duplicated().any():
        raise ValueError("Healthy ECG filenames are missing or duplicated")

    total_needed = sum(len(rows) for rows in patient_ecgs.values())
    if len(pool) < total_needed:
        raise ValueError(f"Need {total_needed} healthy ECGs, found {len(pool)}")
    sampled = pool.sample(n=total_needed, random_state=seed, replace=False).reset_index(
        drop=True
    )
    sampled["source_cohort"] = "healthy"
    sampled["cohort"] = "healthy"
    sampled["label_name"] = "健康"
    sampled["label_id"] = LABEL_MAP["健康"]
    sampled["label"] = LABEL_MAP["健康"]
    sampled["selected_by"] = f"random_ecg_sample_seed_{seed}"

    by_split = {}
    start = 0
    for split in SPLITS:
        end = start + len(patient_ecgs[split])
        by_split[split] = sampled.iloc[start:end].copy()
        start = end
    return by_split


def summarize(
    patient_by_split: dict[str, pd.DataFrame], healthy_by_split: dict[str, pd.DataFrame], args
) -> dict:
    patient = pd.concat(patient_by_split.values(), ignore_index=True)
    return {
        "patient_source": str(args.patient_dir),
        "healthy_source": str(args.healthy_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "patient_selection": "all ECGs; one label per ECG; no original_row_index deduplication",
        "healthy_selection": "one unique healthy ECG per patient ECG, without replacement",
        "patient_split_rule": "preserve source patient-grouped train/val/test",
        "healthy_patient_grouping_verified": False,
        "healthy_patient_grouping_note": "Healthy metadata has no patient ID; ECG filenames are unique, but patient-level separation cannot be checked.",
        "label_map": LABEL_MAP,
        "patient_ecgs": int(len(patient)),
        "healthy_ecgs": int(sum(map(len, healthy_by_split.values()))),
        "total_ecgs": int(2 * len(patient)),
        "original_label_rows": int(patient["original_row_index"].nunique()),
        "known_patient_ids": int(patient["patient_sn"].nunique()),
        "splits": {
            split: {
                "patient_ecgs": int(len(patient_by_split[split])),
                "healthy_ecgs": int(len(healthy_by_split[split])),
                "total_ecgs": int(len(patient_by_split[split]) + len(healthy_by_split[split])),
                "label_counts": {
                    name: int((patient_by_split[split]["label_name"] == name).sum())
                    if name != "健康"
                    else int(len(healthy_by_split[split]))
                    for name in LABEL_MAP
                },
            }
            for split in SPLITS
        },
    }


def write_split(
    split: str, patient: pd.DataFrame, healthy: pd.DataFrame, out_dir: Path, seed: int
) -> pd.DataFrame:
    meta = pd.concat([patient, healthy], ignore_index=True, sort=False)
    meta = meta.sample(
        frac=1.0, random_state=seed + {"train": 101, "val": 107, "test": 113}[split]
    ).reset_index(drop=True)
    labels = np.zeros((len(meta), len(LABEL_MAP)), dtype=np.int64)
    labels[np.arange(len(meta)), meta["label_id"].to_numpy(dtype=np.int64)] = 1
    np.save(out_dir / f"{split}_labels.npy", labels)

    for suffix in ARRAY_SUFFIXES:
        exemplar = np.load(
            Path(patient["source_dataset"].iloc[0]) / f"{split}_{suffix}.npy",
            mmap_mode="r",
        )
        dst = np.lib.format.open_memmap(
            out_dir / f"{split}_{suffix}.npy",
            mode="w+",
            dtype=exemplar.dtype,
            shape=(len(meta), *exemplar.shape[1:]),
        )
        indexed = meta.reset_index().rename(columns={"index": "out_index"})
        for (src_dir, src_split), group in indexed.groupby(
            ["source_dataset", "source_split"], sort=False
        ):
            src = np.load(Path(src_dir) / f"{src_split}_{suffix}.npy", mmap_mode="r")
            output_indices = group["out_index"].to_numpy(dtype=np.int64)
            source_indices = group["source_index"].to_numpy(dtype=np.int64)
            for start in range(0, len(group), 128):
                end = start + 128
                dst[output_indices[start:end]] = src[source_indices[start:end]]
        dst.flush()
        del dst

    meta["split"] = split
    meta.to_csv(out_dir / f"{split}_metadata.csv", index=False)
    return meta


def main() -> None:
    args = parse_args()
    args.patient_dir = args.patient_dir.resolve()
    args.healthy_dir = args.healthy_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    patient = load_patient_ecgs(args.patient_dir)
    healthy = sample_healthy(args.healthy_dir, patient, args.seed)
    summary = summarize(patient, healthy, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    all_meta = []
    for split in SPLITS:
        all_meta.append(write_split(split, patient[split], healthy[split], args.output_dir, args.seed))
        print(f"Wrote {split}: {len(all_meta[-1])} ECGs", flush=True)
    pd.concat(all_meta, ignore_index=True).to_csv(
        args.output_dir / "matched_metadata.csv", index=False
    )
    (args.output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "label_mapping.json").write_text(
        json.dumps(LABEL_MAP, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
