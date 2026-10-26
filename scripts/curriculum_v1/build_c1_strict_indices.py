#!/usr/bin/env python3
"""Build a leakage-controlled index view for Curriculum V1 C1 training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 0
VAL_FRACTION = 0.10
BASE_DIR = Path("datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct")
LOCKED_DIR = Path("datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1")
OUT_DIR = Path("datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict")
SOURCE_CODE = {"train": 0, "val": 1}


def entity_id(frame: pd.DataFrame) -> pd.Series:
    stem = frame["csv_name"].astype(str).str.replace(r"\.csv$", "", regex=True)
    suffix = stem.str.rsplit("_", n=1).str[-1]
    if suffix.isna().any() or suffix.eq("").any():
        raise RuntimeError("Unable to derive an identity suffix from csv_name")
    return frame["cohort"].astype(str).str.lower() + ":" + suffix


def read_base() -> pd.DataFrame:
    parts = []
    for split in ("train", "val"):
        frame = pd.read_csv(BASE_DIR / f"{split}_metadata.csv")
        frame["source_split"] = split
        frame["source_index"] = np.arange(len(frame), dtype=np.int64)
        parts.append(frame)
    full = pd.concat(parts, ignore_index=True)
    full["entity_id"] = entity_id(full)
    return full


def read_locked_entities() -> dict[str, set[str]]:
    result = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(LOCKED_DIR / f"{split}_metadata.csv")
        result[split] = set(entity_id(frame))
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = result[left].intersection(result[right])
        if overlap:
            raise RuntimeError(f"Locked suffix identities overlap between {left} and {right}: {len(overlap)}")
    return result


def choose_validation_entities(candidates: pd.DataFrame) -> set[str]:
    rng = np.random.default_rng(SEED)
    selected: set[str] = set()
    entity_cohort = candidates[["entity_id", "cohort"]].drop_duplicates("entity_id")
    for cohort, group in entity_cohort.groupby("cohort", sort=True):
        values = np.array(sorted(group["entity_id"].astype(str)), dtype=object)
        rng.shuffle(values)
        count = int(round(len(values) * VAL_FRACTION))
        selected.update(values[:count].tolist())
    return selected


def write_index(split: str, frame: pd.DataFrame) -> None:
    frame = frame.sort_values(["source_split", "source_index"], kind="stable").reset_index(drop=True)
    np.save(OUT_DIR / f"{split}_source_code.npy", frame["source_split"].map(SOURCE_CODE).to_numpy(np.uint8))
    np.save(OUT_DIR / f"{split}_source_index.npy", frame["source_index"].to_numpy(np.int64))
    frame.to_csv(OUT_DIR / f"{split}_metadata.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    full = read_base()
    locked = read_locked_entities()

    excluded = locked["val"].union(locked["test"])
    eligible = full[~full["entity_id"].isin(excluded)].copy()
    forced_train = locked["train"]
    free = eligible[~eligible["entity_id"].isin(forced_train)]
    val_entities = choose_validation_entities(free)

    c1_val = eligible[eligible["entity_id"].isin(val_entities)].copy()
    c1_train = eligible[~eligible["entity_id"].isin(val_entities)].copy()
    c1_train["curriculum_split"] = "train"
    c1_val["curriculum_split"] = "val"

    train_entities = set(c1_train["entity_id"])
    val_entities_actual = set(c1_val["entity_id"])
    if train_entities.intersection(val_entities_actual):
        raise RuntimeError("C1 train and validation identities overlap")
    if train_entities.intersection(excluded) or val_entities_actual.intersection(excluded):
        raise RuntimeError("Locked downstream validation/test identity leaked into C1")
    missing_forced = forced_train.difference(train_entities)
    if missing_forced:
        raise RuntimeError(f"Locked downstream training identities missing from C1 train: {len(missing_forced)}")

    write_index("train", c1_train)
    write_index("val", c1_val)
    manifest = pd.concat([c1_train, c1_val], ignore_index=True)
    manifest.to_csv(OUT_DIR / "assignment_manifest.csv", index=False)
    manifest_hash = hashlib.sha256((OUT_DIR / "assignment_manifest.csv").read_bytes()).hexdigest()

    summary = {
        "version": "curriculum_v1_c1_strict",
        "seed": SEED,
        "validation_fraction_by_cohort": VAL_FRACTION,
        "base_data_dir": str(BASE_DIR),
        "locked_fourclass_dir": str(LOCKED_DIR),
        "source_code": SOURCE_CODE,
        "identity_key": "lowercase cohort + ':' + csv_name suffix after final underscore",
        "rules": [
            "exclude locked four-class val/test identities from all C1 data",
            "force locked four-class train identities into C1 train",
            "split remaining identities, not records, into C1 train/val",
        ],
        "base_records": len(full),
        "excluded_records": int(full["entity_id"].isin(excluded).sum()),
        "train_records": len(c1_train),
        "val_records": len(c1_val),
        "train_entities": len(train_entities),
        "val_entities": len(val_entities_actual),
        "cross_split_entity_overlap": len(train_entities.intersection(val_entities_actual)),
        "locked_val_test_overlap_in_c1": len((train_entities | val_entities_actual).intersection(excluded)),
        "cohort_records": {
            "train": c1_train["cohort"].value_counts().to_dict(),
            "val": c1_val["cohort"].value_counts().to_dict(),
        },
        "assignment_manifest_sha256": manifest_hash,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
