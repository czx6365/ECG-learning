from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the immutable Curriculum V1 data split.")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/curriculum_v1/data_contract.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    data_dir = Path(contract["main_dataset_dir"])

    manifest_path = data_dir / "split_manifest.csv"
    actual_manifest_hash = sha256(manifest_path)
    expected_manifest_hash = contract["split_manifest_sha256"]
    if actual_manifest_hash != expected_manifest_hash:
        raise RuntimeError(
            f"Frozen manifest hash mismatch: expected {expected_manifest_hash}, got {actual_manifest_hash}"
        )

    entities: dict[str, set[str]] = {}
    audit: dict[str, object] = {
        "contract": str(args.contract),
        "dataset": str(data_dir),
        "manifest_sha256": actual_manifest_hash,
        "splits": {},
    }

    for split in SPLITS:
        metadata = pd.read_csv(data_dir / f"{split}_metadata.csv")
        labels = np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")
        data = np.load(data_dir / f"{split}_data.npy", mmap_mode="r")
        chans = np.load(data_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
        times = np.load(data_dir / f"{split}_data_in_times.npy", mmap_mode="r")

        expected_size = int(contract["expected_split_sizes"][split])
        lengths = {len(metadata), len(labels), len(data), len(chans), len(times)}
        if lengths != {expected_size}:
            raise RuntimeError(f"Length mismatch in {split}: {sorted(lengths)}; expected {expected_size}")
        if labels.shape != (expected_size, 4):
            raise RuntimeError(f"Unexpected label shape in {split}: {labels.shape}")
        if not np.all(labels.sum(axis=1) == 1):
            raise RuntimeError(f"Non one-hot labels found in {split}")
        if not np.array_equal(labels.argmax(axis=1), metadata["label_id"].to_numpy(dtype=np.int64)):
            raise RuntimeError(f"Labels and metadata disagree in {split}")
        if metadata["entity_id"].isna().any():
            raise RuntimeError(f"Missing entity_id in {split}")

        entities[split] = set(metadata["entity_id"].astype(str))
        healthy = metadata[metadata["cohort"].eq("healthy")]
        if healthy["entity_id"].duplicated().any():
            raise RuntimeError(f"Duplicate healthy identity inside {split}")

        audit["splits"][split] = {
            "samples": expected_size,
            "entities": len(entities[split]),
            "patients": int(metadata["cohort"].eq("patient").sum()),
            "healthy": int(metadata["cohort"].eq("healthy").sum()),
            "label_counts": metadata["label_name"].value_counts().to_dict(),
        }

    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        count = len(entities[left].intersection(entities[right]))
        overlap[f"{left}_{right}"] = count
        if count:
            raise RuntimeError(f"Identity leakage between {left} and {right}: {count}")
    audit["cross_split_entity_overlap"] = overlap

    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
