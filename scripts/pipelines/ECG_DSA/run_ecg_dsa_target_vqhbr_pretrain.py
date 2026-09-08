import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.project_paths import ECG_DSA_RAW_SOURCE_DIR


DEFAULT_ZIP_NAMES = [
    "data.zip",
    "ECG_DSA20260324xml.zip",
    "上次遗漏的ECG.zip",
]
DEFAULT_WAVEFORM_OUTPUT_DIR = Path(
    "datasets/ecg_datasets/ECG_DSA/pretrain_waveforms/target_raw_v1"
)
DEFAULT_QRS_OUTPUT_DIR = Path("datasets/ecg_datasets/ECG_DSA/pretrain_qrs/target_raw_v1")
DEFAULT_LABEL_COLS = [
    "overall_max_stenosis",
    "lm_max_stenosis",
    "lad_max_stenosis",
    "lca_max_stenosis",
    "rca_max_stenosis",
]
PASSTHROUGH_FILES = [
    "all_metadata.csv",
    "matched_metadata.csv",
    "missing_metadata.csv",
    "train_metadata.csv",
    "val_metadata.csv",
    "train_aux_regression_labels.npy",
    "val_aux_regression_labels.npy",
    "prepare_summary.json",
]
REQUIRED_QRS_FILES = [
    "train_data.npy",
    "train_data_in_chans.npy",
    "train_data_in_times.npy",
    "val_data.npy",
    "val_data_in_chans.npy",
    "val_data_in_times.npy",
]


def default_zip_paths():
    return [ECG_DSA_RAW_SOURCE_DIR / name for name in DEFAULT_ZIP_NAMES]


def run_step(cmd, env=None):
    print(f"\n>>> {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def build_env():
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def safe_reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_waveforms(args, env):
    cmd = [
        sys.executable,
        "scripts/data_prep/ECG_DSA/prepare_ecg_dsa_target_pretrain_batch.py",
        "--zip_path",
        *[str(path) for path in args.zip_path],
        "--label_xlsx",
        str(args.label_xlsx),
        "--output_dir",
        str(args.waveform_output_dir),
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--target_length",
        str(args.target_length),
        "--expected_record_count",
        str(args.expected_record_count),
        "--expected_matched_count",
        str(args.expected_matched_count),
        "--label_cols",
        *args.label_cols,
    ]
    if args.require_all_aux_labels:
        cmd.append("--require_all_aux_labels")
    if args.reset_output_dir:
        cmd.append("--reset_output_dir")
    run_step(cmd, env)


def tokenize_qrs(args, env):
    cmd = [
        sys.executable,
        "scripts/data_prep/ECG_DSA/tokenize_pretrain_qrs.py",
        "--input_dir",
        str(args.waveform_output_dir),
        "--output_dir",
        str(args.qrs_output_dir),
        "--splits",
        "train",
        "val",
    ]
    run_step(cmd, env)


def copy_passthrough_files(waveform_dir: Path, qrs_dir: Path):
    qrs_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in PASSTHROUGH_FILES:
        source = waveform_dir / name
        if source.exists():
            shutil.copy2(source, qrs_dir / name)
            copied.append(name)
    return copied


def inspect_split(qrs_dir: Path, split: str) -> dict:
    data = np.load(qrs_dir / f"{split}_data.npy", mmap_mode="r")
    in_chans = np.load(qrs_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
    in_times = np.load(qrs_dir / f"{split}_data_in_times.npy", mmap_mode="r")
    aux = np.load(qrs_dir / f"{split}_aux_regression_labels.npy", mmap_mode="r")

    return {
        "sample_count": int(len(data)),
        "data_shape": [int(dim) for dim in data.shape],
        "in_chans_shape": [int(dim) for dim in in_chans.shape],
        "in_times_shape": [int(dim) for dim in in_times.shape],
        "aux_shape": [int(dim) for dim in aux.shape],
    }


def verify_qrs_output(qrs_dir: Path):
    missing_files = [name for name in REQUIRED_QRS_FILES if not (qrs_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing QRS output files: {missing_files}")

    summary = {
        "required_files": REQUIRED_QRS_FILES,
        "passthrough_files": [name for name in PASSTHROUGH_FILES if (qrs_dir / name).exists()],
        "train": inspect_split(qrs_dir, "train"),
        "val": inspect_split(qrs_dir, "val"),
    }
    with open(qrs_dir / "data_check_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a target-table-only ECG_DSA QRS pretraining dataset with auxiliary regression labels."
    )
    parser.add_argument(
        "--zip_path",
        type=Path,
        nargs="+",
        default=default_zip_paths(),
        help="Exact ECG_DSA ZIP files to include.",
    )
    parser.add_argument(
        "--label_xlsx",
        type=Path,
        default=ECG_DSA_RAW_SOURCE_DIR / "ECG-DSA-target.xlsx",
        help="ECG_DSA target Excel file.",
    )
    parser.add_argument(
        "--waveform_output_dir",
        type=Path,
        default=DEFAULT_WAVEFORM_OUTPUT_DIR,
        help="Output directory for waveform pretraining data.",
    )
    parser.add_argument(
        "--qrs_output_dir",
        type=Path,
        default=DEFAULT_QRS_OUTPUT_DIR,
        help="Output directory for tokenized QRS pretraining data.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_length", type=int, default=1000)
    parser.add_argument("--expected_record_count", type=int, default=17662)
    parser.add_argument("--expected_matched_count", type=int, default=17661)
    parser.add_argument("--label_cols", nargs="+", default=DEFAULT_LABEL_COLS)
    parser.add_argument("--require_all_aux_labels", action="store_true", default=False)
    parser.add_argument("--reset_output_dir", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    env = build_env()
    args.zip_path = [path.resolve() for path in args.zip_path]
    args.label_xlsx = args.label_xlsx.resolve()
    args.waveform_output_dir = args.waveform_output_dir.resolve()
    args.qrs_output_dir = args.qrs_output_dir.resolve()

    if args.reset_output_dir:
        safe_reset_dir(args.waveform_output_dir)
        safe_reset_dir(args.qrs_output_dir)
    else:
        args.waveform_output_dir.mkdir(parents=True, exist_ok=True)
        args.qrs_output_dir.mkdir(parents=True, exist_ok=True)

    prepare_waveforms(args, env)
    tokenize_qrs(args, env)
    copied = copy_passthrough_files(args.waveform_output_dir, args.qrs_output_dir)
    summary = verify_qrs_output(args.qrs_output_dir)

    print("\nCopied passthrough files:", copied, flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
