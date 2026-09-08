import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Allow running as `python scripts/pipelines/ECG_DSA/run_ecg_dsa_vqhbr_pretrain.py`.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.project_paths import ECG_DSA_RAW_SOURCE_DIR
from scripts.data_prep.ECG_DSA.ecg_zip_utils import build_member_table


DEFAULT_ZIP_NAMES = [
    "data.zip",
    "ECG_DSA20260324xml.zip",
    "上次遗漏的ECG.zip",
]
DEFAULT_EXPECTED_CSV_COUNT = 32684
DEFAULT_WAVEFORM_OUTPUT_DIR = Path(
    "datasets/ecg_datasets/ECG_DSA/pretrain_waveforms/all_raw_v1"
)
DEFAULT_QRS_OUTPUT_DIR = Path("datasets/ecg_datasets/ECG_DSA/pretrain_qrs/all_raw_v1")
DEFAULT_TRAIN_OUTPUT_DIR = Path("checkpoints/vqhbr/ECG_DSA_all_raw_v1")
DEFAULT_TRAIN_LOG_DIR = Path("log/vqhbr/ECG_DSA_all_raw_v1")
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
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return env


def require_exists(path: Path, desc: str):
    if not path.exists():
        raise FileNotFoundError(f"{desc} does not exist: {path}")


def safe_reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_csv_scope(zip_paths: list[Path]) -> dict:
    metadata = build_member_table(zip_paths)
    per_zip_counts = {
        source_zip: int(count)
        for source_zip, count in metadata["source_zip"].value_counts().sort_index().items()
    }
    summary = {
        "zip_files": [path.name for path in zip_paths],
        "csv_count": int(len(metadata)),
        "per_zip_counts": per_zip_counts,
    }
    return summary


def prepare_waveforms(args, env):
    cmd = [
        sys.executable,
        "scripts/data_prep/ECG_DSA/prepare_ecg_zip_pretrain_batch.py",
        "--zip_path",
        *[str(path) for path in args.zip_path],
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
    ]
    run_step(cmd, env)


def tokenize_qrs(args, env):
    cmd = [
        sys.executable,
        "heartlang/preprocessing/qrs_tokenizer.py",
        "--dataset_name",
        "ECG_DSA",
        "--input_dirs",
        str(args.waveform_output_dir),
        "--output_dirs",
        str(args.qrs_output_dir),
        "--splits",
        "train",
        "val",
    ]
    run_step(cmd, env)


def inspect_split(qrs_dir: Path, split: str) -> dict:
    data = np.load(qrs_dir / f"{split}_data.npy", mmap_mode="r")
    in_chans = np.load(qrs_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
    in_times = np.load(qrs_dir / f"{split}_data_in_times.npy", mmap_mode="r")

    if len(data) != len(in_chans) or len(data) != len(in_times):
        raise ValueError(
            f"{split} split length mismatch: "
            f"data={len(data)}, in_chans={len(in_chans)}, in_times={len(in_times)}"
        )

    zero_sample_count = 0
    nonzero_token_count = 0
    chunk_size = 256
    for start in range(0, len(data), chunk_size):
        end = min(start + chunk_size, len(data))
        chunk = np.asarray(data[start:end])
        sample_nonzero = np.any(chunk != 0, axis=(1, 2))
        zero_sample_count += int((~sample_nonzero).sum())
        nonzero_token_count += int(np.any(chunk != 0, axis=2).sum())

    return {
        "sample_count": int(len(data)),
        "data_shape": [int(dim) for dim in data.shape],
        "data_dtype": str(data.dtype),
        "in_chans_shape": [int(dim) for dim in in_chans.shape],
        "in_chans_dtype": str(in_chans.dtype),
        "in_times_shape": [int(dim) for dim in in_times.shape],
        "in_times_dtype": str(in_times.dtype),
        "zero_sample_count": zero_sample_count,
        "nonzero_token_count": nonzero_token_count,
    }


def verify_qrs_output(qrs_dir: Path) -> dict:
    missing_files = [name for name in REQUIRED_QRS_FILES if not (qrs_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing QRS output files: {missing_files}")

    summary = {
        "required_files": REQUIRED_QRS_FILES,
        "train": inspect_split(qrs_dir, "train"),
        "val": inspect_split(qrs_dir, "val"),
    }
    summary_path = qrs_dir / "data_check_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def print_train_command(args):
    cmd = [
        args.torchrun_path,
        "--nnodes=1",
        f"--nproc_per_node={args.nproc_per_node}",
        "heartlang/cli/run_vqhbr_training.py",
        "--output_dir",
        str(args.train_output_dir),
        "--log_dir",
        str(args.train_log_dir),
        "--model",
        "vqhbr",
        "--pretrain_data_dirs",
        str(args.qrs_output_dir),
        "--codebook_n_emd",
        str(args.codebook_n_emd),
        "--codebook_emd_dim",
        str(args.codebook_emd_dim),
        "--quantize_kmeans_init",
        "--batch_size",
        str(args.batch_size),
        "--opt",
        "adamw",
        "--opt_betas",
        "0.9",
        "0.99",
        "--weight_decay",
        "1e-4",
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--epochs",
        str(args.epochs),
        "--save_ckpt_freq",
        str(args.save_ckpt_freq),
        "--world_size",
        str(args.world_size),
        "--val_freq",
        str(args.val_freq),
        "--lr",
        str(args.lr),
        "--min_lr",
        str(args.min_lr),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if args.init_checkpoint:
        cmd += ["--init_checkpoint", str(args.init_checkpoint)]
    print("\nSuggested training command:", flush=True)
    print(" ".join(cmd), flush=True)


def train_vqhbr(args, env):
    cmd = [
        args.torchrun_path,
        "--nnodes=1",
        f"--nproc_per_node={args.nproc_per_node}",
        "heartlang/cli/run_vqhbr_training.py",
        "--output_dir",
        str(args.train_output_dir),
        "--log_dir",
        str(args.train_log_dir),
        "--model",
        "vqhbr",
        "--pretrain_data_dirs",
        str(args.qrs_output_dir),
        "--codebook_n_emd",
        str(args.codebook_n_emd),
        "--codebook_emd_dim",
        str(args.codebook_emd_dim),
        "--quantize_kmeans_init",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--opt",
        "adamw",
        "--opt_betas",
        "0.9",
        "0.99",
        "--weight_decay",
        "1e-4",
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--epochs",
        str(args.epochs),
        "--save_ckpt_freq",
        str(args.save_ckpt_freq),
        "--world_size",
        str(args.world_size),
        "--val_freq",
        str(args.val_freq),
        "--lr",
        str(args.lr),
        "--min_lr",
        str(args.min_lr),
        "--device",
        args.device,
    ]
    if args.init_checkpoint:
        cmd += ["--init_checkpoint", str(args.init_checkpoint)]
    run_step(cmd, env)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ECG-DSA all-raw VQHBR pretraining data and validate the QRS output."
    )
    parser.add_argument(
        "--zip_path",
        type=Path,
        nargs="+",
        default=default_zip_paths(),
        help="Exact ECG_DSA ZIP files to include in VQHBR pretraining.",
    )
    parser.add_argument(
        "--expected_csv_count",
        type=int,
        default=DEFAULT_EXPECTED_CSV_COUNT,
        help="Expected number of CSV waveforms across all ZIP files.",
    )
    parser.add_argument(
        "--waveform_output_dir",
        type=Path,
        default=DEFAULT_WAVEFORM_OUTPUT_DIR,
        help="Output directory for unlabeled waveform pretraining data.",
    )
    parser.add_argument(
        "--qrs_output_dir",
        type=Path,
        default=DEFAULT_QRS_OUTPUT_DIR,
        help="Output directory for tokenized QRS pretraining data.",
    )
    parser.add_argument(
        "--train_output_dir",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT_DIR,
        help="Suggested checkpoint output directory for VQHBR training.",
    )
    parser.add_argument(
        "--train_log_dir",
        type=Path,
        default=DEFAULT_TRAIN_LOG_DIR,
        help="Suggested log directory for VQHBR training.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_length", type=int, default=1000)
    parser.add_argument("--nproc_per_node", type=int, default=8)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--save_ckpt_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--codebook_n_emd", type=int, default=8192)
    parser.add_argument("--codebook_emd_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torchrun_path", type=str, default="torchrun")
    parser.add_argument(
        "--init_checkpoint",
        type=Path,
        default=None,
        help="Optional VQ-HBR checkpoint used to initialize the codebook/model before training.",
    )
    parser.add_argument(
        "--skip_waveform_prepare",
        action="store_true",
        help="Reuse existing waveform arrays instead of rebuilding them from ZIP files.",
    )
    parser.add_argument(
        "--skip_qrs_tokenize",
        action="store_true",
        help="Reuse existing QRS token arrays instead of running QRS tokenization.",
    )
    parser.add_argument(
        "--run_training",
        action="store_true",
        help="Start VQ-HBR training after waveform/QRS preparation. By default only prepares data and prints the command.",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Remove and rebuild waveform/QRS output directories before processing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.zip_path = [path.resolve() for path in args.zip_path]
    args.waveform_output_dir = args.waveform_output_dir.resolve()
    args.qrs_output_dir = args.qrs_output_dir.resolve()
    args.train_output_dir = args.train_output_dir.resolve()
    args.train_log_dir = args.train_log_dir.resolve()
    if args.init_checkpoint is not None:
        args.init_checkpoint = args.init_checkpoint.resolve()

    for zip_path in args.zip_path:
        require_exists(zip_path, "ZIP file")

    csv_scope = collect_csv_scope(args.zip_path)
    print(json.dumps(csv_scope, indent=2, ensure_ascii=False), flush=True)
    if csv_scope["csv_count"] != args.expected_csv_count:
        raise ValueError(
            f"CSV count mismatch: expected {args.expected_csv_count}, "
            f"got {csv_scope['csv_count']}"
        )

    if args.force_rebuild:
        safe_reset_dir(args.waveform_output_dir)
        safe_reset_dir(args.qrs_output_dir)
    else:
        args.waveform_output_dir.mkdir(parents=True, exist_ok=True)
        args.qrs_output_dir.mkdir(parents=True, exist_ok=True)

    env = build_env()
    if not args.skip_waveform_prepare:
        prepare_waveforms(args, env)
    if not args.skip_qrs_tokenize:
        tokenize_qrs(args, env)
    check_summary = verify_qrs_output(args.qrs_output_dir)
    print(json.dumps(check_summary, indent=2, ensure_ascii=False), flush=True)
    if args.run_training:
        train_vqhbr(args, env)
    else:
        print_train_command(args)


if __name__ == "__main__":
    main()
