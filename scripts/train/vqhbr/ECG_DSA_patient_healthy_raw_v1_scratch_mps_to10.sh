#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-heartlang/bin/python}"
DEVICE="${DEVICE:-mps}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"

"${PYTHON_BIN}" heartlang/cli/run_vqhbr_training.py \
  --output_dir checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v1_scratch_milestones \
  --log_dir log/vqhbr/ECG_DSA_patient_healthy_raw_v1_scratch_milestones \
  --model vqhbr \
  --pretrain_data_dirs datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v1 \
  --codebook_n_emd 8192 \
  --codebook_emd_dim 128 \
  --quantize_kmeans_init \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --opt adamw \
  --opt_betas 0.9 0.99 \
  --weight_decay 1e-4 \
  --warmup_epochs 1 \
  --epochs 10 \
  --save_ckpt_freq 1 \
  --world_size 1 \
  --val_freq 1 \
  --lr 1e-04 \
  --min_lr 1e-06 \
  --device "${DEVICE}" \
  --no_auto_resume
