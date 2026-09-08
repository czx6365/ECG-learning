#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-heartlang/bin/python}"
DEVICE="${DEVICE:-mps}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-3832}"

"${PYTHON_BIN}" scripts/eval/codebook/compare_ecg_dsa_patient_healthy_milestones.py \
  --checkpoint_dir checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v1_milestones \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v1 \
  --output_root outputs/codebook_eval/ECG_DSA_patient_healthy_raw_v1_milestones \
  --epochs 1 3 5 10 \
  --split val \
  --max_samples "${MAX_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --python_bin "${PYTHON_BIN}"
