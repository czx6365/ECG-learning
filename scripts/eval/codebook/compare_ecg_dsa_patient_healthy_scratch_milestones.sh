#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-heartlang/bin/python}"
DEVICE="${DEVICE:-mps}"

"${PYTHON_BIN}" scripts/eval/codebook/compare_ecg_dsa_patient_healthy_milestones.py \
  --checkpoint_dir checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v1_scratch_milestones \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v1 \
  --output_root outputs/codebook_eval/ECG_DSA_patient_healthy_raw_v1_scratch_milestones \
  --epochs 1 2 3 5 10 \
  --split val \
  --max_samples 3832 \
  --batch_size 8 \
  --device "${DEVICE}" \
  "$@"
