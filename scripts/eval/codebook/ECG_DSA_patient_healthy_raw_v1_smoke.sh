#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-heartlang/bin/python}"
DEVICE="${DEVICE:-mps}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-3832}"

"${PYTHON_BIN}" scripts/audit_codebook.py \
  --checkpoint_path checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v1_smoke/checkpoint-1.pth \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v1 \
  --split val \
  --output_dir outputs/codebook_eval/ECG_DSA_patient_healthy_raw_v1_smoke/val \
  --max_samples "${MAX_SAMPLES}" \
  --reconstruction_max_samples "${MAX_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --codebook_size 8192 \
  --codebook_dim 128 \
  --device "${DEVICE}" \
  --top_heatmap_codes 200 \
  --top_prototype_codes 20 \
  --top_associated_codes 20 \
  --prototype_overlays 20 \
  --pairwise_sample_size 2048
