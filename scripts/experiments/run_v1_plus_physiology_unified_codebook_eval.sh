#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

BASE_CKPT="checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth"
V1PHYS_CKPT="checkpoints/vqhbr/morphology_v1_plus_physiology_1024_dim64_bs256_e50/checkpoint-best.pth"
DATA="datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct"
PHYS_CACHE="results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct"
OUT="results/information_flow/codebook/MorphologyV2_vs_V1PlusPhysiology_val_shift4"

for checkpoint in "$BASE_CKPT" "$V1PHYS_CKPT"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
done
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing output: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

.venv-heartlang/bin/python scripts/eval/codebook/evaluate_morphology_only_comparison.py \
  --run "Morphology-V2,morphology_v2,$BASE_CKPT" \
  --run "V1-Plus-Physiology,morphology_v2,$V1PHYS_CKPT" \
  --data_path "$DATA" \
  --split val \
  --output_dir "$OUT" \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --batch_size 256 \
  --max_shift 4 \
  --corr_threshold 0.9 \
  --device cuda

.venv-heartlang/bin/python scripts/eval/codebook/evaluate_high_corr_phys_sid_disagreement.py \
  --data_path "$DATA" \
  --phys_cache "$PHYS_CACHE" \
  --v1_sid "$OUT/Morphology-V2/val_sid_assignments.npy" \
  --v2_sid "$OUT/V1-Plus-Physiology/val_sid_assignments.npy" \
  --v1_name "Morphology-V2" \
  --v2_name "V1-Plus-Physiology" \
  --output_dir "$OUT/high_corr_phys_analysis" \
  --split val \
  --max_shift 4 \
  --corr_threshold 0.9 \
  --codebook_size 1024

echo "V1 + Physiology unified codebook evaluation complete: $OUT"
