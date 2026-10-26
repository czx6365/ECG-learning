#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

BASE_CKPT="checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth"
NUMERIC8_CKPT="checkpoints/vqhbr/morphology_v2_numeric8_1024_dim64_bs256_e50/checkpoint-best.pth"
NEW_CKPT="checkpoints/vqhbr/morphology_v2_stable4_contrast_diverse_1024_dim64_bs256_e50/checkpoint-50.pth"
DATA="datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct"
PHYS="results/ecg_physiology_features_expanded_v1/patient_healthy_raw_v2_clean_le5pct"
OUT="results/morphology_v2_stable4_contrast_diverse_unified_val_shift4"

for checkpoint in "$BASE_CKPT" "$NUMERIC8_CKPT" "$NEW_CKPT"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
done

mkdir -p "$OUT"

.venv-heartlang/bin/python scripts/eval/codebook/evaluate_morphology_only_comparison.py \
  --run "Morphology-V2,morphology_v2,$BASE_CKPT" \
  --run "Numeric8,morphology_v2,$NUMERIC8_CKPT" \
  --run "Stable4-Contrast-Diverse,morphology_v2,$NEW_CKPT" \
  --data_path "$DATA" \
  --split val \
  --output_dir "$OUT" \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --batch_size 256 \
  --max_shift 4 \
  --corr_threshold 0.9 \
  --device cuda

.venv-heartlang/bin/python scripts/eval/codebook/evaluate_sid_continuous_variance.py \
  --phys_dir "$PHYS" \
  --split val \
  --run "Morphology-V2,$OUT/Morphology-V2/val_sid_assignments.npy" \
  --run "Numeric8,$OUT/Numeric8/val_sid_assignments.npy" \
  --run "Stable4-Contrast-Diverse,$OUT/Stable4-Contrast-Diverse/val_sid_assignments.npy" \
  --output_dir "$OUT/continuous_variance_stable4" \
  --codebook_size 1024 \
  --exclude_features qrs_duration_ms st_deviation st_slope_per_s t_peak_time_ms rr_interval_ms

.venv-heartlang/bin/python scripts/eval/codebook/evaluate_ze_zq_feature_probe.py \
  --run "Morphology-V2,$BASE_CKPT" \
  --run "Numeric8,$NUMERIC8_CKPT" \
  --run "Stable4-Contrast-Diverse,$NEW_CKPT" \
  --data_dir "$DATA" \
  --phys_dir "$PHYS" \
  --output_dir "$OUT/ze_zq_stable4_probe" \
  --batch_records 128 \
  --max_train_tokens 250000 \
  --max_val_tokens 250000 \
  --ridge_alpha 0.001 \
  --device cuda

echo "Stable4 contrast/diversity unified evaluation complete: $OUT"
