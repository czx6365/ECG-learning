#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

TRAIN_OUT="checkpoints/vqhbr/curriculum_v2_1024_dim64_bs256_e100"
BASE_CKPT="checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth"
CURR_CKPT="$TRAIN_OUT/checkpoint-100.pth"
DATA="datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct"
PHYS_CACHE="results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct"
OUT="results/morphology_v2_codebook/MorphologyV2_vs_CurriculumV2_val_shift4"

if [ ! -f "$CURR_CKPT" ]; then
  echo "Missing completed Curriculum-V2 checkpoint: $CURR_CKPT" >&2
  exit 1
fi
if [ ! -f "$BASE_CKPT" ]; then
  echo "Missing Morphology-V2 baseline checkpoint: $BASE_CKPT" >&2
  exit 1
fi

mkdir -p "$OUT"

.venv-heartlang/bin/python scripts/eval/codebook/evaluate_morphology_only_comparison.py \
  --run "Morphology-V2,morphology_v2,$BASE_CKPT" \
  --run "Curriculum-V2,morphology_v2,$CURR_CKPT" \
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
  --v2_sid "$OUT/Curriculum-V2/val_sid_assignments.npy" \
  --v1_name "Morphology-V2" \
  --v2_name "Curriculum-V2" \
  --output_dir "$OUT" \
  --split val \
  --max_shift 4 \
  --corr_threshold 0.9 \
  --codebook_size 1024

echo "Curriculum-V2 comparison complete: $OUT"
