#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

OUT="results/morphology_only_codebook/Hclean_vs_morphology_only_val_shift4"
mkdir -p "$OUT"

nohup .venv-heartlang/bin/python scripts/eval/codebook/evaluate_morphology_only_comparison.py \
  --run H_clean,contextual_vqhbr,checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly/checkpoint-50.pth \
  --run Morphology_only,morphology_only,checkpoints/vqhbr/morphology_only_v2_clean_1024_dim64_bs256_e50_finalonly/checkpoint-50.pth \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --split val \
  --output_dir "$OUT" \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --batch_size 256 \
  --max_shift 4 \
  --corr_threshold 0.9 \
  --device cuda \
  > "$OUT/eval_stdout.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/eval.pid"
cat "$OUT/eval.pid"
