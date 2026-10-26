#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

.venv-heartlang/bin/python scripts/eval/codebook/evaluate_morphology_only_comparison.py \
  --run "H-clean,contextual_vqhbr,checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly/checkpoint-50.pth" \
  --run "Old-Morphology-only,morphology_only,checkpoints/vqhbr/morphology_only_v2_clean_1024_dim64_bs256_e50_finalonly/checkpoint-50.pth" \
  --run "C1-strict,morphology_only,checkpoints/curriculum_v1/c1_word_recon_vq_1024_dim64_bs256_e50/checkpoint-best.pth" \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --split val \
  --output_dir results/curriculum_v1/c1_word_recon_vq_1024_dim64_bs256_e50/historical_hclean_val_comparison \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --batch_size 256 \
  --max_shift 4 \
  --corr_threshold 0.9 \
  --device cuda
