#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

.venv-heartlang/bin/python scripts/eval/codebook/evaluate_codebook_alignment_robustness.py \
  --run Morphology-V2,morphology_v2,checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth \
  --run V1+Phys,morphology_v2,checkpoints/vqhbr/morphology_v1_plus_physiology_1024_dim64_bs256_e50/checkpoint-best.pth \
  --run NoContinuous,morphology_v2,checkpoints/vqhbr/morphology_v2_no_continuous_1024_dim64_bs256_e50/checkpoint-best.pth \
  --run H-clean,contextual_vqhbr,checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly/checkpoint-50.pth \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --split val \
  --output_dir results/codebook_alignment_robustness \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --batch_size 256 \
  --pool_size 60000 \
  --anchors 5000 \
  --top_k 30 \
  --feature_points 24 \
  --shift_windows 2 4 6 \
  --thresholds 0.90 0.95 0.98 \
  --random_pairs 200000 \
  --examples_per_model 20 \
  --seed 20260921 \
  --device cuda
