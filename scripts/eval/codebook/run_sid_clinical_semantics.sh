#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

.venv-heartlang/bin/python scripts/eval/codebook/evaluate_sid_clinical_semantics.py \
  --run Morphology-V2,morphology_v2,checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth \
  --run V1+Phys,morphology_v2,checkpoints/vqhbr/morphology_v1_plus_physiology_1024_dim64_bs256_e50/checkpoint-best.pth \
  --run NoContinuous,morphology_v2,checkpoints/vqhbr/morphology_v2_no_continuous_1024_dim64_bs256_e50/checkpoint-best.pth \
  --run H-clean,contextual_vqhbr,checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly/checkpoint-50.pth \
  --data_path datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675 \
  --split test \
  --output_dir results/sid_clinical_semantics \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --batch_size 256 \
  --min_heartbeats 100 \
  --min_patients 20 \
  --max_patient_dominance 0.25 \
  --fdr_alpha 0.05 \
  --bootstrap 1000 \
  --max_pairs 20000 \
  --top_n 50 \
  --waveforms_per_sid 20 \
  --seed 20260922 \
  --device cuda
