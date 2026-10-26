#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

TOKENIZER="checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth"

.venv-heartlang/bin/python scripts/experiments/generate_morphology_v2_sid_dataset.py \
  --source_data_dir datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2 \
  --output_dir datasets/ecg_datasets/ECG_DSA/pretrain_sid/morphology_v2_physaware_raw_v2 \
  --tokenizer_checkpoint "${TOKENIZER}" \
  --splits train val \
  --batch_size 512 \
  --device cuda \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --pad_sid 1024 \
  --overwrite

.venv-heartlang/bin/python scripts/experiments/generate_morphology_v2_sid_dataset.py \
  --source_data_dir datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675 \
  --output_dir datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_morphology_v2 \
  --tokenizer_checkpoint "${TOKENIZER}" \
  --splits train val test \
  --batch_size 512 \
  --device cuda \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --pad_sid 1024 \
  --overwrite
