#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

DATA="datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct"
INDEX="datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict"
OUT="checkpoints/curriculum_v1/c1_word_recon_vq_1024_dim64_bs256_e50"
LOG="log/curriculum_v1/c1_word_recon_vq_1024_dim64_bs256_e50"

.venv-heartlang/bin/python scripts/curriculum_v1/verify_frozen_split.py

if [[ -e "$OUT/checkpoint.pth" || -e "$OUT/log.txt" ]]; then
  echo "Refusing to overwrite existing C1 run: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT" "$LOG"

.venv-heartlang/bin/python scripts/curriculum_v1/train_c1_word_vq.py \
  --pretrain_data_dir "$DATA" \
  --split_index_dir "$INDEX" \
  --output_dir "$OUT" \
  --log_dir "$LOG" \
  --batch_size 256 \
  --epochs 50 \
  --save_ckpt_freq 10 \
  --val_freq 5 \
  --codebook_n_emd 1024 \
  --codebook_emd_dim 64 \
  --token_len 96 \
  --ema_decay 0.99 \
  --opt adamw \
  --lr 5e-5 \
  --min_lr 1e-5 \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --weight_decay_end 1e-4 \
  --device cuda \
  --seed 0 \
  --num_workers 10
