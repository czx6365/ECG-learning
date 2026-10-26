#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

export C2_INDEX_DIR="datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict"
export C2_INIT_CHECKPOINT="checkpoints/curriculum_v1/c1_word_recon_vq_1024_dim64_bs256_e50/checkpoint-best.pth"
DATA="datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct"
CACHE="results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct"
OUT="checkpoints/curriculum_v1/c2_pattern_only_from_c1_e50"
LOG="log/curriculum_v1/c2_pattern_only_from_c1_e50"

.venv-heartlang/bin/python scripts/curriculum_v1/verify_frozen_split.py

if [[ -e "$OUT/checkpoint.pth" || -e "$OUT/log.txt" ]]; then
  echo "Refusing to overwrite existing corrected C2 run: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT" "$LOG"

.venv-heartlang/bin/python scripts/curriculum_v1/train_c2_from_c1.py \
  --pretrain_data_dir "$DATA" \
  --phys_cache_dir "$CACHE" \
  --output_dir "$OUT" \
  --log_dir "$LOG" \
  --batch_size 256 \
  --epochs 50 \
  --save_ckpt_freq 25 \
  --val_freq 25 \
  --codebook_n_emd 1024 \
  --codebook_emd_dim 64 \
  --token_len 96 \
  --fs 100 \
  --phys_head_source z_q \
  --loss_schedule static \
  --lambda_waveform 1.0 \
  --lambda_corr 0.0 \
  --lambda_deriv 0.0 \
  --lambda_vq 1.0 \
  --lambda_phys 0.2 \
  --discrete_alpha 1.0 \
  --continuous_alpha 0.0 \
  --continuous_loss_reduction element \
  --continuous_clip_z 0.0 \
  --ema_decay 0.99 \
  --opt adamw \
  --lr 2.5e-5 \
  --min_lr 5e-6 \
  --warmup_epochs 2 \
  --weight_decay 1e-4 \
  --weight_decay_end 1e-4 \
  --device cuda \
  --seed 0 \
  --num_workers 10
