#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

EXP="morphology_v2_numeric9_1024_dim64_bs256_e50"
OUT="checkpoints/vqhbr/${EXP}"
LOG="log/vqhbr/${EXP}"
CACHE="results/ecg_physiology_features_expanded_v1/patient_healthy_raw_v2_clean_le5pct"

if [[ -e "$OUT" || -e "$LOG" ]]; then
  echo "Refusing to overwrite an existing experiment: $EXP" >&2
  exit 1
fi
mkdir -p "$OUT" "$LOG"

nohup .venv-heartlang/bin/python scripts/experiments/train_morphology_v2_physaware_vq.py \
  --pretrain_data_dir datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --phys_cache_dir "$CACHE" \
  --phys_schema expanded_v1 \
  --output_dir "$OUT" \
  --log_dir "$LOG" \
  --batch_size 256 \
  --epochs 50 \
  --save_ckpt_freq 50 \
  --val_freq 10 \
  --codebook_n_emd 1024 \
  --codebook_emd_dim 64 \
  --token_len 96 \
  --fs 100 \
  --phys_head_source z_q \
  --lambda_waveform 1.0 \
  --lambda_corr 0.2 \
  --lambda_deriv 0.1 \
  --lambda_vq 1.0 \
  --lambda_phys 0.05 \
  --discrete_alpha 0.0 \
  --continuous_alpha 1.0 \
  --continuous_loss_reduction feature \
  --continuous_clip_z 5.0 \
  --ema_decay 0.99 \
  --opt adamw \
  --lr 5e-5 \
  --min_lr 1e-5 \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --weight_decay_end 1e-4 \
  --device cuda \
  --num_workers 10 \
  > "$OUT/train_stdout.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/train.pid"
echo "Started $EXP with PID $(cat "$OUT/train.pid")"
