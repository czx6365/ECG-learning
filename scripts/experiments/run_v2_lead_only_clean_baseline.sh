#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

EXP="morphology_v2_leadV2_only_clean_1024_dim64_bs256_e50"
TRAIN_OUT="checkpoints/vqhbr/${EXP}"
TRAIN_LOG="log/vqhbr/${EXP}"
DATA="datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct"
PHYS_CACHE="results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct"
BASE_CKPT="checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth"
EVAL_OUT="results/morphology_v2_codebook/MorphologyV2_full_vs_leadV2only_val"

mkdir -p "$TRAIN_OUT" "$TRAIN_LOG" "$EVAL_OUT"

.venv-heartlang/bin/python scripts/experiments/train_morphology_v2_physaware_vq.py \
  --pretrain_data_dir "$DATA" \
  --phys_cache_dir "$PHYS_CACHE" \
  --output_dir "$TRAIN_OUT" \
  --log_dir "$TRAIN_LOG" \
  --lead_id 8 \
  --batch_size 256 \
  --epochs 50 \
  --save_ckpt_freq 25 \
  --val_freq 10 \
  --codebook_n_emd 1024 \
  --codebook_emd_dim 64 \
  --token_len 96 \
  --fs 100 \
  --phys_head_source z_q \
  --loss_schedule static \
  --lambda_waveform 1.0 \
  --lambda_corr 0.2 \
  --lambda_deriv 0.1 \
  --lambda_vq 1.0 \
  --lambda_phys 0.2 \
  --continuous_alpha 0.25 \
  --ema_decay 0.99 \
  --opt adamw \
  --lr 5e-5 \
  --min_lr 1e-5 \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --weight_decay_end 1e-4 \
  --device cuda \
  --num_workers 10 \
  --write_assignments

.venv-heartlang/bin/python scripts/eval/codebook/evaluate_morphology_only_comparison.py \
  --run "Morphology-V2-full,morphology_v2,$BASE_CKPT" \
  --run "V2-lead-only,morphology_v2,$TRAIN_OUT/checkpoint-50.pth" \
  --data_path "$DATA" \
  --split val \
  --output_dir "$EVAL_OUT" \
  --lead_id 8 \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --batch_size 256 \
  --max_shift 4 \
  --corr_threshold 0.9 \
  --device cuda

.venv-heartlang/bin/python scripts/eval/codebook/evaluate_high_corr_phys_sid_disagreement.py \
  --data_path "$DATA" \
  --phys_cache "$PHYS_CACHE" \
  --v1_sid "$EVAL_OUT/Morphology-V2-full/val_sid_assignments.npy" \
  --v2_sid "$EVAL_OUT/V2-lead-only/val_sid_assignments.npy" \
  --v1_name "Morphology-V2-full" \
  --v2_name "V2-lead-only" \
  --output_dir "$EVAL_OUT" \
  --split val \
  --lead_id 8 \
  --max_shift 4 \
  --corr_threshold 0.9 \
  --codebook_size 1024

echo "V2-lead-only clean baseline complete: $EVAL_OUT"
