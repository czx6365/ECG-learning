#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

SID_PRETRAIN_DATA="datasets/ecg_datasets/ECG_DSA/pretrain_sid/morphology_v2_physaware_raw_v2"
SID_DOWNSTREAM_DATA="datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_morphology_v2"
PRETRAIN_OUT="checkpoints/pretrain/HeartLangSID_MorphologyV2_SIDOnly_rawv2_e100"
PRETRAIN_LOG="log/pretrain/HeartLangSID_MorphologyV2_SIDOnly_rawv2_e100"
REG_OUT="checkpoints/finetune/ecg_dsa/regression_label4_HeartLangSID_MorphologyV2_SIDOnly_e100_linear"
CLS_OUT="checkpoints/finetune/ecg_dsa/classification_label4_HeartLangSID_MorphologyV2_SIDOnly_e100_linear_weighted_ce"

for required in \
  "$SID_PRETRAIN_DATA/train_sid.npy" \
  "$SID_PRETRAIN_DATA/val_sid.npy" \
  "$SID_DOWNSTREAM_DATA/train_sid.npy" \
  "$SID_DOWNSTREAM_DATA/val_sid.npy" \
  "$SID_DOWNSTREAM_DATA/test_sid.npy"; do
  if [ ! -f "$required" ]; then
    echo "Missing required SID data: $required" >&2
    exit 1
  fi
done

mkdir -p "$PRETRAIN_OUT" "$PRETRAIN_LOG" "$REG_OUT" "$CLS_OUT"

RESUME_ARGS=()
if [ -f "$PRETRAIN_OUT/checkpoint.pth" ]; then
  RESUME_ARGS=(--resume "$PRETRAIN_OUT/checkpoint.pth")
fi

.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_pretraining.py \
  --sid_data_dir "$SID_PRETRAIN_DATA" \
  --output_dir "$PRETRAIN_OUT" \
  --log_dir "$PRETRAIN_LOG" \
  --no_spatial_embedding \
  --no_temporal_embedding \
  --batch_size 256 \
  --epochs 100 \
  --save_ckpt_freq 50 \
  --val_freq 10 \
  --vocab_size 1024 \
  --pad_sid 1024 \
  --seq_len 256 \
  --embed_dim 768 \
  --depth 12 \
  --heads 8 \
  --mlp_dim 1024 \
  --mask_ratio 0.5 \
  --lr 5e-4 \
  --min_lr 1e-5 \
  --warmup_epochs 5 \
  --weight_decay 0.05 \
  --clip_grad 3.0 \
  --num_workers 8 \
  --device cuda \
  "${RESUME_ARGS[@]}"

.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_regression.py \
  --sid_data_dir "$SID_DOWNSTREAM_DATA" \
  --pretrained_checkpoint "$PRETRAIN_OUT/checkpoint-best.pth" \
  --output_dir "$REG_OUT" \
  --no_spatial_embedding \
  --no_temporal_embedding \
  --batch_size 64 \
  --epochs 30 \
  --save_ckpt_freq 10 \
  --vocab_size 1024 \
  --pad_sid 1024 \
  --seq_len 256 \
  --embed_dim 768 \
  --depth 12 \
  --heads 8 \
  --mlp_dim 1024 \
  --task_head_type linear \
  --lr 5e-4 \
  --min_lr 1e-5 \
  --warmup_epochs 5 \
  --weight_decay 0.05 \
  --clip_grad 3.0 \
  --num_workers 8 \
  --device cuda

.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_classification.py \
  --sid_data_dir "$SID_DOWNSTREAM_DATA" \
  --pretrained_checkpoint "$PRETRAIN_OUT/checkpoint-best.pth" \
  --output_dir "$CLS_OUT" \
  --no_spatial_embedding \
  --no_temporal_embedding \
  --batch_size 64 \
  --epochs 30 \
  --save_ckpt_freq 10 \
  --vocab_size 1024 \
  --pad_sid 1024 \
  --seq_len 256 \
  --embed_dim 768 \
  --depth 12 \
  --heads 8 \
  --mlp_dim 1024 \
  --class_weight normalized_inverse_freq \
  --lr 5e-4 \
  --min_lr 1e-5 \
  --warmup_epochs 5 \
  --weight_decay 0.05 \
  --clip_grad 3.0 \
  --num_workers 8 \
  --device cuda

echo "Morphology-V2 SID-only downstream pipeline complete."
