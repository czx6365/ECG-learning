#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

OUT="checkpoints/pretrain/HeartLangSID_MorphologyV2Context_rawv2_e100"
REG="checkpoints/finetune/ecg_dsa/regression_label4_HeartLangSID_MorphologyV2Context_e100_linear"

RESUME_ARGS=()
if [[ -f "${OUT}/checkpoint.pth" ]]; then
  RESUME_ARGS=(--resume "${OUT}/checkpoint.pth")
fi

.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_pretraining.py \
  --sid_data_dir datasets/ecg_datasets/ECG_DSA/pretrain_sid/morphology_v2_physaware_raw_v2 \
  --output_dir "${OUT}" \
  --log_dir log/pretrain/HeartLangSID_MorphologyV2Context_rawv2_e100 \
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

rm -rf "${REG}"

.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_regression.py \
  --sid_data_dir datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_morphology_v2 \
  --pretrained_checkpoint "${OUT}/checkpoint-best.pth" \
  --output_dir "${REG}" \
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
