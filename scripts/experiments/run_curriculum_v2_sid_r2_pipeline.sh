#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

TOKENIZER="checkpoints/vqhbr/curriculum_v2_1024_dim64_bs256_e100/checkpoint-100.pth"
PRETRAIN_SOURCE="datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2"
PRETRAIN_SID="datasets/ecg_datasets/ECG_DSA/pretrain_sid/morphology_curriculum_v2_raw_v2"
DOWNSTREAM_SOURCE="datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675"
DOWNSTREAM_SID="datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_curriculum_v2"
PRETRAIN_OUT="checkpoints/pretrain/HeartLangSID_CurriculumV2Context_rawv2_e100"
PRETRAIN_LOG="log/pretrain/HeartLangSID_CurriculumV2Context_rawv2_e100"
REG_OUT="checkpoints/finetune/ecg_dsa/regression_label4_HeartLangSID_CurriculumV2Context_e100_linear"

if [ ! -f "$TOKENIZER" ]; then
  echo "Missing Curriculum-V2 tokenizer checkpoint: $TOKENIZER" >&2
  exit 1
fi

mkdir -p "$PRETRAIN_SID" "$DOWNSTREAM_SID" "$PRETRAIN_OUT" "$PRETRAIN_LOG" "$REG_OUT"

.venv-heartlang/bin/python scripts/experiments/generate_morphology_v2_sid_dataset.py \
  --source_data_dir "$PRETRAIN_SOURCE" \
  --output_dir "$PRETRAIN_SID" \
  --tokenizer_checkpoint "$TOKENIZER" \
  --splits train val \
  --batch_size 512 \
  --device cuda \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --pad_sid 1024 \
  --symlink_context \
  --overwrite

.venv-heartlang/bin/python scripts/experiments/generate_morphology_v2_sid_dataset.py \
  --source_data_dir "$DOWNSTREAM_SOURCE" \
  --output_dir "$DOWNSTREAM_SID" \
  --tokenizer_checkpoint "$TOKENIZER" \
  --splits train val test \
  --batch_size 512 \
  --device cuda \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --pad_sid 1024 \
  --symlink_context \
  --overwrite

RESUME_ARGS=()
if [ -f "$PRETRAIN_OUT/checkpoint.pth" ]; then
  RESUME_ARGS=(--resume "$PRETRAIN_OUT/checkpoint.pth")
fi

.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_pretraining.py \
  --sid_data_dir "$PRETRAIN_SID" \
  --output_dir "$PRETRAIN_OUT" \
  --log_dir "$PRETRAIN_LOG" \
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
  --sid_data_dir "$DOWNSTREAM_SID" \
  --pretrained_checkpoint "$PRETRAIN_OUT/checkpoint-best.pth" \
  --output_dir "$REG_OUT" \
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

echo "Curriculum-V2 R2 pipeline complete: $REG_OUT/test_regression_summary.json"
