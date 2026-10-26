#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

TOKENIZER="checkpoints/vqhbr/morphology_v1_plus_physiology_1024_dim64_bs256_e50/checkpoint-best.pth"
PRETRAINED="checkpoints/pretrain/HeartLangSID_MorphologyV1PlusPhysiology_SIDOnly_rawv2_e100/checkpoint-best.pth"
SID_DATA="datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_morphology_v1_plus_physiology"
REG_OUT="checkpoints/finetune/ecg_dsa/regression_label4_HeartLangSID_MorphologyV1PlusPhysiology_SIDOnly_e100_linear"
CLS_OUT="checkpoints/finetune/ecg_dsa/classification_label4_HeartLangSID_MorphologyV1PlusPhysiology_SIDOnly_e100_linear_weighted_ce"

for required in "$TOKENIZER" "$PRETRAINED"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required checkpoint: $required" >&2
    exit 1
  fi
done

for target in "$SID_DATA" "$REG_OUT" "$CLS_OUT"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing output: $target" >&2
    exit 1
  fi
done

.venv-heartlang/bin/python scripts/experiments/generate_morphology_v2_sid_dataset.py \
  --source_data_dir datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675 \
  --output_dir "$SID_DATA" \
  --tokenizer_checkpoint "$TOKENIZER" \
  --splits train val test \
  --batch_size 512 \
  --device cuda \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --pad_sid 1024

mkdir -p "$REG_OUT"
.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_regression.py \
  --sid_data_dir "$SID_DATA" \
  --pretrained_checkpoint "$PRETRAINED" \
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

mkdir -p "$CLS_OUT"
.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_classification.py \
  --sid_data_dir "$SID_DATA" \
  --pretrained_checkpoint "$PRETRAINED" \
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

echo "V1 + Physiology downstream regression and classification evaluation complete."
