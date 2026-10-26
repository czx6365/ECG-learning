#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

TOKENIZER_OUT="checkpoints/vqhbr/morphology_v1_plus_physiology_1024_dim64_bs256_e50"
TOKENIZER_CKPT="$TOKENIZER_OUT/checkpoint-best.pth"
SID_DATA="datasets/ecg_datasets/ECG_DSA/pretrain_sid/morphology_v1_plus_physiology_raw_v2"
PRETRAIN_OUT="checkpoints/pretrain/HeartLangSID_MorphologyV1PlusPhysiology_SIDOnly_rawv2_e100"
PRETRAIN_LOG="log/pretrain/HeartLangSID_MorphologyV1PlusPhysiology_SIDOnly_rawv2_e100"

if [[ ! -f "$TOKENIZER_OUT/train.pid" ]]; then
  echo "Missing tokenizer training PID: $TOKENIZER_OUT/train.pid" >&2
  exit 1
fi

TRAIN_PID="$(cat "$TOKENIZER_OUT/train.pid")"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  printf '%s tokenizer training is still running (PID %s)\n' "$(date --iso-8601=seconds)" "$TRAIN_PID"
  sleep 60
done

if [[ ! -f "$TOKENIZER_CKPT" ]]; then
  echo "Tokenizer process exited without a best checkpoint: $TOKENIZER_CKPT" >&2
  exit 1
fi

if [[ -e "$SID_DATA" ]]; then
  echo "Refusing to overwrite existing SID dataset: $SID_DATA" >&2
  exit 1
fi

if [[ -e "$PRETRAIN_OUT" || -e "$PRETRAIN_LOG" ]]; then
  echo "Refusing to overwrite existing HeartLang pretraining output" >&2
  exit 1
fi

.venv-heartlang/bin/python scripts/experiments/generate_morphology_v2_sid_dataset.py \
  --source_data_dir datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2 \
  --output_dir "$SID_DATA" \
  --tokenizer_checkpoint "$TOKENIZER_CKPT" \
  --splits train val \
  --batch_size 512 \
  --device cuda \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --pad_sid 1024

mkdir -p "$PRETRAIN_OUT" "$PRETRAIN_LOG"

.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_pretraining.py \
  --sid_data_dir "$SID_DATA" \
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
  --device cuda

echo "V1 + Physiology SID-only HeartLang pretraining complete."
