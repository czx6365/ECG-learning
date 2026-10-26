#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

TOKENIZER="checkpoints/vqhbr/morphology_v2_no_continuous_1024_dim64_bs256_e50/checkpoint-best.pth"
SID_PRETRAIN_DATA="datasets/ecg_datasets/ECG_DSA/pretrain_sid/morphology_v2_no_continuous_raw_v2"
SID_DOWNSTREAM_DATA="datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_morphology_v2_no_continuous"
PRETRAIN_OUT="checkpoints/pretrain/HeartLangSID_MorphologyV2NoContinuous_SIDOnly_rawv2_e100"
PRETRAIN_LOG="log/pretrain/HeartLangSID_MorphologyV2NoContinuous_SIDOnly_rawv2_e100"
REG_OUT="checkpoints/finetune/ecg_dsa/regression_label4_HeartLangSID_MorphologyV2NoContinuous_SIDOnly_e100_linear"
CLS_OUT="checkpoints/finetune/ecg_dsa/classification_label4_HeartLangSID_MorphologyV2NoContinuous_SIDOnly_e100_linear_weighted_ce"
MANIFEST="results/information_flow/no_continuous_pipeline_manifest.txt"

if [[ ! -f "$TOKENIZER" ]]; then
  echo "Missing tokenizer checkpoint: $TOKENIZER" >&2
  exit 1
fi

for target in "$SID_PRETRAIN_DATA" "$SID_DOWNSTREAM_DATA" "$PRETRAIN_OUT" "$PRETRAIN_LOG" "$REG_OUT" "$CLS_OUT"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing output: $target" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$MANIFEST")"
{
  echo "experiment=NoContinuous_SIDOnly_full_pipeline"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable-no-git-metadata)"
  echo "seed=0"
  echo "tokenizer_checkpoint=$TOKENIZER"
  echo "pretrain_epochs=100"
  echo "downstream_epochs=30"
  echo "spatial_embedding=false"
  echo "temporal_embedding=false"
} > "$MANIFEST"

.venv-heartlang/bin/python scripts/experiments/generate_morphology_v2_sid_dataset.py \
  --source_data_dir datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2 \
  --output_dir "$SID_PRETRAIN_DATA" \
  --tokenizer_checkpoint "$TOKENIZER" \
  --splits train val \
  --batch_size 512 \
  --device cuda \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --pad_sid 1024

.venv-heartlang/bin/python scripts/experiments/generate_morphology_v2_sid_dataset.py \
  --source_data_dir datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675 \
  --output_dir "$SID_DOWNSTREAM_DATA" \
  --tokenizer_checkpoint "$TOKENIZER" \
  --splits train val test \
  --batch_size 512 \
  --device cuda \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --pad_sid 1024

mkdir -p "$PRETRAIN_OUT" "$PRETRAIN_LOG"
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
  --seed 0

mkdir -p "$REG_OUT"
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
  --device cuda \
  --seed 0

mkdir -p "$CLS_OUT"
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
  --device cuda \
  --seed 0

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "pretrained_checkpoint=$PRETRAIN_OUT/checkpoint-best.pth"
  echo "regression_summary=$REG_OUT/test_regression_summary.json"
  echo "classification_summary=$CLS_OUT/test_classification_summary.json"
} >> "$MANIFEST"

echo "NoContinuous SID-only pretrain and downstream pipeline complete."
