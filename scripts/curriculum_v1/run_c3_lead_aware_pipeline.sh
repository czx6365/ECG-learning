#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

SID_DATA="datasets/ecg_datasets/ECG_DSA/pretrain_sid/curriculum_v1_c2_pattern_only_strict"
DOWNSTREAM_SID="datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_patient_split_v1_c2_pattern_only"
INIT_MLM="checkpoints/curriculum_v1/c2_pattern_only_sid_mlm_e100/checkpoint-best.pth"
PRETRAIN_OUT="checkpoints/curriculum_v1/c3_lead_aware_sid_mlm_e40"
PRETRAIN_LOG="log/curriculum_v1/c3_lead_aware_sid_mlm_e40"
PROBE_OUT="checkpoints/curriculum_v1/c3_lead_aware_fourclass_linear_probe_e30"
RUN_DIR="results/curriculum_v1/c3_lead_aware_pipeline"

.venv-heartlang/bin/python scripts/curriculum_v1/verify_frozen_split.py

for required in "$SID_DATA/train_sid.npy" "$SID_DATA/val_sid.npy" "$DOWNSTREAM_SID/train_sid.npy" "$DOWNSTREAM_SID/val_sid.npy" "$DOWNSTREAM_SID/test_sid.npy" "$INIT_MLM"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done
for target in "$PRETRAIN_OUT" "$PRETRAIN_LOG" "$PROBE_OUT"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing C3 output: $target" >&2
    exit 1
  fi
done
mkdir -p "$RUN_DIR"

{
  echo "experiment=curriculum_v1_c3_lead_aware_sentence_mlm"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "seed=0"
  echo "fixed_tokenizer=checkpoints/curriculum_v1/c2_pattern_only_from_c1_e50/checkpoint-best.pth"
  echo "fixed_sid_data=$SID_DATA"
  echo "initialized_sentence_checkpoint=$INIT_MLM"
  echo "sentence_epochs=40"
  echo "sentence_lr=1e-4"
  echo "spatial_embedding=true"
  echo "temporal_embedding=false"
  echo "sentence_objective=masked_sid_modeling_only"
  echo "vascular_auxiliary=false"
  echo "word_encoder_updated=false"
  echo "codebook_updated=false"
  echo "probe_epochs=30"
  echo "probe_backbone_frozen=true"
  echo "probe_head=linear_768_to_4"
  echo "probe_objective=unweighted_four_output_mse"
  echo "primary_metric=uniform_average_four_output_r2"
  echo "c1_overall_r2_baseline=0.21526321888058036"
  echo "pattern_only_c2_overall_r2_baseline=0.21967520297749854"
} > "$RUN_DIR/protocol.txt"

mkdir -p "$PRETRAIN_OUT" "$PRETRAIN_LOG"
.venv-heartlang/bin/python scripts/curriculum_v1/train_c3_lead_aware_sid_pretraining.py \
  --sid_data_dir "$SID_DATA" \
  --output_dir "$PRETRAIN_OUT" \
  --log_dir "$PRETRAIN_LOG" \
  --resume "$INIT_MLM" \
  --no_temporal_embedding \
  --batch_size 256 \
  --epochs 40 \
  --save_ckpt_freq 20 \
  --val_freq 5 \
  --vocab_size 1024 \
  --pad_sid 1024 \
  --seq_len 256 \
  --embed_dim 768 \
  --depth 12 \
  --heads 8 \
  --mlp_dim 1024 \
  --mask_ratio 0.5 \
  --lr 1e-4 \
  --min_lr 1e-5 \
  --warmup_epochs 2 \
  --weight_decay 0.05 \
  --clip_grad 3.0 \
  --num_workers 8 \
  --device cuda \
  --seed 0

mkdir -p "$PROBE_OUT"
.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_regression.py \
  --sid_data_dir "$DOWNSTREAM_SID" \
  --pretrained_checkpoint "$PRETRAIN_OUT/checkpoint-best.pth" \
  --output_dir "$PROBE_OUT" \
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

cp "$PROBE_OUT/test_regression_summary.json" "$RUN_DIR/test_regression_summary.json"
{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "result=$PROBE_OUT/test_regression_summary.json"
} >> "$RUN_DIR/protocol.txt"

echo "C3 lead-aware Sentence MLM and frozen four-output linear probe complete: $PROBE_OUT/test_regression_summary.json"
