#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

TOKENIZER="checkpoints/curriculum_v1/c2_discrete_patterns_from_c1_e50/checkpoint-best.pth"
BASE_DATA="datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct"
INDEX_DIR="datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict"
FULL_SID="datasets/ecg_datasets/ECG_DSA/pretrain_sid/curriculum_v1_c2_full_hclean"
STRICT_SID="datasets/ecg_datasets/ECG_DSA/pretrain_sid/curriculum_v1_c2_strict"
DOWNSTREAM_DATA="datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1"
DOWNSTREAM_SID="datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_patient_split_v1_c2"
PRETRAIN_OUT="checkpoints/curriculum_v1/c2_sid_mlm_e100"
PRETRAIN_LOG="log/curriculum_v1/c2_sid_mlm_e100"
PROBE_OUT="checkpoints/curriculum_v1/c2_fourclass_linear_probe_e30"
RUN_DIR="results/curriculum_v1/c2_sid_linear_probe"

.venv-heartlang/bin/python scripts/curriculum_v1/verify_frozen_split.py

for required in "$TOKENIZER" "$BASE_DATA/train_data.npy" "$INDEX_DIR/summary.json" "$DOWNSTREAM_DATA/train_data.npy"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done
for target in "$FULL_SID" "$STRICT_SID" "$DOWNSTREAM_SID" "$PRETRAIN_OUT" "$PRETRAIN_LOG" "$PROBE_OUT"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing output: $target" >&2
    exit 1
  fi
done
mkdir -p "$RUN_DIR"

{
  echo "experiment=curriculum_v1_c2_sid_linear_probe"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "seed=0"
  echo "tokenizer=$TOKENIZER"
  echo "tokenizer_frozen=true"
  echo "pretrain_epochs=100"
  echo "probe_epochs=30"
  echo "probe_backbone_frozen=true"
  echo "probe_head=linear_768_to_4"
  echo "objective=unweighted_four_output_mse"
  echo "primary_metric=uniform_average_four_output_r2"
  echo "spatial_embedding=false"
  echo "temporal_embedding=false"
  echo "c1_overall_r2_baseline=0.21526321888058036"
} > "$RUN_DIR/protocol.txt"

.venv-heartlang/bin/python scripts/curriculum_v1/generate_c2_sid_dataset.py \
  --source_data_dir "$BASE_DATA" \
  --output_dir "$FULL_SID" \
  --tokenizer_checkpoint "$TOKENIZER" \
  --splits train val \
  --batch_size 512 \
  --device cuda \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --pad_sid 1024 \
  --symlink_context

.venv-heartlang/bin/python scripts/curriculum_v1/materialize_c1_strict_sid.py \
  --full_sid_dir "$FULL_SID" \
  --source_data_dir "$BASE_DATA" \
  --index_dir "$INDEX_DIR" \
  --output_dir "$STRICT_SID"

.venv-heartlang/bin/python scripts/curriculum_v1/generate_c2_sid_dataset.py \
  --source_data_dir "$DOWNSTREAM_DATA" \
  --output_dir "$DOWNSTREAM_SID" \
  --tokenizer_checkpoint "$TOKENIZER" \
  --splits train val test \
  --batch_size 512 \
  --device cuda \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --pad_sid 1024 \
  --symlink_context

mkdir -p "$PRETRAIN_OUT" "$PRETRAIN_LOG"
.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_pretraining.py \
  --sid_data_dir "$STRICT_SID" \
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

mkdir -p "$PROBE_OUT"
.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_regression.py \
  --sid_data_dir "$DOWNSTREAM_SID" \
  --pretrained_checkpoint "$PRETRAIN_OUT/checkpoint-best.pth" \
  --output_dir "$PROBE_OUT" \
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

cp "$PROBE_OUT/test_regression_summary.json" "$RUN_DIR/test_regression_summary.json"
{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "result=$PROBE_OUT/test_regression_summary.json"
} >> "$RUN_DIR/protocol.txt"

echo "C2 fixed-SID four-class linear probe complete: $PROBE_OUT/test_regression_summary.json"
