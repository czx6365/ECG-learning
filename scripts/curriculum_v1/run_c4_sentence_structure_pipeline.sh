#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

PRETRAIN_SID="datasets/ecg_datasets/ECG_DSA/pretrain_sid/curriculum_v1_c21_pattern_shift_strict"
DOWNSTREAM_SID="datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_patient_split_v1_c21_pattern_shift"
INIT_MLM="checkpoints/curriculum_v1/c21_pattern_shift_sid_mlm_e100/checkpoint-best.pth"
RUN_DIR="results/curriculum_v1/c4_sentence_structure_pipeline"

C4A_MLM="checkpoints/curriculum_v1/c4a_temporal_sid_mlm_e40"
C4A_LOG="log/curriculum_v1/c4a_temporal_sid_mlm_e40"
C4A_PROBE="checkpoints/curriculum_v1/c4a_temporal_c22b_probe_e30"
C4B_MLM="checkpoints/curriculum_v1/c4b_cross_lead_sid_mlm_e40"
C4B_LOG="log/curriculum_v1/c4b_cross_lead_sid_mlm_e40"
C4B_PROBE="checkpoints/curriculum_v1/c4b_cross_lead_c22b_probe_e30"

.venv-heartlang/bin/python scripts/curriculum_v1/verify_frozen_split.py
for required in \
  "$PRETRAIN_SID/train_sid.npy" "$PRETRAIN_SID/val_sid.npy" \
  "$DOWNSTREAM_SID/train_sid.npy" "$DOWNSTREAM_SID/val_sid.npy" "$DOWNSTREAM_SID/test_sid.npy" \
  "$INIT_MLM"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
for target in "$C4A_MLM" "$C4A_LOG" "$C4A_PROBE" "$C4B_MLM" "$C4B_LOG" "$C4B_PROBE"; do
  [[ ! -e "$target" ]] || { echo "Refusing to overwrite existing output: $target" >&2; exit 1; }
done
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/protocol.txt" <<EOF
experiment=C4a_temporal_and_C4b_cross_lead
started_at=$(date --iso-8601=seconds)
seed=0
fixed_tokenizer=C2.1_pattern_shift
fixed_pretrain_sid=$PRETRAIN_SID
fixed_downstream_sid=$DOWNSTREAM_SID
initialized_sentence_checkpoint=$INIT_MLM
sentence_epochs=40
sentence_lr=1e-4
C4a_spatial_embedding=false
C4a_temporal_embedding=true
C4a_objective=ordinary_sentence_mlm
C4b_spatial_embedding=false
C4b_temporal_embedding=false
C4b_objective=ordinary_mlm_plus_0.1_whole_lead_masked_CE
vascular_auxiliary=false
word_encoder_updated=false
codebook_updated=false
probe=shared_mlp_768_256_4_variance_normalized_mse
probe_backbone_frozen=true
probe_epochs=30
primary_metric=uniform_average_four_output_r2
EOF

mkdir -p "$C4A_MLM" "$C4A_LOG"
.venv-heartlang/bin/python scripts/curriculum_v1/train_c4a_temporal_sid_pretraining.py \
  --sid_data_dir "$PRETRAIN_SID" --output_dir "$C4A_MLM" --log_dir "$C4A_LOG" --resume "$INIT_MLM" \
  --no_spatial_embedding --batch_size 256 --epochs 40 --save_ckpt_freq 20 --val_freq 5 \
  --vocab_size 1024 --pad_sid 1024 --seq_len 256 --embed_dim 768 --depth 12 --heads 8 --mlp_dim 1024 \
  --mask_ratio 0.5 --lr 1e-4 --min_lr 1e-5 --warmup_epochs 2 --weight_decay 0.05 --clip_grad 3 \
  --num_workers 8 --device cuda --seed 0

mkdir -p "$C4A_PROBE"
.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_regression.py \
  --sid_data_dir "$DOWNSTREAM_SID" --pretrained_checkpoint "$C4A_MLM/checkpoint-best.pth" --output_dir "$C4A_PROBE" \
  --no_spatial_embedding --batch_size 64 --epochs 30 --save_ckpt_freq 10 \
  --vocab_size 1024 --pad_sid 1024 --seq_len 256 --embed_dim 768 --depth 12 --heads 8 --mlp_dim 1024 \
  --task_head_type mlp2 --task_head_hidden_dim 256 --task_head_dropout 0.1 \
  --loss_type variance_normalized_mse --lr 5e-4 --min_lr 1e-5 --warmup_epochs 5 \
  --weight_decay 0.05 --clip_grad 3 --num_workers 8 --device cuda --seed 0
cp "$C4A_PROBE/test_regression_summary.json" "$RUN_DIR/c4a_test_regression_summary.json"

mkdir -p "$C4B_MLM" "$C4B_LOG"
.venv-heartlang/bin/python scripts/curriculum_v1/train_c4b_cross_lead_sid_pretraining.py \
  --sid_data_dir "$PRETRAIN_SID" --output_dir "$C4B_MLM" --log_dir "$C4B_LOG" --resume "$INIT_MLM" \
  --no_spatial_embedding --no_temporal_embedding --cross_lead_weight 0.1 \
  --batch_size 128 --epochs 40 --save_ckpt_freq 20 --val_freq 5 \
  --vocab_size 1024 --pad_sid 1024 --seq_len 256 --embed_dim 768 --depth 12 --heads 8 --mlp_dim 1024 \
  --mask_ratio 0.5 --lr 1e-4 --min_lr 1e-5 --warmup_epochs 2 --weight_decay 0.05 --clip_grad 3 \
  --num_workers 8 --device cuda --seed 0

mkdir -p "$C4B_PROBE"
.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_regression.py \
  --sid_data_dir "$DOWNSTREAM_SID" --pretrained_checkpoint "$C4B_MLM/checkpoint-best.pth" --output_dir "$C4B_PROBE" \
  --no_spatial_embedding --no_temporal_embedding --batch_size 64 --epochs 30 --save_ckpt_freq 10 \
  --vocab_size 1024 --pad_sid 1024 --seq_len 256 --embed_dim 768 --depth 12 --heads 8 --mlp_dim 1024 \
  --task_head_type mlp2 --task_head_hidden_dim 256 --task_head_dropout 0.1 \
  --loss_type variance_normalized_mse --lr 5e-4 --min_lr 1e-5 --warmup_epochs 5 \
  --weight_decay 0.05 --clip_grad 3 --num_workers 8 --device cuda --seed 0
cp "$C4B_PROBE/test_regression_summary.json" "$RUN_DIR/c4b_test_regression_summary.json"

echo "completed_at=$(date --iso-8601=seconds)" >> "$RUN_DIR/protocol.txt"
echo "C4a and C4b complete"
