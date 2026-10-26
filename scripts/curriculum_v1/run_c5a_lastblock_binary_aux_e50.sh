#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main

SID_DATA="datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_patient_split_v1_c21_pattern_shift"
PRETRAIN="checkpoints/curriculum_v1/c21_pattern_shift_sid_mlm_e100/checkpoint-best.pth"
RESUME="checkpoints/curriculum_v1/c5a_lastblock_binary_aux_probe_e30/checkpoint-30.pth"
OUT="checkpoints/curriculum_v1/c5a_lastblock_binary_aux_probe_e50"
RUN="results/curriculum_v1/c5a_lastblock_binary_aux_e50"

.venv-heartlang/bin/python scripts/curriculum_v1/verify_frozen_split.py
for required in "$SID_DATA/train_sid.npy" "$SID_DATA/val_sid.npy" "$SID_DATA/test_sid.npy" "$PRETRAIN" "$RESUME"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite existing output: $OUT" >&2; exit 1; }
mkdir -p "$RUN"
cat > "$RUN/protocol.txt" <<EOF
experiment=C5a_last_transformer_block_binary_aux_e50
started_at=$(date --iso-8601=seconds)
seed=0
fixed_sid=$SID_DATA
initialized_backbone=$PRETRAIN
resume_checkpoint=$RESUME
resume_epoch=30
target_epoch=50
optimizer_state=resumed
lr_schedule=cosine_horizon_extended_to_50
trainable=last_transformer_block_plus_final_layernorm_plus_shared_mlp
frozen=all_earlier_transformer_blocks_plus_sid_embedding
spatial_embedding=false
temporal_embedding=false
head_peak_lr=1e-4
backbone_peak_lr=1e-5
loss=four_output_variance_normalized_mse_plus_0.2_binary_disease_healthy_mse
checkpoint_selection=validation_uniform_average_four_output_r2_including_epoch30_baseline
EOF

.venv-heartlang/bin/python scripts/curriculum_v1/train_c5a_lastblock_binary_aux.py \
  --sid_data_dir "$SID_DATA" --pretrained_checkpoint "$PRETRAIN" --resume_checkpoint "$RESUME" --output_dir "$OUT" \
  --no_spatial_embedding --no_temporal_embedding --batch_size 64 --epochs 50 --save_ckpt_freq 10 \
  --vocab_size 1024 --pad_sid 1024 --seq_len 256 --embed_dim 768 --depth 12 --heads 8 --mlp_dim 1024 \
  --task_head_type mlp2 --task_head_hidden_dim 256 --task_head_dropout 0.1 \
  --loss_type variance_normalized_mse --lr 1e-4 --min_lr 1e-6 --backbone_lr 1e-5 --backbone_min_lr 1e-7 \
  --binary_aux_weight 0.2 --warmup_epochs 2 --weight_decay 0.05 --clip_grad 3 \
  --num_workers 8 --device cuda --seed 0

cp "$OUT/test_regression_summary.json" "$RUN/test_regression_summary.json"
echo "completed_at=$(date --iso-8601=seconds)" >> "$RUN/protocol.txt"
echo "C5a e50 complete"
