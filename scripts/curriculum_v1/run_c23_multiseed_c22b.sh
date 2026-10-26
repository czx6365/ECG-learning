#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main

SID_DATA="datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_patient_split_v1_c21_pattern_shift"
PRETRAIN="checkpoints/curriculum_v1/c21_pattern_shift_sid_mlm_e100/checkpoint-best.pth"
RUN_DIR="results/curriculum_v1/c23_multiseed_c22b"
mkdir -p "$RUN_DIR"

for SEED in 1 2; do
  OUT="checkpoints/curriculum_v1/c22b_variance_mlp2_probe_e30_seed${SEED}"
  if [[ -f "$OUT/test_regression_summary.json" ]]; then
    echo "Seed $SEED already complete; reusing $OUT"
    continue
  fi
  if [[ -e "$OUT" ]]; then
    echo "Refusing to overwrite incomplete output: $OUT" >&2
    exit 1
  fi
  .venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_regression.py \
    --sid_data_dir "$SID_DATA" --pretrained_checkpoint "$PRETRAIN" --output_dir "$OUT" \
    --no_spatial_embedding --no_temporal_embedding --batch_size 64 --epochs 30 --save_ckpt_freq 10 \
    --vocab_size 1024 --pad_sid 1024 --seq_len 256 --embed_dim 768 --depth 12 --heads 8 --mlp_dim 1024 \
    --task_head_type mlp2 --task_head_hidden_dim 256 --task_head_dropout 0.1 \
    --loss_type variance_normalized_mse --lr 5e-4 --min_lr 1e-5 --warmup_epochs 5 \
    --weight_decay 0.05 --clip_grad 3 --num_workers 8 --device cuda --seed "$SEED" \
    2>&1 | tee "$RUN_DIR/seed${SEED}.log"
done

.venv-heartlang/bin/python scripts/curriculum_v1/ensemble_c22b_seeds.py \
  --run_dirs \
    checkpoints/curriculum_v1/c22b_variance_mlp2_probe_e30 \
    checkpoints/curriculum_v1/c22b_variance_mlp2_probe_e30_seed1 \
    checkpoints/curriculum_v1/c22b_variance_mlp2_probe_e30_seed2 \
  --output_dir "$RUN_DIR"
