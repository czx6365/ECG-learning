#!/usr/bin/env bash
# 四架构相同预算的中等规模训练检查；只使用严格预训练 train/val。
set -euo pipefail
cd /home/ubuntu/HeartLang-main
PY=.venv-heartlang/bin/python
BASE=results/ecg_representation_comparison

for MODE in waveform sid_only hybrid ze_only; do
  OUT="$BASE/experiment2_medium_${MODE}_seed0_e5"
  if [[ -e "$OUT" || -e "${OUT}.log" ]]; then
    echo "已有输出，拒绝覆盖: $OUT" >&2
    exit 1
  fi
  echo "START $MODE $(date -u +%FT%TZ)" >&2
  "$PY" scripts/experiments/train_representation_e2_mlm.py \
    --mode "$MODE" --seed 0 --epochs 5 --batch 64 \
    --max_train 8192 --max_val 1024 --lr 0.0005 \
    --output "$OUT" > "${OUT}.log" 2>&1
  test -f "$OUT/COMPLETE"
  echo "COMPLETE $MODE $(date -u +%FT%TZ)" >&2
done
