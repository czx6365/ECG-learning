#!/usr/bin/env bash
# 正式 E2 匹配预训练与下游评估入口。仅在用户确认多天训练后手动启动。
set -euo pipefail
cd /home/ubuntu/HeartLang-main
PY=.venv-heartlang/bin/python
BASE=results/ecg_representation_comparison

# 每个输入模式与随机种子各用独立目录；任何已有结果都拒绝覆盖。
for MODE in waveform sid_only hybrid ze_only; do
  for SEED in 0 1 2; do
    PRETRAIN="$BASE/experiment2_matched_${MODE}_seed${SEED}"
    DOWNSTREAM="$BASE/experiment2_downstream_${MODE}_seed${SEED}"
    if [[ -e "$PRETRAIN" || -e "$DOWNSTREAM" ]]; then
      echo "已有输出目录，停止以保护产物: $PRETRAIN 或 $DOWNSTREAM" >&2
      exit 1
    fi
    "$PY" scripts/experiments/train_representation_e2_mlm.py \
      --mode "$MODE" --seed "$SEED" --epochs 100 --batch 64 \
      --lr 0.0005 --output "$PRETRAIN" \
      > "${PRETRAIN}.log" 2>&1
    test -f "$PRETRAIN/COMPLETE"
    "$PY" scripts/experiments/evaluate_representation_e2.py \
      --pretrained "$PRETRAIN/checkpoint-best.pth" \
      --output "$DOWNSTREAM" --epochs 30 --batch 64 --hidden 256 \
      --lr 0.001 --seeds 0 1 2 \
      > "${DOWNSTREAM}.log" 2>&1
    test -f "$DOWNSTREAM/COMPLETE"
  done
done
