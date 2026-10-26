#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

TRAIN_OUT="checkpoints/vqhbr/curriculum_v2_1024_dim64_bs256_e100"
TRAIN_PID_FILE="$TRAIN_OUT/train.pid"
EVAL_LOG="$TRAIN_OUT/post_eval.log"

if [ ! -f "$TRAIN_PID_FILE" ]; then
  echo "Missing training PID file: $TRAIN_PID_FILE" >&2
  exit 1
fi

TRAIN_PID="$(cat "$TRAIN_PID_FILE")"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done

if [ ! -f "$TRAIN_OUT/checkpoint-100.pth" ]; then
  echo "Training process ended without checkpoint-100.pth; post-evaluation skipped." >> "$EVAL_LOG"
  exit 1
fi

scripts/experiments/run_curriculum_v2_post_eval.sh >> "$EVAL_LOG" 2>&1
