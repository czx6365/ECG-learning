#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
RUN_DIR="results/curriculum_v1/c21_pattern_shift_train"
mkdir -p "$RUN_DIR"
nohup bash scripts/curriculum_v1/run_c21_train.sh > "$RUN_DIR/train_stdout.log" 2>&1 &
echo $! > "$RUN_DIR/train.pid"
echo "Started C2.1 tokenizer training pid=$(cat "$RUN_DIR/train.pid")"
