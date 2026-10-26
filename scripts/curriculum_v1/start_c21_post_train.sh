#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
RUN_DIR=results/curriculum_v1/c21_pattern_shift_pipeline
mkdir -p "$RUN_DIR"
nohup bash scripts/curriculum_v1/run_c21_post_train.sh > "$RUN_DIR/pipeline.log" 2>&1 &
echo $! > "$RUN_DIR/pipeline.pid"
echo "Started C2.1 post-training pipeline pid=$(cat "$RUN_DIR/pipeline.pid")"
