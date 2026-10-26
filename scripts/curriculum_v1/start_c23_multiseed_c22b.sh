#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
RUN_DIR="results/curriculum_v1/c23_multiseed_c22b"
mkdir -p "$RUN_DIR"
nohup bash scripts/curriculum_v1/run_c23_multiseed_c22b.sh > "$RUN_DIR/pipeline.log" 2>&1 &
echo $! > "$RUN_DIR/pipeline.pid"
echo "Started C2.2b multiseed pipeline pid=$(cat "$RUN_DIR/pipeline.pid")"
