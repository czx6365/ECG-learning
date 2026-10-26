#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
RUN="results/curriculum_v1/c5b_lite_joint"
mkdir -p "$RUN"
nohup bash scripts/curriculum_v1/run_c5b_lite_pipeline.sh > "$RUN/pipeline.log" 2>&1 &
echo $! > "$RUN/pipeline.pid"
echo "Started C5b-lite pid=$(cat "$RUN/pipeline.pid")"
