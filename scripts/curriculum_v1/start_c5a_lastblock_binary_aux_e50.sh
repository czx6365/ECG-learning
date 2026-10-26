#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
RUN="results/curriculum_v1/c5a_lastblock_binary_aux_e50"
mkdir -p "$RUN"
nohup bash scripts/curriculum_v1/run_c5a_lastblock_binary_aux_e50.sh > "$RUN/pipeline.log" 2>&1 &
echo $! > "$RUN/pipeline.pid"
echo "Started C5a e50 pid=$(cat "$RUN/pipeline.pid")"
