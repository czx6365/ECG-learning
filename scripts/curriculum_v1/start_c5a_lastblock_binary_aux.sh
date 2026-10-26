#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
RUN="results/curriculum_v1/c5a_lastblock_binary_aux"
mkdir -p "$RUN"
nohup bash scripts/curriculum_v1/run_c5a_lastblock_binary_aux.sh > "$RUN/pipeline.log" 2>&1 &
echo $! > "$RUN/pipeline.pid"
echo "Started C5a pid=$(cat "$RUN/pipeline.pid")"
