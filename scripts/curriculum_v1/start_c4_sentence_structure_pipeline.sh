#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
RUN_DIR="results/curriculum_v1/c4_sentence_structure_pipeline"
mkdir -p "$RUN_DIR"
nohup bash scripts/curriculum_v1/run_c4_sentence_structure_pipeline.sh > "$RUN_DIR/pipeline.log" 2>&1 &
echo $! > "$RUN_DIR/pipeline.pid"
echo "Started C4 sentence-structure pipeline pid=$(cat "$RUN_DIR/pipeline.pid")"
