#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

RUN_DIR="results/curriculum_v1/c3_lead_aware_pipeline"
mkdir -p "$RUN_DIR"
if [[ -f "$RUN_DIR/pipeline.pid" ]] && kill -0 "$(cat "$RUN_DIR/pipeline.pid")" 2>/dev/null; then
  echo "C3 lead-aware pipeline is already running with PID $(cat "$RUN_DIR/pipeline.pid")"
  exit 0
fi
if [[ -f "$RUN_DIR/test_regression_summary.json" ]]; then
  echo "C3 lead-aware pipeline is already complete"
  exit 0
fi

nohup scripts/curriculum_v1/run_c3_lead_aware_pipeline.sh > "$RUN_DIR/pipeline.log" 2>&1 < /dev/null &
echo "$!" > "$RUN_DIR/pipeline.pid"
echo "Started C3 lead-aware pipeline with PID $(cat "$RUN_DIR/pipeline.pid")"
