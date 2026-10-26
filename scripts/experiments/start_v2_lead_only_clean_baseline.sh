#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

OUT="checkpoints/vqhbr/morphology_v2_leadV2_only_clean_1024_dim64_bs256_e50"
mkdir -p "$OUT"

if [ -f "$OUT/pipeline.pid" ] && kill -0 "$(cat "$OUT/pipeline.pid")" 2>/dev/null; then
  echo "V2-lead-only baseline is already running with PID $(cat "$OUT/pipeline.pid")"
  exit 0
fi

nohup scripts/experiments/run_v2_lead_only_clean_baseline.sh \
  > "$OUT/pipeline.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/pipeline.pid"
echo "Started V2-lead-only baseline with PID $(cat "$OUT/pipeline.pid")"
