#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

OUT="checkpoints/pretrain/HeartLangSID_MorphologyV2_SIDOnly_rawv2_e100"
mkdir -p "$OUT"

if [ -f "$OUT/pipeline.pid" ] && kill -0 "$(cat "$OUT/pipeline.pid")" 2>/dev/null; then
  echo "Morphology-V2 SID-only pipeline is already running with PID $(cat "$OUT/pipeline.pid")"
  exit 0
fi

nohup scripts/experiments/run_morphology_v2_sid_only_downstream_pipeline.sh \
  > "$OUT/pipeline.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/pipeline.pid"
echo "Started Morphology-V2 SID-only pipeline with PID $(cat "$OUT/pipeline.pid")"
