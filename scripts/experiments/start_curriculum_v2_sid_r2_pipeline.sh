#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

PIPELINE_OUT="checkpoints/pretrain/HeartLangSID_CurriculumV2Context_rawv2_e100"
mkdir -p "$PIPELINE_OUT"

if [ -f "$PIPELINE_OUT/pipeline.pid" ] && kill -0 "$(cat "$PIPELINE_OUT/pipeline.pid")" 2>/dev/null; then
  echo "Curriculum-V2 SID R2 pipeline is already running with PID $(cat "$PIPELINE_OUT/pipeline.pid")"
  exit 0
fi

nohup scripts/experiments/run_curriculum_v2_sid_r2_pipeline.sh \
  > "$PIPELINE_OUT/pipeline.log" 2>&1 < /dev/null &
echo "$!" > "$PIPELINE_OUT/pipeline.pid"
echo "Started Curriculum-V2 SID R2 pipeline with PID $(cat "$PIPELINE_OUT/pipeline.pid")"
