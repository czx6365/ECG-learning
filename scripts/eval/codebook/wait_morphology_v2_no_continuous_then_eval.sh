#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

CHECKPOINT="checkpoints/vqhbr/morphology_v2_no_continuous_1024_dim64_bs256_e50/checkpoint-50.pth"
WAIT_LOG="log/vqhbr/morphology_v2_no_continuous_post_eval_wait.log"
mkdir -p "$(dirname "$WAIT_LOG")"

while [[ ! -f "$CHECKPOINT" ]]; do
  printf '%s training incomplete; waiting for checkpoint-50\n' "$(date --iso-8601=seconds)" >> "$WAIT_LOG"
  sleep 120
done

while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; do
  printf '%s checkpoint ready but GPU busy; waiting to evaluate\n' "$(date --iso-8601=seconds)" >> "$WAIT_LOG"
  sleep 60
done

printf '%s launching Morphology-V2 no-continuous evaluation\n' "$(date --iso-8601=seconds)" >> "$WAIT_LOG"
bash scripts/eval/codebook/run_morphology_v2_no_continuous_post_eval.sh >> "$WAIT_LOG" 2>&1
