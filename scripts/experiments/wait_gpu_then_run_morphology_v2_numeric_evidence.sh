#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

WAIT_LOG="log/vqhbr/morphology_v2_numeric9_wait.log"
mkdir -p "$(dirname "$WAIT_LOG")"

while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; do
  printf '%s GPU busy; waiting for existing compute processes to finish\n' "$(date --iso-8601=seconds)" >> "$WAIT_LOG"
  sleep 60
done

printf '%s GPU idle; launching morphology_v2_numeric9\n' "$(date --iso-8601=seconds)" >> "$WAIT_LOG"
bash scripts/experiments/run_morphology_v2_numeric_evidence.sh >> "$WAIT_LOG" 2>&1
