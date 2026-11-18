#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
RUN=results/e22d_adaptive_difference_finetune
SCRIPT=scripts/curriculum_v1/run_e22d_adaptive_difference_finetune.py
PY=.venv-heartlang/bin/python
mkdir -p "$RUN"
echo $$ > "$RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$RUN/exit_code.txt"; if [[ $code -ne 0 ]]; then echo failed > "$RUN/status.txt"; fi' EXIT
for stage in cache smoke downstream report; do
  if [[ -f "$RUN/COMPLETE" ]]; then break; fi
  "$PY" -u "$SCRIPT" --stage "$stage" 2>&1 | tee -a "$RUN/${stage}.log"
done
echo 0 > "$RUN/exit_code.txt"
