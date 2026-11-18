#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
run=results/e29_e30_comparison
mkdir -p "$run"
exec 9>"$run/pipeline.lock"
flock -n 9
trap 'echo "$?" > "$run/exit_code.txt"' EXIT
for stage in e29 e30; do
  echo "$stage" > "$run/status.txt"
  if [[ "$stage" == e29 ]]; then script=run_e29_pooling; else script=run_e30; fi
  .venv-heartlang/bin/python -u "scripts/curriculum_v1/$script.py" > "$run/$stage.log" 2>&1
done
echo complete > "$run/status.txt"
