#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
RUN=results/e27_e28_checks
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || exit 1
echo $$ > "$RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$RUN/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "$RUN/status.txt"; fi' EXIT
for stage in e27 e28; do
  if [[ "$stage" == e27 ]]; then OUT=results/e27_mask_diagnostics; SCRIPT=scripts/curriculum_v1/run_e27.py; else OUT=results/e28_context_teacher_pilot; SCRIPT=scripts/curriculum_v1/run_e28.py; fi
  mkdir -p "$OUT"
  if [[ ! -f "$OUT/COMPLETE" ]]; then
    echo "$stage" > "$RUN/status.txt"
    .venv-heartlang/bin/python -u "$SCRIPT" > "$OUT/pipeline.log" 2>&1
  fi
done
echo complete > "$RUN/status.txt"
echo 0 > "$RUN/exit_code.txt"
