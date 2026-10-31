#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
PY=.venv-heartlang/bin/python
RUN=results/e18b_downstream_text_auxiliary
SCRIPT=scripts/curriculum_v1/run_e18b_downstream_text_auxiliary.py
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || { echo 'E18-B pipeline already running' >&2; exit 1; }
echo $$ > "$RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$RUN/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "$RUN/status.txt"; fi' EXIT

"$PY" -u "$SCRIPT" audit --output_dir "$RUN" > "$RUN/audit.log" 2>&1
"$PY" -u "$SCRIPT" smoke --output_dir "$RUN" > "$RUN/smoke.log" 2>&1
"$PY" -u "$SCRIPT" train --output_dir "$RUN" > "$RUN/train.log" 2>&1
"$PY" -u "$SCRIPT" report --output_dir "$RUN" > "$RUN/report.log" 2>&1
echo 0 > "$RUN/exit_code.txt"
