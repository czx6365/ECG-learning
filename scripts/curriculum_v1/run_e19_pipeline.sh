#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
PY=.venv-heartlang/bin/python
RUN=results/e19_hierarchical_text_concepts
SCRIPT=scripts/curriculum_v1/run_e19_hierarchical_text_concepts.py
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || { echo 'E19 pipeline already running' >&2; exit 1; }
echo $$ > "$RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$RUN/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "$RUN/status.txt"; fi' EXIT

"$PY" -u "$SCRIPT" audit --output_dir "$RUN" > "$RUN/audit.log" 2>&1
"$PY" -u "$SCRIPT" cache --output_dir "$RUN" > "$RUN/cache.log" 2>&1
"$PY" -u "$SCRIPT" smoke --output_dir "$RUN" > "$RUN/smoke.log" 2>&1
"$PY" -u "$SCRIPT" train --output_dir "$RUN" > "$RUN/train.log" 2>&1
"$PY" -u "$SCRIPT" report --output_dir "$RUN" > "$RUN/report.log" 2>&1
echo 0 > "$RUN/exit_code.txt"
