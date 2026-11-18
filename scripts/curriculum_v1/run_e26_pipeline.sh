#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
RUN=results/e26_information_retention
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || exit 1
echo $$ > "$RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$RUN/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "$RUN/status.txt"; fi' EXIT
.venv-heartlang/bin/python -u scripts/curriculum_v1/run_e26.py > "$RUN/pipeline.log" 2>&1
