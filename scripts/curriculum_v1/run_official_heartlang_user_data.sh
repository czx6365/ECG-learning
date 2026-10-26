#!/usr/bin/env bash
set -u
cd /home/ubuntu/HeartLang-main || exit 1
RUN=results/heartlang_official_user_data_20261019
SCRIPT=scripts/curriculum_v1/run_official_heartlang_user_data.py
PY=.venv-heartlang/bin/python
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || { echo "official HeartLang pipeline already running" >&2; exit 1; }
echo $$ > "$RUN/pipeline.pid"
: > "$RUN/pipeline.log"
run_stage() {
  local s="$1"
  echo "$s" > "$RUN/status.txt"
  echo "[$(date -Is)] START $s" | tee -a "$RUN/pipeline.log"
  "$PY" -u "$SCRIPT" "$s" --output_dir "$RUN" --epochs 100 > "$RUN/$s.log" 2>&1
  local code=$?
  echo "[$(date -Is)] END $s code=$code" | tee -a "$RUN/pipeline.log"
  if [ "$code" -ne 0 ]; then echo "$code" > "$RUN/exit_code.txt"; echo "failed:$s" > "$RUN/status.txt"; exit "$code"; fi
}
run_stage audit
run_stage cache
run_stage pretrain
run_stage features
run_stage downstream
run_stage test
run_stage report
echo 0 > "$RUN/exit_code.txt"
