#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
RUN=results/e15_quality_aware_mlm
PY=.venv-heartlang/bin/python
mkdir -p "$RUN"
exec > >(tee -a "$RUN/pipeline.log") 2>&1
trap 'ec=$?; echo "$ec" > "$RUN/exit_code.txt"; if [ "$ec" -ne 0 ]; then echo failed > "$RUN/status.txt"; fi' EXIT
stage () {
  local name="$1"
  echo "$name" > "$RUN/status.txt"
  echo "[$(date -Is)] stage=$name"
  "$PY" scripts/curriculum_v1/run_e15_quality_aware_mlm.py "$name" --output_dir "$RUN" --epochs 100 > "$RUN/$name.log" 2>&1
}
stage audit
echo awaiting_audit_review > "$RUN/status.txt"
touch "$RUN/AUDIT_READY"
echo "[$(date -Is)] audit ready; waiting for independent visual review marker"
while [ ! -f "$RUN/AUDIT_REVIEWED" ]; do sleep 30; done
stage smoke
stage pretrain
stage fixed_mlm
stage features
stage downstream
stage report
echo 0 > "$RUN/exit_code.txt"
echo complete > "$RUN/status.txt"
