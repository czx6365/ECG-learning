#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
PY=.venv-heartlang/bin/python
SCRIPT=scripts/curriculum_v1/run_e23_downstream_age_auxiliary.py
ROOT_RUN=results/e23_downstream_age_auxiliary_weight_sweep

run_one() {
  local weight="$1"
  local tag="$2"
  local run="$ROOT_RUN/$tag"
  mkdir -p "$run"
  exec 9>"$run/pipeline.lock"
  flock -n 9 || { echo "E23 age-weight $weight already running" >&2; return 1; }
  echo $$ > "$run/pipeline.pid"
  echo "$weight" > "$run/age_weight.txt"
  trap 'code=$?; echo "$code" > "'$run'/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "'$run'/status.txt"; fi' RETURN

  echo audit_patient_age_targets > "$run/status.txt"
  "$PY" -u "$SCRIPT" audit --output_dir "$run" --age_weight "$weight" > "$run/audit.log" 2>&1
  echo smoke_exact_e12_inference_and_age_gradients > "$run/status.txt"
  "$PY" -u "$SCRIPT" smoke --output_dir "$run" --age_weight "$weight" > "$run/smoke.log" 2>&1
  echo train_e23_age_auxiliary_seeds_0_1_2 > "$run/status.txt"
  "$PY" -u "$SCRIPT" train --output_dir "$run" --age_weight "$weight" > "$run/train.log" 2>&1
  echo report_validation_only > "$run/status.txt"
  "$PY" -u "$SCRIPT" report --output_dir "$run" --age_weight "$weight" > "$run/report.log" 2>&1
  echo 0 > "$run/exit_code.txt"
  trap - RETURN
  exec 9>&-
}

mkdir -p "$ROOT_RUN"
echo $$ > "$ROOT_RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$ROOT_RUN/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "$ROOT_RUN/status.txt"; fi' EXIT
echo weight_0.05 > "$ROOT_RUN/status.txt"
run_one 0.05 w0p05
echo weight_0.15 > "$ROOT_RUN/status.txt"
run_one 0.15 w0p15
echo complete > "$ROOT_RUN/status.txt"
echo 0 > "$ROOT_RUN/exit_code.txt"
