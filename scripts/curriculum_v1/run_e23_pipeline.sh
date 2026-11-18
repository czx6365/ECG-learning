#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
PY=.venv-heartlang/bin/python
RUN=results/e23_downstream_age_auxiliary
SCRIPT=scripts/curriculum_v1/run_e23_downstream_age_auxiliary.py
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || { echo 'E23 pipeline already running' >&2; exit 1; }
echo $$ > "$RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$RUN/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "$RUN/status.txt"; fi' EXIT

echo audit_patient_age_targets > "$RUN/status.txt"
"$PY" -u "$SCRIPT" audit --output_dir "$RUN" > "$RUN/audit.log" 2>&1
echo smoke_exact_e12_inference_and_age_gradients > "$RUN/status.txt"
"$PY" -u "$SCRIPT" smoke --output_dir "$RUN" > "$RUN/smoke.log" 2>&1
echo train_e23_age_auxiliary_seeds_0_1_2 > "$RUN/status.txt"
"$PY" -u "$SCRIPT" train --output_dir "$RUN" > "$RUN/train.log" 2>&1
echo report_validation_only > "$RUN/status.txt"
"$PY" -u "$SCRIPT" report --output_dir "$RUN" > "$RUN/report.log" 2>&1
echo 0 > "$RUN/exit_code.txt"
