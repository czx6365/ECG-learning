#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
PY=.venv-heartlang/bin/python
RUN=results/e24_downstream_age_sex_auxiliary
SCRIPT=scripts/curriculum_v1/run_e24_downstream_age_sex_auxiliary.py
mkdir -p "$RUN"; exec 9>"$RUN/pipeline.lock"; flock -n 9 || { echo 'E24 pipeline already running' >&2; exit 1; }
echo $$ > "$RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$RUN/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "$RUN/status.txt"; fi' EXIT
for stage in audit smoke train report; do
  case "$stage" in audit) status=audit_patient_age_and_sex_targets;; smoke) status=smoke_exact_e12_inference_and_demographic_gradients;; train) status=train_e24_age_sex_seeds_0_1_2;; report) status=report_validation_only;; esac
  echo "$status" > "$RUN/status.txt"; "$PY" -u "$SCRIPT" "$stage" --output_dir "$RUN" > "$RUN/$stage.log" 2>&1
done
echo 0 > "$RUN/exit_code.txt"
