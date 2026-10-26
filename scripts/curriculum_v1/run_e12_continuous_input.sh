#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
PY=.venv-heartlang/bin/python
RUN=results/curriculum_v1/e12_continuous_ze_sentence_20261018
SMOKE=results/curriculum_v1/e12_continuous_ze_sentence_20261018_smoke
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || { echo 'E12 pipeline already running' >&2; exit 1; }
echo $$ > "$RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$RUN/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "$RUN/status.txt"; fi' EXIT
SCRIPT=scripts/curriculum_v1/run_e12_continuous_input.py

if [[ ! -f "$SMOKE/COMPLETE" ]]; then
  rm -rf "$SMOKE"
  "$PY" -u "$SCRIPT" smoke --output_dir "$SMOKE" > "$RUN/smoke.log" 2>&1
fi
echo audit_no_val_test_overlap > "$RUN/status.txt"
"$PY" -u "$SCRIPT" audit --output_dir "$RUN" > "$RUN/audit.log" 2>&1
echo caching_frozen_e3_ze > "$RUN/status.txt"
"$PY" -u "$SCRIPT" cache --output_dir "$RUN" --batch_size 128 > "$RUN/cache.log" 2>&1
echo pretraining_e12a_scratch_e100 > "$RUN/status.txt"
"$PY" -u "$SCRIPT" pretrain_a --output_dir "$RUN" --epochs 100 > "$RUN/pretrain_a.log" 2>&1
echo pretraining_e12b_e3_warmstart_e100 > "$RUN/status.txt"
"$PY" -u "$SCRIPT" pretrain_b --output_dir "$RUN" --epochs 100 > "$RUN/pretrain_b.log" 2>&1
echo fixed_mask_mlm_comparison > "$RUN/status.txt"
"$PY" -u "$SCRIPT" fixed_mlm --output_dir "$RUN" > "$RUN/fixed_mlm.log" 2>&1
echo extracting_e12a_downstream_features > "$RUN/status.txt"
"$PY" -u "$SCRIPT" features_a --output_dir "$RUN" > "$RUN/features_a.log" 2>&1
echo extracting_e12b_downstream_features > "$RUN/status.txt"
"$PY" -u "$SCRIPT" features_b --output_dir "$RUN" > "$RUN/features_b.log" 2>&1
echo training_downstream_validation_only > "$RUN/status.txt"
"$PY" -u "$SCRIPT" downstream --output_dir "$RUN" > "$RUN/downstream.log" 2>&1
echo testing_validation_selected_e12_only > "$RUN/status.txt"
"$PY" -u "$SCRIPT" test --output_dir "$RUN" > "$RUN/test.log" 2>&1
echo generating_report > "$RUN/status.txt"
"$PY" -u "$SCRIPT" report --output_dir "$RUN" > "$RUN/report.log" 2>&1
echo 0 > "$RUN/exit_code.txt"
