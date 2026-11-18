#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
PY=.venv-heartlang/bin/python
RUN=results/e22_three_beat_grouped_sid
SCRIPT=scripts/curriculum_v1/run_e22_three_beat_grouped_sid.py
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || { echo 'E22 pipeline already running' >&2; exit 1; }
echo $$ > "$RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$RUN/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "$RUN/status.txt"; fi' EXIT

echo audit_same_lead_three_beat_groups > "$RUN/status.txt"
"$PY" -u "$SCRIPT" audit --output_dir "$RUN" > "$RUN/audit.log" 2>&1
echo smoke_group_vq_and_group_mlm > "$RUN/status.txt"
"$PY" -u "$SCRIPT" smoke --output_dir "$RUN" > "$RUN/smoke.log" 2>&1
echo train_new_three_beat_codebook_e50 > "$RUN/status.txt"
"$PY" -u "$SCRIPT" codebook --output_dir "$RUN" --codebook_epochs 50 > "$RUN/codebook.log" 2>&1
echo cache_group_ze_and_sid > "$RUN/status.txt"
"$PY" -u "$SCRIPT" cache --output_dir "$RUN" > "$RUN/cache.log" 2>&1
echo train_group_sid_mlm_e100 > "$RUN/status.txt"
"$PY" -u "$SCRIPT" pretrain --output_dir "$RUN" --mlm_epochs 100 > "$RUN/pretrain.log" 2>&1
echo fixed_mask_group_sid_validation > "$RUN/status.txt"
"$PY" -u "$SCRIPT" fixed_mlm --output_dir "$RUN" > "$RUN/fixed_mlm.log" 2>&1
echo extract_grouped_ecg_features > "$RUN/status.txt"
"$PY" -u "$SCRIPT" features --output_dir "$RUN" > "$RUN/features.log" 2>&1
echo downstream_e12_head_seeds_0_1_2 > "$RUN/status.txt"
"$PY" -u "$SCRIPT" downstream --output_dir "$RUN" > "$RUN/downstream.log" 2>&1
echo report_validation_only > "$RUN/status.txt"
"$PY" -u "$SCRIPT" report --output_dir "$RUN" > "$RUN/report.log" 2>&1
echo 0 > "$RUN/exit_code.txt"
