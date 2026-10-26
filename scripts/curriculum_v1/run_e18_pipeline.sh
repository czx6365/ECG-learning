#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
PY=.venv-heartlang/bin/python
RUN=results/e18_ecg_text_contrastive
SCRIPT=scripts/curriculum_v1/run_e18_ecg_text_contrastive.py
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || { echo 'E18 pipeline already running' >&2; exit 1; }
echo $$ > "$RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$RUN/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "$RUN/status.txt"; fi' EXIT

echo audit_text_alignment > "$RUN/status.txt"
"$PY" -u "$SCRIPT" audit --output_dir "$RUN" > "$RUN/audit.log" 2>&1
echo encode_frozen_chinese_medbert > "$RUN/status.txt"
"$PY" -u "$SCRIPT" encode_text --output_dir "$RUN" > "$RUN/encode_text.log" 2>&1
echo leakage_and_gradient_smoke > "$RUN/status.txt"
"$PY" -u "$SCRIPT" smoke --output_dir "$RUN" > "$RUN/smoke.log" 2>&1
echo pretrain_joint_mlm_ecg_text_e100 > "$RUN/status.txt"
"$PY" -u "$SCRIPT" pretrain --output_dir "$RUN" --epochs 100 > "$RUN/pretrain.log" 2>&1
echo fixed_mask_mlm_and_text_evaluation > "$RUN/status.txt"
"$PY" -u "$SCRIPT" fixed_mlm --output_dir "$RUN" > "$RUN/fixed_mlm.log" 2>&1
echo extract_downstream_ecg_only_features > "$RUN/status.txt"
"$PY" -u "$SCRIPT" features --output_dir "$RUN" > "$RUN/features.log" 2>&1
echo downstream_validation_seeds_0_1_2 > "$RUN/status.txt"
"$PY" -u "$SCRIPT" downstream --output_dir "$RUN" > "$RUN/downstream.log" 2>&1
echo report_validation_only > "$RUN/status.txt"
"$PY" -u "$SCRIPT" report --output_dir "$RUN" > "$RUN/report.log" 2>&1
echo 0 > "$RUN/exit_code.txt"
