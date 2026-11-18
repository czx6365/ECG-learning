#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
PY=.venv-heartlang/bin/python
RUN=results/e22c_pretraining_only_local_decoder
SCRIPT=scripts/curriculum_v1/run_e22c_pretraining_only_local_decoder.py
mkdir -p "$RUN";exec 9>"$RUN/pipeline.lock";flock -n 9||{ echo 'E22-C pipeline already running' >&2;exit 1; };echo $$>"$RUN/pipeline.pid"
trap 'code=$?;echo "$code">"$RUN/exit_code.txt";if [[ "$code" != 0 ]];then echo failed>"$RUN/status.txt";fi' EXIT
for stage in audit smoke pretrain fixed_mlm features downstream report;do
  echo "$stage">"$RUN/status.txt";"$PY" -u "$SCRIPT" "$stage" --output_dir "$RUN" --epochs 100 >"$RUN/$stage.log" 2>&1
done
echo 0>"$RUN/exit_code.txt"
