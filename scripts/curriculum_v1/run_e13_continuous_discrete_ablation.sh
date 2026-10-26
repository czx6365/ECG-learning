#!/usr/bin/env bash
set -u
cd /home/ubuntu/HeartLang-main || exit 1

RUN=results/e13_continuous_discrete_ablation
PY=scripts/curriculum_v1/run_e13_continuous_discrete_ablation.py
PYTHON=.venv-heartlang/bin/python
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || { echo "E13 pipeline already running" >&2; exit 1; }
echo $$ > "$RUN/pipeline.pid"
: > "$RUN/pipeline.log"

run_stage() {
  local stage="$1"
  echo "[$(date -Is)] START $stage" | tee -a "$RUN/pipeline.log"
  echo "$stage" > "$RUN/status.txt"
  "$PYTHON" -u "$PY" "$stage" --output_dir "$RUN" --epochs 100 > "$RUN/${stage}.log" 2>&1
  local code=$?
  echo "[$(date -Is)] END $stage code=$code" | tee -a "$RUN/pipeline.log"
  if [ "$code" -ne 0 ]; then
    echo "$code" > "$RUN/exit_code.txt"
    echo "failed:$stage" > "$RUN/status.txt"
    exit "$code"
  fi
}

# Mandatory protocol and leakage checks.
run_stage smoke
run_stage audit
run_stage prepare_a

# Priority experiments requested by the protocol: A and C first.
run_stage features_a
run_stage downstream_a
run_stage reuse_b
run_stage pretrain_c
run_stage features_c
run_stage downstream_c

# Hybrid and raw-waveform comparisons follow after A/C are complete.
run_stage pretrain_d
run_stage features_d
run_stage downstream_d
run_stage cache_wave
run_stage pretrain_e
run_stage features_e
run_stage downstream_e

# Val-only architecture selection, then one locked test and report.
run_stage test
run_stage report
echo 0 > "$RUN/exit_code.txt"
echo "complete" > "$RUN/status.txt"

