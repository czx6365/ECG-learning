#!/usr/bin/env bash
set -u

ROOT=/home/ubuntu/HeartLang-main
OUT="$ROOT/results/e15_selective_mlm"
PY="$ROOT/.venv-heartlang/bin/python"
SCRIPT="$ROOT/scripts/curriculum_v1/run_e15_selective_mlm.py"

mkdir -p "$OUT"
exec 9>"$OUT/pipeline.lock"
if ! flock -n 9; then
  echo "another E15-C pipeline holds the lock" >&2
  exit 2
fi
echo $$ > "$OUT/pipeline.pid"

finish() {
  code=$?
  echo "$code" > "$OUT/exit_code.txt"
  if [ "$code" -ne 0 ]; then
    echo "failed" > "$OUT/status.txt"
  fi
}
trap finish EXIT

run_stage() {
  stage="$1"
  echo "$stage" > "$OUT/status.txt"
  printf '[%s] start %s\n' "$(date -Iseconds)" "$stage" | tee -a "$OUT/pipeline.log"
  "$PY" "$SCRIPT" "$stage" --output_dir "$OUT" --epochs 100 > "$OUT/$stage.log" 2>&1
  code=$?
  printf '[%s] end %s code=%s\n' "$(date -Iseconds)" "$stage" "$code" | tee -a "$OUT/pipeline.log"
  return "$code"
}

run_stage meta || exit $?
run_stage pretrain || exit $?
run_stage fixed_mlm || exit $?
run_stage features || exit $?
run_stage downstream || exit $?
run_stage report || exit $?
echo complete > "$OUT/status.txt"

