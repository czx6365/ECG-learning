#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

OUT="checkpoints/curriculum_v1/c1_word_recon_vq_1024_dim64_bs256_e50"
mkdir -p "$OUT"

if [[ -f "$OUT/train.pid" ]] && kill -0 "$(cat "$OUT/train.pid")" 2>/dev/null; then
  echo "C1 is already running with PID $(cat "$OUT/train.pid")"
  exit 0
fi
if [[ -e "$OUT/checkpoint.pth" || -e "$OUT/log.txt" ]]; then
  echo "Refusing to overwrite existing C1 run: $OUT" >&2
  exit 1
fi

nohup scripts/curriculum_v1/run_c1_train.sh > "$OUT/train_stdout.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/train.pid"
echo "Started C1 with PID $(cat "$OUT/train.pid")"
