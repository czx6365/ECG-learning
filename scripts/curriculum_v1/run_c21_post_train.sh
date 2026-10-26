#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
TRAIN_PID=$(cat results/curriculum_v1/c21_pattern_shift_train/train.pid)
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
CKPT=checkpoints/curriculum_v1/c21_pattern_shift_from_c2_e50/checkpoint-best.pth
if [[ ! -f "$CKPT" ]] || ! grep -q 'Training time' results/curriculum_v1/c21_pattern_shift_train/train_stdout.log; then
  echo "C2.1 training failed or incomplete" >&2; exit 1
fi
.venv-heartlang/bin/python scripts/curriculum_v1/evaluate_c21_tokenizers.py \
  --output results/curriculum_v1/c21_pattern_shift_train/tokenizer_comparison.json --batch_size 256
bash scripts/curriculum_v1/run_c21_sid_linear_probe.sh
