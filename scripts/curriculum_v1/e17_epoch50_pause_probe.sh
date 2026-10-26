#!/usr/bin/env bash
set -u

ROOT=/home/ubuntu/HeartLang-main
OUT="$ROOT/results/e17_adversarial_masking"
SRC="$ROOT/scripts/curriculum_v1/run_e17_adversarial_masking.py"
PY="$ROOT/.venv-heartlang/bin/python"
INTERIM="$OUT/interim_epoch50"
LOG="$OUT/epoch50_pause_probe.log"

exec 8>"$OUT/epoch50_pause_probe.lock"
if ! flock -n 8; then
  exit 0
fi

echo $$ > "$OUT/epoch50_pause_probe.pid"
printf '[%s] watcher started\n' "$(date -Iseconds)" >> "$LOG"

while :; do
  pipeline_pid=$(cat "$OUT/pipeline.pid" 2>/dev/null || true)
  train_pid=""
  if [ -n "$pipeline_pid" ]; then
    train_pid=$(pgrep -P "$pipeline_pid" -f 'run_e17_adversarial_masking.py pretrain' | head -n 1 || true)
  fi

  if [ -z "$train_pid" ] || ! kill -0 "$train_pid" 2>/dev/null; then
    printf '[%s] training process unavailable before epoch 50\n' "$(date -Iseconds)" >> "$LOG"
    echo watcher_failed > "$OUT/epoch50_pause_status.txt"
    exit 1
  fi

  checkpoint="$OUT/mlm/adversarial/checkpoint-50.pth"
  if grep -q '^\{"epoch": 50,' "$OUT/pretrain.log" 2>/dev/null && [ -s "$checkpoint" ]; then
    # Wait until torch.save has finished writing the checkpoint.
    size_a=$(stat -c %s "$checkpoint" 2>/dev/null || echo 0)
    sleep 1
    size_b=$(stat -c %s "$checkpoint" 2>/dev/null || echo 0)
    if [ "$size_a" -gt 100000000 ] && [ "$size_a" = "$size_b" ]; then
      worker_pids=$(pgrep -P "$train_pid" || true)
      if [ -n "$worker_pids" ]; then
        kill -STOP $worker_pids 2>/dev/null || true
      fi
      kill -STOP "$train_pid"
      {
        echo "$train_pid"
        echo "$worker_pids"
      } > "$OUT/epoch50_paused_pids.txt"
      echo paused_epoch50_interim_running > "$OUT/status.txt"
      echo paused > "$OUT/epoch50_pause_status.txt"
      printf '[%s] paused training pid=%s after checkpoint-50 size=%s\n' "$(date -Iseconds)" "$train_pid" "$size_b" >> "$LOG"

      mkdir -p "$INTERIM/mlm/adversarial"
      cp -p "$checkpoint" "$INTERIM/mlm/adversarial/checkpoint-best.pth"
      cp -p "$checkpoint" "$INTERIM/mlm/adversarial/checkpoint-50.pth"
      ln -sfn "$OUT/mask_meta" "$INTERIM/mask_meta"
      cat > "$INTERIM/INTERIM_PROTOCOL.txt" <<EOF
Epoch-50 validation-only downstream probe requested by the user.
Source checkpoint: $checkpoint
The main E17 pretraining process is SIGSTOP-paused with its optimizer and RNG state retained in memory.
No Test evaluation is run. Final E17 outputs are not overwritten.
EOF

      printf '[%s] start fixed_mlm\n' "$(date -Iseconds)" >> "$LOG"
      "$PY" "$SRC" fixed_mlm --output_dir "$INTERIM" --epochs 50 > "$INTERIM/fixed_mlm.log" 2>&1 || {
        echo interim_fixed_mlm_failed > "$OUT/epoch50_pause_status.txt"; exit 2;
      }
      printf '[%s] start features\n' "$(date -Iseconds)" >> "$LOG"
      "$PY" "$SRC" features --output_dir "$INTERIM" --epochs 50 > "$INTERIM/features.log" 2>&1 || {
        echo interim_features_failed > "$OUT/epoch50_pause_status.txt"; exit 3;
      }
      printf '[%s] start downstream\n' "$(date -Iseconds)" >> "$LOG"
      "$PY" "$SRC" downstream --output_dir "$INTERIM" --epochs 50 > "$INTERIM/downstream.log" 2>&1 || {
        echo interim_downstream_failed > "$OUT/epoch50_pause_status.txt"; exit 4;
      }
      echo complete > "$INTERIM/COMPLETE"
      echo paused_epoch50_interim_complete > "$OUT/status.txt"
      echo interim_complete > "$OUT/epoch50_pause_status.txt"
      printf '[%s] interim epoch-50 downstream complete; main training remains paused\n' "$(date -Iseconds)" >> "$LOG"
      exit 0
    fi
  fi
  sleep 1
done
