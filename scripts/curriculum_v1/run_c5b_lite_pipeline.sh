#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main

RAW="datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1"
SID="datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_patient_split_v1_c21_pattern_shift"
TOKENIZER="checkpoints/curriculum_v1/c21_pattern_shift_from_c2_e50/checkpoint-best.pth"
C5A="checkpoints/curriculum_v1/c5a_lastblock_binary_aux_probe_e50/checkpoint-best.pth"
RUN="results/curriculum_v1/c5b_lite_joint"
CACHE="$RUN/morphology_cache"
SMOKE="checkpoints/curriculum_v1/c5b_lite_joint_smoke_v2"
OUT="checkpoints/curriculum_v1/c5b_lite_joint_e25"

mkdir -p "$RUN"
on_exit() {
  code=$?
  if [[ $code -ne 0 ]]; then
    echo failed > "$RUN/status.txt"
    echo "$code" > "$RUN/exit_code.txt"
  fi
}
trap on_exit EXIT
echo preparing > "$RUN/status.txt"
.venv-heartlang/bin/python scripts/curriculum_v1/verify_frozen_split.py > "$RUN/frozen_split_audit.log" 2>&1
for required in \
  "$RAW/train_data.npy" "$RAW/val_data.npy" "$RAW/test_data.npy" \
  "$SID/train_sid.npy" "$SID/val_sid.npy" "$SID/test_sid.npy" \
  "$TOKENIZER" "$C5A"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
for target in "$SMOKE" "$OUT" "$RUN/COMPLETE"; do
  [[ ! -e "$target" ]] || { echo "Refusing to overwrite existing target: $target" >&2; exit 1; }
done

cat > "$RUN/protocol.txt" <<EOF
experiment=C5b_lite_joint
seed=0
raw_data=$RAW
fixed_old_sid=$SID
tokenizer_init=$TOKENIZER
sentence_and_head_init=$C5A
train_val_test=6640/1344/1366
stage1_epochs=1-5_bridge_warmup_tokenizer_frozen
stage2_epochs=6-10_encoder_adaptation_codebook_frozen
stage3_epochs=11-25_encoder_adaptation_codebook_slow_ema
bridge=SID_embedding_plus_0.1_times_zero_initialized_linear_zq_64_to_768
sentence_trainable=last_transformer_block_plus_final_layernorm
sentence_frozen=first_11_blocks_plus_sid_cls_position_embeddings
tokenizer_trainable=encoder_from_epoch6
decoder_and_pattern_heads=frozen
codebook_update=EMA_epoch11_plus
ema_decay=0.999
loss=four_variance_MSE+0.2_binary_MSE+0.1_C21_tokenizer+0.05_old_SID_anchor
C21_tokenizer_loss=waveform+VQ+0.2_pattern+0.05_shift_latent+0.05_shift_SID
batch_size=8
gradient_accumulation=8
effective_batch_size=64
head_bridge_lr=1e-4
sentence_last_lr=1e-5
morphology_encoder_lr=1e-6
checkpoint_selection=validation_uniform_average_four_output_R2
test_evaluation=once_after_model_selection
EOF

if [[ ! -e "$CACHE/alignment_audit.json" ]]; then
  .venv-heartlang/bin/python scripts/curriculum_v1/prepare_c5b_lite_data.py \
    --raw_data_dir "$RAW" --fixed_sid_dir "$SID" --output_dir "$CACHE" --chunk_records 256 \
    > "$RUN/prepare.log" 2>&1
else
  echo "Reusing validated morphology cache: $CACHE" > "$RUN/prepare.log"
fi

echo smoke > "$RUN/status.txt"
.venv-heartlang/bin/python scripts/curriculum_v1/train_c5b_lite_joint.py \
  --raw_data_dir "$RAW" --fixed_sid_dir "$SID" --morphology_cache_dir "$CACHE" \
  --tokenizer_checkpoint "$TOKENIZER" --c5a_checkpoint "$C5A" --output_dir "$SMOKE" \
  --epochs 3 --stage1_epochs 1 --stage2_epochs 1 --batch_size 4 --accum_iter 1 --num_workers 0 \
  --max_train_records 32 --max_val_records 16 --max_test_records 16 \
  --max_train_steps 1 --max_eval_steps 1 --device cuda --seed 0 \
  > "$RUN/smoke.log" 2>&1

.venv-heartlang/bin/python - <<'PY'
import json, torch
from pathlib import Path
p=Path('checkpoints/curriculum_v1/c5b_lite_joint_smoke_v2')
assert (p/'test_regression_summary.json').is_file()
c=torch.load(p/'checkpoint-3.pth',map_location='cpu',weights_only=False)['model']
new=c['tokenizer.quantize.embedding.weight']; old=c['initial_codebook']
delta=(new-old).abs().max().item()
if not delta > 0:
    raise RuntimeError('Smoke failed: slow EMA stage did not update codebook')
hist=(p/'training_history.csv').read_text()
for stage in ('bridge_warmup','encoder_adaptation','slow_ema_joint'):
    assert stage in hist, stage
print(json.dumps({'smoke_passed':True,'codebook_max_abs_delta':delta}))
PY

echo training > "$RUN/status.txt"
.venv-heartlang/bin/python scripts/curriculum_v1/train_c5b_lite_joint.py \
  --raw_data_dir "$RAW" --fixed_sid_dir "$SID" --morphology_cache_dir "$CACHE" \
  --tokenizer_checkpoint "$TOKENIZER" --c5a_checkpoint "$C5A" --output_dir "$OUT" \
  --epochs 25 --stage1_epochs 5 --stage2_epochs 5 --batch_size 8 --accum_iter 8 --num_workers 8 \
  --head_lr 1e-4 --sentence_lr 1e-5 --encoder_lr 1e-6 --warmup_epochs 2 \
  --lambda_binary 0.2 --lambda_tokenizer 0.1 --lambda_anchor 0.05 \
  --bridge_alpha 0.1 --anchor_temperature 0.1 --shift_temperature 0.1 --shift_radius 4 \
  --ema_decay 0.999 --device cuda --seed 0 \
  > "$RUN/train.log" 2>&1

echo tokenizer_evaluation > "$RUN/status.txt"
.venv-heartlang/bin/python scripts/curriculum_v1/evaluate_c5b_lite_tokenizer.py \
  --c5b_checkpoint "$OUT/tokenizer-checkpoint-best.pth" \
  --output "$RUN/tokenizer_comparison.json" --batch_size 256 \
  > "$RUN/tokenizer_eval.log" 2>&1

.venv-heartlang/bin/python scripts/curriculum_v1/summarize_c5b_lite.py \
  --run_dir "$RUN" --checkpoint_dir "$OUT" > "$RUN/summary.log" 2>&1

cp "$OUT/test_regression_summary.json" "$RUN/test_regression_summary.json"
echo complete > "$RUN/status.txt"
date --iso-8601=seconds > "$RUN/COMPLETE"
echo 0 > "$RUN/exit_code.txt"
echo "C5b-lite complete"
