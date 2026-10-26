#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
EXP=e3_healthy_patient_replay_20261011
RUN=results/curriculum_v1/$EXP
mkdir -p "$RUN"
exec 9>"$RUN/pipeline.lock"
flock -n 9 || { echo 'E3 pipeline already running' >&2; exit 1; }
echo $$ > "$RUN/pipeline.pid"
trap 'code=$?; echo "$code" > "$RUN/exit_code.txt"; if [[ "$code" != 0 ]]; then echo failed > "$RUN/status.txt"; fi' EXIT
PY=.venv-heartlang/bin/python
DATA=datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct
CACHE=results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct
PARENT=datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict
MAIN=datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1
INIT=checkpoints/curriculum_v1/healthy_only_c21_full_20261008_c21_shift_e50/checkpoint-best.pth
C21=checkpoints/curriculum_v1/${EXP}_adapt_e30
FULL=datasets/ecg_datasets/ECG_DSA/pretrain_sid/${EXP}_full
STRICT=datasets/ecg_datasets/ECG_DSA/pretrain_sid/${EXP}_strict_mixed
DOWN=datasets/ecg_datasets/ECG_DSA/qrs_sid/$EXP
MLM=checkpoints/curriculum_v1/${EXP}_sid_mlm_e100
LINEAR=checkpoints/curriculum_v1/${EXP}_linear_e30
MLP=checkpoints/curriculum_v1/${EXP}_mlp2_e30
for target in "$C21" "$FULL" "$STRICT" "$DOWN" "$MLM" "$LINEAR" "$MLP"; do
  if [[ -e "$target" ]]; then echo "Refusing to overwrite $target" >&2; exit 1; fi
done
[[ -f "$RUN/reference_threeway/downstream_results.json" && -f "$RUN/preflight_checks.json" ]]
$PY scripts/curriculum_v1/verify_frozen_split.py > "$RUN/frozen_split_audit.json"
export C2_INDEX_DIR="$PARENT" C2_INIT_CHECKPOINT="$INIT" C21_INIT_CHECKPOINT="$INIT"
export C21_SHIFT_LATENT_WEIGHT=.05 C21_SHIFT_SID_WEIGHT=.05 C21_SHIFT_TEMPERATURE=.1
echo e3_patient_heavy_adaptation > "$RUN/status.txt"
$PY -u scripts/curriculum_v1/train_e3_healthy_patient_adaptation.py \
  --pretrain_data_dir "$DATA" --phys_cache_dir "$CACHE" --batch_size 256 --epochs 30 \
  --codebook_n_emd 1024 --codebook_emd_dim 64 --token_len 96 --fs 100 \
  --phys_head_source z_q --loss_schedule static --lambda_waveform 1 --lambda_corr 0 --lambda_deriv 0 \
  --lambda_vq 1 --lambda_phys .2 --discrete_alpha 1 --continuous_alpha 0 \
  --continuous_loss_reduction element --continuous_clip_z 0 --ema_decay .99 --opt adamw \
  --lr 5e-6 --min_lr 1e-6 --warmup_epochs 2 --weight_decay 1e-4 --weight_decay_end 1e-4 \
  --device cuda --seed 0 --num_workers 8 --output_dir "$C21" \
  --log_dir "log/curriculum_v1/${EXP}_adapt" --save_ckpt_freq 5 --val_freq 5 \
  > "$RUN/adaptation_train.log" 2>&1
cp "$C21/adaptation_budget.json" "$RUN/adaptation_budget.json"
echo tokenizer_validation_audit > "$RUN/status.txt"
$PY -u scripts/curriculum_v1/evaluate_vocabulary_pilot.py \
  --checkpoint "$C21/checkpoint-best.pth" \
  --validation_manifest checkpoints/curriculum_v1/vocabulary_pilot_20261007_mixed/tokenizer_validation_manifest.csv \
  --output "$RUN/tokenizer_balanced_validation.json" > "$RUN/tokenizer_balanced_eval.log" 2>&1
echo generating_fixed_sid > "$RUN/status.txt"
$PY -u scripts/curriculum_v1/generate_c2_sid_dataset.py \
  --source_data_dir "$DATA" --output_dir "$FULL" --tokenizer_checkpoint "$C21/checkpoint-best.pth" \
  --splits train val --batch_size 256 --device cuda --codebook_size 1024 --codebook_dim 64 \
  --pad_sid 1024 --symlink_context > "$RUN/pretrain_sid.log" 2>&1
# Sentence MLM deliberately uses the SAME strict mixed pool as C2.1 baseline.
# Adapted tokenizer stays frozen; sentence training uses the original strict mixed pool.
$PY scripts/curriculum_v1/materialize_c1_strict_sid.py \
  --full_sid_dir "$FULL" --source_data_dir "$DATA" --index_dir "$PARENT" --output_dir "$STRICT" \
  > "$RUN/sentence_split_audit.json"
$PY -u scripts/curriculum_v1/generate_c2_sid_dataset.py \
  --source_data_dir "$MAIN" --output_dir "$DOWN" --tokenizer_checkpoint "$C21/checkpoint-best.pth" \
  --splits train val test --batch_size 256 --device cuda --codebook_size 1024 --codebook_dim 64 \
  --pad_sid 1024 --symlink_context > "$RUN/downstream_sid.log" 2>&1

echo frozen_e3_tokenizer_sentence_mlm > "$RUN/status.txt"
$PY -u scripts/experiments/train_heartlang_sid_pretraining.py \
  --sid_data_dir "$STRICT" --output_dir "$MLM" --log_dir "log/curriculum_v1/${EXP}_mlm" \
  --no_spatial_embedding --no_temporal_embedding --batch_size 256 --epochs 100 \
  --save_ckpt_freq 50 --val_freq 10 --vocab_size 1024 --pad_sid 1024 --seq_len 256 \
  --embed_dim 768 --depth 12 --heads 8 --mlp_dim 1024 --mask_ratio .5 \
  --lr 5e-4 --min_lr 1e-5 --warmup_epochs 5 --weight_decay .05 --clip_grad 3 \
  --num_workers 8 --device cuda --seed 0 > "$RUN/mlm_train.log" 2>&1

for head in linear mlp2; do
  echo "fourlabel_${head}" > "$RUN/status.txt"
  if [[ "$head" == linear ]]; then
    OUT="$LINEAR"; LOSS=mse
  else
    OUT="$MLP"; LOSS=variance_normalized_mse
  fi
  $PY -u scripts/experiments/train_heartlang_sid_regression.py \
    --sid_data_dir "$DOWN" --pretrained_checkpoint "$MLM/checkpoint-best.pth" --output_dir "$OUT" \
    --no_spatial_embedding --no_temporal_embedding --batch_size 64 --epochs 30 --save_ckpt_freq 10 \
    --vocab_size 1024 --pad_sid 1024 --seq_len 256 --embed_dim 768 --depth 12 --heads 8 --mlp_dim 1024 \
    --task_head_type "$head" --task_head_hidden_dim 256 --task_head_dropout .1 --loss_type "$LOSS" \
    --lr 5e-4 --min_lr 1e-5 --warmup_epochs 5 --weight_decay .05 --clip_grad 3 \
    --num_workers 8 --device cuda --seed 0 > "$RUN/${head}_train.log" 2>&1
  cp "$OUT/test_regression_summary.json" "$RUN/${head}_test_regression_summary.json"
done
echo generating_complete_report > "$RUN/status.txt"
$PY scripts/curriculum_v1/summarize_e3_pipeline.py --run_dir "$RUN" > "$RUN/report_generation.log" 2>&1
echo complete > "$RUN/status.txt"
touch "$RUN/COMPLETE"
