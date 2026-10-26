#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main
EXP=vocabulary_pilot_20261007
RES=results/curriculum_v1/$EXP
mkdir -p "$RES"
exec 9>"$RES/pipeline.lock"
flock -n 9 || { echo 'Pilot already running'; exit 1; }
echo $$ > "$RES/pipeline.pid"
trap 'code=$?; echo "$code" > "$RES/exit_code.txt"' EXIT
.venv-heartlang/bin/python scripts/curriculum_v1/verify_frozen_split.py
export VOCAB_EPOCHS=50 VOCAB_SAMPLES_PER_EPOCH=12288 VOCAB_HEALTHY_EPOCHS=10
export C2_INDEX_DIR=datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict
for mode in mixed natural healthy_then_patient; do
  export VOCAB_MODE=$mode
  OUT=checkpoints/curriculum_v1/${EXP}_${mode}
  if [[ -e "$OUT" ]]; then
    echo "Refusing to overwrite $OUT" >&2; exit 1
  fi
  echo "Training $mode" > "$RES/status.txt"
  .venv-heartlang/bin/python -u scripts/curriculum_v1/train_vocabulary_pilot.py \
    --pretrain_data_dir datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
    --phys_cache_dir results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct \
    --output_dir "$OUT" --log_dir "log/curriculum_v1/${EXP}_${mode}" \
    --batch_size 256 --epochs 50 --save_ckpt_freq 10 --val_freq 5 \
    --codebook_n_emd 1024 --codebook_emd_dim 64 --token_len 96 --fs 100 \
    --phys_head_source z_q --loss_schedule static \
    --lambda_waveform 1 --lambda_corr 0 --lambda_deriv 0 --lambda_vq 1 \
    --lambda_phys 0.2 --discrete_alpha 1 --continuous_alpha 0 \
    --continuous_loss_reduction element --continuous_clip_z 0 \
    --ema_decay 0.99 --opt adamw --lr 2.5e-5 --min_lr 5e-6 --warmup_epochs 2 \
    --weight_decay 1e-4 --weight_decay_end 1e-4 --device cuda --seed 0 --num_workers 6 \
    > "$RES/${mode}_train.log" 2>&1
  echo "Evaluating $mode" > "$RES/status.txt"
  .venv-heartlang/bin/python -u scripts/curriculum_v1/evaluate_vocabulary_pilot.py \
    --checkpoint "$OUT/checkpoint-best.pth" \
    --validation_manifest "$OUT/tokenizer_validation_manifest.csv" \
    --output "$RES/${mode}_validation.json" \
    > "$RES/${mode}_eval.log" 2>&1
done
.venv-heartlang/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('results/curriculum_v1/vocabulary_pilot_20261007')
out={mode:json.loads((p/f'{mode}_validation.json').read_text()) for mode in ('natural','mixed','healthy_then_patient')}
# Identical draw streams are a required part of the sequential-vs-mixed comparison.
configs={mode:json.loads((Path('checkpoints/curriculum_v1')/f'vocabulary_pilot_20261007_{mode}'/'config.json').read_text())['sampling'] for mode in ('mixed','healthy_then_patient')}
for key in ('patient_draw_sha256','healthy_draw_sha256','patient_exposures','healthy_exposures'):
    assert configs['mixed'][key] == configs['healthy_then_patient'][key], key
(p/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
lines=['# Vocabulary pilot: validation only','', '| Mode | z_e Ridge R2 | z_q Ridge R2 | z_e Mean+Max R2 | z_q Mean+Max R2 |','|---|---:|---:|---:|---:|']
for mode,result in out.items():
    values=[result['probes'][key]['best']['overall_r2'] for key in ('z_e','z_q','z_e_mean_max','z_q_mean_max')]
    lines.append('| '+mode+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
lines.extend(['','Matched mixed/curriculum cohort draw hashes verified. No test waveform evaluation.','These are frozen-tokenizer Ridge probes; do not compare numerically to full HeartLang scores.','Expansion and full HeartLang pretraining require review of validation signals.'])
(p/'comparison.md').write_text('\n'.join(lines)+'\n')
PY
echo complete > "$RES/status.txt"
touch "$RES/COMPLETE"
