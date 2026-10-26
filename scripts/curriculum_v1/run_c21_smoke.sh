#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/HeartLang-main

export C2_INDEX_DIR="datasets/ecg_datasets/ECG_DSA/pretrain_indices/curriculum_v1_c1_strict"
export C2_INIT_CHECKPOINT="checkpoints/curriculum_v1/c2_pattern_only_from_c1_e50/checkpoint-best.pth"
export C21_INIT_CHECKPOINT="$C2_INIT_CHECKPOINT"
export C21_SHIFT_LATENT_WEIGHT=0.05
export C21_SHIFT_SID_WEIGHT=0.05
export C21_SHIFT_TEMPERATURE=0.1

OUT="checkpoints/curriculum_v1/c21_shift_smoke"
rm -rf "$OUT"
mkdir -p "$OUT"

.venv-heartlang/bin/python scripts/curriculum_v1/train_c21_shift_invariant.py \
  --pretrain_data_dir datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --phys_cache_dir results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct \
  --output_dir "$OUT" --batch_size 8 --epochs 1 --save_ckpt_freq 1 --val_freq 1 \
  --max_train_steps_per_epoch 2 --max_eval_steps 1 \
  --codebook_n_emd 1024 --codebook_emd_dim 64 --token_len 96 --fs 100 \
  --phys_head_source z_q --loss_schedule static \
  --lambda_waveform 1 --lambda_corr 0 --lambda_deriv 0 --lambda_vq 1 \
  --lambda_phys 0.2 --discrete_alpha 1 --continuous_alpha 0 \
  --continuous_loss_reduction element --continuous_clip_z 0 \
  --ema_decay 0.99 --opt adamw --lr 2.5e-5 --min_lr 5e-6 --warmup_epochs 0 \
  --weight_decay 1e-4 --weight_decay_end 1e-4 --device cuda --seed 0 --num_workers 0

python3 - <<'PY'
import csv, json
from pathlib import Path
p=Path('checkpoints/curriculum_v1/c21_shift_smoke')
cfg=json.loads((p/'config.json').read_text())
rows=list(csv.DictReader((p/'training_history.csv').open()))
required=['train_shift_latent_loss','train_shift_sid_loss','train_shift_sid_agreement','val_shift_sid_agreement']
assert cfg['loss']['corr']==0 and cfg['loss']['derivative']==0
assert cfg['loss']['shift_latent']==0.05 and cfg['loss']['shift_sid']==0.05
assert all(k in rows[-1] for k in required), (required, rows[-1].keys())
assert float(rows[-1]['train_shift_radius']) == 1
print(json.dumps({k: rows[-1][k] for k in required + ['train_shift_radius']}, indent=2))
PY
