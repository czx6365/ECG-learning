#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

FEATURE_ROOT="results/information_flow/mean_max_representations"
PROBE_OUT="results/information_flow/mean_max_probes"
TOKENIZER="checkpoints/vqhbr/morphology_v2_no_continuous_1024_dim64_bs256_e50/checkpoint-50.pth"
SID_DATA="datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_morphology_v2_no_continuous"
HEARTLANG="checkpoints/pretrain/HeartLangSID_MorphologyV2NoContinuous_SIDOnly_rawv2_e100/checkpoint-best.pth"

for path in "$FEATURE_ROOT" "$PROBE_OUT"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite existing output: $path" >&2
    exit 1
  fi
done

.venv-heartlang/bin/python scripts/experiments/extract_information_flow_representations.py \
  --run "NoContinuous,$TOKENIZER,$SID_DATA,$HEARTLANG" \
  --waveform_data_dir datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675 \
  --output_dir "$FEATURE_ROOT" \
  --batch_size 64 \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --device cuda

.venv-heartlang/bin/python scripts/experiments/train_information_flow_linear_probes.py \
  --feature_root "$FEATURE_ROOT" \
  --run NoContinuous \
  --representation z_e \
  --representation z_e_mean_max \
  --representation z_q \
  --representation z_q_mean_max \
  --output_dir "$PROBE_OUT" \
  --batch_size 64 \
  --epochs 30 \
  --lr 5e-4 \
  --min_lr 1e-5 \
  --warmup_lr 1e-6 \
  --warmup_epochs 5 \
  --weight_decay 0.05 \
  --clip_grad 3.0 \
  --class_weight normalized_inverse_freq \
  --seed 0 \
  --device cuda

sha256sum \
  scripts/experiments/extract_information_flow_representations.py \
  scripts/experiments/train_information_flow_linear_probes.py \
  "$TOKENIZER" \
  "$HEARTLANG" \
  > "$PROBE_OUT/execution_sha256.txt"

echo "NoContinuous mean versus mean+max probe complete."
