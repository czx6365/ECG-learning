#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/HeartLang-main

INFO_ROOT="results/information_flow"
PIPELINE_PID_FILE="$INFO_ROOT/no_continuous_pipeline.pid"
V1PHYS_EVAL="$INFO_ROOT/codebook/MorphologyV2_vs_V1PlusPhysiology_val_shift4"
FEATURE_ROOT="$INFO_ROOT/representations"
PROBE_OUT="$INFO_ROOT/probes"
MANIFEST="$INFO_ROOT/execution_manifest.txt"

if [[ ! -f "$PIPELINE_PID_FILE" ]]; then
  echo "Missing NoContinuous pipeline PID file: $PIPELINE_PID_FILE" >&2
  exit 1
fi

PIPELINE_PID="$(cat "$PIPELINE_PID_FILE")"
while kill -0 "$PIPELINE_PID" 2>/dev/null; do
  printf '%s waiting for NoContinuous pipeline PID %s\n' "$(date --iso-8601=seconds)" "$PIPELINE_PID"
  sleep 60
done

required_outputs=(
  "checkpoints/pretrain/HeartLangSID_MorphologyV2NoContinuous_SIDOnly_rawv2_e100/checkpoint-best.pth"
  "checkpoints/finetune/ecg_dsa/regression_label4_HeartLangSID_MorphologyV2NoContinuous_SIDOnly_e100_linear/test_regression_summary.json"
  "checkpoints/finetune/ecg_dsa/classification_label4_HeartLangSID_MorphologyV2NoContinuous_SIDOnly_e100_linear_weighted_ce/test_classification_summary.json"
)
for required in "${required_outputs[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "NoContinuous pipeline ended without required output: $required" >&2
    exit 1
  fi
done

bash scripts/experiments/run_v1_plus_physiology_unified_codebook_eval.sh

.venv-heartlang/bin/python scripts/experiments/extract_information_flow_representations.py \
  --run "Morphology-V2,checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth,datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_morphology_v2,checkpoints/pretrain/HeartLangSID_MorphologyV2_SIDOnly_rawv2_e100/checkpoint-best.pth" \
  --run "V1-Plus-Physiology,checkpoints/vqhbr/morphology_v1_plus_physiology_1024_dim64_bs256_e50/checkpoint-best.pth,datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_morphology_v1_plus_physiology,checkpoints/pretrain/HeartLangSID_MorphologyV1PlusPhysiology_SIDOnly_rawv2_e100/checkpoint-best.pth" \
  --run "NoContinuous,checkpoints/vqhbr/morphology_v2_no_continuous_1024_dim64_bs256_e50/checkpoint-50.pth,datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_morphology_v2_no_continuous,checkpoints/pretrain/HeartLangSID_MorphologyV2NoContinuous_SIDOnly_rawv2_e100/checkpoint-best.pth" \
  --waveform_data_dir datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675 \
  --output_dir "$FEATURE_ROOT" \
  --batch_size 64 \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --device cuda

.venv-heartlang/bin/python scripts/experiments/train_information_flow_linear_probes.py \
  --feature_root "$FEATURE_ROOT" \
  --run Morphology-V2 \
  --run V1-Plus-Physiology \
  --run NoContinuous \
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

.venv-heartlang/bin/python scripts/experiments/summarize_information_flow_results.py

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "git_commit=unavailable-no-git-metadata"
  echo "seed=0"
  echo "early_stopping=disabled"
  echo "probe_best_selection=regression:val_r2,classification:val_macro_roc_auc"
  echo "feature_standardization=false"
  echo "pooling=z_e/z_q:masked_mean,HeartLang:CLS"
  echo "v1phys_codebook_table=$V1PHYS_EVAL/comparison_table.csv"
  echo "probe_table=$PROBE_OUT/representation_probe_table.csv"
  echo "code_and_checkpoint_sha256:"
  sha256sum \
    scripts/experiments/train_morphology_v2_physaware_vq.py \
    scripts/eval/codebook/evaluate_morphology_only_comparison.py \
    scripts/experiments/train_heartlang_sid_pretraining.py \
    scripts/experiments/train_heartlang_sid_regression.py \
    scripts/experiments/train_heartlang_sid_classification.py \
    scripts/experiments/extract_information_flow_representations.py \
    scripts/experiments/train_information_flow_linear_probes.py \
    scripts/experiments/summarize_information_flow_results.py \
    checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth \
    checkpoints/vqhbr/morphology_v1_plus_physiology_1024_dim64_bs256_e50/checkpoint-best.pth \
    checkpoints/vqhbr/morphology_v2_no_continuous_1024_dim64_bs256_e50/checkpoint-50.pth \
    checkpoints/pretrain/HeartLangSID_MorphologyV2_SIDOnly_rawv2_e100/checkpoint-best.pth \
    checkpoints/pretrain/HeartLangSID_MorphologyV1PlusPhysiology_SIDOnly_rawv2_e100/checkpoint-best.pth \
    checkpoints/pretrain/HeartLangSID_MorphologyV2NoContinuous_SIDOnly_rawv2_e100/checkpoint-best.pth
} > "$MANIFEST"

echo "Information-flow experiment suite complete."
