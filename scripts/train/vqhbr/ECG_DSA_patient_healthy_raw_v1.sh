#!/bin/bash

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
WORLD_SIZE="${WORLD_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-100}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-10}"
VAL_FREQ="${VAL_FREQ:-10}"
LR="${LR:-5e-5}"
MIN_LR="${MIN_LR:-1e-5}"

"${PYTHON_BIN}" scripts/pipelines/ECG_DSA/run_ecg_dsa_vqhbr_pretrain.py \
    --zip_path \
        "/Users/czx/医疗数据/患者/data_0521.zip" \
        "/Users/czx/医疗数据/健康用户/data.zip" \
        "/Users/czx/医疗数据/健康用户/data_1.zip" \
    --expected_csv_count 38323 \
    --waveform_output_dir datasets/ecg_datasets/ECG_DSA/pretrain_waveforms/patient_healthy_raw_v1 \
    --qrs_output_dir datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v1 \
    --train_output_dir checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v1 \
    --train_log_dir log/vqhbr/ECG_DSA_patient_healthy_raw_v1 \
    --train_ratio 0.9 \
    --val_ratio 0.1 \
    --seed 0 \
    --target_length 1000 \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --world_size "${WORLD_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --save_ckpt_freq "${SAVE_CKPT_FREQ}" \
    --val_freq "${VAL_FREQ}" \
    --codebook_n_emd 8192 \
    --codebook_emd_dim 128 \
    --lr "${LR}" \
    --min_lr "${MIN_LR}" \
    --device "${DEVICE}" \
    --init_checkpoint checkpoints/vqhbr/MIMIC-IV/checkpoint-100.pth \
    --run_training
