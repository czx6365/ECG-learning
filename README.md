# ECG-learning

This repository contains ECG codebook learning experiments based on HeartLang / VQ-HBR. The current focus is not downstream classification first, but the quality of discrete ECG words:

> Can a SID / ECG word represent heartbeat segments with similar morphology?

The code is adapted from the ICLR 2025 HeartLang project, "Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model". This repository adds ECG_DSA data processing, VQ-HBR codebook experiments, SID quality evaluation, artifact filtering, shift-invariant morphology distance, and polarity-aware VQ-HBR training.

## Repository Contents

```text
heartlang/
  Core model, training entry points, VQ-HBR, quantizer, tokenizer, and utilities.

scripts/
  Data preparation, VQ-HBR training, HeartLang pretraining, and codebook/SID evaluation scripts.

checkpoints/
  Lightweight logs and metadata only. Model weight files are not included.

log/
  Training logs.

outputs/
  Codebook evaluation results, SID quality summaries, CSV/JSON metrics, and selected figures.

results/
  Downstream linear-classifier result files.
```

## Not Included

Large data artifacts are intentionally excluded from Git:

```text
datasets/
.venv/
.venv-heartlang/
*.pth
*.pt
*.ckpt
*.npy
*.npz
*.h5
*.hdf5
*.mat
events.out.tfevents*
token_assignments.csv
```

This repository is meant to keep code, experiment configuration, logs, summaries, and discussion-ready figures. Raw ECG datasets, generated numpy arrays, TensorBoard event streams, full token-assignment dumps, and model weights should be managed separately.

## How to Run / Scripts

The following commands are the main scripts used in the current ECG_DSA experiments. They are written for the remote project layout:

```bash
cd /home/ubuntu/HeartLang-main
conda activate base
source .venv-heartlang/bin/activate
```

Before running training, make sure the excluded artifacts have been restored locally on the server:

```text
datasets/ecg_datasets/ECG_DSA/
checkpoints/vqhbr/*.pth
checkpoints/pretrain/*.pth
```

### 1. Build ECG_DSA Waveform Dataset

Prepare waveform-level ECG_DSA data from raw ECG files.

```bash
python scripts/data_prep/ECG_DSA/make_ecg_dsa_dataset.py \
  --zip_path /path/to/raw_ecg_files.zip \
  --label_xlsx /path/to/label.xlsx \
  --output_dir datasets/ecg_datasets/ECG_DSA/waveforms/base
```

The exact input paths depend on where the raw patient ECG files and labels are mounted.

### 2. Build QRS / Heartbeat Token Dataset

Convert ECG waveforms into QRS heartbeat-token data used by VQ-HBR.

```bash
python scripts/data_prep/ECG_DSA/prepare_ecg_zip_pretrain_batch.py \
  --zip_path /path/to/raw_ecg_files.zip \
  --output_dir datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2
```

If a labeled downstream dataset is needed, copy labels after building the QRS dataset:

```bash
python scripts/data_prep/ECG_DSA/copy_ecg_dsa_labels.py \
  --source_dir datasets/ecg_datasets/ECG_DSA/waveforms/base \
  --target_dir datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label3
```

### 3. Build H Clean QRS Dataset

H clean removes suspicious heartbeat tokens, including long flat segments and badly shifted R peaks.

```bash
python scripts/data_prep/ECG_DSA/make_clean_qrs_dataset.py \
  --src datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2 \
  --dst datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --splits train val \
  --max_flat_run 50 \
  --peak_idx_min 40 \
  --peak_idx_max 65
```

### 4. Train H Clean VQ-HBR Codebook

This is the main clean-codebook experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python heartlang/cli/run_vqhbr_training.py \
  --model vqhbr \
  --pretrain_data_dirs datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --output_dir checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly \
  --log_dir log/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly \
  --codebook_n_emd 1024 \
  --codebook_emd_dim 64 \
  --batch_size 256 \
  --epochs 50 \
  --save_ckpt_freq 50 \
  --val_freq 10 \
  --lr 1e-4 \
  --min_lr 1e-5 \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --device cuda \
  --seed 0 \
  --no_auto_resume
```

### 5. Evaluate SID / Codebook Quality

Basic SID quality evaluation:

```bash
python scripts/eval/codebook/evaluate_sid_quality.py \
  --checkpoint_path checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly/checkpoint-50.pth \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --split val \
  --output_dir outputs/sid_quality/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_H_1024_dim64_val \
  --codebook_size 1024 \
  --codebook_dim 64
```

Shift-invariant pattern-distance evaluation:

```bash
python scripts/eval/codebook/evaluate_shift_invariant_sid_distance.py \
  --checkpoint_path checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly/checkpoint-50.pth \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --split val \
  --output_dir outputs/sid_quality/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_H_1024_dim64_val_shift_corr_pm6 \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --max_shift 6
```

Use `--max_shift 2`, `--max_shift 4`, and `--max_shift 6` to compare different allowed local shifts.

### 6. Generate H Discussion Figures

Create the compact visualization pack used for discussion:

```bash
python scripts/eval/codebook/make_h_visual_discussion_pack.py
```

Key output:

```text
outputs/sid_quality/discussion_H_visual_pack/
```

### 7. Inspect Artifact Filtering

Visualize what would be filtered by H clean:

```bash
python scripts/eval/codebook/inspect_artifact_filter.py \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2 \
  --split val \
  --output_dir outputs/sid_quality/artifact_filter_v2_val_le5pct_actual \
  --max_flat_run 50 \
  --peak_idx_min 40 \
  --peak_idx_max 65
```

Sweep filter thresholds:

```bash
python scripts/eval/codebook/sweep_artifact_filter_thresholds.py \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2 \
  --split val \
  --output_dir outputs/sid_quality/artifact_filter_v2_val_sweep
```

### 8. Train H-Resume Control

This controls for extra training epochs without polarity loss:

```bash
CUDA_VISIBLE_DEVICES=0 python heartlang/cli/run_vqhbr_training.py \
  --model vqhbr \
  --pretrain_data_dirs datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --init_checkpoint checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly/checkpoint-50.pth \
  --output_dir checkpoints/vqhbr/H_resume_control_no_polarity_e50_finalonly \
  --log_dir log/vqhbr/H_resume_control_no_polarity_e50_finalonly \
  --codebook_n_emd 1024 \
  --codebook_emd_dim 64 \
  --batch_size 256 \
  --epochs 50 \
  --save_ckpt_freq 50 \
  --val_freq 10 \
  --lr 1e-4 \
  --min_lr 1e-5 \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --device cuda \
  --seed 0 \
  --no_auto_resume
```

### 9. Train Polarity-Aware Split-5

Split-5 starts from the H clean checkpoint and only targets the five most reliable polarity-mixed SIDs.

```bash
CUDA_VISIBLE_DEVICES=0 python heartlang/cli/run_vqhbr_training.py \
  --model vqhbr \
  --pretrain_data_dirs datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --init_checkpoint checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly/checkpoint-50.pth \
  --output_dir checkpoints/vqhbr/H_polarity_split5_lam0p05_warm5_e50_finalonly \
  --log_dir checkpoints/vqhbr/H_polarity_split5_lam0p05_warm5_e50_finalonly/tb \
  --codebook_n_emd 1024 \
  --codebook_emd_dim 64 \
  --batch_size 256 \
  --epochs 50 \
  --save_ckpt_freq 50 \
  --val_freq 10 \
  --lr 1e-4 \
  --min_lr 1e-5 \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --device cuda \
  --seed 0 \
  --no_auto_resume \
  --polarity_split_sids 823,517,565,65,218 \
  --polarity_dead_code_usage_csv outputs/sid_quality/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_H_1024_dim64_val_shift_corr_pm6/sid_usage_summary.csv \
  --polarity_loss_weight 0.05 \
  --polarity_margin 0.1 \
  --polarity_warmup_epochs 5
```

### 10. Train Polarity-Aware Split-9

Split-9 uses the same H clean checkpoint, not Split-5, and targets nine polarity-mixed SIDs.

```bash
CUDA_VISIBLE_DEVICES=0 python heartlang/cli/run_vqhbr_training.py \
  --model vqhbr \
  --pretrain_data_dirs datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --init_checkpoint checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly/checkpoint-50.pth \
  --output_dir checkpoints/vqhbr/H_polarity_split9_lam0p05_warm5_e50_finalonly \
  --log_dir checkpoints/vqhbr/H_polarity_split9_lam0p05_warm5_e50_finalonly/tb \
  --codebook_n_emd 1024 \
  --codebook_emd_dim 64 \
  --batch_size 256 \
  --epochs 50 \
  --save_ckpt_freq 50 \
  --val_freq 10 \
  --lr 1e-4 \
  --min_lr 1e-5 \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --device cuda \
  --seed 0 \
  --no_auto_resume \
  --polarity_split_sids 823,578,493,200,218,858,517,565,65 \
  --polarity_dead_code_usage_csv outputs/sid_quality/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_H_1024_dim64_val_shift_corr_pm6/sid_usage_summary.csv \
  --polarity_loss_weight 0.05 \
  --polarity_margin 0.1 \
  --polarity_warmup_epochs 5
```

### 11. Evaluate Polarity-Aware Models

Evaluate H clean, H-resume control, Split-5, or Split-9 with the same validation data:

```bash
python scripts/eval/codebook/evaluate_polarity_aware_vqhbr.py \
  --checkpoint_path checkpoints/vqhbr/H_polarity_split9_lam0p05_warm5_e50_finalonly/checkpoint-50.pth \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --split val \
  --output_dir outputs/sid_quality/H_polarity_split9_lam0p05_warm5_e50_full_eval \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --target_pairs 823:3,578:6,493:8,200:9,218:10,858:15,517:16,565:17,65:22
```

For baseline or H-resume control, omit `--target_pairs`.

### 12. HeartLang Pretraining with Split-9 Tokenizer

Use the Split-9 VQ-HBR checkpoint as the tokenizer:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python heartlang/cli/run_heartlang_pretraining.py \
  --model HeartLang \
  --tokenizer_model vqhbr \
  --tokenizer_weight checkpoints/vqhbr/H_polarity_split9_lam0p05_warm5_e50_finalonly/checkpoint-50.pth \
  --pretrain_data_dirs datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2 \
  --output_dir checkpoints/pretrain/HeartLang_pretrain_ECG_DSA_v2_split9_vqhbr1024_dim64_e100 \
  --log_dir log/pretrain/HeartLang_pretrain_ECG_DSA_v2_split9_vqhbr1024_dim64_e100 \
  --batch_size 256 \
  --epochs 100 \
  --save_ckpt_freq 50 \
  --lr 5e-4 \
  --warmup_epochs 5 \
  --min_lr 1e-5 \
  --weight_decay 0.05 \
  --codebook_size 1024 \
  --codebook_dim 64 \
  --auto_tokenizer_config \
  --num_workers 8 \
  --device cuda \
  --seed 0 \
  --no_auto_resume \
  > checkpoints/pretrain/HeartLang_pretrain_ECG_DSA_v2_split9_vqhbr1024_dim64_e100/train_stdout.log 2>&1 &
```

### 13. Downstream Linear Classification

Train a linear classification head on the ECG_DSA three-class task:

```bash
CUDA_VISIBLE_DEVICES=0 python heartlang/cli/run_class_finetuning.py \
  --dataset_dir datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label3 \
  --output_dir checkpoints/finetune/ecg_dsa/finetune_data_0521_label3_multiclass_linear_split9_pretrain_e100_multiclass \
  --log_dir log/finetune/finetune_data_0521_label3_multiclass_linear_split9_pretrain_e100_multiclass \
  --model HeartLang_finetune_base \
  --finetune checkpoints/pretrain/HeartLang_pretrain_ECG_DSA_v2_split9_vqhbr1024_dim64_e100/checkpoint-100.pth \
  --trainable linear \
  --task_head_type linear \
  --pooling_type cls \
  --use_mean_pooling \
  --batch_size 256 \
  --epochs 30 \
  --lr 5e-3 \
  --warmup_epochs 10 \
  --weight_decay 0.05 \
  --nb_classes 3 \
  --no_auto_resume \
  --disable_amp \
  --device cuda \
  --seed 0
```

Evaluate the saved classifier:

```bash
CUDA_VISIBLE_DEVICES=0 python heartlang/cli/run_class_finetuning.py \
  --eval \
  --dataset_dir datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label3 \
  --output_dir checkpoints/finetune/ecg_dsa/finetune_data_0521_label3_multiclass_linear_split9_pretrain_e100_multiclass_eval_best_cli \
  --log_dir log/finetune/finetune_data_0521_label3_multiclass_linear_split9_pretrain_e100_multiclass_eval_best_cli \
  --model HeartLang_finetune_base \
  --resume checkpoints/finetune/ecg_dsa/finetune_data_0521_label3_multiclass_linear_split9_pretrain_e100_multiclass/checkpoint-best.pth \
  --trainable linear \
  --task_head_type linear \
  --pooling_type cls \
  --use_mean_pooling \
  --batch_size 256 \
  --nb_classes 3 \
  --disable_amp \
  --device cuda \
  --seed 0
```

## Main Experiments

### VQ-HBR Codebook Experiments

| Experiment | Setting | Purpose |
|---|---|---|
| B | 1024 codes, dim64, batch 256, no balancing, 30/50 epochs | Main V2 scratch baseline |
| C | B + usage balancing loss | Test whether balancing improves code usage |
| D | B + dead code restart | Test whether restarting inactive codes improves utilization |
| E | B, 50 epochs, final checkpoint only | Stable no-balancing control |
| F | MIMIC init + 1024 codes, dim64, no balancing | Reuse MIMIC encoder/decoder ability, learn a new 1024-word codebook |
| G | MIMIC init + 1024 codes, dim128, no balancing | Test larger code dimension with MIMIC initialization |
| H | V2-clean, 1024 codes, dim64, no balancing, 50 epochs | Retrain after removing suspicious heartbeat artifacts |
| H-resume control | H clean checkpoint + 50 more epochs, no polarity loss | Control for extra training epochs |
| Split-5 | H clean checkpoint + polarity-aware training on 5 SID | Conservative polarity-aware codebook refinement |
| Split-9 | H clean checkpoint + polarity-aware training on 9 SID | Broader polarity-aware codebook refinement |

## H Clean Dataset

H clean removes heartbeat tokens that do not look like normal centered QRS segments. The filtering is intentionally limited to about 5%, so it is not deleting rare heartbeat patterns in bulk.

Removed token types:

1. Long flat segments: `max_flat_run >= 50`
2. R peak too early: `peak_idx < 40`
3. R peak too late: `peak_idx > 65`

Human interpretation:

- Some original V2 heartbeat tokens were cut off, shifted, padded, or nearly flat.
- These artifacts can pollute SID clusters and make a single ECG word contain unrelated shapes.
- H clean removes the worst boundary/preprocessing artifacts before retraining the codebook.

## Key Codebook Results

### V2 Original vs H Clean

| Experiment | Active Codes | Perplexity | Intra RMSE | Inter/Intra RMSE | Intra Corr Dist | Corr Separation | Silhouette |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 original no balancing | 780 | 703.59 | 1.1134 | 1.1517 | 0.7232 | 1.2456 | -0.4669 |
| V2 clean H | 730 | 652.17 | 0.9696 | 1.3048 | 0.5880 | 1.5105 | -0.3123 |

Conclusion:

H clean slightly reduces code coverage, but same-SID heartbeat segments become more compact and different SIDs become more separable. Filtering about 5% suspicious tokens improves ECG word semantic purity.

## Shift-Invariant Pattern Distance

ECG is a time-series signal. Two heartbeat segments may have the same morphology but be shifted by a few samples. To avoid over-penalizing small temporal offsets, the evaluation includes local shift-invariant correlation distance.

For two heartbeat segments:

1. Shift one segment within a small window, such as `+-2`, `+-4`, or `+-6` samples.
2. Compute correlation distance for each shift.
3. Use the minimum distance as the morphology distance.

This evaluates whether heartbeat patterns are similar, instead of requiring exact point-by-point alignment.

Important note:

`+-2`, `+-4`, and `+-6` are units of sampling points in the 96-point heartbeat token, not seconds or milliseconds.

## Polarity-Aware VQ-HBR

Some high-error SIDs were found to mix positive and negative polarity heartbeat patterns. To address this, polarity-aware VQ-HBR was implemented as a real training method, not as post-hoc relabeling.

Basic idea:

```text
H clean checkpoint
        |
        v
Find polarity-mixed target SIDs
        |
        v
Reuse inactive / low-usage codes as paired negative codes
        |
        v
Positive heartbeat stays near the original SID code
Negative heartbeat is encouraged to move near the paired code
        |
        v
Continue updating Encoder, Decoder, quantizer, and EMA codebook
```

Training uses a small polarity margin loss:

```text
positive latent -> closer to original SID code
negative latent -> closer to paired code
both groups -> farther from the opposite paired code
```

The final evaluation uses true nearest-neighbor SID assignment from the trained model. It does not manually rewrite labels at evaluation time.

### Split-5

Target SIDs:

```text
823, 517, 565, 65, 218
```

Split-5 is the conservative experiment. These SIDs showed clearer positive/negative separation in post-hoc analysis.

### Split-9

Target SIDs:

```text
823, 578, 493, 200, 218, 858, 517, 565, 65
```

Split-9 is broader. It tests whether more polarity-mixed SIDs can be improved, but some of them may also contain extra morphology mixing, not only polarity mixing.

## Important Evaluation Scripts

```text
scripts/eval/codebook/evaluate_sid_quality.py
scripts/eval/codebook/evaluate_shift_invariant_sid_distance.py
scripts/eval/codebook/evaluate_polarity_aware_vqhbr.py
scripts/eval/codebook/analyze_worst_sid.py
scripts/eval/codebook/inspect_artifact_filter.py
scripts/eval/codebook/make_h_visual_discussion_pack.py
scripts/eval/codebook/sweep_artifact_filter_thresholds.py
```

These scripts evaluate:

- codebook utilization
- active code count
- perplexity
- reconstruction quality
- intra / inter distance
- shift-invariant correlation distance
- per-SID intra distance
- polarity consistency
- paired positive-negative separation
- worst-SID visualization

## Current Conclusion

Current findings:

1. The original V2 codebook is not random. Heartbeats inside the same SID are generally more similar than heartbeats from different SIDs.
2. The original V2 codebook is polluted by a small number of badly cut, flat, shifted, or boundary-artifact heartbeat tokens.
3. H clean improves SID compactness and SID separation after removing about 5% suspicious heartbeat tokens.
4. Local shift-invariant correlation distance is important because heartbeat morphology can be similar even with small temporal offsets.
5. Polarity-aware VQ-HBR is a real codebook training modification. It updates the encoder, decoder, quantizer, and EMA codebook, instead of only rewriting SID labels.
6. At the current stage, VQ-HBR codebook quality is the primary research target. HeartLang MLM pretraining and downstream linear classification are secondary checks, not the main success criterion.

## Reproducibility Notes

To fully reproduce the experiments, prepare the excluded artifacts separately:

```text
ECG_DSA raw dataset
V2 clean QRS dataset
VQ-HBR checkpoints
HeartLang pretrain checkpoints
Python/conda environment
```

The repository keeps the code and discussion-ready results. Large datasets and model weights are intentionally not committed.

## Original Reference

This work builds on HeartLang:

```bibtex
@inproceedings{
jin2025reading,
title={Reading Your Heart: Learning {ECG} Words and Sentences via Pre-training {ECG} Language Model},
author={Jiarui Jin and Haoyu Wang and Hongyan Li and Jun Li and Jiahui Pan and Shenda Hong},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=6Hz1Ko087B},
}
```
