# Curriculum V1

This directory contains only the new staged curriculum training entry points.
Historical scripts under `scripts/experiments/` and `scripts/train/` remain unchanged.

## Frozen baseline

- Baseline snapshot: `snapshots/baseline_20260926_before_curriculum`
- Data contract: `configs/curriculum_v1/data_contract.json`
- Split builder: `scripts/build_ecg_dsa_label4_patient_healthy_curriculum_split.py`
- Split verifier: `scripts/curriculum_v1/verify_frozen_split.py`

## Non-negotiable data rules

1. Training code may optimize parameters from the `train` split only.
2. The `val` split is evaluation-only and selects checkpoints or schedules.
3. The `test` split is read only for the final locked evaluation.
4. Patient identity is `patient_sn`; healthy identity is the suffix after the final underscore in `csv_name`.
5. Every launcher must write below the four `curriculum_v1` artifact roots in the data contract.
6. Launchers must run the frozen-split verifier before training.

## C1 entry points

- `build_c1_strict_indices.py`: builds the grouped index view and excludes locked four-class validation/test identities.
- `train_c1_word_vq.py`: trains only heartbeat waveform MSE and EMA VQ commitment.
- `run_c1_train.sh`: foreground reproducible C1 run.
- `start_c1_train.sh`: background launcher for the same run.

Later stage launchers will be added here without replacing historical entry points.
