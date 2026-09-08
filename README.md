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
