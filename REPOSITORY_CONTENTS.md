# ECG-learning Repository Contents

This repository contains the HeartLang/VQ-HBR code, evaluation scripts, and lightweight experiment logs/results.

Large artifacts are intentionally excluded unless explicitly published with Git LFS:

- datasets/
- virtual environments
- checkpoint weight files (*.pth, *.pt, *.ckpt), except documented Git LFS releases such as E12-A
- large array/data files (*.npy, *.npz, *.h5, *.mat)
- TensorBoard event files
- full token assignment dumps

Checkpoint directories may still contain lightweight metadata such as log.txt, train_stdout.log, and polarity_init_summary.json.

The current project checkpoint release is documented in
`checkpoints/pretrain/E12-A_continuous_ze_sid_mlm/README.md` and stored with Git LFS.

The retained `outputs/`, `log/`, and `results/` files are the lightweight summaries, figures, CSV/JSON metrics, and training logs needed to review the current ECG codebook experiments.
