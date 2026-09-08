# ECG-learning Repository Contents

This repository contains the HeartLang/VQ-HBR code, evaluation scripts, and lightweight experiment logs/results.

Large artifacts are intentionally excluded for now:

- datasets/
- virtual environments
- checkpoint weight files (*.pth, *.pt, *.ckpt)
- large array/data files (*.npy, *.npz, *.h5, *.mat)
- TensorBoard event files
- full token assignment dumps

Checkpoint directories may still contain lightweight metadata such as log.txt, train_stdout.log, and polarity_init_summary.json.

The retained `outputs/`, `log/`, and `results/` files are the lightweight summaries, figures, CSV/JSON metrics, and training logs needed to review the current ECG codebook experiments.
