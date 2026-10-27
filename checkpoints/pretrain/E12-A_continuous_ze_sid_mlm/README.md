# E12-A continuous z_e masked SID checkpoint

`checkpoint-best.pth` is the validation-selected E12-A sentence-pretraining checkpoint used by the downstream ECG_DSA experiments.

## Model and training protocol

- Input: frozen E3 MorphologyEncoder continuous `z_e` features (`64` dimensions per valid heartbeat).
- Target: frozen `1024 x 64` E3 codebook SID.
- Objective: cross-entropy over 50% randomly masked valid heartbeat positions.
- Padding: explicit key-padding mask; padding is never an SID target.
- Sentence model: the E12-A Transformer initialized from scratch.
- Pretraining data: 87,066 train ECGs and 8,941 validation ECGs.
- Training: 100 epochs, batch size 256, seed 0.
- Optimizer: AdamW, learning rate `5e-4` to `1e-5`, 5 warmup epochs, weight decay `0.05`, gradient clipping `3.0`.
- Selection: minimum validation masked-SID cross-entropy; best epoch 100.

## Validation evidence

- Training-run best validation CE: `1.2768266847`.
- Fixed-mask validation CE: `1.2785441207`.
- Fixed-mask masked SID accuracy: `0.6005020244`.
- Downstream four-output Validation Overall R-squared, mean over seeds 0/1/2: `0.3829476476`.

The downstream result used frozen Transformer Mean+Max pooling plus the E11 regional residual head. No Test result was used to select this checkpoint.

## Integrity

- File size: `155463072` bytes.
- SHA-256: `7d6d681489dccb0d0a8687d6a4ebcb6cc4b845007762019634845eb4f86a2c61`.
- Storage: Git LFS.

This checkpoint depends on the matching frozen E3 MorphologyEncoder/codebook and the continuous-input E12-A model code. It is not the original HeartLang discrete-token checkpoint.

## Use from a fresh checkout

Install Git LFS before cloning, or run `git lfs pull` after cloning. Both
required checkpoints are published in this repository:

- E3 heartbeat encoder/codebook: `checkpoints/vqhbr/E3_morphology_encoder_codebook/checkpoint-best.pth`
- E12-A sentence encoder: `checkpoints/pretrain/E12-A_continuous_ze_sid_mlm/checkpoint-best.pth`

The portable model definition is `heartlang/models/modeling_e12a.py`. To check
the complete raw-heartbeat path and create a 1536-dimensional Mean+Max feature:

```bash
python scripts/curriculum_v1/e12a_quickstart.py \
  --input your_ecg_sentences.npz \
  --output e12a_features.npy
```

The NPZ contract is:

- `waveforms`: float array `[N, 256, 96]`, one padded ECG sentence per row;
- `lead_ids`: integer array `[N, 256]`, with `0` for padding;
- `time_ids`: integer array `[N, 256]`, with `0` for padding.

The output `[N, 1536]` can be passed to a new downstream head. To fine-tune the
E12-A backbone instead of freezing it, call `load_e12a(..., train=True)` and
include its parameters in the optimizer. The E3 encoder and codebook must stay
frozen to match the published protocol.
