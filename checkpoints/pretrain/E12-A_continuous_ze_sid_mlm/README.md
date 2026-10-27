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
