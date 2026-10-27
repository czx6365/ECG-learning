# Frozen E3 morphology encoder and codebook

This checkpoint is the exact frozen heartbeat model required by the published
E12-A checkpoint. It converts padded 96-sample heartbeats into continuous
64-dimensional `z_e` features and supplies the matching 1024-entry SID
codebook.

- Model: E3 healthy/patient replay morphology encoder.
- Training milestone: adaptation epoch 30.
- Codebook: `1024 x 64`.
- File size: `24991756` bytes.
- SHA-256: `d39b40f5e73e2317b6b9fb564f79071cf17d61aebbe76177cd9bdac95a356d25`.
- Storage: Git LFS.

Keep this model frozen when reproducing or fine-tuning E12-A. See the E12-A
[usage guide](../../pretrain/E12-A_continuous_ze_sid_mlm/README.md) and
`scripts/curriculum_v1/e12a_quickstart.py`.
