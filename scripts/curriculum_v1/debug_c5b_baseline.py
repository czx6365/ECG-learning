#!/usr/bin/env python3
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch

ROOT = Path('/home/ubuntu/HeartLang-main')
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'/'curriculum_v1')]
import train_c5b_lite_joint as c

args = SimpleNamespace(
    pad_sid=1024, vocab_size=1024, bridge_alpha=0.1,
    anchor_temperature=0.1, shift_temperature=0.1, shift_radius=4,
    codebook_dim=64, token_len=96, ema_decay=0.999,
    seq_len=256, embed_dim=768, depth=12, heads=8, dim_head=64,
    mlp_dim=1024, dropout=0.1, emb_dropout=0.1,
    tokenizer_checkpoint=ROOT/'checkpoints/curriculum_v1/c21_pattern_shift_from_c2_e50/checkpoint-best.pth',
    c5a_checkpoint=ROOT/'checkpoints/curriculum_v1/c5a_lastblock_binary_aux_probe_e50/checkpoint-best.pth',
)
raw = ROOT/'datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1'
sid_dir = ROOT/'datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_patient_split_v1_c21_pattern_shift'
cache = ROOT/'results/curriculum_v1/c5b_lite_joint/morphology_cache'
ds = c.JointDataset(raw, sid_dir, cache, 'val')
items = [ds[i] for i in range(16)]
wave = torch.tensor(np.stack([x[0] for x in items]), dtype=torch.float32, device='cuda')
old = torch.tensor(np.stack([x[1] for x in items]), dtype=torch.long, device='cuda')
ch = torch.tensor(np.stack([x[2] for x in items]), dtype=torch.long, device='cuda')
ti = torch.tensor(np.stack([x[3] for x in items]), dtype=torch.long, device='cuda')
model = c.C5bLiteJoint(args).cuda().eval()
with torch.no_grad():
    p_ref = model.regressor(old, in_chan_matrix=ch, in_time_matrix=ti)
    valid, beats, z, zq, online, qloss = model.tokenizer.encode_valid(wave, in_chan_matrix=ch, in_time_matrix=ti)
    zfull = torch.zeros(wave.shape[0], wave.shape[1], zq.shape[-1], device='cuda')
    zfull[valid] = zq
    p_manual_fixed = model.sentence_forward(old, torch.zeros_like(zfull), old < 1024)
    online_full = torch.full_like(old, 1024); online_full[valid] = online
    p_online = model.sentence_forward(online_full, zfull, valid)
    p_online_zero = model.sentence_forward(online_full, torch.zeros_like(zfull), valid)
print('ref_manual_fixed_maxdiff', (p_ref-p_manual_fixed).abs().max().item())
print('online_all_agree', (online_full==old).float().mean().item())
print('valid_mask_agree', (valid==(old<1024)).float().mean().item())
print('ref_online_zero_mean_abs', (p_ref-p_online_zero).abs().mean().item())
print('ref_online_mean_abs', (p_ref-p_online).abs().mean().item())
print('pred_ref_0', p_ref[0].tolist())
print('pred_online_0', p_online[0].tolist())
