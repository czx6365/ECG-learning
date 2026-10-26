#!/usr/bin/env python3
"""只读训练和验证集，检查 E2 checkpoint 到下游 CLS 的链路。"""
import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

from ecg_representation_e2 import ECGRepresentationMLM, encode_independent_teacher
from evaluate_representation_e2 import DATA, SID
from evaluate_vocabulary_pilot import load_model
import run_representation_e1_fusion as e1
import numpy as np

PILOT = ROOT / "results/ecg_representation_comparison/experiment2_hybrid_random_batch64_pilot1024/checkpoint-best.pth"
payload = torch.load(PILOT, map_location="cpu", weights_only=False)
model = ECGRepresentationMLM(payload["args"]["mode"])
model.load_state_dict(payload["model"], strict=True)
model.eval().requires_grad_(False)
teacher = load_model(e1.TOKENIZER, torch.device("cpu")).eval().requires_grad_(False)

with torch.no_grad():
    for split in ("train", "val"):
        w = torch.from_numpy(np.asarray(np.load(DATA / f"{split}_data.npy", mmap_mode="r")[:2]).copy()).float()
        s = torch.from_numpy(np.asarray(np.load(SID / f"{split}_sid.npy", mmap_mode="r")[:2]).copy()).long()
        c = torch.from_numpy(np.asarray(np.load(DATA / f"{split}_data_in_chans.npy", mmap_mode="r")[:2]).copy()).long()
        t = torch.from_numpy(np.asarray(np.load(DATA / f"{split}_data_in_times.npy", mmap_mode="r")[:2]).copy()).long()
        valid, ze, _zq, _online = encode_independent_teacher(teacher, w, c, t)
        assert torch.equal(valid, (s >= 0) & (s < 1024))
        hidden, key_valid = model(w, s, ze, valid, torch.zeros_like(valid), c, t, return_features=True)
        assert hidden.shape == (2, 257, 768) and key_valid.shape == (2, 257)
        assert torch.isfinite(hidden[:, 0]).all()
        print(f"{split}: CLS {tuple(hidden[:, 0].shape)}, valid={valid.sum().item()}")
print("E2 下游特征链路 smoke 通过；未使用测试集。")
