#!/usr/bin/env python3
"""E2 真实 ECG 单 batch warm-start pilot；只读取训练划分。"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

from ecg_representation_e2 import ECGRepresentationMLM, encode_independent_teacher, random_valid_mask, warmstart_from_sid_mlm
from evaluate_vocabulary_pilot import load_model

DATA = ROOT / "datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1"
SID = ROOT / "datasets/ecg_datasets/ECG_DSA/qrs_sid/e3_healthy_patient_replay_20261011"
TOKENIZER = ROOT / "checkpoints/curriculum_v1/e3_healthy_patient_replay_20261011_adapt_e30/checkpoint-best.pth"
MLM = ROOT / "checkpoints/curriculum_v1/e3_healthy_patient_replay_20261011_sid_mlm_e100/checkpoint-best.pth"
OUT = ROOT / "results/ecg_representation_comparison/experiment2_warmstart_pilot"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"拒绝覆盖已有 pilot: {OUT}")
    OUT.mkdir(parents=True)
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    take = slice(0, 2)
    wave = torch.from_numpy(np.asarray(np.load(DATA / "train_data.npy", mmap_mode="r")[take]).copy()).to(device)
    chans = torch.from_numpy(np.asarray(np.load(DATA / "train_data_in_chans.npy", mmap_mode="r")[take]).copy()).long().to(device)
    times = torch.from_numpy(np.asarray(np.load(DATA / "train_data_in_times.npy", mmap_mode="r")[take]).copy()).long().to(device)
    sid = torch.from_numpy(np.asarray(np.load(SID / "train_sid.npy", mmap_mode="r")[take]).copy()).long().to(device)
    assert wave.shape == (2, 256, 96) and sid.shape == (2, 256)
    tokenizer = load_model(TOKENIZER, device).eval().requires_grad_(False)
    with torch.no_grad():
        valid, ze, _zq, online_sid = encode_independent_teacher(tokenizer, wave, chans, times)
    assert torch.equal(valid, sid < 1024)
    agreement = float((online_sid[valid] == sid[valid]).float().mean().item())
    mask = random_valid_mask(valid, 0.5, torch.Generator(device="cuda").manual_seed(7))
    # 教师逐心拍编码必须独立：隐藏心拍剧烈变化后，所有可见位置不变。
    changed = wave.clone()
    changed[mask] = torch.randn_like(changed[mask]) * 100
    with torch.no_grad():
        valid2, ze2, _zq2, sid2 = encode_independent_teacher(tokenizer, changed, chans, times)
    assert torch.equal(valid2, valid)
    assert torch.allclose(ze2[valid & ~mask], ze[valid & ~mask], atol=1e-6)
    assert torch.equal(sid2[valid & ~mask], online_sid[valid & ~mask])
    del changed, ze2, sid2
    torch.cuda.empty_cache()
    output = {"source": {"tokenizer": str(TOKENIZER), "sentence": str(MLM),
                         "tokenizer_sha256": sha(TOKENIZER), "sentence_sha256": sha(MLM)},
              "train_only": True, "batch": 2, "valid_tokens": int(valid.sum()),
              "masked_tokens": int(mask.sum()), "online_vs_fixed_sid_agreement": agreement,
              "teacher_visible_independence": True, "models": {}}
    for mode in ("waveform", "sid_only", "hybrid", "ze_only"):
        torch.cuda.reset_peak_memory_stats()
        model = ECGRepresentationMLM(mode).to(device)
        init = warmstart_from_sid_mlm(model, MLM)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = model.mlm_loss(wave, sid, ze, valid, mask, chans, times)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        duration = time.monotonic() - started
        output["models"][mode] = {"loss_one_step": float(loss.item()),
                                  "duration_seconds": duration,
                                  "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                                  "total_parameters": sum(p.numel() for p in model.parameters()),
                                  "warmstart": init}
        print(mode, json.dumps({k: v for k, v in output["models"][mode].items() if k != "warmstart"}), flush=True)
        del model, optimizer, loss
        torch.cuda.empty_cache()
    (OUT / "pilot.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "COMPLETE").touch()


if __name__ == "__main__":
    main()
