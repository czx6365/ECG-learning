#!/usr/bin/env python3
"""E2 冻结句子模型，统一运行四分类与患者狭窄回归。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

import run_representation_e1_fusion as e1
from ecg_representation_e2 import ECGRepresentationMLM, encode_independent_teacher
from evaluate_vocabulary_pilot import load_model

DATA = e1.ROOT / "datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1"
SID = e1.META


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def extract(model, tokenizer, split, batch_size, device, indices=None):
    wave = np.load(DATA / f"{split}_data.npy", mmap_mode="r")
    sid = np.load(SID / f"{split}_sid.npy", mmap_mode="r")
    chans = np.load(DATA / f"{split}_data_in_chans.npy", mmap_mode="r")
    times = np.load(DATA / f"{split}_data_in_times.npy", mmap_mode="r")
    selected = np.arange(len(wave)) if indices is None else np.asarray(indices, dtype=np.int64)
    out = np.empty((len(selected), model.embed_dim), dtype=np.float32)
    for start in range(0, len(selected), batch_size):
        end = min(start + batch_size, len(selected))
        ix = selected[start:end]
        w = torch.from_numpy(np.asarray(wave[ix]).copy()).float().to(device)
        s = torch.from_numpy(np.asarray(sid[ix]).copy()).long().to(device)
        c = torch.from_numpy(np.asarray(chans[ix]).copy()).long().to(device)
        t = torch.from_numpy(np.asarray(times[ix]).copy()).long().to(device)
        valid, ze, _zq, _sid = encode_independent_teacher(tokenizer, w, c, t)
        if not torch.equal(valid, (s >= 0) & (s < 1024)):
            raise ValueError(f"{split}: 固定 SID 与心拍有效位不一致")
        hidden, key_valid = model(w, s, ze, valid, torch.zeros_like(valid), c, t, return_features=True)
        assert key_valid.shape == (len(w), 257)
        out[start:end] = hidden[:, 0].float().cpu().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError(f"{split}: CLS 非有限")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    ckpt_path = args.pretrained.resolve() if args.pretrained.is_absolute() else (ROOT / args.pretrained).resolve()
    out = args.output.resolve() if args.output.is_absolute() else (ROOT / args.output).resolve()
    if out.exists():
        raise FileExistsError(f"拒绝覆盖: {out}")
    out.mkdir(parents=True)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mode = payload["args"]["mode"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGRepresentationMLM(mode)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False).to(device)
    tokenizer = load_model(e1.TOKENIZER, device).eval().requires_grad_(False)
    data, audit = e1.load_data()
    for split in ("train", "val", "test"):
        # 保存冻结 CLS 缓存以便逐样本复核；该操作不拟合或更新参数。
        cls = extract(model, tokenizer, split, args.batch, device)
        np.save(out / f"{split}_cls.npy", cls)
        data[split]["arrays"]["heartlang"] = cls
    e1.FEATURES = {f"E2-{mode}": ("heartlang",)}
    e1.RUN = out
    # E1 的结果保存器要求超参数可直接写入 JSON；CLI Path 先转换成字符串。
    training_args = SimpleNamespace(**{key: str(value) if isinstance(value, Path) else value
                                       for key, value in vars(args).items()})
    source = {"pretrained_checkpoint": str(ckpt_path), "pretrained_sha256": sha(ckpt_path),
              "tokenizer_checkpoint": str(e1.TOKENIZER), "mode": mode,
              "split_manifest_sha256": audit["manifest_sha256"]}
    e1.dump(out / "data_audit.json", audit)
    e1.dump(out / "sources.json", source)
    results = []
    for task in ("classification", "stenosis"):
        for seed in args.seeds:
            results.append(e1.train_one(data, f"E2-{mode}", task, seed, training_args, device, source_info=source))
            e1.dump(out / "partial_summary.json", results)
    e1.dump(out / "summary.json", results)
    (out / "COMPLETE").touch()


if __name__ == "__main__":
    main()
