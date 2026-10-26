#!/usr/bin/env python3
"""E9: frozen E3 HeartLang MeanMax + frozen E3 z_e MeanMax fusion.

Primary comparison uses validation four-output uniform-average R2 for checkpoint
selection.  Test is evaluated once after the selected state is restored.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]
import run_e4_aggregation as e4
import run_e7_objective as e7


EXP = "e9_ze_fusion_r2_20261018"
DEFAULT_RUN = ROOT / "results/curriculum_v1" / EXP
DATA = ROOT / "datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675_patient_split_v1"
SID = ROOT / "datasets/ecg_datasets/ECG_DSA/qrs_sid/e3_healthy_patient_replay_20261011"
TOKENIZER = ROOT / "checkpoints/curriculum_v1/e3_healthy_patient_replay_20261011_adapt_e30/checkpoint-best.pth"
MLM = ROOT / "checkpoints/curriculum_v1/e3_healthy_patient_replay_20261011_sid_mlm_e100/checkpoint-best.pth"
E4_FEATURES = ROOT / "outputs/curriculum_v1/e4_e3_aggregation_20261011"
LABEL_NAMES = ["OMI", "NOMI", "Clinical Control", "Healthy"]
METHODS = ["e8_r2", "e8_zero_pad_1664", "ze_only", "e9_fusion"]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_status(run_dir: Path, value: str) -> None:
    (run_dir / "status.txt").write_text(value, encoding="utf-8")
    print(value, flush=True)


def stable_ids(frame: pd.DataFrame, split: str) -> list[str]:
    def val(row, key):
        value = row.get(key, "")
        return "" if pd.isna(value) else str(value)
    ids = []
    for idx, row in frame.iterrows():
        ids.append("|".join([
            split, str(idx), val(row, "cohort"), val(row, "entity_id"),
            val(row, "csv_name"), val(row, "source_index"),
        ]))
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{split}: stable sample ids are not unique")
    return ids


def write_id_file(path: Path, ids: list[str]) -> str:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "sample_id"])
        for i, value in enumerate(ids):
            w.writerow([i, value])
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


@torch.inference_mode()
def extract_ze_features(run_dir: Path, device: torch.device, batch_size: int = 32) -> dict:
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = e4.load_model(TOKENIZER, device)
    tokenizer.eval().requires_grad_(False)
    before = {k: v.detach().cpu().clone() for k, v in tokenizer.state_dict().items()}
    audit = {
        "tokenizer_checkpoint": str(TOKENIZER),
        "tokenizer_sha256": sha256(TOKENIZER),
        "encoder_output": "L2-normalized z_e from MorphologyEncoder, before VQ",
        "pooling": "masked mean + masked max over valid heartbeats",
        "valid_rule": "channel>0 & time>0 & abs(waveform).amax(-1)>1e-8",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        waves = np.load(DATA / f"{split}_data.npy", mmap_mode="r")
        chans = np.load(DATA / f"{split}_data_in_chans.npy", mmap_mode="r")
        times = np.load(DATA / f"{split}_data_in_times.npy", mmap_mode="r")
        labels = np.load(DATA / f"{split}_labels.npy", mmap_mode="r")
        sid_labels = np.load(SID / f"{split}_labels.npy", mmap_mode="r")
        cached_labels = np.load(E4_FEATURES / f"{split}_labels.npy", mmap_mode="r")
        if not (np.array_equal(labels, sid_labels) and np.array_equal(labels, cached_labels)):
            raise RuntimeError(f"{split}: label/order mismatch between waveform, SID and E4 cache")
        source_meta = DATA / f"{split}_metadata.csv"
        sid_meta = SID / f"{split}_metadata.csv"
        if sha256(source_meta) != sha256(sid_meta):
            raise RuntimeError(f"{split}: metadata mismatch between waveform and SID data")
        frame = pd.read_csv(source_meta)
        if len(frame) != len(waves):
            raise RuntimeError(f"{split}: metadata length mismatch")
        ids = stable_ids(frame, split)
        id_sha = write_id_file(feature_dir / f"{split}_sample_ids.csv", ids)

        output = np.empty((len(waves), 128), dtype=np.float32)
        valid_counts = np.empty(len(waves), dtype=np.int16)
        norm_min, norm_max, norm_sum, norm_count = float("inf"), -float("inf"), 0.0, 0
        for start in range(0, len(waves), batch_size):
            end = min(start + batch_size, len(waves))
            wave_np = np.asarray(waves[start:end], dtype=np.float32)
            chan_np = np.asarray(chans[start:end])
            time_np = np.asarray(times[start:end])
            valid_np = (chan_np > 0) & (time_np > 0) & (np.abs(wave_np).max(axis=-1) > 1e-8)
            x = torch.from_numpy(wave_np.copy()).to(device)
            c = torch.from_numpy(chan_np.copy()).long().to(device)
            t = torch.from_numpy(time_np.copy()).long().to(device)
            valid_t = torch.from_numpy(valid_np).to(device)
            model_valid = tokenizer.valid_mask(x, c, t)
            if not torch.equal(valid_t, model_valid):
                raise RuntimeError(f"{split}: valid mask mismatch at rows {start}:{end}")
            ze = tokenizer.encode_to_quantizer_features(x, c, t).float()
            if ze.shape != (end - start, 256, 64):
                raise RuntimeError(f"{split}: unexpected z_e shape {ze.shape}")
            if valid_t.any():
                norms = ze[valid_t].norm(dim=-1)
                norm_min = min(norm_min, float(norms.min()))
                norm_max = max(norm_max, float(norms.max()))
                norm_sum += float(norms.sum())
                norm_count += int(norms.numel())
            count = valid_t.sum(1)
            mean = (ze * valid_t.unsqueeze(-1)).sum(1) / count.clamp_min(1).unsqueeze(-1)
            maximum = ze.masked_fill(~valid_t.unsqueeze(-1), -torch.inf).amax(1)
            maximum[~valid_t.any(1)] = 0
            pooled = torch.cat([mean, maximum], dim=1)
            if not torch.isfinite(pooled).all():
                raise RuntimeError(f"{split}: non-finite pooled z_e")
            output[start:end] = pooled.cpu().numpy()
            valid_counts[start:end] = count.cpu().numpy().astype(np.int16)
            if start == 0 or end == len(waves) or end % (batch_size * 20) == 0:
                print(f"{split}: z_e extracted {end}/{len(waves)}", flush=True)
        old = np.load(E4_FEATURES / f"{split}_z_e_mean_max.npy")
        if old.shape != output.shape:
            raise RuntimeError(f"{split}: E4 z_e cache shape mismatch")
        max_diff = float(np.max(np.abs(old - output)))
        if not np.allclose(old, output, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"{split}: fresh z_e differs from E4 cache, max_abs_diff={max_diff}")
        np.save(feature_dir / f"{split}_ze_mean_max.npy", output)
        np.save(feature_dir / f"{split}_valid_heartbeat_counts.npy", valid_counts)
        mlm = np.load(E4_FEATURES / f"{split}_heartlang_mean_max.npy", mmap_mode="r")
        if mlm.shape != (len(waves), 1536):
            raise RuntimeError(f"{split}: MLM cache shape mismatch {mlm.shape}")
        audit["splits"][split] = {
            "records": len(waves), "sample_id_sha256": id_sha,
            "source_metadata_sha256": sha256(source_meta),
            "labels_sha256": sha256(DATA / f"{split}_labels.npy"),
            "e4_mlm_feature_sha256": sha256(E4_FEATURES / f"{split}_heartlang_mean_max.npy"),
            "fresh_vs_e4_ze_max_abs_diff": max_diff,
            "valid_heartbeats": int(valid_counts.sum()),
            "empty_ecg": int((valid_counts == 0).sum()),
            "valid_count_min": int(valid_counts.min()),
            "valid_count_max": int(valid_counts.max()),
            "valid_count_mean": float(valid_counts.mean()),
            "ze_token_norm_min": norm_min,
            "ze_token_norm_max": norm_max,
            "ze_token_norm_mean": norm_sum / max(norm_count, 1),
            "ze_feature_mean": float(output.mean()),
            "ze_feature_std": float(output.std()),
            "ze_feature_l2_mean": float(np.linalg.norm(output, axis=1).mean()),
            "mlm_feature_mean": float(np.asarray(mlm).mean()),
            "mlm_feature_std": float(np.asarray(mlm).std()),
            "mlm_feature_l2_mean": float(np.linalg.norm(np.asarray(mlm), axis=1).mean()),
        }
    after = tokenizer.state_dict()
    for key, value in before.items():
        if not torch.equal(value, after[key].cpu()):
            raise RuntimeError(f"Frozen tokenizer changed during extraction: {key}")
    audit["frozen_tokenizer_unchanged"] = True
    audit["fresh_features_match_existing_e4_cache"] = True
    dump(feature_dir / "feature_audit.json", audit)
    return audit


class DualHead(nn.Module):
    def __init__(self, input_dim: int, seed: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(0.1))
        self.main = nn.Linear(128, 4)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(83000 + int(seed))
            self.aux = nn.Linear(128, 3)

    def forward(self, x):
        return self.main(self.shared(x))

    def training_outputs(self, x):
        h = self.shared(x)
        return self.main(h), self.aux(h)


def patient_auxiliary_loss(logits, target):
    patient = target != 3
    return nn.functional.cross_entropy(logits[patient], target[patient]) if patient.any() else logits.sum() * 0


def paired_initial_state(seed: int) -> dict:
    e4.set_seed(seed)
    return {k: v.detach().cpu().clone() for k, v in DualHead(1664, seed).state_dict().items()}


def initialize_method(method: str, seed: int, device: torch.device) -> DualHead:
    full = paired_initial_state(seed)
    input_dim = {"e8_r2": 1536, "e8_zero_pad_1664": 1664, "ze_only": 128, "e9_fusion": 1664}[method]
    model = DualHead(input_dim, seed)
    target = model.state_dict()
    for key in target:
        if key == "shared.0.weight":
            if method == "e8_r2":
                target[key].copy_(full[key][:, :1536])
            elif method == "ze_only":
                target[key].copy_(full[key][:, 1536:])
            else:
                target[key].copy_(full[key])
        else:
            target[key].copy_(full[key])
    model.load_state_dict(target, strict=True)
    # Reset the stochastic training stream after dimension-dependent temporary
    # module construction so paired methods receive the same dropout sequence.
    e4.set_seed(seed)
    return model.to(device)


def build_inputs(mlm: dict, ze: dict, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    result = {m: {} for m in METHODS}
    for split in ("train", "val", "test"):
        a = torch.from_numpy(np.asarray(mlm[split])).float().to(device)
        b = torch.from_numpy(np.asarray(ze[split])).float().to(device)
        zeros = torch.zeros_like(b)
        result["e8_r2"][split] = a
        result["e8_zero_pad_1664"][split] = torch.cat([a, zeros], dim=1)
        result["ze_only"][split] = b
        result["e9_fusion"][split] = torch.cat([a, b], dim=1)
    return result


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    raw = np.concatenate([model(x[i:i + 96]).cpu().numpy() for i in range(0, len(y), 96)])
    metrics, scores = e7.classification(raw, y.cpu().numpy(), "CE")
    return metrics, scores, raw


def train_one(method, seed, x, labels, run_dir, args, device):
    out = run_dir / "checkpoints" / method / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=False)
    model = initialize_method(method, seed, device)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "initial_state.pth")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    target = labels["train"].argmax(1)
    steps = len(target) // args.batch_size
    generator = torch.Generator().manual_seed(seed)
    history, best_r2, best_state, best_epoch = [], -float("inf"), None, None
    first_grad = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(target), generator=generator).to(device)
        main_losses, aux_losses = [], []
        for step in range(steps):
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            lr = e4.lr_for_step((epoch - 1) * steps + step, args.epochs * steps, args.warmup_epochs * steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            main, aux = model.training_outputs(x["train"][idx])
            loss_main = nn.functional.cross_entropy(main, target[idx])
            loss_aux = patient_auxiliary_loss(aux, target[idx])
            loss = loss_main + 0.2 * loss_aux
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method}/seed{seed}: non-finite loss")
            loss.backward()
            if epoch == 1 and step == 0:
                grad = model.shared[0].weight.grad.detach()
                first_grad = {"total": float(grad.norm())}
                if method in ("e9_fusion", "e8_zero_pad_1664"):
                    first_grad.update({
                        "mlm_columns": float(grad[:, :1536].norm()),
                        "ze_columns": float(grad[:, 1536:].norm()),
                    })
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            main_losses.append(float(loss_main))
            aux_losses.append(float(loss_aux))
        val, _, _ = evaluate(model, x["val"], labels["val"])
        row = {"epoch": epoch, "train_main_ce": float(np.mean(main_losses)),
               "train_patient_ce": float(np.mean(aux_losses)), "val": val}
        history.append(row)
        if val["r2"] > best_r2:
            best_r2, best_epoch = val["r2"], epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"{method}/seed{seed}: no best checkpoint")
    model.load_state_dict(best_state)
    val, val_scores, val_raw = evaluate(model, x["val"], labels["val"])
    # The only test evaluation in this run occurs here, after validation selection.
    test, test_scores, test_raw = evaluate(model, x["test"], labels["test"])
    torch.save({"model": {k: v.cpu() for k, v in best_state.items()}, "best_epoch": best_epoch,
                "selection": "validation four-output uniform-average R2", "method": method, "seed": seed},
               out / "checkpoint-best.pth")
    dump(out / "history.json", history)
    pred_dir = run_dir / "predictions" / method / f"seed{seed}"
    pred_dir.mkdir(parents=True, exist_ok=False)
    np.save(pred_dir / "val_logits.npy", val_raw)
    np.save(pred_dir / "val_probabilities.npy", val_scores)
    np.save(pred_dir / "test_logits.npy", test_raw)
    np.save(pred_dir / "test_probabilities.npy", test_scores)
    result = {
        "method": method, "seed": seed, "input_dim": int(x["train"].shape[1]),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch, "selection": "val Overall R2", "lambda_aux": 0.2,
        "first_batch_shared_weight_gradient_norm": first_grad, "val": val, "test": test,
    }
    dump(out / "summary.json", result)
    print(json.dumps({"method": method, "seed": seed, "best_epoch": best_epoch,
                      "val_r2": val["r2"], "test_r2": test["r2"]}), flush=True)
    return result


def aggregate(results: list[dict]) -> dict:
    output = {"per_seed": results, "methods": {}}
    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        block = {}
        for split in ("val", "test"):
            keys = ["r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy"]
            block[split] = {}
            for key in keys:
                vals = np.array([r[split][key] for r in rows], dtype=float)
                block[split][key] = {"mean": float(vals.mean()), "std_ddof1": float(vals.std(ddof=1)), "values": vals.tolist()}
            block[split]["per_output_r2"] = {}
            for label in LABEL_NAMES:
                vals = np.array([r[split]["per_output_r2"][label] for r in rows], dtype=float)
                block[split]["per_output_r2"][label] = {"mean": float(vals.mean()), "std_ddof1": float(vals.std(ddof=1)), "values": vals.tolist()}
        block["trainable_parameters"] = rows[0]["trainable_parameters"]
        output["methods"][method] = block
    base = output["methods"]["e8_r2"]["test"]["r2"]["mean"]
    matched = output["methods"]["e8_zero_pad_1664"]["test"]["r2"]["mean"]
    fusion = output["methods"]["e9_fusion"]["test"]["r2"]["mean"]
    output["contrasts"] = {
        "e9_minus_e8_r2_test_mean": fusion - base,
        "e9_minus_parameter_matched_zero_pad_test_mean": fusion - matched,
        "selection_rule": "all contrasts are descriptive after per-seed validation-R2 selection; no test-based model selection",
    }
    return output


def write_csvs(run_dir: Path, results: list[dict], summary: dict) -> None:
    metrics = run_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    with (metrics / "results_by_seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "parameters", "best_epoch", "split", "overall_r2", *[f"{x}_r2" for x in LABEL_NAMES], "mse", "mae", "macro_auc", "macro_f1", "accuracy"])
        for r in results:
            for split in ("val", "test"):
                m = r[split]
                w.writerow([r["method"], r["seed"], r["trainable_parameters"], r["best_epoch"], split, m["r2"],
                            *[m["per_output_r2"][x] for x in LABEL_NAMES], m["mse"], m["mae"], m["macro_roc_auc"], m["macro_f1"], m["accuracy"]])
    with (metrics / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["method", "parameters", "split", "metric", "mean", "std_ddof1"])
        for method, b in summary["methods"].items():
            for split in ("val", "test"):
                for metric in ("r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy"):
                    w.writerow([method, b["trainable_parameters"], split, metric, b[split][metric]["mean"], b[split][metric]["std_ddof1"]])
    with (metrics / "per_class_r2.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["method", "split", "class", "mean_r2", "std_ddof1"])
        for method, b in summary["methods"].items():
            for split in ("val", "test"):
                for label in LABEL_NAMES:
                    x = b[split]["per_output_r2"][label]
                    w.writerow([method, split, label, x["mean"], x["std_ddof1"]])


def write_report(run_dir: Path, summary: dict, feature_audit: dict) -> None:
    names = {"e8_r2": "E8-R2", "e8_zero_pad_1664": "E8 Zero-Pad 1664", "ze_only": "z_e Only", "e9_fusion": "E9 Fusion"}
    lines = [
        "# E9：连续 z_e 与 Sentence MLM Mean+Max 融合实验", "",
        "## 结论", "",
    ]
    e8 = summary["methods"]["e8_r2"]["test"]["r2"]["mean"]
    e9 = summary["methods"]["e9_fusion"]["test"]["r2"]["mean"]
    matched = summary["methods"]["e8_zero_pad_1664"]["test"]["r2"]["mean"]
    direction = "提高" if e9 > e8 else "降低"
    lines += [f"E9 Fusion 三种子 Test Overall R² 均值为 **{e9:.6f}**，E8-R2 为 **{e8:.6f}**，{direction} **{e9-e8:+.6f}**。",
              f"相对同参数量 Zero-Pad 对照（{matched:.6f}）差值为 **{e9-matched:+.6f}**。所有 checkpoint 只按验证 Overall R² 选择。", ""]
    lines += ["## 实际数据流", "", "`ECG→冻结E3 MorphologyEncoder→z_e Mean+Max(128)` 与 `固定SID→冻结E3 Sentence Transformer→最后层有效token Mean+Max(1536)` 拼接为1664维；只训练共享128维层、四分类主头和患者三分类辅助头。", ""]
    lines += ["## 冻结特征与对齐审计", "", f"Tokenizer: `{TOKENIZER}`  ", f"Sentence MLM: `{MLM}`  ", "E3 tokenizer 参数提取前后完全一致；新提取 z_e 与 E4 原缓存逐项 allclose；waveform/SID/E4 标签完全一致；metadata 哈希一致。", ""]
    lines += ["| split | records | valid heartbeats | empty ECG | z_e norm mean | fresh/E4 max diff |", "|---|---:|---:|---:|---:|---:|"]
    for split, x in feature_audit["splits"].items():
        lines.append(f"| {split} | {x['records']} | {x['valid_heartbeats']} | {x['empty_ecg']} | {x['ze_token_norm_mean']:.8f} | {x['fresh_vs_e4_ze_max_abs_diff']:.3g} |")
    lines += ["", "## 三种子结果", "", "| 方法 | 参数 | Val R² mean±sd | Test R² mean±sd | ΔTest vs E8 |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        b = summary["methods"][method]
        vr, tr = b["val"]["r2"], b["test"]["r2"]
        lines.append(f"| {names[method]} | {b['trainable_parameters']} | {vr['mean']:.6f} ± {vr['std_ddof1']:.6f} | {tr['mean']:.6f} ± {tr['std_ddof1']:.6f} | {tr['mean']-e8:+.6f} |")
    lines += ["", "## Test 各类别 R²", "", "| 方法 | OMI | NOMI | Clinical Control | Healthy | Overall |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        b = summary["methods"][method]["test"]
        lines.append("| " + names[method] + " | " + " | ".join([f"{b['per_output_r2'][x]['mean']:.6f}" for x in LABEL_NAMES] + [f"{b['r2']['mean']:.6f}"]) + " |")
    lines += ["", "## 训练协议", "", "- Seed 0/1/2；30 epoch；batch 64；AdamW；峰值LR 5e-4；5轮warmup；cosine到1e-5；weight decay 0.05；clip 3。", "- 损失为四分类 CE + 0.2×患者三分类 CE；Healthy 对辅助头零梯度。", "- E8与E9共享部分采用配对初始化；E8使用Fusion初始矩阵前1536列，z_e Only使用后128列。", "- E8 Zero-Pad 与 E9 使用完全相同1664维模型和初始权重，末128维分别为零或真实z_e，用于排除名义参数量差异。", "- 每个epoch仅评估val；恢复val R²最佳状态后，test只评估一次。", ""]
    lines += ["## 解释边界", "", "该实验可判断冻结连续形态表示在当前MLM Mean+Max之外是否提供可解码增量；三种子仍不能替代外部队列。Softmax分数未经温度校准，R²是队列级拟合指标，不是单个患者置信区间。", ""]
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def smoke(run_dir: Path, device: torch.device) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    e4.set_seed(0)
    y = torch.eye(4, device=device).repeat(2, 1)
    mlm = {s: np.random.default_rng(0).normal(size=(8, 1536)).astype(np.float32) for s in ("train", "val", "test")}
    ze = {s: np.random.default_rng(1).normal(size=(8, 128)).astype(np.float32) for s in ("train", "val", "test")}
    inputs = build_inputs(mlm, ze, device)
    a = initialize_method("e9_fusion", 0, device)
    b = initialize_method("e8_zero_pad_1664", 0, device)
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key])
    logits, aux = a.training_outputs(inputs["e9_fusion"]["train"])
    loss = nn.functional.cross_entropy(logits, y.argmax(1)) + 0.2 * patient_auxiliary_loss(aux, y.argmax(1))
    loss.backward()
    assert torch.isfinite(loss)
    dump(run_dir / "smoke.json", {"passed": True, "fusion_shape": list(inputs["e9_fusion"]["train"].shape), "paired_full_initialization": True})
    (run_dir / "COMPLETE").touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=3.0)
    args = parser.parse_args()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_only:
        smoke(args.output_dir, device)
        return
    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "pipeline.pid").write_text(str(__import__("os").getpid()))
    start = time.time()
    protocol = {
        "experiment": EXP, "methods": METHODS, "seeds": [0, 1, 2],
        "primary_metric": "four-output uniform-average R2 on softmax probabilities vs one-hot",
        "selection": "validation Overall R2 only; earliest strict maximum",
        "test_policy": "once per method/seed after checkpoint selection",
        "architecture": "E8 1536 or z_e 128 or concat 1664 -> Linear128/GELU/Dropout.1 -> main4 + patient_aux3",
        "loss": "CE_main + .2*CE_patient_aux; healthy masked from auxiliary",
        "epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "AdamW",
        "lr": args.lr, "min_lr": args.min_lr, "warmup_lr": args.warmup_lr,
        "warmup_epochs": args.warmup_epochs, "weight_decay": args.weight_decay, "clip": args.clip,
        "feature_standardization": False, "tokenizer": str(TOKENIZER), "heartlang": str(MLM),
        "e4_features": str(E4_FEATURES), "no_encoder_codebook_or_mlm_updates": True,
        "parameter_control": "e8_zero_pad_1664 has identical architecture/initial state to E9 but zero last128 inputs",
    }
    dump(run_dir / "protocol.json", protocol)
    sources = [TOKENIZER, MLM, ROOT / "scripts/curriculum_v1/run_e4_aggregation.py",
               ROOT / "scripts/curriculum_v1/run_e7_objective.py", Path(__file__)]
    source_hashes = {str(p): sha256(p) for p in sources}
    dump(run_dir / "source_fingerprints.json", source_hashes)
    try:
        set_status(run_dir, "extracting_and_auditing_ze")
        feature_audit = extract_ze_features(run_dir, device)
        mlm = {s: np.load(E4_FEATURES / f"{s}_heartlang_mean_max.npy", mmap_mode="r") for s in ("train", "val", "test")}
        ze = {s: np.load(run_dir / "features" / f"{s}_ze_mean_max.npy", mmap_mode="r") for s in ("train", "val", "test")}
        labels = {s: torch.from_numpy(np.load(DATA / f"{s}_labels.npy").astype(np.float32)).to(device) for s in ("train", "val", "test")}
        if [len(labels[s]) for s in ("train", "val", "test")] != [6640, 1344, 1366]:
            raise RuntimeError("Frozen split counts changed")
        inputs = build_inputs(mlm, ze, device)
        results = []
        for method in METHODS:
            for seed in (0, 1, 2):
                set_status(run_dir, f"training_{method}_seed{seed}")
                result = train_one(method, seed, inputs[method], labels, run_dir, args, device)
                results.append(result)
                dump(run_dir / "partial_results.json", results)
        summary = aggregate(results)
        dump(run_dir / "metrics" / "aggregate_summary.json", summary)
        write_csvs(run_dir, results, summary)
        write_report(run_dir, summary, feature_audit)
        if any(sha256(p) != source_hashes[str(p)] for p in sources):
            raise RuntimeError("A frozen source changed during E9")
        dump(run_dir / "completion_audit.json", {
            "source_hashes_unchanged": True, "split_counts": {s: len(labels[s]) for s in labels},
            "all_runs": len(results), "elapsed_seconds": time.time() - start,
            "test_evaluations_per_run": 1, "selection_uses_test": False,
        })
        set_status(run_dir, "complete")
        (run_dir / "exit_code.txt").write_text("0")
        (run_dir / "COMPLETE").touch()
    except BaseException:
        set_status(run_dir, "failed")
        (run_dir / "exit_code.txt").write_text("1")
        raise


if __name__ == "__main__":
    main()
