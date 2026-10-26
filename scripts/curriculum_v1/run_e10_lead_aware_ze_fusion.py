#!/usr/bin/env python3
"""E10: lead-aware frozen z_e fusion for four-output probability R2.

The frozen E3 morphology encoder and frozen E3 sentence model are reused.
Only the lead aggregation branch and the E8 dual-head classifier are trained.
Checkpoint selection uses validation uniform-average R2; test is evaluated once.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
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
import run_e9_ze_fusion_r2 as e9


EXP = "e10_lead_aware_ze_fusion_20261018"
DEFAULT_RUN = ROOT / "results/curriculum_v1" / EXP
DATA = e9.DATA
SID = e9.SID
TOKENIZER = e9.TOKENIZER
MLM = e9.MLM
E4_FEATURES = e9.E4_FEATURES
E9_RUN = ROOT / "results/curriculum_v1/e9_ze_fusion_r2_20261018"
LABEL_NAMES = e9.LABEL_NAMES
LEAD_CODES = [1, 2, 3, 7, 8, 9, 10, 11, 12]
LEAD_NAMES = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
NEW_METHODS = ["e10a_repeat_control", "e10a_lead", "e10b_repeat_control", "e10b_lead"]
ALL_METHODS = ["e9_fusion", *NEW_METHODS]


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
    return e9.stable_ids(frame, split)


def write_id_file(path: Path, ids: list[str]) -> str:
    return e9.write_id_file(path, ids)


@torch.inference_mode()
def extract_lead_features(run_dir: Path, device: torch.device, batch_size: int = 32) -> dict:
    """Freshly extract frozen z_e and pool within each of the nine actual leads."""
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = e4.load_model(TOKENIZER, device)
    tokenizer.eval().requires_grad_(False)
    before = {k: v.detach().cpu().clone() for k, v in tokenizer.state_dict().items()}
    audit = {
        "tokenizer_checkpoint": str(TOKENIZER),
        "tokenizer_sha256": sha256(TOKENIZER),
        "lead_codes": LEAD_CODES,
        "lead_names": LEAD_NAMES,
        "lead_mapping": dict(zip(map(str, LEAD_CODES), LEAD_NAMES)),
        "encoder_output": "L2-normalized z_e [record,256,64], before VQ",
        "valid_rule": "channel>0 & time>0 & abs(waveform).amax(-1)>1e-8",
        "pooling": "per-lead masked mean+max, empty lead slot is exactly zero",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        waves = np.load(DATA / f"{split}_data.npy", mmap_mode="r")
        chans = np.load(DATA / f"{split}_data_in_chans.npy", mmap_mode="r")
        times = np.load(DATA / f"{split}_data_in_times.npy", mmap_mode="r")
        labels = np.load(DATA / f"{split}_labels.npy", mmap_mode="r")
        sid_labels = np.load(SID / f"{split}_labels.npy", mmap_mode="r")
        e4_labels = np.load(E4_FEATURES / f"{split}_labels.npy", mmap_mode="r")
        if not (np.array_equal(labels, sid_labels) and np.array_equal(labels, e4_labels)):
            raise RuntimeError(f"{split}: label/order mismatch")
        source_meta, sid_meta = DATA / f"{split}_metadata.csv", SID / f"{split}_metadata.csv"
        if sha256(source_meta) != sha256(sid_meta):
            raise RuntimeError(f"{split}: waveform/SID metadata mismatch")
        frame = pd.read_csv(source_meta)
        if len(frame) != len(waves):
            raise RuntimeError(f"{split}: metadata length mismatch")
        id_sha = write_id_file(feature_dir / f"{split}_sample_ids.csv", stable_ids(frame, split))

        n = len(waves)
        lead_out = np.zeros((n, 9, 128), dtype=np.float32)
        lead_present = np.zeros((n, 9), dtype=bool)
        lead_counts = np.zeros((n, 9), dtype=np.int16)
        global_out = np.zeros((n, 128), dtype=np.float32)
        observed_codes: set[int] = set()
        invalid_nonzero_chan = 0
        norm_sum, norm_count = 0.0, 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            wave_np = np.asarray(waves[start:end], dtype=np.float32)
            chan_np = np.asarray(chans[start:end])
            time_np = np.asarray(times[start:end])
            valid_np = (chan_np > 0) & (time_np > 0) & (np.abs(wave_np).max(axis=-1) > 1e-8)
            invalid_nonzero_chan += int(((chan_np > 0) & ~valid_np).sum())
            observed_codes.update(map(int, np.unique(chan_np[valid_np]).tolist()))
            x = torch.from_numpy(wave_np.copy()).to(device)
            c = torch.from_numpy(chan_np.copy()).long().to(device)
            t = torch.from_numpy(time_np.copy()).long().to(device)
            valid = torch.from_numpy(valid_np).to(device)
            if not torch.equal(valid, tokenizer.valid_mask(x, c, t)):
                raise RuntimeError(f"{split}: tokenizer valid mask mismatch at {start}:{end}")
            ze = tokenizer.encode_to_quantizer_features(x, c, t).float()
            if ze.shape != (end - start, 256, 64):
                raise RuntimeError(f"{split}: unexpected z_e shape {tuple(ze.shape)}")
            if valid.any():
                norms = ze[valid].norm(dim=-1)
                norm_sum += float(norms.sum())
                norm_count += int(norms.numel())
            total_count = valid.sum(1)
            total_mean = (ze * valid.unsqueeze(-1)).sum(1) / total_count.clamp_min(1).unsqueeze(-1)
            total_max = ze.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
            total_max[~valid.any(1)] = 0
            global_out[start:end] = torch.cat([total_mean, total_max], 1).cpu().numpy()
            for j, code in enumerate(LEAD_CODES):
                m = valid & (c == code)
                count = m.sum(1)
                mean = (ze * m.unsqueeze(-1)).sum(1) / count.clamp_min(1).unsqueeze(-1)
                maximum = ze.masked_fill(~m.unsqueeze(-1), -torch.inf).amax(1)
                maximum[~m.any(1)] = 0
                lead_out[start:end, j] = torch.cat([mean, maximum], 1).cpu().numpy()
                lead_present[start:end, j] = m.any(1).cpu().numpy()
                lead_counts[start:end, j] = count.cpu().numpy().astype(np.int16)
            if start == 0 or end == n or end % (batch_size * 20) == 0:
                print(f"{split}: lead z_e extracted {end}/{n}", flush=True)

        if sorted(observed_codes) != LEAD_CODES:
            raise RuntimeError(f"{split}: actual lead codes {sorted(observed_codes)} != expected {LEAD_CODES}")
        if np.any(lead_out[~lead_present] != 0):
            raise RuntimeError(f"{split}: missing lead slot is not exactly zero")
        if np.any(lead_counts.sum(1) == 0) and np.any(lead_out[lead_counts.sum(1) == 0] != 0):
            raise RuntimeError(f"{split}: empty ECG has nonzero lead features")

        # Reconstruct the global pool from the nine lead pools and token counts.
        weighted_mean = (lead_out[:, :, :64] * lead_counts[:, :, None]).sum(1)
        weighted_mean /= lead_counts.sum(1).clip(min=1)[:, None]
        lead_max = np.where(lead_present[:, :, None], lead_out[:, :, 64:], -np.inf).max(1)
        lead_max[~lead_present.any(1)] = 0
        reconstructed = np.concatenate([weighted_mean, lead_max], axis=1).astype(np.float32)
        reconstruction_diff = float(np.max(np.abs(reconstructed - global_out)))
        if not np.allclose(reconstructed, global_out, rtol=2e-5, atol=2e-6):
            raise RuntimeError(f"{split}: per-lead pools cannot reconstruct global pool, diff={reconstruction_diff}")

        e9_global_path = E9_RUN / "features" / f"{split}_ze_mean_max.npy"
        e4_global_path = E4_FEATURES / f"{split}_z_e_mean_max.npy"
        e9_global = np.load(e9_global_path, mmap_mode="r")
        e4_global = np.load(e4_global_path, mmap_mode="r")
        e9_diff = float(np.max(np.abs(global_out - e9_global)))
        e4_diff = float(np.max(np.abs(global_out - e4_global)))
        if not np.allclose(global_out, e9_global, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"{split}: fresh global pool differs from E9, diff={e9_diff}")
        if not np.allclose(global_out, e4_global, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"{split}: fresh global pool differs from E4, diff={e4_diff}")
        mlm = np.load(E4_FEATURES / f"{split}_heartlang_mean_max.npy", mmap_mode="r")
        if mlm.shape != (n, 1536):
            raise RuntimeError(f"{split}: unexpected MLM feature shape {mlm.shape}")
        np.save(feature_dir / f"{split}_lead_ze_mean_max.npy", lead_out)
        np.save(feature_dir / f"{split}_lead_present.npy", lead_present)
        np.save(feature_dir / f"{split}_lead_counts.npy", lead_counts)
        np.save(feature_dir / f"{split}_global_ze_mean_max.npy", global_out)
        audit["splits"][split] = {
            "records": n,
            "sample_id_sha256": id_sha,
            "observed_valid_lead_codes": sorted(observed_codes),
            "observed_valid_lead_names": LEAD_NAMES,
            "valid_heartbeats": int(lead_counts.sum()),
            "empty_ecg": int((lead_counts.sum(1) == 0).sum()),
            "invalid_nonzero_channel_tokens": invalid_nonzero_chan,
            "present_records_per_lead": dict(zip(LEAD_NAMES, lead_present.sum(0).astype(int).tolist())),
            "valid_heartbeats_per_lead": dict(zip(LEAD_NAMES, lead_counts.sum(0).astype(int).tolist())),
            "missing_records_per_lead": dict(zip(LEAD_NAMES, (~lead_present).sum(0).astype(int).tolist())),
            "ze_token_norm_mean": norm_sum / max(norm_count, 1),
            "lead_reconstruction_max_abs_diff": reconstruction_diff,
            "fresh_vs_e9_global_max_abs_diff": e9_diff,
            "fresh_vs_e4_global_max_abs_diff": e4_diff,
        }
    for key, value in before.items():
        if not torch.equal(value, tokenizer.state_dict()[key].cpu()):
            raise RuntimeError(f"Frozen tokenizer changed during extraction: {key}")
    audit["frozen_tokenizer_unchanged"] = True
    audit["fresh_global_matches_e9_and_e4"] = True
    dump(feature_dir / "feature_audit.json", audit)
    return audit


class E10A(nn.Module):
    """Flatten nine lead pools and linearly project 1152 -> 128."""
    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91000 + int(seed))
            self.lead_project = nn.Linear(9 * 128, 128)
        self.classifier = e9.DualHead(1664, seed)
        self.classifier.load_state_dict(e9.paired_initial_state(seed), strict=True)

    def encode(self, mlm, lead, mask):
        del mask
        return torch.cat([mlm, self.lead_project(lead.flatten(1))], dim=1)

    def forward(self, mlm, lead, mask):
        return self.classifier(self.encode(mlm, lead, mask))

    def training_outputs(self, mlm, lead, mask):
        return self.classifier.training_outputs(self.encode(mlm, lead, mask))


class E10B(nn.Module):
    """Learn lead importance with masked additive attention over nine pools."""
    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(92000 + int(seed))
            self.lead_embedding = nn.Parameter(torch.empty(9, 128))
            nn.init.normal_(self.lead_embedding, std=0.02)
            self.attention = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = e9.DualHead(1664, seed)
        self.classifier.load_state_dict(e9.paired_initial_state(seed), strict=True)

    def pool(self, lead, mask):
        score = self.attention(lead + self.lead_embedding.unsqueeze(0)).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        weight = torch.softmax(score, dim=1) * mask.float()
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-12)
        pooled = (weight.unsqueeze(-1) * lead).sum(1)
        return pooled, weight

    def encode(self, mlm, lead, mask):
        pooled, weight = self.pool(lead, mask)
        return torch.cat([mlm, pooled], dim=1), weight

    def forward(self, mlm, lead, mask):
        x, _ = self.encode(mlm, lead, mask)
        return self.classifier(x)

    def training_outputs(self, mlm, lead, mask):
        x, _ = self.encode(mlm, lead, mask)
        return self.classifier.training_outputs(x)


def family_of(method: str) -> str:
    return "a" if method.startswith("e10a") else "b"


def build_model(method: str, seed: int, device: torch.device) -> nn.Module:
    model = E10A(seed) if family_of(method) == "a" else E10B(seed)
    e4.set_seed(seed)  # identical dropout/order stream within each paired family
    return model.to(device)


def state_sha(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def input_for(method: str, lead: torch.Tensor, mask: torch.Tensor, global_ze: torch.Tensor) -> torch.Tensor:
    if method.endswith("lead"):
        return lead
    # Same-parameter controls retain E9 global z_e but repeat it into present
    # lead slots. They contain no lead-specific morphology information.
    return global_ze.unsqueeze(1).expand(-1, 9, -1) * mask.unsqueeze(-1)


@torch.no_grad()
def evaluate(model, method, arrays, labels):
    model.eval()
    raw_chunks, weight_chunks = [], []
    for i in range(0, len(labels), 96):
        m = arrays["mask"][i:i + 96]
        lead = input_for(method, arrays["lead"][i:i + 96], m, arrays["global"][i:i + 96])
        if isinstance(model, E10B):
            x, weight = model.encode(arrays["mlm"][i:i + 96], lead, m)
            raw = model.classifier(x)
            weight_chunks.append(weight.cpu().numpy())
        else:
            raw = model(arrays["mlm"][i:i + 96], lead, m)
        raw_chunks.append(raw.cpu().numpy())
    raw = np.concatenate(raw_chunks)
    metrics, scores = e7.classification(raw, labels.cpu().numpy(), "CE")
    weights = np.concatenate(weight_chunks) if weight_chunks else None
    return metrics, scores, raw, weights


def train_one(method, seed, data, labels, run_dir, args, device):
    out = run_dir / "checkpoints" / method / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=False)
    model = build_model(method, seed, device)
    initial_sha = state_sha(model)
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
            lr = e4.lr_for_step((epoch - 1) * steps + step, args.epochs * steps,
                                args.warmup_epochs * steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            mask = data["train"]["mask"][idx]
            lead = input_for(method, data["train"]["lead"][idx], mask, data["train"]["global"][idx])
            main, aux = model.training_outputs(data["train"]["mlm"][idx], lead, mask)
            loss_main = nn.functional.cross_entropy(main, target[idx])
            loss_aux = e9.patient_auxiliary_loss(aux, target[idx])
            loss = loss_main + 0.2 * loss_aux
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method}/seed{seed}: non-finite loss")
            loss.backward()
            if epoch == 1 and step == 0:
                if isinstance(model, E10A):
                    branch_norm = float(model.lead_project.weight.grad.norm())
                else:
                    branch_norm = float(sum((p.grad.norm() ** 2 for p in [model.lead_embedding, *model.attention.parameters()] if p.grad is not None), torch.tensor(0., device=device)).sqrt())
                first_grad = {
                    "branch": branch_norm,
                    "classifier": float(model.classifier.shared[0].weight.grad.norm()),
                }
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            main_losses.append(float(loss_main))
            aux_losses.append(float(loss_aux))
        val, _, _, _ = evaluate(model, method, data["val"], labels["val"])
        history.append({"epoch": epoch, "train_main_ce": float(np.mean(main_losses)),
                        "train_patient_ce": float(np.mean(aux_losses)), "val": val})
        if val["r2"] > best_r2:
            best_r2, best_epoch = val["r2"], epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"{method}/seed{seed}: no selected checkpoint")
    model.load_state_dict(best_state)
    val, val_scores, val_raw, val_weights = evaluate(model, method, data["val"], labels["val"])
    # The only test evaluation is here, after validation-only selection.
    test, test_scores, test_raw, test_weights = evaluate(model, method, data["test"], labels["test"])
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
    attention = None
    if test_weights is not None:
        np.save(pred_dir / "val_attention.npy", val_weights)
        np.save(pred_dir / "test_attention.npy", test_weights)
        present = data["test"]["mask"].cpu().numpy()
        attention = {}
        for j, name in enumerate(LEAD_NAMES):
            values = test_weights[present[:, j], j]
            attention[name] = {"mean_when_present": float(values.mean()), "std_when_present": float(values.std())}
    result = {
        "method": method, "seed": seed, "family": family_of(method),
        "source": "lead_specific" if method.endswith("lead") else "global_ze_repeated_same_parameter_control",
        "input": "MLM1536 + branch128", "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch, "selection": "val Overall R2", "lambda_aux": 0.2,
        "initial_state_sha256": initial_sha, "first_batch_gradient_norm": first_grad,
        "val": val, "test": test, "test_attention": attention,
    }
    dump(out / "summary.json", result)
    print(json.dumps({"method": method, "seed": seed, "best_epoch": best_epoch,
                      "val_r2": val["r2"], "test_r2": test["r2"]}), flush=True)
    return result


def load_e9_results() -> list[dict]:
    path = E9_RUN / "metrics/aggregate_summary.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    rows = [copy.deepcopy(r) for r in source["per_seed"] if r["method"] == "e9_fusion"]
    if sorted(r["seed"] for r in rows) != [0, 1, 2]:
        raise RuntimeError("E9 does not contain exactly seeds 0/1/2")
    for row in rows:
        row["source_experiment"] = str(E9_RUN)
        row["reused_without_retraining"] = True
    return rows


def aggregate(results: list[dict]) -> dict:
    output = {"per_seed": results, "methods": {}}
    for method in ALL_METHODS:
        rows = [r for r in results if r["method"] == method]
        if len(rows) != 3:
            raise RuntimeError(f"{method}: expected three seeds, got {len(rows)}")
        block = {"trainable_parameters": rows[0]["trainable_parameters"]}
        for split in ("val", "test"):
            block[split] = {}
            for key in ("r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy"):
                vals = np.array([r[split][key] for r in rows], dtype=float)
                block[split][key] = {"mean": float(vals.mean()), "std_ddof1": float(vals.std(ddof=1)), "values": vals.tolist()}
            block[split]["per_output_r2"] = {}
            for label in LABEL_NAMES:
                vals = np.array([r[split]["per_output_r2"][label] for r in rows], dtype=float)
                block[split]["per_output_r2"][label] = {"mean": float(vals.mean()), "std_ddof1": float(vals.std(ddof=1)), "values": vals.tolist()}
        output["methods"][method] = block
    mean = lambda method: output["methods"][method]["test"]["r2"]["mean"]
    output["contrasts"] = {
        "e10a_lead_minus_e9": mean("e10a_lead") - mean("e9_fusion"),
        "e10a_lead_minus_same_parameter_repeat_control": mean("e10a_lead") - mean("e10a_repeat_control"),
        "e10b_lead_minus_e9": mean("e10b_lead") - mean("e9_fusion"),
        "e10b_lead_minus_same_parameter_repeat_control": mean("e10b_lead") - mean("e10b_repeat_control"),
        "e10b_lead_minus_e10a_lead": mean("e10b_lead") - mean("e10a_lead"),
        "note": "descriptive test contrasts after independent validation-R2 checkpoint selection",
    }
    return output


def write_csvs(run_dir: Path, results: list[dict], summary: dict) -> None:
    metrics = run_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    with (metrics / "results_by_seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "parameters", "best_epoch", "split", "overall_r2",
                    *[f"{x}_r2" for x in LABEL_NAMES], "mse", "mae", "macro_auc", "macro_f1", "accuracy"])
        for r in results:
            for split in ("val", "test"):
                m = r[split]
                w.writerow([r["method"], r["seed"], r["trainable_parameters"], r["best_epoch"], split,
                            m["r2"], *[m["per_output_r2"][x] for x in LABEL_NAMES], m["mse"], m["mae"],
                            m["macro_roc_auc"], m["macro_f1"], m["accuracy"]])
    with (metrics / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "parameters", "test_r2_mean", "test_r2_sd", "macro_auc_mean", "macro_auc_sd"])
        for method in ALL_METHODS:
            b = summary["methods"][method]
            w.writerow([method, b["trainable_parameters"], b["test"]["r2"]["mean"],
                        b["test"]["r2"]["std_ddof1"], b["test"]["macro_roc_auc"]["mean"],
                        b["test"]["macro_roc_auc"]["std_ddof1"]])


def write_report(run_dir: Path, summary: dict, audit: dict, results: list[dict]) -> None:
    names = {
        "e9_fusion": "E9 全局 z_e",
        "e10a_repeat_control": "E10-A 等参数全局重复对照",
        "e10a_lead": "E10-A 导联独立池化",
        "e10b_repeat_control": "E10-B 等参数全局重复对照",
        "e10b_lead": "E10-B 导联 Attention",
    }
    c = summary["contrasts"]
    best = max((summary["methods"][m]["test"]["r2"]["mean"], m) for m in ALL_METHODS)[1]
    lines = [
        "# E10：Lead-aware z_e Fusion（导联感知连续特征融合）", "",
        "## 结论", "",
        f"三种子 Test Overall R² 均值最高的方法是 **{names[best]}**（{summary['methods'][best]['test']['r2']['mean']:.6f}）。",
        f"E10-A 相对 E9 的差值为 **{c['e10a_lead_minus_e9']:+.6f}**，相对其等参数对照为 **{c['e10a_lead_minus_same_parameter_repeat_control']:+.6f}**。",
        f"E10-B 相对 E9 的差值为 **{c['e10b_lead_minus_e9']:+.6f}**，相对其等参数对照为 **{c['e10b_lead_minus_same_parameter_repeat_control']:+.6f}**；Attention 相对 E10-A 为 **{c['e10b_lead_minus_e10a_lead']:+.6f}**。",
        "这些差值是验证集选模后的描述性测试结果；三种子结果不足以替代外部验证。", "",
        "## 数据链路", "",
        "`ECG → 冻结E3 MorphologyEncoder → z_e[256,64] → 按I/II/III/V1–V6分组 → 每导联Mean+Max[128]`。",
        "E10-A 将九个导联池化结果展平为1152维，再用 Linear(1152,128) 投影；E10-B 使用带导联嵌入的掩码 Attention 聚合为128维。两者均与冻结 E4 MLM Mean+Max 1536维拼接，再进入原 E8 双头分类器。", "",
        "## 导联与缺失审计", "",
        "实际有效导联编码严格为 `1,2,3,7,8,9,10,11,12`，对应 `I,II,III,V1,V2,V3,V4,V5,V6`。padding、缺失导联均通过 mask 排除；空 ECG 的九个导联特征和 Attention 输出均为零。", "",
        "| split | records | valid beats | empty ECG | missing per lead | fresh/E9 max diff |", "|---|---:|---:|---:|---|---:|",
    ]
    for split in ("train", "val", "test"):
        a = audit["splits"][split]
        missing = ", ".join(f"{k}:{v}" for k, v in a["missing_records_per_lead"].items())
        lines.append(f"| {split} | {a['records']} | {a['valid_heartbeats']} | {a['empty_ecg']} | {missing} | {a['fresh_vs_e9_global_max_abs_diff']:.3g} |")
    lines += ["", "## 三种子总体结果", "",
              "| 方法 | 参数量 | Test Overall R² mean±sd | Macro AUC mean±sd | ΔR² vs E9 |",
              "|---|---:|---:|---:|---:|"]
    e9_mean = summary["methods"]["e9_fusion"]["test"]["r2"]["mean"]
    for method in ALL_METHODS:
        b = summary["methods"][method]
        lines.append(f"| {names[method]} | {b['trainable_parameters']} | {b['test']['r2']['mean']:.6f} ± {b['test']['r2']['std_ddof1']:.6f} | {b['test']['macro_roc_auc']['mean']:.6f} ± {b['test']['macro_roc_auc']['std_ddof1']:.6f} | {b['test']['r2']['mean']-e9_mean:+.6f} |")
    lines += ["", "## Test 四类别 R²", "",
              "| 方法 | OMI | NOMI | Clinical Control | Healthy | Overall |", "|---|---:|---:|---:|---:|---:|"]
    for method in ALL_METHODS:
        b = summary["methods"][method]["test"]
        vals = [b["per_output_r2"][x]["mean"] for x in LABEL_NAMES]
        lines.append(f"| {names[method]} | " + " | ".join(f"{v:.6f}" for v in vals) + f" | {b['r2']['mean']:.6f} |")
    lines += ["", "## 每个 seed", "",
              "| 方法 | seed | best epoch | Val R² | Test R² | Macro AUC |", "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(results, key=lambda x: (ALL_METHODS.index(x["method"]), x["seed"])):
        lines.append(f"| {names[r['method']]} | {r['seed']} | {r['best_epoch']} | {r['val']['r2']:.6f} | {r['test']['r2']:.6f} | {r['test']['macro_roc_auc']:.6f} |")
    lines += ["", "## 等参数对照", "",
              "A/B 各自的对照与真实导联模型具有完全相同的结构、参数量和逐 seed 初始状态。对照把 E9 的全局 z_e 复制到所有存在的导联槽，因此保留全局连续形态信息，却不含导联间差异。真实模型相对该对照的差值才用于判断导联信息本身的增量。", "",
              "## 训练协议", "",
              "- 固定患者级 train/val/test = 6640/1344/1366；冻结 Encoder、Codebook 和 MLM。",
              "- seed 0/1/2；30 epoch；batch64；AdamW；峰值LR 5e-4；5轮warmup；cosine至1e-5；weight decay 0.05；clip 3。",
              "- 损失为四分类 CE + 0.2×患者三分类 CE；输出经 Softmax 后与 one-hot 标签计算四输出 uniform-average R²。",
              "- 每个 epoch 只评估验证集；按 Val Overall R² 选择 checkpoint，恢复后测试集只评估一次。", ""]
    b_rows = [r for r in results if r["method"] == "e10b_lead"]
    if b_rows:
        lines += ["## E10-B 测试集平均 Attention（导联存在时）", "",
                  "| lead | seed0 | seed1 | seed2 | mean |", "|---|---:|---:|---:|---:|"]
        for lead in LEAD_NAMES:
            vals = [r["test_attention"][lead]["mean_when_present"] for r in sorted(b_rows, key=lambda x: x["seed"])]
            lines.append(f"| {lead} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {np.mean(vals):.4f} |")
        lines.append("")
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def smoke(run_dir: Path, device: torch.device) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    e4.set_seed(0)
    mlm = torch.randn(5, 1536, device=device)
    lead = torch.randn(5, 9, 128, device=device)
    mask = torch.tensor([[1] * 9, [1, 0, 1, 0, 1, 0, 1, 0, 1], [0] * 9, [1] * 8 + [0], [0, 1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.bool, device=device)
    lead = lead * mask.unsqueeze(-1)
    global_ze = torch.randn(5, 128, device=device)
    global_ze[2] = 0
    checks = {}
    for family, cls in (("a", E10A), ("b", E10B)):
        lead_method, control_method = f"e10{family}_lead", f"e10{family}_repeat_control"
        m1, m2 = cls(0).to(device), cls(0).to(device)
        if state_sha(m1) != state_sha(m2):
            raise RuntimeError(f"E10-{family.upper()}: paired initialization mismatch")
        for method, model in ((lead_method, m1), (control_method, m2)):
            z = input_for(method, lead, mask, global_ze)
            main, aux = model.training_outputs(mlm, z, mask)
            loss = nn.functional.cross_entropy(main, torch.tensor([0, 1, 2, 3, 0], device=device)) + 0.2 * e9.patient_auxiliary_loss(aux, torch.tensor([0, 1, 2, 3, 0], device=device))
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method}: smoke loss is non-finite")
            checks[method] = {"parameters": sum(p.numel() for p in model.parameters()), "loss": float(loss)}
        if checks[lead_method]["parameters"] != checks[control_method]["parameters"]:
            raise RuntimeError(f"E10-{family.upper()}: parameter control mismatch")
    b = E10B(0).to(device)
    repeated = input_for("e10b_repeat_control", lead, mask, global_ze)
    pooled, weights = b.pool(repeated, mask)
    if not torch.allclose(pooled[mask.any(1)], global_ze[mask.any(1)], atol=2e-6, rtol=1e-6):
        raise RuntimeError("E10-B repeat control does not preserve global z_e")
    if not torch.equal(pooled[~mask.any(1)], torch.zeros_like(pooled[~mask.any(1)])):
        raise RuntimeError("E10-B empty ECG output is not zero")
    if not torch.isfinite(weights).all():
        raise RuntimeError("E10-B attention contains non-finite values")
    dump(run_dir / "smoke.json", {"passed": True, "checks": checks,
                                   "attention_mask_and_empty_ecg_passed": True,
                                   "repeat_control_preserves_global_ze": True})
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
    (run_dir / "pipeline.pid").write_text(str(os.getpid()))
    start = time.time()
    protocol = {
        "experiment": EXP, "methods": ALL_METHODS, "new_methods": NEW_METHODS, "seeds": [0, 1, 2],
        "primary_metric": "softmax probabilities vs one-hot, four-output uniform-average R2",
        "selection": "validation Overall R2 only; earliest strict maximum",
        "test_policy": "once per new method/seed after selection; E9 metrics reused from completed experiment",
        "lead_codes": LEAD_CODES, "lead_names": LEAD_NAMES,
        "e10a": "9x128 lead MeanMax -> flatten1152 -> Linear128 -> concat MLM1536 -> E8 dual head",
        "e10b": "9x128 lead MeanMax + learned lead embedding -> masked additive attention ->128 -> concat MLM1536 -> E8 dual head",
        "same_parameter_controls": "replace each lead-specific vector by repeated E9 global z_e in present slots; exact same architecture and initialization",
        "loss": "CE4 + 0.2*patient-only CE3", "epochs": args.epochs, "batch_size": args.batch_size,
        "optimizer": "AdamW", "lr": args.lr, "min_lr": args.min_lr, "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay, "clip": args.clip, "feature_standardization": False,
        "tokenizer": str(TOKENIZER), "sentence_mlm": str(MLM), "e4_features": str(E4_FEATURES),
        "e9_result": str(E9_RUN), "encoder_codebook_mlm_frozen": True,
    }
    dump(run_dir / "protocol.json", protocol)
    sources = [TOKENIZER, MLM, ROOT / "scripts/curriculum_v1/run_e4_aggregation.py",
               ROOT / "scripts/curriculum_v1/run_e7_objective.py",
               ROOT / "scripts/curriculum_v1/run_e9_ze_fusion_r2.py", Path(__file__)]
    source_hashes = {str(p): sha256(p) for p in sources}
    dump(run_dir / "source_fingerprints.json", source_hashes)
    try:
        set_status(run_dir, "extracting_and_auditing_lead_ze")
        audit = extract_lead_features(run_dir, device)
        labels = {s: torch.from_numpy(np.load(DATA / f"{s}_labels.npy").astype(np.float32)).to(device) for s in ("train", "val", "test")}
        if [len(labels[s]) for s in ("train", "val", "test")] != [6640, 1344, 1366]:
            raise RuntimeError("Frozen patient split counts changed")
        data = {}
        for split in ("train", "val", "test"):
            data[split] = {
                "mlm": torch.from_numpy(np.asarray(np.load(E4_FEATURES / f"{split}_heartlang_mean_max.npy", mmap_mode="r"))).float().to(device),
                "lead": torch.from_numpy(np.load(run_dir / "features" / f"{split}_lead_ze_mean_max.npy")).float().to(device),
                "mask": torch.from_numpy(np.load(run_dir / "features" / f"{split}_lead_present.npy")).bool().to(device),
                "global": torch.from_numpy(np.load(run_dir / "features" / f"{split}_global_ze_mean_max.npy")).float().to(device),
            }

        # Audit exact per-family paired initialization and parameter counts.
        pair_audit = {}
        for family in ("a", "b"):
            for seed in (0, 1, 2):
                lead_model = build_model(f"e10{family}_lead", seed, device)
                control_model = build_model(f"e10{family}_repeat_control", seed, device)
                if state_sha(lead_model) != state_sha(control_model):
                    raise RuntimeError(f"E10-{family.upper()} seed{seed}: initial states differ")
                if sum(p.numel() for p in lead_model.parameters()) != sum(p.numel() for p in control_model.parameters()):
                    raise RuntimeError(f"E10-{family.upper()} seed{seed}: parameter counts differ")
                pair_audit[f"e10{family}_seed{seed}"] = {"initial_state_sha256": state_sha(lead_model),
                                                          "parameters": sum(p.numel() for p in lead_model.parameters())}
                del lead_model, control_model
        dump(run_dir / "paired_control_audit.json", pair_audit)

        results = load_e9_results()
        dump(run_dir / "partial_results.json", results)
        for method in NEW_METHODS:
            for seed in (0, 1, 2):
                set_status(run_dir, f"training_{method}_seed{seed}")
                results.append(train_one(method, seed, data, labels, run_dir, args, device))
                dump(run_dir / "partial_results.json", results)
        summary = aggregate(results)
        dump(run_dir / "metrics/aggregate_summary.json", summary)
        write_csvs(run_dir, results, summary)
        write_report(run_dir, summary, audit, results)
        if any(sha256(p) != source_hashes[str(p)] for p in sources):
            raise RuntimeError("A frozen source changed during E10")
        dump(run_dir / "completion_audit.json", {
            "source_hashes_unchanged": True, "split_counts": {s: len(labels[s]) for s in labels},
            "new_runs": len(NEW_METHODS) * 3, "reused_e9_runs": 3,
            "test_evaluations_per_new_run": 1, "selection_uses_test": False,
            "elapsed_seconds": time.time() - start,
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
