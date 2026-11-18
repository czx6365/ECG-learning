#!/usr/bin/env python3
"""E21: polarity/location-aware report concepts inside E12-A SID MLM.

The experiment starts from the exact E12-A random initialization.  During
pretraining, every ECG is still optimized with 50% masked-SID CE.  ECGs from
downstream-Train patients that have a same-row Chinese report also receive a
training-only structured concept BCE.  Validation and downstream inference use
ECG alone.  The downstream architecture is the unchanged E12-A Mean+Max + E11
region residual classifier.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

import run_e12_continuous_input as e12
import run_e18_ecg_text_contrastive as e18
import run_e20_polarity_region_concepts as e20


E12_RUN = ROOT / "results/curriculum_v1/e12_continuous_ze_sentence_20261018"
DEFAULT_RUN = ROOT / "results/e21_mlm_polarity_region_text_concepts"
CONCEPT_WEIGHT = 0.1
CONCEPT_WARMUP_START = 5
CONCEPT_WARMUP_END = 10


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha_state(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_status(run: Path, value: str) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "status.txt").write_text(value + "\n", encoding="utf-8")
    print(value, flush=True)


def concept_lambda(epoch: int) -> float:
    if epoch <= CONCEPT_WARMUP_START:
        return 0.0
    if epoch >= CONCEPT_WARMUP_END:
        return CONCEPT_WEIGHT
    return CONCEPT_WEIGHT * (epoch - CONCEPT_WARMUP_START) / (CONCEPT_WARMUP_END - CONCEPT_WARMUP_START)


def build_audit(run: Path) -> None:
    out = run / "concepts"
    if (out / "COMPLETE").exists():
        return
    e18.build_alignment(run)
    out.mkdir(parents=True, exist_ok=False)
    pairs = pd.read_csv(run / "text_alignment/train_report_pairs.csv", low_memory=False)
    report_rows = [json.loads(line) for line in
                   (run / "text_alignment/report_texts.jsonl").read_text(encoding="utf-8").splitlines()]
    n_reports, n_concepts = len(report_rows), len(e20.CONCEPT_NAMES)
    targets = np.zeros((n_reports, n_concepts), dtype=np.float32)
    weights = np.zeros((n_reports, n_concepts), dtype=np.float32)
    report_text = pairs.groupby("report_id", sort=False)["report_text"].first().to_dict()
    event_rows, conflict_rows = [], []
    counts = {"train_paired_ecgs": int(len(pairs)), "train_unique_reports": int(len(report_text)),
              "reports_with_events": 0, "asserted": 0, "negated": 0,
              "uncertain": 0, "suggestive": 0, "conflicting_concepts": 0}
    for report_id, text in report_text.items():
        target, weight, events, conflicts = e20.report_targets(str(text))
        targets[int(report_id)], weights[int(report_id)] = target, weight
        counts["reports_with_events"] += int(bool(events))
        counts["conflicting_concepts"] += len(conflicts)
        for event in events:
            counts[event["status"]] = counts.get(event["status"], 0) + 1
            if len(event_rows) < 500:
                event_rows.append({"report_id": int(report_id), **event})
        for concept in conflicts:
            conflict_rows.append({"report_id": int(report_id), "concept": concept})
    if len(report_text) == 0 or counts["negated"] == 0 or counts["uncertain"] == 0:
        raise RuntimeError(f"Concept audit did not exercise report parsing: {counts}")
    np.save(out / "report_targets.npy", targets)
    np.save(out / "report_weights.npy", weights)
    pd.DataFrame(event_rows).to_csv(out / "parsed_event_examples.csv", index=False)
    pd.DataFrame(conflict_rows, columns=["report_id", "concept"]).to_csv(out / "conflicts.csv", index=False)
    prevalence = []
    for j, name in enumerate(e20.CONCEPT_NAMES):
        prevalence.append({
            "concept": name,
            "positive_reports": int(((targets[:, j] > 0.5) & (weights[:, j] > 0)).sum()),
            "positive_weight": float((targets[:, j] * weights[:, j]).sum()),
            "negative_weight": float(((1.0 - targets[:, j]) * weights[:, j]).sum()),
            "supervised_weight": float(weights[:, j].sum()),
        })
    pd.DataFrame(prevalence).to_csv(out / "concept_prevalence.csv", index=False)
    alignment = json.loads((run / "text_alignment/alignment_audit.json").read_text())
    parser_checks = e20.parser_self_check()
    dump(out / "parser_self_check.json", parser_checks)
    dump(out / "audit.json", {
        "experiment": "E21 report concepts during E12-A masked-SID pretraining",
        "text_mapping": "all non-empty diagnosis_1..6 in an Excel row are concatenated; every ECG in that row shares the composite report",
        "allowed_text_fields": ["file_name", "patient_sn", "diagnosis_1..6"],
        "forbidden_fields_used": [],
        "concepts": {"count": n_concepts, "morphologies": list(e20.MORPHOLOGIES),
                     "regions": list(e20.REGIONS), "polarity_aware": True},
        "counts": counts,
        "alignment": alignment,
        "text_scope": "auxiliary loss only for pretraining-Train ECGs belonging to downstream-Train patients",
        "pretraining_validation_text_used": False,
        "downstream_validation_test_text_used": False,
        "model_selection": "minimum ECG-only fixed-mask MLM validation CE",
        "parser_self_check_passed": True,
    })
    (out / "COMPLETE").touch()


class ConceptMLM(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.ecg, self.init_audit = e12.initialize_continuous("scratch", device)
        # A separate RNG scope preserves the bit-identical E12-A backbone init.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(212100)
            self.concept_head = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, len(e20.CONCEPT_NAMES)))

    def encode(self, ze: torch.Tensor, valid: torch.Tensor, mask: torch.Tensor):
        tokens = self.ecg.backbone.forward_feature(ze, valid, mask)[:, 1:]
        logits = self.ecg.lm_head(tokens[mask & valid])
        count = valid.sum(1).clamp_min(1).unsqueeze(-1)
        mean = (tokens * valid.unsqueeze(-1)).sum(1) / count
        maximum = tokens.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
        maximum = torch.where(valid.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        concepts = self.concept_head(torch.cat([mean, maximum], 1))
        return logits, concepts

    def token_features(self, ze: torch.Tensor, valid: torch.Tensor):
        return self.ecg.extract_tokens(ze, valid)


def concept_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor,
                 pos_weight: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none",
                                             pos_weight=pos_weight.float())
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def build_pos_weight(targets: np.ndarray, weights: np.ndarray, device: torch.device) -> torch.Tensor:
    positive = (targets * weights).sum(0)
    negative = ((1.0 - targets) * weights).sum(0)
    value = np.clip(negative / np.maximum(positive, 1e-6), 1.0, 10.0).astype(np.float32)
    return torch.from_numpy(value).to(device)


def load_model(path: Path, device: torch.device) -> ConceptMLM:
    model = ConceptMLM(device).to(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def save_checkpoint(path: Path, model: ConceptMLM, optimizer, scaler, epoch: int,
                    best: float, best_epoch: int | None, config: dict) -> None:
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best_val_mlm_ce": best, "best_epoch": best_epoch,
                "config": config}, path)


def smoke(run: Path, device: torch.device) -> None:
    out = run / "smoke.json"
    if out.exists():
        return
    build_audit(run)
    model = ConceptMLM(device).to(device)
    reference, _ = e12.initialize_continuous("scratch", torch.device("cpu"))
    same_init = sha_state(model.ecg.state_dict()) == sha_state(reference.state_dict())
    ze = torch.randn(6, 256, 64, device=device)
    valid = torch.zeros(6, 256, dtype=torch.bool, device=device); valid[:, :48] = True
    mask = torch.zeros_like(valid); mask[:, [2, 7, 13, 29, 41]] = True
    model.eval()
    with torch.no_grad():
        logits_a, concepts_a = model.encode(ze, valid, mask)
        changed = ze.clone(); changed[mask] = torch.randn_like(changed[mask]) * 1e5
        logits_b, concepts_b = model.encode(changed, valid, mask)
    leak = max(float((logits_a - logits_b).abs().max()), float((concepts_a - concepts_b).abs().max()))
    targets_np = np.load(run / "concepts/report_targets.npy")
    weights_np = np.load(run / "concepts/report_weights.npy")
    eligible = np.flatnonzero(weights_np.sum(1) > 0)[:6]
    targets = torch.from_numpy(targets_np[eligible]).to(device)
    weights = torch.from_numpy(weights_np[eligible]).to(device)
    pos_weight = build_pos_weight(targets_np, weights_np, device)
    model.train(); model.zero_grad(set_to_none=True)
    logits, concepts = model.encode(ze, valid, mask)
    loss = F.cross_entropy(logits.float(), torch.randint(0, 1024, (len(logits),), device=device))
    loss = loss + CONCEPT_WEIGHT * concept_loss(concepts, targets, weights, pos_weight)
    loss.backward()
    result = {
        "same_e12a_backbone_initialization": same_init,
        "masked_ze_mutation_delta": leak,
        "backbone_concept_joint_gradient_norm": float(model.ecg.backbone.input_proj.weight.grad.norm()),
        "concept_head_gradient_norm": float(model.concept_head[1].weight.grad.norm()),
        "finite_loss": bool(torch.isfinite(loss)),
        "validation_text_used": False, "downstream_text_used": False,
        "explicit_padding_key_mask": True,
    }
    if not same_init or leak != 0 or not result["finite_loss"] or min(
            result["backbone_concept_joint_gradient_norm"], result["concept_head_gradient_norm"]) <= 0:
        raise RuntimeError(f"Smoke failed: {result}")
    dump(out, result)


@torch.inference_mode()
def validate_mlm(model: ConceptMLM, dataset: e18.AlignedDataset, device: torch.device) -> dict:
    model.eval(); loss_sum = 0.0; correct = tokens = 0
    loader = DataLoader(dataset, batch_size=384, shuffle=False, num_workers=8, pin_memory=True)
    for step, (ze_np, sid_np, _report, _sample_index) in enumerate(loader):
        ze = ze_np.float().to(device); sid = sid_np.long().to(device)
        valid = (sid >= 0) & (sid < 1024)
        mask = torch.from_numpy(e12.fixed_masks(sid_np.numpy(), step * 384)).to(device)
        labels = sid[mask]
        logits, _ = model.encode(ze, valid, mask)
        loss_sum += float(F.cross_entropy(logits.float(), labels, reduction="sum"))
        correct += int((logits.argmax(-1) == labels).sum()); tokens += int(labels.numel())
    return {"mlm_ce": loss_sum / tokens, "mlm_accuracy": correct / tokens, "masked_tokens": tokens}


def pretrain(run: Path, device: torch.device, epochs: int) -> None:
    out = run / "mlm"
    if (out / "COMPLETE").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    train_ds = e18.AlignedDataset(run, "train")
    val_ds = e18.AlignedDataset(run, "val")
    concept_targets_np = np.load(run / "concepts/report_targets.npy")
    concept_weights_np = np.load(run / "concepts/report_weights.npy")
    concept_targets = torch.from_numpy(concept_targets_np).float().to(device)
    concept_weights = torch.from_numpy(concept_weights_np).float().to(device)
    pos_weight = build_pos_weight(concept_targets_np, concept_weights_np, device)
    model = ConceptMLM(device).to(device)
    reference, source_audit = e12.initialize_continuous("scratch", torch.device("cpu"))
    init = {"same_e12a_backbone_initialization": sha_state(model.ecg.state_dict()) == sha_state(reference.state_dict()),
            "backbone_sha256": sha_state(model.ecg.state_dict()), "e12_initialization": source_audit,
            "e12_checkpoint_loaded": False}
    if not init["same_e12a_backbone_initialization"]:
        raise RuntimeError("Backbone initialization differs from E12-A")
    dump(out / "initialization_audit.json", init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, best_epoch, start_epoch = float("inf"), None, 1
    last = out / "checkpoint-last.pth"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"]); scaler.load_state_dict(ckpt["scaler"])
        best, best_epoch, start_epoch = float(ckpt["best_val_mlm_ce"]), ckpt["best_epoch"], int(ckpt["epoch"]) + 1
        print(f"resume from completed epoch {start_epoch - 1}", flush=True)
    config = {"epochs": epochs, "batch_size": 256, "mask_ratio": 0.5,
              "concept_weight": CONCEPT_WEIGHT, "concept_ramp": "epochs1-5=0; epochs6-10 linear; epochs10-100=0.1",
              "optimizer": "AdamW peak5e-4 min1e-5 warmup5 wd0.05 clip3",
              "model_selection": "minimum ECG-only fixed-mask MLM validation CE",
              "validation_text_used": False, "downstream_text_used": False}
    history = out / "training_history.jsonl"
    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        sampler = e18.StratifiedBatchSampler(np.asarray(train_ds.report), 256, epoch)
        loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=8, pin_memory=True)
        lam = concept_lambda(epoch)
        paired_scale = lam * len(sampler) / max(1, sampler.paired_batches)
        model.train(); sums = {"mlm_loss": 0.0, "mlm_correct": 0, "tokens": 0,
                               "concept_loss": 0.0, "paired_batches": 0, "batches": 0}
        for step, (ze_np, sid_np, report_np, _sample_index) in enumerate(loader):
            ze = ze_np.float().to(device, non_blocking=True); sid = sid_np.long().to(device, non_blocking=True)
            report = report_np.long().to(device, non_blocking=True); valid = (sid >= 0) & (sid < 1024)
            mask = e18.deterministic_mask(valid, epoch, step); labels = sid[mask]
            global_step = (epoch - 1) * len(loader) + step
            lr = e12.lr_at(global_step, epochs * len(loader), 5 * len(loader), 5e-4, 1e-5, 1e-6)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, concept_logits = model.encode(ze, valid, mask)
                mlm = F.cross_entropy(logits.float(), labels)
                paired = report >= 0
                if paired.all():
                    aux = concept_loss(concept_logits, concept_targets[report], concept_weights[report], pos_weight)
                    loss = mlm + paired_scale * aux
                elif paired.any():
                    raise RuntimeError("Stratified batch mixed paired and unpaired records")
                else:
                    aux, loss = None, mlm
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(optimizer); scaler.update()
            with torch.no_grad():
                sums["mlm_loss"] += float(mlm) * labels.numel()
                sums["mlm_correct"] += int((logits.argmax(-1) == labels).sum())
                sums["tokens"] += int(labels.numel()); sums["batches"] += 1
                if aux is not None:
                    sums["concept_loss"] += float(aux); sums["paired_batches"] += 1
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == len(loader):
                print(f"train epoch={epoch} {step+1}/{len(loader)} mlm={float(mlm):.5f} "
                      f"concept={float(aux) if aux is not None else float('nan'):.5f} lambda={lam:.4f}", flush=True)
        row = {"epoch": epoch, "train_mlm_ce": sums["mlm_loss"] / sums["tokens"],
               "train_mlm_accuracy": sums["mlm_correct"] / sums["tokens"],
               "train_concept_ce": sums["concept_loss"] / max(1, sums["paired_batches"]),
               "concept_lambda_epoch_mean": lam, "paired_batch_loss_multiplier": paired_scale,
               "paired_batches": sums["paired_batches"], "total_batches": sums["batches"]}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            val = validate_mlm(model, val_ds, device)
            row.update({f"val_{k}": value for k, value in val.items()})
            if val["mlm_ce"] < best:
                best, best_epoch = val["mlm_ce"], epoch
                save_checkpoint(out / "checkpoint-best.pth", model, optimizer, scaler, epoch, best, best_epoch, config)
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        save_checkpoint(last, model, optimizer, scaler, epoch, best, best_epoch, config)
        if epoch in (50, epochs):
            save_checkpoint(out / f"checkpoint-{epoch}.pth", model, optimizer, scaler, epoch, best, best_epoch, config)
        print(json.dumps(row), flush=True)
    dump(out / "summary.json", {"epochs": epochs, "best_epoch": best_epoch, "best_val_mlm_ce": best,
                                 "elapsed_seconds_this_process": time.time() - started,
                                 "train_records": len(train_ds), "val_records": len(val_ds), **config})
    (out / "COMPLETE").touch()


@torch.inference_mode()
def fixed_mlm(run: Path, device: torch.device) -> None:
    path = run / "fixed_mask_mlm_comparison.json"
    if path.exists():
        return
    stats = validate_mlm(load_model(run / "mlm/checkpoint-best.pth", device), e18.AlignedDataset(run, "val"), device)
    baseline = json.loads((E12_RUN / "fixed_mask_mlm_comparison.json").read_text())["e12a_scratch"]
    dump(path, {"mask_policy": "same E12 stateless 50% valid-position mask",
                "e12a": baseline, "e21_mlm_concepts": stats,
                "checkpoint": str(run / "mlm/checkpoint-best.pth")})


@torch.inference_mode()
def extract_features(run: Path, device: torch.device) -> None:
    out = run / "features"
    if (out / "COMPLETE").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = e12.e4.load_model(e12.TOKENIZER, device).eval().requires_grad_(False)
    model = load_model(run / "mlm/checkpoint-best.pth", device).eval().requires_grad_(False)
    audit = {"pooling": "valid-token mean+max", "text_used": False, "splits": {}}
    for split in ("train", "val", "test"):
        waves = np.load(e12.DOWNSTREAM / f"{split}_data.npy", mmap_mode="r")
        chans = np.load(e12.DOWNSTREAM / f"{split}_data_in_chans.npy", mmap_mode="r")
        times = np.load(e12.DOWNSTREAM / f"{split}_data_in_times.npy", mmap_mode="r")
        result = np.empty((len(waves), 1536), np.float32)
        for start in range(0, len(waves), 32):
            end = min(start + 32, len(waves))
            x = torch.from_numpy(np.asarray(waves[start:end], dtype=np.float32).copy()).to(device)
            c = torch.from_numpy(np.asarray(chans[start:end]).copy()).long().to(device)
            t = torch.from_numpy(np.asarray(times[start:end]).copy()).long().to(device)
            valid = tokenizer.valid_mask(x, c, t)
            ze = tokenizer.encode_to_quantizer_features(x, c, t).float()
            tokens = model.token_features(ze, valid).float()
            count = valid.sum(1).clamp_min(1).unsqueeze(-1)
            mean = (tokens * valid.unsqueeze(-1)).sum(1) / count
            maximum = tokens.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
            maximum[~valid.any(1)] = 0
            result[start:end] = torch.cat([mean, maximum], 1).cpu().numpy()
        np.save(out / f"{split}_heartlang_mean_max.npy", result)
        audit["splits"][split] = {"records": len(result), "shape": list(result.shape),
                                   "sha256": e18.sha256(out / f"{split}_heartlang_mean_max.npy")}
    dump(out / "audit.json", audit); (out / "COMPLETE").touch()


def aggregate(items: list[dict]) -> dict:
    keys = ("r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy", "brier_score", "ece15_top_label")
    result = {key: {"mean": float(np.mean([row["val"][key] for row in items])),
                    "sd": float(np.std([row["val"][key] for row in items], ddof=1))} for key in keys}
    result["per_output_r2"] = {
        name: {"mean": float(np.mean([row["val"]["per_output_r2"][name] for row in items])),
               "sd": float(np.std([row["val"]["per_output_r2"][name] for row in items], ddof=1))}
        for name in e12.LABEL_NAMES}
    result["patient_mean_r2"] = {
        "mean": float(np.mean([np.mean([row["val"]["per_output_r2"][name] for name in e12.LABEL_NAMES[:3]]) for row in items])),
        "sd": float(np.std([np.mean([row["val"]["per_output_r2"][name] for name in e12.LABEL_NAMES[:3]]) for row in items], ddof=1))}
    return result


def downstream(run: Path, device: torch.device) -> None:
    path = run / "validation_summary.json"
    if path.exists():
        return
    rows = []
    for seed in (0, 1, 2):
        item = e18.train_downstream_seed(run, seed, device)
        item["method"] = "e21_mlm_polarity_region_concepts"
        dump(run / "downstream" / f"seed{seed}" / "val_summary.json", item)
        rows.append(item)
    base_rows = json.loads((E12_RUN / "downstream/validation_selection.json").read_text())["methods"]["e12a_region"]["per_seed"]
    dump(path, {"validation_only": True, "test_run": False,
                "e12a_historical": {"aggregate": aggregate(base_rows), "per_seed": base_rows},
                "e21": {"aggregate": aggregate(rows), "per_seed": rows}})


def report(run: Path) -> None:
    summary = json.loads((run / "validation_summary.json").read_text())
    fixed = json.loads((run / "fixed_mask_mlm_comparison.json").read_text())
    audit = json.loads((run / "concepts/audit.json").read_text())
    base, new = summary["e12a_historical"]["aggregate"], summary["e21"]["aggregate"]
    metric_rows = []
    for label, key in (("E12-A", "e12a_historical"), ("E21", "e21")):
        for item in summary[key]["per_seed"]:
            val = item["val"]
            metric_rows.append({"method": label, "statistic": "seed", "seed": int(item.get("seed", 0)),
                                "overall_r2": val["r2"], "patient_mean_r2": np.mean([val["per_output_r2"][x] for x in e12.LABEL_NAMES[:3]]),
                                **{f"{name.lower().replace(' ', '_')}_r2": val["per_output_r2"][name] for name in e12.LABEL_NAMES},
                                "macro_auc": val["macro_roc_auc"], "macro_f1": val["macro_f1"]})
        agg = summary[key]["aggregate"]
        for statistic in ("mean", "sd"):
            metric_rows.append({"method": label, "statistic": statistic, "seed": np.nan,
                                "overall_r2": agg["r2"][statistic], "patient_mean_r2": agg["patient_mean_r2"][statistic],
                                **{f"{name.lower().replace(' ', '_')}_r2": agg["per_output_r2"][name][statistic] for name in e12.LABEL_NAMES},
                                "macro_auc": agg["macro_roc_auc"][statistic], "macro_f1": agg["macro_f1"][statistic]})
    pd.DataFrame(metric_rows).to_csv(run / "metrics_summary.csv", index=False)
    def row(label: str, x: dict) -> str:
        return (f"| {label} | {x['r2']['mean']:.6f} ± {x['r2']['sd']:.6f} | {x['patient_mean_r2']['mean']:.6f} | "
                + " | ".join(f"{x['per_output_r2'][name]['mean']:.6f}" for name in e12.LABEL_NAMES)
                + f" | {x['macro_roc_auc']['mean']:.6f} | {x['macro_f1']['mean']:.6f} |")
    text = ["# E21：MLM阶段的极性与区域报告概念监督", "", "## 方法", "",
            "E21从与E12-A完全相同的随机初始化训练100轮。在50% Masked SID CE之外，仅对下游Train患者中有报告的预训练ECG增加结构化概念BCE。",
            "概念区分肯定、否定、可能/考虑等不确定性，并细化到真实导联与E11五个区域。Validation与下游均只输入ECG。", "",
            "- Epoch 1–5：仅SID MLM；Epoch 6–10：概念权重线性增加；Epoch 10–100：概念权重0.1。",
            "- Checkpoint只按不含文字的固定Mask MLM Validation CE选择。",
            "- 下游使用原始E12-A四分类Mean+Max+E11区域融合，seed 0/1/2；未运行Test。", "",
            "## 审计", "", f"- 训练配对ECG：{audit['counts']['train_paired_ecgs']}；唯一报告：{audit['counts']['train_unique_reports']}。",
            f"- 概念数：{audit['concepts']['count']}；否定事件：{audit['counts']['negated']}；不确定事件：{audit['counts']['uncertain']}。",
            "- 未使用description、造影、PCI、肌钙蛋白、主诉、现病史、出院诊断或疾病标签。", "",
            "## 固定Mask MLM", "", "| 方法 | Val CE | Masked SID Accuracy |", "|---|---:|---:|",
            f"| E12-A | {fixed['e12a']['loss']:.6f} | {fixed['e12a']['masked_sid_accuracy']:.6f} |",
            f"| E21 | {fixed['e21_mlm_concepts']['mlm_ce']:.6f} | {fixed['e21_mlm_concepts']['mlm_accuracy']:.6f} |", "",
            "## 下游Validation（三个seed）", "",
            "| 方法 | Overall R² | 患者三类均值 | OMI | NOMI | Clinical Control | Healthy | Macro AUC | Macro F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|", row("E12-A", base), row("E21", new), "",
            f"E21相对E12-A：Overall R² **{new['r2']['mean']-base['r2']['mean']:+.6f}**；患者三类平均R² **{new['patient_mean_r2']['mean']-base['patient_mean_r2']['mean']:+.6f}**。", ""]
    (run / "README.md").write_text("\n".join(text), encoding="utf-8")
    set_status(run, "complete"); (run / "COMPLETE").touch()


def protocol(run: Path, epochs: int) -> None:
    path = run / "protocol.json"
    if path.exists():
        return
    dump(path, {"experiment": "E21 MLM-stage polarity/location-aware report concepts",
                "baseline": "E12-A read only", "epochs": epochs, "pretrain_seed": 0,
                "same_e12a_initialization": True, "frozen_e3_encoder_codebook": True,
                "continuous_ze": True, "mask_ratio": 0.5, "sid_classes": 1024,
                "concept_count": len(e20.CONCEPT_NAMES), "concept_weight": CONCEPT_WEIGHT,
                "downstream": "unchanged E12-A four-class Mean+Max1536 + E11 region residual; seeds0,1,2",
                "evaluation": "Validation only; Test prohibited"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "smoke", "pretrain", "fixed_mlm", "features", "downstream", "report"])
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    run = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    run.mkdir(parents=True, exist_ok=True); protocol(run, args.epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "audit": build_audit(run)
    elif args.stage == "smoke": smoke(run, device)
    elif args.stage == "pretrain": pretrain(run, device, args.epochs)
    elif args.stage == "fixed_mlm": fixed_mlm(run, device)
    elif args.stage == "features": extract_features(run, device)
    elif args.stage == "downstream": downstream(run, device)
    elif args.stage == "report": report(run)


if __name__ == "__main__":
    main()
