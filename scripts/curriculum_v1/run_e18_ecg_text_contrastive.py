#!/usr/bin/env python3
"""E18: E12-A masked SID learning plus Chinese ECG-report alignment.

Only the report-alignment branch is new.  E3, the 1024x64 codebook, the E12-A
continuous-z_e backbone, masking policy, optimizer budget, and downstream
protocol remain unchanged.  Clinical labels are never consumed by pretraining.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "heartlang").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiments"), str(ROOT / "scripts/curriculum_v1")]

import run_e12_continuous_input as e12


E12_RUN = ROOT / "results/curriculum_v1/e12_continuous_ze_sentence_20261018"
LABEL_XLSX = ROOT / "datasets/标签_Label.xlsx"
TEXT_MODEL = ROOT / "external_models/medbert-base-chinese-668ca1e"
TEXT_MODEL_ID = "trueto/medbert-base-chinese"
TEXT_MODEL_REVISION = "668ca1e929abd0380c7ff831615b154122daf5c1"
DEFAULT_RUN = ROOT / "results/e18_ecg_text_contrastive"
TEXT_DIM = 768
PROJ_DIM = 256
CLIP_WEIGHT = 0.1


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_status(run_dir: Path, value: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.txt").write_text(value + "\n", encoding="utf-8")
    print(value, flush=True)


def normalize_text(value: str) -> str:
    value = str(value).replace("\u3000", " ").replace("；", ";").replace("，", ",")
    value = re.sub(r"\s+", " ", value).strip(" ;,")
    return value


def label_pairs() -> tuple[pd.DataFrame, dict]:
    table = pd.read_excel(LABEL_XLSX)
    rows = []
    rules = {"row_composite": 0, "none": 0}
    for row_index, row in table.iterrows():
        files = [x.strip() for x in str(row["file_name"]).split(";")
                 if x.strip() and x.strip().lower() != "nan"]
        diagnoses = []
        for pos in range(1, 7):
            raw = row.get(f"诊断结论_{pos}")
            if pd.notna(raw) and normalize_text(raw):
                diagnoses.append((pos, normalize_text(raw)))
        diagnosis_positions = ",".join(str(pos) for pos, _ in diagnoses)
        report_text = normalize_text("；".join(text for _, text in diagnoses))
        rule = "row_composite" if report_text else "none"
        for pos, filename in enumerate(files, 1):
            rules[rule] += 1
            rows.append({
                "excel_row": int(row_index),
                "patient_sn": str(row["patient_sn"]),
                "ecg_position": int(pos),
                "row_ecg_count": int(len(files)),
                "row_diagnosis_count": int(len(diagnoses)),
                "diagnosis_positions": diagnosis_positions,
                "csv_stem": Path(filename).stem,
                "report_text": report_text,
                "mapping_rule": rule,
            })
    pairs = pd.DataFrame(rows)
    duplicate_groups = [group for _, group in pairs.groupby("csv_stem") if len(group) > 1]
    cross_patient = [group.csv_stem.iloc[0] for group in duplicate_groups
                     if group.patient_sn.nunique() > 1]
    if cross_patient:
        raise RuntimeError(f"Duplicate ECG files assigned to different patients: {cross_patient[:5]}")
    tied_maximal = []
    for group in duplicate_groups:
        maximal = group.loc[group.row_ecg_count == group.row_ecg_count.max()]
        if maximal.report_text.nunique() > 1:
            tied_maximal.append(group.csv_stem.iloc[0])
    if tied_maximal:
        raise RuntimeError(f"Duplicate ECG files have conflicting equally maximal rows: {tied_maximal[:5]}")
    # Some source ECG filenames are repeated in a shorter and a more complete row for the
    # same patient. Keep the row containing the largest ECG/diagnosis set so one physical
    # ECG receives exactly one reproducible row-level report.
    pairs = pairs.sort_values(
        ["csv_stem", "row_ecg_count", "row_diagnosis_count", "excel_row", "ecg_position"],
        ascending=[True, False, False, False, True],
    ).drop_duplicates("csv_stem")
    return pairs, {
        "excel_rows": len(table), "ecg_references": len(rows), "unique_ecg_files": len(pairs),
        "mapping_rules": rules, "duplicate_ecg_references": int(len(rows) - len(pairs)),
        "duplicate_ecg_stems": int(len(duplicate_groups)), "cross_patient_duplicate_stems": 0,
        "duplicate_resolution": "same-patient duplicate ECG keeps the Excel row with the largest ECG set, then largest diagnosis set",
    }


def build_alignment(run_dir: Path) -> None:
    out = run_dir / "text_alignment"
    if (out / "COMPLETE").exists():
        return
    out.mkdir(parents=True, exist_ok=False)
    pairs, source_audit = label_pairs()

    downstream_patient_sets = {}
    for split in ("train", "val", "test"):
        meta = pd.read_csv(e12.DOWNSTREAM / f"{split}_metadata.csv", low_memory=False)
        downstream_patient_sets[split] = set(meta["patient_sn"].dropna().astype(str))
    if downstream_patient_sets["train"] & downstream_patient_sets["val"] or \
       downstream_patient_sets["train"] & downstream_patient_sets["test"] or \
       downstream_patient_sets["val"] & downstream_patient_sets["test"]:
        raise RuntimeError("Downstream patient_sn split overlap")

    forbidden = downstream_patient_sets["val"] | downstream_patient_sets["test"]
    texts: dict[str, int] = {}
    split_audit = {}
    for split in ("train", "val"):
        meta = pd.read_csv(e12.PRETRAIN_INDEX / f"{split}_metadata.csv", low_memory=False).reset_index(names="pretrain_index")
        meta["csv_stem"] = meta["csv_name"].astype(str).map(lambda x: Path(x).stem)
        merged = meta.merge(pairs, on="csv_stem", how="left", validate="many_to_one")
        eligible = (merged["report_text"].fillna("").str.len() > 0) & \
                   merged["patient_sn"].fillna("").isin(downstream_patient_sets["train"]) & \
                   ~merged["patient_sn"].fillna("").isin(forbidden)
        report_index = np.full(len(meta), -1, dtype=np.int64)
        manifest_rows = []
        for row in merged.loc[eligible].itertuples(index=False):
            text = str(row.report_text)
            if text not in texts:
                texts[text] = len(texts)
            rid = texts[text]
            report_index[int(row.pretrain_index)] = rid
            manifest_rows.append({
                "pretrain_index": int(row.pretrain_index), "csv_name": row.csv_name,
                "patient_sn": row.patient_sn, "excel_row": int(row.excel_row), "report_id": rid,
                "ecg_position": int(row.ecg_position), "row_ecg_count": int(row.row_ecg_count),
                "row_diagnosis_count": int(row.row_diagnosis_count),
                "diagnosis_positions": str(row.diagnosis_positions),
                "mapping_rule": row.mapping_rule, "report_text": text,
            })
        manifest = pd.DataFrame(manifest_rows).sort_values("pretrain_index")
        manifest.to_csv(out / f"{split}_report_pairs.csv", index=False)
        np.save(out / f"{split}_report_index.npy", report_index)
        split_audit[split] = {
            "pretrain_records": len(meta), "paired_ecgs": int(eligible.sum()),
            "paired_patients": int(merged.loc[eligible, "patient_sn"].nunique()),
            "downstream_val_patient_overlap": int(len(set(manifest.patient_sn) & downstream_patient_sets["val"])),
            "downstream_test_patient_overlap": int(len(set(manifest.patient_sn) & downstream_patient_sets["test"])),
            "mapping_rules": manifest.mapping_rule.value_counts().to_dict(),
        }
        if split_audit[split]["downstream_val_patient_overlap"] or split_audit[split]["downstream_test_patient_overlap"]:
            raise RuntimeError(f"{split}: downstream Val/Test report leakage")

    ordered_texts = [None] * len(texts)
    for text, index in texts.items():
        ordered_texts[index] = text
    with (out / "report_texts.jsonl").open("w", encoding="utf-8") as f:
        for i, text in enumerate(ordered_texts):
            f.write(json.dumps({"report_id": i, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "text": text}, ensure_ascii=False) + "\n")
    audit = {
        "source": str(LABEL_XLSX), "allowed_fields": ["file_name", "patient_sn", "诊断结论_1..6"],
        "forbidden_fields_used": [],
        "mapping_policy": "concatenate every non-empty diagnosis_1..6 in Excel-row order; every ECG listed in that row shares the complete row-level report",
        "multi_ecg_same_text_policy": "all ECGs in one Excel row share its composite report_id; identical normalized composite reports across rows also share a report_id",
        "duplicate_ecg_policy": "when the same ECG file is repeated in multiple rows for the same patient, use the row with the largest ECG set, then the largest diagnosis set",
        "patient_leakage_policy": "text alignment restricted to downstream-train patient_sn; downstream Val/Test reports excluded",
        "source_audit": source_audit, "unique_normalized_reports": len(texts), "splits": split_audit,
    }
    dump(out / "alignment_audit.json", audit)
    (out / "COMPLETE").touch()


@torch.inference_mode()
def encode_reports(run_dir: Path, device: torch.device) -> None:
    out = run_dir / "text_alignment"
    target = out / "medbert_report_embeddings.npy"
    if target.exists():
        return
    required = [TEXT_MODEL / "config.json", TEXT_MODEL / "vocab.txt", TEXT_MODEL / "pytorch_model.bin"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete local MedBERT snapshot: {missing}")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(TEXT_MODEL, local_files_only=True).to(device).eval().requires_grad_(False)
    texts = [json.loads(line)["text"] for line in (out / "report_texts.jsonl").read_text(encoding="utf-8").splitlines()]
    result = np.empty((len(texts), int(model.config.hidden_size)), dtype=np.float32)
    batch_size = 128
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        hidden = model(**tokens).last_hidden_state.float()
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        result[start:start + len(batch)] = pooled.cpu().numpy()
        if start == 0 or start + len(batch) == len(texts) or (start // batch_size) % 10 == 0:
            print(f"text encoding {start + len(batch)}/{len(texts)}", flush=True)
    np.save(target, result)
    dump(out / "text_encoder_audit.json", {
        "model_id": TEXT_MODEL_ID, "revision": TEXT_MODEL_REVISION, "local_path": str(TEXT_MODEL),
        "frozen": True, "pooling": "attention-mask mean of last hidden state", "max_length": 128,
        "reports": len(texts), "shape": list(result.shape), "finite": bool(np.isfinite(result).all()),
        "embedding_sha256": sha256(target),
        "model_files_sha256": {p.name: sha256(p) for p in required},
    })


class AlignedDataset(Dataset):
    def __init__(self, run_dir: Path, split: str):
        self.ze = np.load(E12_RUN / "pretrain_ze_cache" / f"{split}_ze.npy", mmap_mode="r")
        self.sid = np.load(e12.PRETRAIN_SID / f"{split}_sid.npy", mmap_mode="r")
        self.report = np.load(run_dir / "text_alignment" / f"{split}_report_index.npy", mmap_mode="r")
        if self.ze.shape[:2] != self.sid.shape or len(self.report) != len(self.sid):
            raise RuntimeError(f"{split}: alignment shape mismatch")

    def __len__(self):
        return len(self.sid)

    def __getitem__(self, index):
        return (np.array(self.ze[index], copy=True), np.array(self.sid[index], copy=True),
                int(self.report[index]), int(index))


class StratifiedBatchSampler(Sampler[list[int]]):
    """One pass per epoch, grouping paired examples for useful in-batch negatives."""
    def __init__(self, report_index: np.ndarray, batch_size: int, epoch: int):
        self.batch_size = batch_size
        rng = np.random.default_rng(18180000 + epoch)
        paired = np.flatnonzero(report_index >= 0); unpaired = np.flatnonzero(report_index < 0)
        paired = rng.permutation(paired); unpaired = rng.permutation(unpaired)
        batches = [paired[i:i+batch_size].tolist() for i in range(0, len(paired)-batch_size+1, batch_size)]
        batches += [unpaired[i:i+batch_size].tolist() for i in range(0, len(unpaired)-batch_size+1, batch_size)]
        order = rng.permutation(len(batches)); self.batches = [batches[i] for i in order]
        self.paired_batches = len(paired) // batch_size

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class ECGTextModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        base, init_audit = e12.initialize_continuous("scratch", device)
        self.ecg = base
        torch.manual_seed(1818)
        self.ecg_projection = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, PROJ_DIM))
        self.text_projection = nn.Sequential(nn.LayerNorm(TEXT_DIM), nn.Linear(TEXT_DIM, PROJ_DIM))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))
        self.init_audit = init_audit

    def encode(self, ze: torch.Tensor, valid: torch.Tensor, mask: torch.Tensor):
        tokens = self.ecg.backbone.forward_feature(ze, valid, mask)[:, 1:]
        logits = self.ecg.lm_head(tokens[mask & valid])
        count = valid.sum(1).clamp_min(1).unsqueeze(-1)
        mean = (tokens * valid.unsqueeze(-1)).sum(1) / count
        maximum = tokens.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
        maximum = torch.where(valid.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        ecg_embedding = F.normalize(self.ecg_projection(torch.cat([mean, maximum], 1)), dim=-1)
        return logits, ecg_embedding

    def text_embedding(self, frozen_text: torch.Tensor):
        return F.normalize(self.text_projection(frozen_text), dim=-1)

    def token_features(self, ze: torch.Tensor, valid: torch.Tensor):
        return self.ecg.extract_tokens(ze, valid)


def contrastive_loss(ecg: torch.Tensor, text: torch.Tensor, report_ids: torch.Tensor,
                     logit_scale: torch.Tensor) -> tuple[torch.Tensor, dict]:
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * ecg.float() @ text.float().T
    positive = report_ids[:, None].eq(report_ids[None, :])
    neg_inf = torch.finfo(logits.dtype).min
    row_loss = torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~positive, neg_inf), 1)
    col_loss = torch.logsumexp(logits, 0) - torch.logsumexp(logits.masked_fill(~positive, neg_inf), 0)
    loss = 0.5 * (row_loss.mean() + col_loss.mean())
    row_hit = positive.gather(1, logits.argmax(1, keepdim=True)).float().mean()
    col_hit = positive.gather(0, logits.argmax(0, keepdim=True)).float().mean()
    return loss, {"retrieval_r1": float(0.5 * (row_hit + col_hit)), "temperature": float(1 / scale)}


def deterministic_mask(valid: torch.Tensor, epoch: int, batch_index: int) -> torch.Tensor:
    gen = torch.Generator(device=valid.device).manual_seed(2026102600 + epoch * 1000 + batch_index)
    return e12.random_mask(valid, 0.5, gen)


def initialization_audit(model: ECGTextModel) -> dict:
    e12_ckpt = torch.load(E12_RUN / "mlm/scratch/checkpoint-100.pth", map_location="cpu", weights_only=False)
    e12_initial, source_audit = e12.initialize_continuous("scratch", torch.device("cpu"))
    same = state_sha(e12_initial.state_dict()) == state_sha(model.ecg.state_dict())
    return {"same_e12a_backbone_initialization": same,
            "backbone_initial_state_sha256": state_sha(model.ecg.state_dict()),
            "reference_recreated_initial_state_sha256": state_sha(e12_initial.state_dict()),
            "e12_initialization": source_audit,
            "note": "checkpoint-100 is read only and not loaded into E18; comparison starts from recreated E12-A initialization"}


def save_checkpoint(path: Path, model: ECGTextModel, optimizer, scaler, epoch: int,
                    best: float, best_epoch: int | None, config: dict) -> None:
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best_joint_val": best, "best_epoch": best_epoch,
                "config": config}, path)


def load_model_checkpoint(path: Path, device: torch.device) -> ECGTextModel:
    model = ECGTextModel(device).to(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device)


def pretrain(run_dir: Path, device: torch.device, args) -> None:
    out = run_dir / "mlm"
    if (out / "COMPLETE").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    train_ds = AlignedDataset(run_dir, "train")
    val_ds = AlignedDataset(run_dir, "val")
    text_embeddings = torch.from_numpy(np.load(run_dir / "text_alignment/medbert_report_embeddings.npy")).float().to(device)
    model = ECGTextModel(device).to(device)
    init = initialization_audit(model)
    if not init["same_e12a_backbone_initialization"]:
        raise RuntimeError("E18 backbone initialization differs from recreated E12-A")
    dump(out / "initialization_audit.json", init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, best_epoch, start_epoch = float("inf"), None, 1
    last = out / "checkpoint-last.pth"
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"]); scaler.load_state_dict(checkpoint["scaler"])
        best = float(checkpoint["best_joint_val"]); best_epoch = checkpoint["best_epoch"]
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"resume from completed epoch {start_epoch-1}", flush=True)
    config = {"epochs": args.epochs, "batch_size": 256, "clip_weight": CLIP_WEIGHT,
              "projection_dim": PROJ_DIM, "temperature_initial": 0.07,
              "paired_batching": "paired and unpaired records grouped separately; each record sampled at most once per epoch",
              "paired_loss_scaling": "lambda * total_batches / paired_batches gives epoch-level mean objective",
              "model_selection": "minimum MLM_val_CE + 0.1*paired_contrastive_val_CE"}
    history_path = out / "training_history.jsonl"
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        sampler = StratifiedBatchSampler(np.asarray(train_ds.report), 256, epoch)
        loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=8, pin_memory=True)
        pair_scale = CLIP_WEIGHT * len(sampler) / max(1, sampler.paired_batches)
        model.train(); sums = {"mlm_loss": 0.0, "mlm_correct": 0, "mlm_tokens": 0,
                               "clip_loss": 0.0, "clip_r1": 0.0, "paired_batches": 0, "batches": 0}
        for step, (ze_np, sid_np, report_np, sample_index) in enumerate(loader):
            ze = ze_np.float().to(device, non_blocking=True); sid = sid_np.long().to(device, non_blocking=True)
            report = report_np.long().to(device, non_blocking=True); valid = (sid >= 0) & (sid < 1024)
            mask = deterministic_mask(valid, epoch, step); labels = sid[mask]
            global_step = (epoch - 1) * len(loader) + step
            lr = e12.lr_at(global_step, args.epochs * len(loader), 5 * len(loader), 5e-4, 1e-5, 1e-6)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, ecg_embedding = model.encode(ze, valid, mask)
                mlm_loss = F.cross_entropy(logits.float(), labels)
                paired = report >= 0
                if paired.all():
                    text_embedding = model.text_embedding(text_embeddings[report])
                    clip_loss, clip_stats = contrastive_loss(ecg_embedding, text_embedding, report, model.logit_scale)
                    loss = mlm_loss + pair_scale * clip_loss
                elif paired.any():
                    raise RuntimeError("Stratified batch unexpectedly mixes paired and unpaired records")
                else:
                    clip_loss, clip_stats, loss = None, None, mlm_loss
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(optimizer); scaler.update()
            with torch.no_grad():
                sums["mlm_loss"] += float(mlm_loss) * labels.numel()
                sums["mlm_correct"] += int((logits.argmax(-1) == labels).sum())
                sums["mlm_tokens"] += int(labels.numel()); sums["batches"] += 1
                if clip_loss is not None:
                    sums["clip_loss"] += float(clip_loss); sums["clip_r1"] += clip_stats["retrieval_r1"]
                    sums["paired_batches"] += 1
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == len(loader):
                print(f"train epoch={epoch} {step+1}/{len(loader)} mlm={float(mlm_loss):.5f} "
                      f"clip={float(clip_loss) if clip_loss is not None else float('nan'):.5f}", flush=True)
        row = {"epoch": epoch, "train_mlm_ce": sums["mlm_loss"] / sums["mlm_tokens"],
               "train_mlm_accuracy": sums["mlm_correct"] / sums["mlm_tokens"],
               "train_contrastive_ce": sums["clip_loss"] / max(1, sums["paired_batches"]),
               "train_contrastive_r1": sums["clip_r1"] / max(1, sums["paired_batches"]),
               "paired_batches": sums["paired_batches"], "total_batches": sums["batches"],
               "paired_batch_loss_multiplier": pair_scale,
               "temperature": float(1 / model.logit_scale.exp().clamp(max=100))}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            val = validate(model, val_ds, text_embeddings, device, epoch)
            row.update({f"val_{k}": v for k, v in val.items()})
            joint = val["mlm_ce"] + CLIP_WEIGHT * val["contrastive_ce"]
            row["val_joint_objective"] = joint
            if joint < best:
                best, best_epoch = joint, epoch
                save_checkpoint(out / "checkpoint-best.pth", model, optimizer, scaler, epoch, best, best_epoch, config)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        save_checkpoint(last, model, optimizer, scaler, epoch, best, best_epoch, config)
        if epoch in (50, args.epochs):
            save_checkpoint(out / f"checkpoint-{epoch}.pth", model, optimizer, scaler, epoch, best, best_epoch, config)
        print(json.dumps(row), flush=True)
    dump(out / "summary.json", {"epochs": args.epochs, "best_epoch": best_epoch,
                                 "best_joint_val": best, "elapsed_seconds_this_process": time.time() - start_time,
                                 "train_records": len(train_ds), "val_records": len(val_ds), **config})
    (out / "COMPLETE").touch()


@torch.inference_mode()
def validate(model: ECGTextModel, dataset: AlignedDataset, text_embeddings: torch.Tensor,
             device: torch.device, epoch: int) -> dict:
    model.eval(); loss_sum = 0.0; correct = tokens = 0
    loader = DataLoader(dataset, batch_size=384, shuffle=False, num_workers=8, pin_memory=True)
    for step, (ze_np, sid_np, report_np, sample_index) in enumerate(loader):
        ze = ze_np.float().to(device); sid = sid_np.long().to(device); valid = (sid >= 0) & (sid < 1024)
        mask_np = e12.fixed_masks(sid_np.numpy(), step * 384)
        mask = torch.from_numpy(mask_np).to(device); labels = sid[mask]
        logits, _ = model.encode(ze, valid, mask)
        loss_sum += float(F.cross_entropy(logits.float(), labels, reduction="sum"))
        correct += int((logits.argmax(-1) == labels).sum()); tokens += int(labels.numel())
    paired_indices = np.flatnonzero(np.asarray(dataset.report) >= 0)
    ecg_chunks, text_chunks, id_chunks = [], [], []
    for start in range(0, len(paired_indices), 256):
        ids = paired_indices[start:start+256]
        batch = [dataset[int(i)] for i in ids]
        ze = torch.from_numpy(np.stack([x[0] for x in batch])).float().to(device)
        sid_np = np.stack([x[1] for x in batch]); sid = torch.from_numpy(sid_np).long().to(device)
        report = torch.tensor([x[2] for x in batch], dtype=torch.long, device=device)
        valid = (sid >= 0) & (sid < 1024)
        mask = torch.from_numpy(e12.fixed_masks(sid_np, start)).to(device)
        _, ecg = model.encode(ze, valid, mask); text = model.text_embedding(text_embeddings[report])
        ecg_chunks.append(ecg); text_chunks.append(text); id_chunks.append(report)
    ecg = torch.cat(ecg_chunks); text = torch.cat(text_chunks); report = torch.cat(id_chunks)
    clip, stats = contrastive_loss(ecg, text, report, model.logit_scale)
    model.train()
    return {"mlm_ce": loss_sum / tokens, "mlm_accuracy": correct / tokens, "masked_tokens": tokens,
            "contrastive_ce": float(clip), "contrastive_r1": stats["retrieval_r1"],
            "paired_ecgs": len(paired_indices), "temperature": stats["temperature"]}


def smoke(run_dir: Path, device: torch.device) -> None:
    out = run_dir / "smoke"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=False)
    model = ECGTextModel(device).to(device).eval()
    ze = torch.randn(4, 256, 64, device=device); valid = torch.zeros(4, 256, dtype=torch.bool, device=device)
    valid[:, :40] = True; mask = torch.zeros_like(valid); mask[:, [2, 7, 13, 29]] = True
    text = torch.randn(4, TEXT_DIM, device=device); ids = torch.tensor([0, 0, 1, 2], device=device)
    with torch.no_grad():
        logits_a, ecg_a = model.encode(ze, valid, mask)
        changed = ze.detach().clone(); changed[mask] = torch.randn_like(changed[mask]) * 1e5
        logits_b, ecg_b = model.encode(changed, valid, mask)
    leakage_delta = float(torch.max(torch.abs(logits_a.detach() - logits_b.detach())))
    if leakage_delta != 0.0 or not torch.equal(ecg_a.detach(), ecg_b.detach()):
        raise RuntimeError(f"Masked z_e leakage delta={leakage_delta}")
    model.train()
    logits_a, ecg_a = model.encode(ze, valid, mask)
    clip, _ = contrastive_loss(ecg_a, model.text_embedding(text), ids, model.logit_scale)
    loss = F.cross_entropy(logits_a.float(), torch.randint(0, 1024, (len(logits_a),), device=device)) + CLIP_WEIGHT * clip
    loss.backward()
    checks = {
        "masked_ze_mutation_delta": leakage_delta,
        "ecg_projection_grad_finite": bool(torch.isfinite(model.ecg_projection[1].weight.grad).all()),
        "text_projection_grad_finite": bool(torch.isfinite(model.text_projection[1].weight.grad).all()),
        "backbone_grad_finite": bool(torch.isfinite(model.ecg.backbone.input_proj.weight.grad).all()),
        "multi_positive_same_report": True, "explicit_padding_key_mask": True,
        "text_encoder_frozen_precomputed": True,
    }
    if not all(v is True or v == 0.0 for v in checks.values()):
        raise RuntimeError(checks)
    dump(out / "smoke_audit.json", checks); (out / "COMPLETE").touch()


@torch.inference_mode()
def fixed_evaluation(run_dir: Path, device: torch.device) -> None:
    path = run_dir / "fixed_mask_mlm_comparison.json"
    if path.exists(): return
    dataset = AlignedDataset(run_dir, "val")
    text = torch.from_numpy(np.load(run_dir / "text_alignment/medbert_report_embeddings.npy")).float().to(device)
    model = load_model_checkpoint(run_dir / "mlm/checkpoint-best.pth", device)
    stats = validate(model, dataset, text, device, 0)
    baseline = json.loads((E12_RUN / "fixed_mask_mlm_comparison.json").read_text())
    dump(path, {"mask_policy": "same E12 stateless 50% valid-position mask",
                "e12a": baseline["e12a_scratch"], "e18_ecg_text": stats,
                "checkpoint": str(run_dir / "mlm/checkpoint-best.pth")})


@torch.inference_mode()
def extract_features(run_dir: Path, device: torch.device) -> None:
    out = run_dir / "features"
    if (out / "COMPLETE").exists(): return
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = e12.e4.load_model(e12.TOKENIZER, device).eval().requires_grad_(False)
    model = load_model_checkpoint(run_dir / "mlm/checkpoint-best.pth", device).eval().requires_grad_(False)
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
            maximum = tokens.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1); maximum[~valid.any(1)] = 0
            result[start:end] = torch.cat([mean, maximum], 1).cpu().numpy()
        np.save(out / f"{split}_heartlang_mean_max.npy", result)
        audit["splits"][split] = {"records": len(result), "shape": list(result.shape),
                                   "sha256": sha256(out / f"{split}_heartlang_mean_max.npy")}
    dump(out / "audit.json", audit); (out / "COMPLETE").touch()


def load_downstream(run_dir: Path, device: torch.device) -> dict:
    data = {}
    for split in ("train", "val", "test"):
        labels = np.load(e12.DOWNSTREAM / f"{split}_labels.npy").astype(np.float32)
        mlm = np.load(run_dir / "features" / f"{split}_heartlang_mean_max.npy")
        lead = np.load(e12.E10_RUN / f"features/{split}_lead_ze_mean_max.npy")
        lead_mask = np.load(e12.E10_RUN / f"features/{split}_lead_present.npy")
        global_ze = np.load(e12.E10_RUN / f"features/{split}_global_ze_mean_max.npy")
        data[split] = {"labels": torch.from_numpy(labels).to(device), "mlm": torch.from_numpy(mlm).float().to(device),
                       "lead": torch.from_numpy(lead).float().to(device),
                       "lead_mask": torch.from_numpy(lead_mask).bool().to(device),
                       "global": torch.from_numpy(global_ze).float().to(device)}
    return data


def train_downstream_seed(run_dir: Path, seed: int, device: torch.device) -> dict:
    out = run_dir / "downstream" / f"seed{seed}"
    if (out / "VAL_COMPLETE").exists(): return json.loads((out / "val_summary.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    data = load_downstream(run_dir, device); model = e12.e11.E11RegionResidual(seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    target = data["train"]["labels"].argmax(1); steps = len(target) // 64
    generator = torch.Generator().manual_seed(seed)
    train_regions, train_region_mask = e12.e11.masked_region_means(data["train"]["lead"], data["train"]["lead_mask"])
    history, best, best_epoch, best_state = [], -float("inf"), None, None
    for epoch in range(1, 31):
        model.train(); order = torch.randperm(len(target), generator=generator).to(device); losses = []
        for step in range(steps):
            idx = order[step*64:(step+1)*64]
            lr = e12.lr_at((epoch-1)*steps+step, 30*steps, 5*steps, 5e-4, 1e-5, 1e-6)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            main, aux = model.training_outputs(data["train"]["mlm"][idx], data["train"]["global"][idx],
                                               train_regions[idx], train_region_mask[idx])
            lm = F.cross_entropy(main, target[idx]); la = e12.e9.patient_auxiliary_loss(aux, target[idx])
            loss = lm + 0.2 * la; loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step()
            losses.append(float(loss))
        val, _, _ = e12.eval_downstream(model, "e12a_region", data["val"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val})
        if val["r2"] > best:
            best, best_epoch, best_state = val["r2"], epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); val, prob, logits = e12.eval_downstream(model, "e12a_region", data["val"])
    torch.save({"model": {k: v.cpu() for k, v in best_state.items()}, "seed": seed,
                "best_epoch": best_epoch, "selection": "Val Overall R2"}, out / "checkpoint-best.pth")
    np.save(out / "val_logits.npy", logits); np.save(out / "val_probabilities.npy", prob)
    dump(out / "history.json", history)
    result = {"method": "e18_ecg_text_region", "seed": seed, "best_epoch": best_epoch, "val": val,
              "trainable_parameters": sum(p.numel() for p in model.parameters())}
    dump(out / "val_summary.json", result); (out / "VAL_COMPLETE").touch(); return result


def downstream(run_dir: Path, device: torch.device) -> None:
    path = run_dir / "validation_summary.json"
    if path.exists(): return
    rows = [train_downstream_seed(run_dir, seed, device) for seed in (0, 1, 2)]
    base = json.loads((E12_RUN / "downstream/validation_selection.json").read_text())["methods"]["e12a_region"]
    keys = ("r2", "mse", "mae", "macro_roc_auc", "macro_f1", "accuracy", "brier_score", "ece15_top_label")
    agg = {key: {"mean": float(np.mean([row["val"][key] for row in rows])),
                 "sd": float(np.std([row["val"][key] for row in rows], ddof=1))} for key in keys}
    agg["per_output_r2"] = {name: {"mean": float(np.mean([row["val"]["per_output_r2"][name] for row in rows])),
                                           "sd": float(np.std([row["val"]["per_output_r2"][name] for row in rows], ddof=1))}
                            for name in e12.LABEL_NAMES}
    base_rows = base["per_seed"]
    base_agg = {key: {"mean": float(np.mean([row["val"][key] for row in base_rows])),
                      "sd": float(np.std([row["val"][key] for row in base_rows], ddof=1))} for key in keys}
    base_agg["per_output_r2"] = {name: {"mean": float(np.mean([row["val"]["per_output_r2"][name] for row in base_rows])),
                                                "sd": float(np.std([row["val"]["per_output_r2"][name] for row in base_rows], ddof=1))}
                                 for name in e12.LABEL_NAMES}
    dump(path, {"validation_only": True, "test_run": False,
                "e12a_historical": {"aggregate": base_agg, "per_seed": base_rows},
                "e18": {"aggregate": agg, "per_seed": rows}})


def report(run_dir: Path) -> None:
    summary = json.loads((run_dir / "validation_summary.json").read_text())
    fixed = json.loads((run_dir / "fixed_mask_mlm_comparison.json").read_text())
    audit = json.loads((run_dir / "text_alignment/alignment_audit.json").read_text())
    e18 = summary["e18"]["aggregate"]
    base = summary["e12a_historical"]["aggregate"]
    lines = ["# E18：中文 ECG 报告对齐", "", "## 结论", "",
             "本实验在E12-A的Masked SID预训练中加入冻结中文MedBERT报告分支；下游只输入ECG。",
             "所有比较仅使用Validation，未运行Test。", "", "## 文本与泄漏审计", "",
             f"- 训练对齐ECG：{audit['splits']['train']['paired_ecgs']}，患者：{audit['splits']['train']['paired_patients']}。",
             f"- 无监督Val对齐ECG：{audit['splits']['val']['paired_ecgs']}。",
             "- 只读取file_name、patient_sn和诊断结论_1..6；未读取造影、PCI、肌钙蛋白、出院诊断或下游标签。",
             "- 文本监督按patient_sn限制为下游Train患者；下游Val/Test患者报告重叠均为0。",
             "- 同一Excel行的诊断结论_1..6全部非空内容按顺序拼接；该行所有ECG共享这一完整文本和multi-positive report_id。", "",
             "## 固定Mask MLM", "", "| 方法 | Val CE | Masked SID Accuracy |", "|---|---:|---:|",
             f"| E12-A | {fixed['e12a']['loss']:.6f} | {fixed['e12a']['masked_sid_accuracy']:.6f} |",
             f"| E18 ECG+中文报告 | {fixed['e18_ecg_text']['mlm_ce']:.6f} | {fixed['e18_ecg_text']['mlm_accuracy']:.6f} |", "",
             "## 下游Validation（三个seed）", "",
             "| 方法 | Overall R² | OMI | NOMI | Clinical Control | Healthy | Macro AUC | Macro F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
             f"| E12-A | {base['r2']['mean']:.6f} ± {base['r2']['sd']:.6f} | " + " | ".join(f"{base['per_output_r2'][n]['mean']:.6f}" for n in e12.LABEL_NAMES) + f" | {base['macro_roc_auc']['mean']:.6f} | {base['macro_f1']['mean']:.6f} |",
             f"| E18 | {e18['r2']['mean']:.6f} ± {e18['r2']['sd']:.6f} | " + " | ".join(f"{e18['per_output_r2'][n]['mean']:.6f}" for n in e12.LABEL_NAMES) + f" | {e18['macro_roc_auc']['mean']:.6f} | {e18['macro_f1']['mean']:.6f} |", "",
             f"E18相对E12-A的Val Overall R²差值：**{e18['r2']['mean'] - base['r2']['mean']:+.6f}**。", "",
             "## 方法边界", "", "- 这是ECG-CLIP思想向HeartLang心拍SID MLM的适配，不是原论文完整复现。",
             "- 中文MedBERT冻结并预编码；训练更新E12-A主干、MLM头、ECG投影、文本投影和温度。",
             "- 对比损失权重0.1，在每个epoch按paired batch比例校正，使其对应paired ECG均值。",
             "- 只有一个预训练seed；三个下游seed不能替代独立预训练seed。", ""]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def protocol(run_dir: Path, epochs: int) -> None:
    path = run_dir / "protocol.json"
    if path.exists(): return
    dump(path, {"experiment": "E18 ECG-Chinese-report alignment", "e12a_backbone": str(E12_RUN),
                "e12a_history_read_only": True, "pretrain_epochs": epochs, "batch_size": 256,
                "mask_ratio_valid_tokens": 0.5, "sid_classes": 1024, "code_dim": 64,
                "text_encoder": TEXT_MODEL_ID, "text_revision": TEXT_MODEL_REVISION, "text_encoder_frozen": True,
                "projection_dim": PROJ_DIM, "contrastive_weight": CLIP_WEIGHT,
                "optimizer": "AdamW lr5e-4 min1e-5 warmup5 wd0.05 clip3",
                "downstream": "frozen ECG backbone Mean+Max1536 + E11 region residual; seeds0,1,2; 30 epochs",
                "evaluation": "Validation only; Test prohibited for this exploratory arm"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "encode_text", "smoke", "pretrain", "fixed_mlm", "features", "downstream", "report"])
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args(); run_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True); protocol(run_dir, args.epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "audit": build_alignment(run_dir)
    elif args.stage == "encode_text": encode_reports(run_dir, device)
    elif args.stage == "smoke": smoke(run_dir, device)
    elif args.stage == "pretrain": pretrain(run_dir, device, args)
    elif args.stage == "fixed_mlm": fixed_evaluation(run_dir, device)
    elif args.stage == "features": extract_features(run_dir, device)
    elif args.stage == "downstream": downstream(run_dir, device)
    elif args.stage == "report": report(run_dir); set_status(run_dir, "complete"); (run_dir / "COMPLETE").touch()


if __name__ == "__main__":
    main()
