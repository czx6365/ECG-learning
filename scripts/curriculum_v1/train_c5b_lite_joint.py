#!/usr/bin/env python3
"""C5b-lite: residual continuous bridge with staged tokenizer/EMA adaptation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
sys.path.insert(0, str(ROOT / "scripts" / "curriculum_v1"))

import train_morphology_v2_physaware_vq as morphology
from heartlang.models.modeling_heartlang_sid import HeartLangSIDForRegression
from heartlang.utils.utils import set_seed


LABEL_NAMES = ["OMI", "NOMI", "Clinical Control", "Healthy"]


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def patch_checkpoint_paths() -> None:
    if sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath


class JointDataset(Dataset):
    def __init__(self, raw_dir: Path, sid_dir: Path, cache_dir: Path, split: str):
        self.wave = np.load(raw_dir / f"{split}_data.npy", mmap_mode="r")
        self.sid = np.load(sid_dir / f"{split}_sid.npy", mmap_mode="r")
        self.chans = np.load(raw_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
        self.times = np.load(raw_dir / f"{split}_data_in_times.npy", mmap_mode="r")
        self.labels = np.load(raw_dir / f"{split}_labels.npy", mmap_mode="r").astype(np.float32)
        self.discrete = np.load(cache_dir / f"{split}_phys_discrete.npy", mmap_mode="r")
        n = len(self.wave)
        if not all(len(array) == n for array in (self.sid, self.chans, self.times, self.labels, self.discrete)):
            raise RuntimeError(f"C5b dataset length mismatch for {split}")

    def __len__(self) -> int:
        return len(self.wave)

    def __getitem__(self, index: int):
        return (
            np.array(self.wave[index], copy=True),
            np.array(self.sid[index], copy=True),
            np.array(self.chans[index], copy=True),
            np.array(self.times[index], copy=True),
            np.array(self.labels[index], copy=True),
            np.array(self.discrete[index], copy=True),
        )


def non_circular_shift(beats: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(beats)
    for delta in torch.unique(deltas).tolist():
        mask = deltas == int(delta)
        if delta > 0:
            shifted[mask, delta:] = beats[mask, :-delta]
        elif delta < 0:
            amount = -int(delta)
            shifted[mask, :-amount] = beats[mask, amount:]
        else:
            shifted[mask] = beats[mask]
    return shifted


class C5bLiteJoint(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pad_sid = int(args.pad_sid)
        self.vocab_size = int(args.vocab_size)
        self.bridge_alpha = float(args.bridge_alpha)
        self.anchor_temperature = float(args.anchor_temperature)
        self.shift_temperature = float(args.shift_temperature)
        self.shift_radius = int(args.shift_radius)

        self.tokenizer = morphology.MorphologyV2PhysAwareVQ(
            codebook_size=args.vocab_size,
            codebook_dim=args.codebook_dim,
            token_len=args.token_len,
            ema_decay=args.ema_decay,
            quantize_kmeans_init=False,
            phys_head_source="z_q",
            continuous_mean=[0.0] * len(morphology.CONTINUOUS_FEATURES),
            continuous_std=[1.0] * len(morphology.CONTINUOUS_FEATURES),
        )
        tokenizer_ckpt = torch.load(args.tokenizer_checkpoint, map_location="cpu", weights_only=False)
        result = self.tokenizer.load_state_dict(tokenizer_ckpt["model"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Tokenizer initialization mismatch: {result}")

        self.regressor = HeartLangSIDForRegression(
            num_outputs=4,
            task_head_type="mlp2",
            task_head_hidden_dim=256,
            task_head_dropout=0.1,
            freeze_backbone=False,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            pad_sid=args.pad_sid,
            embed_dim=args.embed_dim,
            depth=args.depth,
            heads=args.heads,
            dim_head=args.dim_head,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout,
            use_spatial_embedding=False,
            use_temporal_embedding=False,
        )
        downstream_ckpt = torch.load(args.c5a_checkpoint, map_location="cpu", weights_only=False)
        result = self.regressor.load_state_dict(downstream_ckpt["model"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"C5a initialization mismatch: {result}")

        self.bridge = nn.Linear(args.codebook_dim, args.embed_dim, bias=False)
        nn.init.zeros_(self.bridge.weight)
        self.register_buffer(
            "initial_codebook",
            self.tokenizer.quantize.embedding.weight.detach().clone(),
            persistent=True,
        )
        self.stage = ""
        self.configure_stage("bridge_warmup", args)

    def configure_stage(self, stage: str, args) -> None:
        self.stage = stage
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.bridge.parameters():
            parameter.requires_grad = True
        for parameter in self.regressor.mlp_head.parameters():
            parameter.requires_grad = True
        for parameter in self.regressor.backbone.transformer.layers[-1].parameters():
            parameter.requires_grad = True
        for parameter in self.regressor.backbone.norm_layer.parameters():
            parameter.requires_grad = True
        encoder_trainable = stage in {"encoder_adaptation", "slow_ema_joint"}
        for parameter in self.tokenizer.encoder.parameters():
            parameter.requires_grad = encoder_trainable
        self.tokenizer.quantize.embedding.update = stage == "slow_ema_joint"
        self.tokenizer.quantize.decay = float(args.ema_decay)
        self.tokenizer.quantize.embedding.decay = float(args.ema_decay)

    def _sample_deltas(self, count: int, device: torch.device, training: bool) -> torch.Tensor:
        choices = torch.cat(
            [
                torch.arange(-self.shift_radius, 0, device=device),
                torch.arange(1, self.shift_radius + 1, device=device),
            ]
        )
        if training:
            return choices[torch.randint(0, len(choices), (count,), device=device)]
        return choices[torch.arange(count, device=device) % len(choices)]

    def sentence_forward(self, sid: torch.Tensor, z_q_full: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        backbone = self.regressor.backbone
        safe_sid = torch.where(valid, sid, torch.full_like(sid, self.pad_sid))
        tokens = backbone.sid_embed(safe_sid)
        tokens = tokens + self.bridge_alpha * self.bridge(z_q_full)
        cls = backbone.cls_token.view(1, 1, -1).expand(tokens.shape[0], 1, -1)
        hidden = torch.cat([cls, tokens], dim=1)
        hidden = backbone.dropout(hidden + backbone.pos_embed[:, : hidden.shape[1]])
        hidden = backbone.transformer(hidden)
        hidden = backbone.norm_layer(hidden)
        return self.regressor.mlp_head(hidden[:, 0])

    def forward(self, wave, old_sid, chans, times, discrete, training: bool):
        valid, beats, z, z_q, online_ids, quant_loss = self.tokenizer.encode_valid(
            wave, in_chan_matrix=chans, in_time_matrix=times
        )
        if beats.numel() == 0:
            raise RuntimeError("C5b batch has no valid heartbeat tokens")
        online_sid = torch.full_like(old_sid, self.pad_sid)
        online_sid[valid] = online_ids
        z_q_full = torch.zeros(
            wave.shape[0], wave.shape[1], z_q.shape[-1], device=wave.device, dtype=z_q.dtype
        )
        z_q_full[valid] = z_q
        # Keep the exact fixed C2.1 SID stream during bridge warm-up.  The
        # original SID dataset was generated under CUDA autocast, while the
        # joint tokenizer runs in full precision; nearest-neighbour ties can
        # therefore change a small fraction of otherwise identical IDs.  The
        # fixed stream gives an exact C5a epoch-0 baseline.  Once encoder
        # adaptation starts, switch to online IDs so tokenizer/codebook
        # changes participate in the downstream path.
        sentence_sid = old_sid if self.stage == "bridge_warmup" else online_sid
        sentence_valid = (old_sid >= 0) & (old_sid < self.vocab_size) if self.stage == "bridge_warmup" else valid
        pred = self.sentence_forward(sentence_sid, z_q_full, sentence_valid)

        rec = self.tokenizer.decoder(z_q)
        waveform_loss = F.mse_loss(rec.float(), beats.float())
        disc_valid = discrete[valid]
        pattern_loss, _, pattern_log = self.tokenizer.physiology_loss(
            z_q,
            disc_valid,
            None,
            None,
            continuous_clip_z=0.0,
            continuous_loss_reduction="element",
            enable_discrete=True,
        )

        deltas = self._sample_deltas(beats.shape[0], beats.device, training)
        shifted = non_circular_shift(beats, deltas)
        z_shift = self.tokenizer.encoder(shifted)
        shift_latent = 1.0 - F.cosine_similarity(z_shift.float(), z.detach().float(), dim=-1).mean()
        current_codebook = F.normalize(
            self.tokenizer.quantize.embedding.weight.detach().float(), dim=-1
        )
        shift_logits = F.normalize(z_shift.float(), dim=-1) @ current_codebook.T
        shift_logits = shift_logits / max(self.shift_temperature, 1e-6)
        shift_sid = F.cross_entropy(shift_logits, online_ids.detach())
        shift_agreement = (shift_logits.argmax(dim=-1) == online_ids.detach()).float().mean()

        old_valid_sid = old_sid[valid]
        anchor_logits = F.normalize(z.float(), dim=-1) @ F.normalize(
            self.initial_codebook.float(), dim=-1
        ).T
        anchor_logits = anchor_logits / max(self.anchor_temperature, 1e-6)
        anchor_loss = F.cross_entropy(anchor_logits, old_valid_sid)
        sid_agreement = (online_ids == old_valid_sid).float().mean()
        tokenizer_loss = (
            waveform_loss
            + quant_loss.float()
            + 0.2 * pattern_loss.float()
            + 0.05 * shift_latent
            + 0.05 * shift_sid
        )
        codebook_cos = F.cosine_similarity(
            F.normalize(self.tokenizer.quantize.embedding.weight.float(), dim=-1),
            F.normalize(self.initial_codebook.float(), dim=-1),
            dim=-1,
        ).mean()
        aux = {
            "waveform_loss": waveform_loss,
            "quant_loss": quant_loss.float(),
            "pattern_loss": pattern_loss.float(),
            "shift_latent_loss": shift_latent,
            "shift_sid_loss": shift_sid,
            "shift_sid_agreement": shift_agreement,
            "anchor_loss": anchor_loss,
            "sid_agreement": sid_agreement,
            "tokenizer_loss": tokenizer_loss,
            "codebook_cosine_to_initial": codebook_cos,
            "online_ids": online_ids.detach(),
            "pattern_acc": {
                key.removeprefix("phys_").removesuffix("_acc"): value
                for key, value in pattern_log.items()
                if key.endswith("_acc")
            },
        }
        return pred, aux


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=Path, required=True)
    parser.add_argument("--fixed_sid_dir", type=Path, required=True)
    parser.add_argument("--morphology_cache_dir", type=Path, required=True)
    parser.add_argument("--tokenizer_checkpoint", type=Path, required=True)
    parser.add_argument("--c5a_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_iter", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--sentence_lr", type=float, default=1e-5)
    parser.add_argument("--encoder_lr", type=float, default=1e-6)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--clip_grad", type=float, default=3.0)
    parser.add_argument("--lambda_binary", type=float, default=0.2)
    parser.add_argument("--lambda_tokenizer", type=float, default=0.1)
    parser.add_argument("--lambda_anchor", type=float, default=0.05)
    parser.add_argument("--bridge_alpha", type=float, default=0.1)
    parser.add_argument("--anchor_temperature", type=float, default=0.1)
    parser.add_argument("--shift_temperature", type=float, default=0.1)
    parser.add_argument("--shift_radius", type=int, default=4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--token_len", type=int, default=96)
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--pad_sid", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb_dropout", type=float, default=0.1)
    parser.add_argument("--max_train_records", type=int, default=0)
    parser.add_argument("--max_val_records", type=int, default=0)
    parser.add_argument("--max_test_records", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_eval_steps", type=int, default=0)
    return parser.parse_args()


def stage_for_epoch(epoch: int, args) -> str:
    if epoch <= args.stage1_epochs:
        return "bridge_warmup"
    if epoch <= args.stage1_epochs + args.stage2_epochs:
        return "encoder_adaptation"
    return "slow_ema_joint"


def cosine_lr(step: int, total: int, warmup: int, peak: float, minimum: float) -> float:
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    per_r2 = {
        name: float(r2_score(target[:, index], pred[:, index]))
        for index, name in enumerate(LABEL_NAMES)
    }
    disease_pred = np.column_stack([pred[:, :3].sum(axis=1), pred[:, 3]])
    disease_target = np.column_stack([target[:, :3].sum(axis=1), target[:, 3]])
    binary_per = {
        "Disease": float(r2_score(disease_target[:, 0], disease_pred[:, 0])),
        "Healthy": float(r2_score(disease_target[:, 1], disease_pred[:, 1])),
    }
    return {
        "r2": float(np.mean(list(per_r2.values()))),
        "per_output_r2": per_r2,
        "mse": float(np.mean((pred - target) ** 2)),
        "mae": float(np.mean(np.abs(pred - target))),
        "binary": {
            "r2": float(np.mean(list(binary_per.values()))),
            "per_output_r2": binary_per,
            "mse": float(np.mean((disease_pred - disease_target) ** 2)),
            "mae": float(np.mean(np.abs(disease_pred - disease_target))),
        },
    }


def variance_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    variance = np.var(np.asarray(labels, dtype=np.float64), axis=0)
    raw = 1.0 / np.maximum(variance, 1e-8)
    return torch.tensor(raw / raw.mean(), dtype=torch.float32, device=device)


def run_epoch(model, loader, optimizer, args, device, epoch, train, output_weights, schedule):
    stage = stage_for_epoch(epoch, args) if epoch > 0 else "bridge_warmup"
    model.configure_stage(stage, args)
    model.train(train)
    model.tokenizer.quantize.embedding.update = bool(train and stage == "slow_ema_joint")
    outputs, targets = [], []
    sums: dict[str, float] = {}
    usage = np.zeros(args.vocab_size, dtype=np.int64)
    steps = 0
    max_steps = args.max_train_steps if train else args.max_eval_steps
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        wave, old_sid, chans, times, target, discrete = batch
        wave = wave.float().to(device, non_blocking=True)
        old_sid = old_sid.long().to(device, non_blocking=True)
        chans = chans.long().to(device, non_blocking=True)
        times = times.long().to(device, non_blocking=True)
        target = target.float().to(device, non_blocking=True)
        discrete = discrete.long().to(device, non_blocking=True)
        pred, aux = model(wave, old_sid, chans, times, discrete, training=train)
        four_loss = ((pred - target).square() * output_weights.view(1, -1)).mean()
        pred_binary = torch.stack([pred[:, :3].sum(dim=1), pred[:, 3]], dim=1)
        target_binary = torch.stack([target[:, :3].sum(dim=1), target[:, 3]], dim=1)
        binary_loss = (pred_binary - target_binary).square().mean()
        joint_enabled = stage != "bridge_warmup"
        total_loss = four_loss + args.lambda_binary * binary_loss
        if joint_enabled:
            total_loss = (
                total_loss
                + args.lambda_tokenizer * aux["tokenizer_loss"]
                + args.lambda_anchor * aux["anchor_loss"]
            )
        if train:
            (total_loss / args.accum_iter).backward()
            should_step = (step + 1) % args.accum_iter == 0 or step + 1 == len(loader) or (
                max_steps and step + 1 == max_steps
            )
            if should_step:
                opt_step = schedule["optimizer_step"]
                for group in optimizer.param_groups:
                    peak = float(group["peak_lr"])
                    group["lr"] = cosine_lr(
                        opt_step,
                        schedule["total_steps"],
                        schedule["warmup_steps"],
                        peak,
                        peak * args.min_lr_ratio,
                    )
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.clip_grad,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                schedule["optimizer_step"] += 1
        values = {
            "loss": total_loss,
            "four_loss": four_loss,
            "binary_loss": binary_loss,
            **{key: value for key, value in aux.items() if torch.is_tensor(value) and value.ndim == 0},
        }
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().item())
        for key, value in aux["pattern_acc"].items():
            sums[f"pattern_{key}_acc"] = sums.get(f"pattern_{key}_acc", 0.0) + float(value.item())
        ids = aux["online_ids"].detach().cpu().numpy()
        usage += np.bincount(ids, minlength=args.vocab_size)[: args.vocab_size]
        outputs.append(pred.detach().cpu())
        targets.append(target.detach().cpu())
        steps += 1
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == len(loader):
            phase = "train" if train else "eval"
            print(
                f"{phase} epoch={epoch} stage={stage} step={step+1}/{len(loader)} "
                f"loss={total_loss.item():.5f} sid_agree={aux['sid_agreement'].item():.4f} "
                f"code_cos={aux['codebook_cosine_to_initial'].item():.6f}",
                flush=True,
            )
    pred_np = torch.cat(outputs).numpy()
    target_np = torch.cat(targets).numpy()
    result = metrics(pred_np, target_np)
    result.update({key: value / max(steps, 1) for key, value in sums.items()})
    probabilities = usage[usage > 0] / max(int(usage.sum()), 1)
    result["active_sid"] = int((usage > 0).sum())
    result["perplexity"] = float(np.exp(-(probabilities * np.log(probabilities)).sum())) if len(probabilities) else 0.0
    result["stage"] = stage
    return result, pred_np, target_np


def write_history(path: Path, row: dict) -> None:
    flat = {}
    for key, value in row.items():
        if isinstance(value, dict):
            flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            flat[key] = value
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def save_checkpoint(model, optimizer, args, epoch, metrics_value, schedule, name: str) -> Path:
    path = args.output_dir / name
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics_value,
            "optimizer_step": schedule["optimizer_step"],
            "args": vars(args),
            "model_type": "c5b_lite_joint",
        },
        path,
    )
    return path


def save_predictions(output_dir: Path, pred: np.ndarray, target: np.ndarray) -> None:
    np.save(output_dir / "test_predictions.npy", pred)
    np.save(output_dir / "test_targets.npy", target)
    with (output_dir / "test_outputs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"pred_{name}" for name in LABEL_NAMES] + [f"target_{name}" for name in LABEL_NAMES])
        writer.writerows(np.concatenate([pred, target], axis=1).tolist())


def make_dataset(dataset: Dataset, maximum: int) -> Dataset:
    return Subset(dataset, range(min(maximum, len(dataset)))) if maximum > 0 else dataset


def main() -> None:
    args = parse_args()
    patch_checkpoint_paths()
    args.raw_data_dir = resolve(args.raw_data_dir)
    args.fixed_sid_dir = resolve(args.fixed_sid_dir)
    args.morphology_cache_dir = resolve(args.morphology_cache_dir)
    args.tokenizer_checkpoint = resolve(args.tokenizer_checkpoint)
    args.c5a_checkpoint = resolve(args.c5a_checkpoint)
    args.output_dir = resolve(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    full_train = JointDataset(args.raw_data_dir, args.fixed_sid_dir, args.morphology_cache_dir, "train")
    full_val = JointDataset(args.raw_data_dir, args.fixed_sid_dir, args.morphology_cache_dir, "val")
    full_test = JointDataset(args.raw_data_dir, args.fixed_sid_dir, args.morphology_cache_dir, "test")
    train_ds = make_dataset(full_train, args.max_train_records)
    val_ds = make_dataset(full_val, args.max_val_records)
    test_ds = make_dataset(full_test, args.max_test_records)
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True),
        "val": DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    }
    model = C5bLiteJoint(args).to(device)
    output_weights = variance_weights(full_train.labels, device)
    head_bridge = list(model.regressor.mlp_head.parameters()) + list(model.bridge.parameters())
    sentence = list(model.regressor.backbone.transformer.layers[-1].parameters()) + list(model.regressor.backbone.norm_layer.parameters())
    encoder = list(model.tokenizer.encoder.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": head_bridge, "lr": args.head_lr, "peak_lr": args.head_lr, "name": "head_bridge"},
            {"params": sentence, "lr": args.sentence_lr, "peak_lr": args.sentence_lr, "name": "sentence_last"},
            {"params": encoder, "lr": args.encoder_lr, "peak_lr": args.encoder_lr, "name": "morphology_encoder"},
        ],
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loaders["train"]) / args.accum_iter)
    schedule = {
        "optimizer_step": 0,
        "total_steps": args.epochs * updates_per_epoch,
        "warmup_steps": args.warmup_epochs * updates_per_epoch,
    }
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_val_test": [len(train_ds), len(val_ds), len(test_ds)],
        "output_loss_weights": output_weights.cpu().tolist(),
        "trainable_counts": {
            "head_bridge": sum(parameter.numel() for parameter in head_bridge),
            "sentence_last": sum(parameter.numel() for parameter in sentence),
            "morphology_encoder": sum(parameter.numel() for parameter in encoder),
        },
        "stage_schedule": {
            "bridge_warmup": [1, args.stage1_epochs],
            "encoder_adaptation": [args.stage1_epochs + 1, args.stage1_epochs + args.stage2_epochs],
            "slow_ema_joint": [args.stage1_epochs + args.stage2_epochs + 1, args.epochs],
        },
        "codebook_update": "EMA only",
        "test_evaluations": 1,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    # Epoch 0 must reproduce the C5a validation baseline before any joint update.
    baseline_schedule = dict(schedule)
    with torch.no_grad():
        baseline_val, _, _ = run_epoch(
            model, loaders["val"], optimizer, args, device, 0, False, output_weights, baseline_schedule
        )
    expected_c5a_val = float(
        torch.load(args.c5a_checkpoint, map_location="cpu", weights_only=False)
        .get("metrics", {})
        .get("r2", float("nan"))
    )
    baseline_audit = {
        "online_vs_fixed_sid_agreement": baseline_val.get("sid_agreement"),
        "joint_epoch0_val_r2": baseline_val["r2"],
        "source_c5a_val_r2": expected_c5a_val,
        "absolute_val_r2_difference": abs(baseline_val["r2"] - expected_c5a_val),
    }
    # Online/fixed agreement is audited rather than required to be bit exact:
    # fixed C2.1 IDs were generated with CUDA autocast, so full-precision
    # nearest-neighbour assignment can differ near Voronoi boundaries.  The
    # actual epoch-0 prediction path above uses fixed IDs and must still
    # reproduce the source C5a validation score exactly on the full split.
    if baseline_val.get("sid_agreement", 0.0) < 0.99:
        raise RuntimeError(f"Online C2.1 SID agreement unexpectedly low: {baseline_audit}")
    if args.max_val_records == 0 and math.isfinite(expected_c5a_val):
        if baseline_audit["absolute_val_r2_difference"] > 1e-5:
            raise RuntimeError(f"Epoch-0 C5a reproduction failed: {baseline_audit}")
    best_val_r2 = float(baseline_val["r2"])
    best_epoch = 0
    save_checkpoint(model, optimizer, args, 0, baseline_val, schedule, "checkpoint-best.pth")
    (args.output_dir / "baseline_validation.json").write_text(
        json.dumps({"metrics": baseline_val, "audit": baseline_audit}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history = args.output_dir / "training_history.csv"
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(
            model, loaders["train"], optimizer, args, device, epoch, True, output_weights, schedule
        )
        with torch.no_grad():
            val_metrics, _, _ = run_epoch(
                model, loaders["val"], optimizer, args, device, epoch, False, output_weights, schedule
            )
        row = {"epoch": epoch, "stage": stage_for_epoch(epoch, args), "train": train_metrics, "val": val_metrics}
        write_history(history, row)
        with (args.output_dir / "log.txt").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = float(val_metrics["r2"])
            best_epoch = epoch
            save_checkpoint(model, optimizer, args, epoch, val_metrics, schedule, "checkpoint-best.pth")
        if epoch in {args.stage1_epochs, args.stage1_epochs + args.stage2_epochs, args.epochs}:
            save_checkpoint(model, optimizer, args, epoch, val_metrics, schedule, f"checkpoint-{epoch}.pth")

    checkpoint = torch.load(args.output_dir / "checkpoint-best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.no_grad():
        test_metrics, prediction, target = run_epoch(
            model, loaders["test"], optimizer, args, device, int(checkpoint["epoch"]), False, output_weights, schedule
        )
    test_metrics.update(
        {
            "best_val_r2": best_val_r2,
            "best_epoch": best_epoch,
            "n_test": len(test_ds),
            "label_names": LABEL_NAMES,
            "c5a_checkpoint": str(args.c5a_checkpoint),
            "tokenizer_checkpoint": str(args.tokenizer_checkpoint),
        }
    )
    (args.output_dir / "test_regression_summary.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_predictions(args.output_dir, prediction, target)
    tokenizer_payload = {
        "model": model.tokenizer.state_dict(),
        "epoch": best_epoch,
        "args": vars(args),
        "model_type": "c5b_lite_exported_tokenizer",
    }
    torch.save(tokenizer_payload, args.output_dir / "tokenizer-checkpoint-best.pth")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Training time: {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
