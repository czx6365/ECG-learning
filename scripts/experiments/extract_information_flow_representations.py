#!/usr/bin/env python3
"""Extract record-level z_e, z_q, and HeartLang representations on fixed splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.models.modeling_heartlang_sid import HeartLangSIDBackbone
from scripts.eval.codebook.evaluate_morphology_only_comparison import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Extract frozen representations for information-flow probes")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME,TOKENIZER_CHECKPOINT,SID_DATA_DIR,HEARTLANG_CHECKPOINT",
    )
    parser.add_argument("--waveform_data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save_token_features",
        action="store_true",
        help="Also save [record, heartbeat, code_dim] z_e/z_q arrays and the valid-heartbeat mask.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_run(spec: str) -> tuple[str, Path, Path, Path]:
    parts = [part.strip() for part in spec.split(",", 3)]
    if len(parts) != 4:
        raise ValueError("--run must be NAME,TOKENIZER_CHECKPOINT,SID_DATA_DIR,HEARTLANG_CHECKPOINT")
    return parts[0], resolve(Path(parts[1])), resolve(Path(parts[2])), resolve(Path(parts[3]))


def load_heartlang(checkpoint: Path, device: torch.device) -> tuple[HeartLangSIDBackbone, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    config = {
        "seq_len": int(saved_args.get("seq_len", 256)),
        "vocab_size": int(saved_args.get("vocab_size", 1024)),
        "pad_sid": int(saved_args.get("pad_sid", 1024)),
        "embed_dim": int(saved_args.get("embed_dim", 768)),
        "depth": int(saved_args.get("depth", 12)),
        "heads": int(saved_args.get("heads", 8)),
        "dim_head": int(saved_args.get("dim_head", 64)),
        "mlp_dim": int(saved_args.get("mlp_dim", 1024)),
        "dropout": float(saved_args.get("dropout", 0.1)),
        "emb_dropout": float(saved_args.get("emb_dropout", 0.1)),
        "use_spatial_embedding": bool(saved_args.get("use_spatial_embedding", False)),
        "use_temporal_embedding": bool(saved_args.get("use_temporal_embedding", False)),
    }
    model = HeartLangSIDBackbone(**config)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    backbone_state = {
        key[len("backbone.") :]: value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError(f"No backbone.* parameters found in {checkpoint}")
    model.load_state_dict(backbone_state, strict=True)
    model.eval().to(device)
    return model, config


def assert_same_array(name: str, waveform_array: np.ndarray, sid_array: np.ndarray) -> None:
    if waveform_array.shape != sid_array.shape:
        raise ValueError(f"{name} shape mismatch: {waveform_array.shape} != {sid_array.shape}")
    if not np.array_equal(np.asarray(waveform_array), np.asarray(sid_array)):
        raise ValueError(f"{name} values differ between waveform and SID datasets")


@torch.inference_mode()
def extract_split(
    tokenizer,
    heartlang,
    waveform_data_dir: Path,
    sid_data_dir: Path,
    split: str,
    batch_size: int,
    codebook_dim: int,
    device: torch.device,
    token_output_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    waves = np.load(waveform_data_dir / f"{split}_data.npy", mmap_mode="r")
    wave_chans = np.load(waveform_data_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
    wave_times = np.load(waveform_data_dir / f"{split}_data_in_times.npy", mmap_mode="r")
    wave_labels = np.load(waveform_data_dir / f"{split}_labels.npy", mmap_mode="r", allow_pickle=True)
    sid = np.load(sid_data_dir / f"{split}_sid.npy", mmap_mode="r")
    sid_chans = np.load(sid_data_dir / f"{split}_data_in_chans.npy", mmap_mode="r")
    sid_times = np.load(sid_data_dir / f"{split}_data_in_times.npy", mmap_mode="r")
    sid_labels = np.load(sid_data_dir / f"{split}_labels.npy", mmap_mode="r", allow_pickle=True)

    if len(waves) != len(sid):
        raise ValueError(f"{split}: record count mismatch: waveform={len(waves)}, sid={len(sid)}")
    assert_same_array(f"{split} labels", wave_labels, sid_labels)
    assert_same_array(f"{split} channel ids", wave_chans, sid_chans)
    assert_same_array(f"{split} time ids", wave_times, sid_times)

    result = {
        "z_e": np.empty((len(waves), codebook_dim), dtype=np.float32),
        "z_q": np.empty((len(waves), codebook_dim), dtype=np.float32),
        "z_e_mean_max": np.empty((len(waves), 2 * codebook_dim), dtype=np.float32),
        "z_q_mean_max": np.empty((len(waves), 2 * codebook_dim), dtype=np.float32),
        "heartlang": np.empty((len(waves), heartlang.embed_dim), dtype=np.float32),
    }
    token_outputs = None
    if token_output_dir is not None:
        token_outputs = {
            "z_e": np.lib.format.open_memmap(
                token_output_dir / f"{split}_z_e_tokens.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(waves), waves.shape[1], codebook_dim),
            ),
            "z_q": np.lib.format.open_memmap(
                token_output_dir / f"{split}_z_q_tokens.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(waves), waves.shape[1], codebook_dim),
            ),
            "valid": np.lib.format.open_memmap(
                token_output_dir / f"{split}_valid_heartbeat_mask.npy",
                mode="w+",
                dtype=np.bool_,
                shape=(len(waves), waves.shape[1]),
            ),
        }
    for start in range(0, len(waves), batch_size):
        end = min(start + batch_size, len(waves))
        x = torch.from_numpy(np.asarray(waves[start:end]).copy()).float().to(device)
        chans = torch.from_numpy(np.asarray(wave_chans[start:end]).copy()).long().to(device)
        times = torch.from_numpy(np.asarray(wave_times[start:end]).copy()).long().to(device)
        sid_batch = torch.from_numpy(np.asarray(sid[start:end]).copy()).long().to(device)

        valid, _beats, z_e, z_q, _ids, _loss = tokenizer.encode_valid(
            x, in_chan_matrix=chans, in_time_matrix=times
        )
        z_e_full = torch.zeros((*valid.shape, codebook_dim), device=device, dtype=z_e.dtype)
        z_q_full = torch.zeros((*valid.shape, codebook_dim), device=device, dtype=z_q.dtype)
        z_e_full[valid] = z_e
        z_q_full[valid] = z_q
        if token_outputs is not None:
            token_outputs["z_e"][start:end] = z_e_full.float().cpu().numpy()
            token_outputs["z_q"][start:end] = z_q_full.float().cpu().numpy()
            token_outputs["valid"][start:end] = valid.cpu().numpy()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).to(z_e_full.dtype)
        z_e_mean = z_e_full.sum(dim=1) / denom
        z_q_mean = z_q_full.sum(dim=1) / denom
        valid_expanded = valid.unsqueeze(-1)
        z_e_max = z_e_full.masked_fill(~valid_expanded, -torch.inf).amax(dim=1)
        z_q_max = z_q_full.masked_fill(~valid_expanded, -torch.inf).amax(dim=1)
        has_valid = valid.any(dim=1, keepdim=True)
        z_e_max = torch.where(has_valid, z_e_max, torch.zeros_like(z_e_max))
        z_q_max = torch.where(has_valid, z_q_max, torch.zeros_like(z_q_max))
        result["z_e"][start:end] = z_e_mean.float().cpu().numpy()
        result["z_q"][start:end] = z_q_mean.float().cpu().numpy()
        result["z_e_mean_max"][start:end] = torch.cat([z_e_mean, z_e_max], dim=-1).float().cpu().numpy()
        result["z_q_mean_max"][start:end] = torch.cat([z_q_mean, z_q_max], dim=-1).float().cpu().numpy()
        result["heartlang"][start:end] = heartlang(
            sid_batch,
            in_chan_matrix=chans,
            in_time_matrix=times,
        ).float().cpu().numpy()
        if start == 0 or end == len(waves) or end % (batch_size * 20) == 0:
            print(f"{split}: extracted {end}/{len(waves)} records", flush=True)

    if token_outputs is not None:
        for output in token_outputs.values():
            output.flush()

    return result, np.asarray(wave_labels).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.waveform_data_dir = resolve(args.waveform_data_dir)
    args.output_dir = resolve(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = {
        "waveform_data_dir": str(args.waveform_data_dir),
        "pooling": "masked mean over valid heartbeat tokens for z_e and z_q; CLS token for HeartLang",
        "splits": ["train", "val", "test"],
        "runs": [],
    }
    for name, tokenizer_checkpoint, sid_data_dir, heartlang_checkpoint in map(parse_run, args.run):
        print(f"Loading run {name}", flush=True)
        run_spec = {
            "name": name,
            "tokenizer_checkpoint": tokenizer_checkpoint,
            "sid_data_dir": sid_data_dir,
            "heartlang_checkpoint": heartlang_checkpoint,
        }
        tokenizer_args = SimpleNamespace(
            codebook_size=args.codebook_size,
            codebook_dim=args.codebook_dim,
        )
        tokenizer, resolved_tokenizer = load_model(
            {
                "name": name,
                "model_type": "morphology_v2",
                "checkpoint": tokenizer_checkpoint,
            },
            tokenizer_args,
            device,
        )
        tokenizer.eval()
        heartlang, heartlang_config = load_heartlang(heartlang_checkpoint, device)
        run_dir = args.output_dir / name
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite representation directory: {run_dir}")
        run_dir.mkdir(parents=True)
        split_records = {}
        for split in ("train", "val", "test"):
            features, labels = extract_split(
                tokenizer,
                heartlang,
                args.waveform_data_dir,
                sid_data_dir,
                split,
                args.batch_size,
                args.codebook_dim,
                device,
                run_dir if args.save_token_features else None,
            )
            for representation, values in features.items():
                np.save(run_dir / f"{split}_{representation}.npy", values)
            np.save(run_dir / f"{split}_labels.npy", labels)
            split_records[split] = int(len(labels))
        metadata = {
            "name": name,
            "tokenizer_checkpoint": str(resolved_tokenizer),
            "sid_data_dir": str(sid_data_dir),
            "heartlang_checkpoint": str(heartlang_checkpoint),
            "heartlang_config": heartlang_config,
            "split_records": split_records,
            "z_e_definition": "raw MorphologyEncoder output before NormEMAVectorQuantizer",
            "z_q_definition": "continuous quantized embedding returned by NormEMAVectorQuantizer",
            "mean_max_definition": "concatenate masked mean and masked max over valid heartbeat tokens; zero vector for records without valid heartbeats",
            "token_features_saved": bool(args.save_token_features),
            "heartlang_definition": "frozen pretrained HeartLang SID-only CLS representation",
        }
        (run_dir / "representation_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest["runs"].append(metadata)
        del tokenizer, heartlang
        torch.cuda.empty_cache()
    (args.output_dir / "extraction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
