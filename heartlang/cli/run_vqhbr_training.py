# --------------------------------------------------------
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model
# By Jiarui Jin and Haoyu Wang
# Based on BEiT-v2, timm, DeiT, DINO and LaBraM code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/935963004/LaBraM
# ---------------------------------------------------------

import argparse
import csv
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
import sys
from timm.models import create_model

# 允许直接以 `python heartlang/cli/run_vqhbr_training.py` 方式运行。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.utils.optim_factory import create_optimizer
from heartlang.engines.engine_for_vqhbr import evaluate, train_one_epoch
from heartlang.utils.utils import NativeScalerWithGradNormCount as NativeScaler, set_seed
import heartlang.utils.utils as utils
from heartlang.utils.QRSDataset import build_ECGpretrain_dataset
from heartlang.models import modeling_vqhbr as modeling_vqhbr


def default_pretrain_data_dirs():
    # 默认读取已经切好的多个 QRS 预训练批次。
    return [
        Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_0"),
        Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_2"),
        Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_3"),
        Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_4"),
        Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_5"),
        Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_6"),
        Path("datasets/ecg_datasets/MIMIC-IV_QRS/batch_7"),
    ]


def has_required_pretrain_files(data_dir: Path, stage: str) -> bool:
    required_files = [
        data_dir / f"{stage}_data.npy",
        data_dir / f"{stage}_data_in_chans.npy",
        data_dir / f"{stage}_data_in_times.npy",
    ]
    return all(path.is_file() for path in required_files)


def resolve_pretrain_data_dirs(input_dirs: list[Path]) -> list[Path]:
    resolved_dirs = []
    seen_dirs = set()

    for raw_dir in input_dirs:
        data_dir = Path(raw_dir).expanduser().resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"Pretrain data dir does not exist: {data_dir}")

        candidate_dirs = []
        if has_required_pretrain_files(data_dir, "train") and has_required_pretrain_files(
            data_dir, "val"
        ):
            candidate_dirs = [data_dir]
        else:
            candidate_dirs = [
                child
                for child in sorted(data_dir.iterdir())
                if child.is_dir()
                and has_required_pretrain_files(child, "train")
                and has_required_pretrain_files(child, "val")
            ]

        if not candidate_dirs:
            raise FileNotFoundError(
                "No usable VQHBR pretrain split files were found under "
                f"{data_dir}. Expected either this directory or its immediate "
                "subdirectories to contain train/val *_data.npy, "
                "*_data_in_chans.npy, and *_data_in_times.npy files."
            )

        for candidate in candidate_dirs:
            candidate_key = str(candidate)
            if candidate_key not in seen_dirs:
                seen_dirs.add(candidate_key)
                resolved_dirs.append(candidate)

    return resolved_dirs


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def load_init_checkpoint(model_without_ddp, checkpoint_path: str, strict: bool = False):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Init checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = extract_model_state_dict(checkpoint)
    incompatible = model_without_ddp.load_state_dict(checkpoint_model, strict=strict)

    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    print(f"Initialized VQHBR weights from {checkpoint_path}")
    print(
        "Init checkpoint load summary: "
        f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
    )
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)


def parse_int_list(raw_value: str) -> list[int]:
    if raw_value is None or raw_value.strip() == "":
        return []
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def choose_polarity_dead_codes(args, target_sids: list[int]) -> tuple[list[int], dict[int, int]]:
    explicit_dead_codes = parse_int_list(args.polarity_dead_codes)
    if explicit_dead_codes:
        if len(explicit_dead_codes) != len(target_sids):
            raise ValueError("--polarity_dead_codes must have the same length as --polarity_split_sids.")
        return explicit_dead_codes, {code: -1 for code in explicit_dead_codes}

    usage_path = Path(args.polarity_dead_code_usage_csv).expanduser().resolve()
    if not usage_path.is_file():
        raise FileNotFoundError(f"Missing dead-code usage csv: {usage_path}")

    usage_counts = {code: 0 for code in range(args.codebook_n_emd)}
    with usage_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = int(row["sid"])
            if 0 <= code < args.codebook_n_emd:
                usage_counts[code] = int(float(row["count"]))

    excluded = set(target_sids)
    ranked_codes = sorted(
        (code for code in range(args.codebook_n_emd) if code not in excluded),
        key=lambda code: (usage_counts.get(code, 0), code),
    )
    dead_codes = ranked_codes[: len(target_sids)]
    if len(dead_codes) < len(target_sids):
        raise RuntimeError("Not enough codebook entries are available for polarity split.")

    if any(usage_counts[code] > 0 for code in dead_codes):
        print("Warning: not enough zero-count dead codes; using lowest-frequency codes.")
    return dead_codes, usage_counts


@torch.no_grad()
def initialize_polarity_code_pairs(model_without_ddp, data_loader_train_list, device, args):
    target_sids = parse_int_list(args.polarity_split_sids)
    if not target_sids:
        return []

    dead_codes, usage_counts = choose_polarity_dead_codes(args, target_sids)
    pairs = list(zip(target_sids, dead_codes))
    print("Polarity-aware paired code mapping:")
    for pos_code, neg_code in pairs:
        print(
            f"  {pos_code} positive -> {pos_code}; "
            f"{pos_code} negative -> {neg_code} "
            f"(H clean baseline count={usage_counts.get(neg_code, -1)})"
        )

    model_without_ddp.eval()
    weight_before = model_without_ddp.quantize.embedding.weight.detach().clone()
    sums_pos = {sid: torch.zeros(args.codebook_emd_dim, device=device) for sid in target_sids}
    sums_neg = {sid: torch.zeros(args.codebook_emd_dim, device=device) for sid in target_sids}
    counts_pos = {sid: 0 for sid in target_sids}
    counts_neg = {sid: 0 for sid in target_sids}
    target_tensor = torch.as_tensor(target_sids, dtype=torch.long, device=device)

    print("Collecting pre-quantization latent means for polarity-aware code init...")
    for data_loader in data_loader_train_list:
        for step, batch in enumerate(data_loader):
            ECG = batch[0].float().to(device, non_blocking=True)
            in_chan_matrix = batch[1].to(device, non_blocking=True)
            in_time_matrix = batch[2].to(device, non_blocking=True)
            valid_token_mask = (
                (in_chan_matrix > 0)
                & (in_time_matrix > 0)
                & ~(ECG == 0).all(dim=-1)
            )
            z = model_without_ddp.encode_to_quantizer_features(
                ECG, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
            )
            z_flat = z.reshape(-1, args.codebook_emd_dim)
            valid_flat = valid_token_mask.reshape(-1)
            z_valid = z_flat[valid_flat]
            if z_valid.numel() == 0:
                continue
            sim = z_valid @ weight_before.to(device).t()
            nearest = torch.argmax(sim, dim=1)
            polarity = model_without_ddp.compute_polarity_labels(ECG).reshape(-1)[valid_flat]
            in_target = (nearest[:, None] == target_tensor[None, :]).any(dim=1)
            if in_target.any():
                for sid in target_sids:
                    sid_mask = nearest == sid
                    pos_mask = sid_mask & (polarity > 0)
                    neg_mask = sid_mask & (polarity < 0)
                    if pos_mask.any():
                        sums_pos[sid] += z_valid[pos_mask].sum(dim=0)
                        counts_pos[sid] += int(pos_mask.sum().item())
                    if neg_mask.any():
                        sums_neg[sid] += z_valid[neg_mask].sum(dim=0)
                        counts_neg[sid] += int(neg_mask.sum().item())
            if step == 0 or (step + 1) % 50 == 0:
                total = sum(counts_pos.values()) + sum(counts_neg.values())
                print(f"  latent init scan step={step + 1}, collected target tokens={total}")

    rows = []
    weight_after = weight_before.to(device).clone()
    for pos_code, neg_code in pairs:
        if counts_pos[pos_code] == 0 or counts_neg[pos_code] == 0:
            raise RuntimeError(
                f"Cannot initialize polarity pair for SID {pos_code}: "
                f"pos={counts_pos[pos_code]}, neg={counts_neg[pos_code]}"
            )
        pos_mean = torch.nn.functional.normalize(
            sums_pos[pos_code] / counts_pos[pos_code], p=2, dim=0
        )
        neg_mean = torch.nn.functional.normalize(
            sums_neg[pos_code] / counts_neg[pos_code], p=2, dim=0
        )
        old_pos = weight_before[pos_code].to(device)
        old_neg = weight_before[neg_code].to(device)
        before = {
            "pos_to_pos_code": float(torch.dot(pos_mean, old_pos).item()),
            "pos_to_neg_code": float(torch.dot(pos_mean, old_neg).item()),
            "neg_to_pos_code": float(torch.dot(neg_mean, old_pos).item()),
            "neg_to_neg_code": float(torch.dot(neg_mean, old_neg).item()),
        }
        weight_after[pos_code] = pos_mean
        weight_after[neg_code] = neg_mean
        after = {
            "pos_to_pos_code": float(torch.dot(pos_mean, weight_after[pos_code]).item()),
            "pos_to_neg_code": float(torch.dot(pos_mean, weight_after[neg_code]).item()),
            "neg_to_pos_code": float(torch.dot(neg_mean, weight_after[pos_code]).item()),
            "neg_to_neg_code": float(torch.dot(neg_mean, weight_after[neg_code]).item()),
        }
        rows.append(
            {
                "positive_code": pos_code,
                "negative_code": neg_code,
                "baseline_negative_code_count": int(usage_counts.get(neg_code, -1)),
                "positive_samples": int(counts_pos[pos_code]),
                "negative_samples": int(counts_neg[pos_code]),
                "before": before,
                "after": after,
                "positive_code_cosine_distance_old_to_new": float(1.0 - torch.dot(old_pos, pos_mean).item()),
                "negative_code_cosine_distance_old_to_new": float(1.0 - torch.dot(old_neg, neg_mean).item()),
                "paired_new_code_cosine_distance": float(1.0 - torch.dot(pos_mean, neg_mean).item()),
            }
        )

    model_without_ddp.quantize.embedding.weight.data.copy_(weight_after.to(model_without_ddp.quantize.embedding.weight.device))
    model_without_ddp.quantize.embedding.embed_avg.data.copy_(model_without_ddp.quantize.embedding.weight.data)
    for row in rows:
        model_without_ddp.quantize.embedding.cluster_size.data[row["positive_code"]] = max(
            row["positive_samples"], 1
        )
        model_without_ddp.quantize.embedding.cluster_size.data[row["negative_code"]] = max(
            row["negative_samples"], 1
        )
        if hasattr(model_without_ddp.quantize, "cluster_size"):
            model_without_ddp.quantize.cluster_size.data[row["positive_code"]] = max(
                row["positive_samples"], 1
            )
            model_without_ddp.quantize.cluster_size.data[row["negative_code"]] = max(
                row["negative_samples"], 1
            )

    model_without_ddp.configure_polarity_pairs(
        pairs=pairs,
        loss_weight=args.polarity_loss_weight,
        margin=args.polarity_margin,
        force_assignment=False,
    )

    if args.output_dir and utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "polarity_init_summary.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "target_sids": target_sids,
                    "pairs": rows,
                    "loss_weight": args.polarity_loss_weight,
                    "margin": args.polarity_margin,
                    "warmup_epochs": args.polarity_warmup_epochs,
                    "dead_code_usage_csv": str(Path(args.polarity_dead_code_usage_csv)),
                },
                handle,
                indent=2,
            )

    print("Polarity code initialization summary:")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    return pairs


def get_args():
    parser = argparse.ArgumentParser("VQ-HBR training", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_ckpt_freq", default=20, type=int)
    parser.add_argument("--val_freq", default=10, type=int)
    parser.add_argument("--codebook_visualization", action="store_true", default=False)
    # VQ tokenizer 模型参数。
    parser.add_argument(
        "--model",
        default="vqhbr",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument(
        "--codebook_n_emd",
        default=8192,
        type=int,
        metavar="MODEL",
        help="number of codebook",
    )
    parser.add_argument(
        "--codebook_emd_dim",
        default=128,
        type=int,
        metavar="MODEL",
        help="dim of code",
    )
    parser.add_argument(
        "--ema_decay",
        default=0.99,
        type=float,
        metavar="MODEL",
        help="ema decay for quantizer",
    )
    parser.add_argument(
        "--quantize_kmeans_init",
        action="store_true",
        help="enable kmeans_init for quantizer",
    )
    parser.add_argument(
        "--codebook_balance_loss_weight",
        default=0.0,
        type=float,
        help="Weight for differentiable batch-level code usage balancing loss.",
    )
    parser.add_argument(
        "--codebook_balance_temperature",
        default=0.1,
        type=float,
        help="Soft-assignment temperature used by code usage balancing loss.",
    )
    parser.add_argument(
        "--restart_unused_codes",
        action="store_true",
        help="Restart unused codebook entries from recent encoder latents after each epoch.",
    )
    parser.add_argument(
        "--restart_unused_min_cluster_size",
        default=1e-6,
        type=float,
        help="Code usage threshold for dead-code restart.",
    )
    parser.add_argument(
        "--restart_unused_freq",
        default=1,
        type=int,
        help="Epoch frequency for dead-code restart.",
    )
    parser.add_argument(
        "--restart_unused_warmup_epochs",
        default=1,
        type=int,
        help="Start dead-code restart after this epoch.",
    )
    parser.add_argument(
        "--restart_unused_max_codes",
        default=0,
        type=int,
        help="Maximum codes to restart each time; 0 means no limit.",
    )
    parser.add_argument(
        "--polarity_split_sids",
        default="",
        type=str,
        help="Comma-separated original SID list for polarity-aware paired-code training.",
    )
    parser.add_argument(
        "--polarity_dead_codes",
        default="",
        type=str,
        help="Optional comma-separated dead/low-usage code list paired with polarity_split_sids.",
    )
    parser.add_argument(
        "--polarity_dead_code_usage_csv",
        default="outputs/sid_quality/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_H_1024_dim64_val_shift_corr_pm6/sid_usage_summary.csv",
        type=str,
        help="H clean usage CSV used to select stable dead or lowest-usage codes.",
    )
    parser.add_argument(
        "--polarity_loss_weight",
        default=0.0,
        type=float,
        help="lambda for polarity margin loss. 0 disables polarity-aware training.",
    )
    parser.add_argument(
        "--polarity_margin",
        default=0.1,
        type=float,
        help="Margin for paired positive/negative code separation.",
    )
    parser.add_argument(
        "--polarity_warmup_epochs",
        default=0,
        type=int,
        help="Epochs that use polarity-gated assignment for paired codes.",
    )

    # 优化器与学习率参数。
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        metavar="LR",
        help="learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )

    # 数据与输出参数。
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument(
        "--pretrain_data_dirs",
        type=Path,
        nargs="+",
        default=default_pretrain_data_dirs(),
        help="One or more QRS batch directories that contain train_data.npy and val_data.npy.",
    )

    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="initialize model weights from a checkpoint without restoring optimizer/epoch state",
    )
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=True,
        help="Enabling distributed evaluation",
    )
    parser.add_argument("--disable_eval", action="store_true", default=False)

    parser.add_argument(
        "--eval", action="store_true", default=False, help="Perform evaluation only"
    )

    parser.add_argument(
        "--start_epoch",
        default=1,
        type=int,
        metavar="N",
        help="start epoch, origin default = 1",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # 分布式训练参数。
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser.parse_args()


def get_model(args, **kwargs):
    # 构建 VQHBR tokenizer，负责“连续波形 -> 离散 codebook token”。
    model = create_model(
        args.model,
        pretrained=False,
        as_tokenzer=False,
        n_code=args.codebook_n_emd,
        code_dim=args.codebook_emd_dim,
        decay=args.ema_decay,
        quantize_kmeans_init=args.quantize_kmeans_init,
        codebook_balance_loss_weight=args.codebook_balance_loss_weight,
        codebook_balance_temperature=args.codebook_balance_temperature,
    )
    return model


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    set_seed(args.seed)

    # 1. 创建 tokenizer 模型本体。
    model = get_model(args)

    # 2. 加载训练/验证数据，输入都是 QRS token 序列。
    pretrain_data_dirs = resolve_pretrain_data_dirs(args.pretrain_data_dirs)
    print("Resolved pretrain data dirs:")
    for pretrain_data_dir in pretrain_data_dirs:
        print(f"  - {pretrain_data_dir}")
    datasets_train = [pretrain_data_dirs]
    datasets_val = [pretrain_data_dirs]

    dataset_train_list = build_ECGpretrain_dataset(datasets_train, stage="train")
    if args.disable_eval:
        dataset_val_list = None
    else:
        dataset_val_list = build_ECGpretrain_dataset(datasets_val, stage="val")

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        # 每个 epoch 的 step 数由总样本数、batch size 和卡数共同决定。
        num_training_steps_per_epoch = (
            sum([len(dataset) for dataset in dataset_train_list])
            // args.batch_size
            // num_tasks
        )

        sampler_train_list = []
        for dataset in dataset_train_list:
            # 分布式训练下，每张卡只读取各自分片。
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
            sampler_train_list.append(sampler_train)
            print("Sampler_train = %s" % str(sampler_train))

        sampler_eval_list = []
        if args.dist_eval:
            # if len(dataset_val) % num_tasks != 0:
            #     print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
            #           'This will slightly alter validation results as extra duplicate entries are added to achieve '
            #           'equal num of samples per-process.')
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
                )
                sampler_eval_list.append(sampler_val)
        else:
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.SequentialSampler(dataset)
                sampler_eval_list.append(sampler_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train_list = []
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        # 每个数据分片使用独立的 DataLoader。
        data_loader_train = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_train_list.append(data_loader_train)

    if dataset_val_list is not None:
        data_loader_val_list = []
        for dataset, sampler in zip(dataset_val_list, sampler_eval_list):
            data_loader_val = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
            data_loader_val_list.append(data_loader_val)
    else:
        data_loader_val_list = None

    model.to(device)
    model_without_ddp = model

    if args.init_checkpoint:
        load_init_checkpoint(model_without_ddp, args.init_checkpoint)

    polarity_pairs = []
    if args.polarity_split_sids:
        if not args.init_checkpoint:
            raise ValueError("polarity-aware training must start from --init_checkpoint.")
        if args.polarity_loss_weight <= 0:
            raise ValueError("--polarity_loss_weight must be > 0 when --polarity_split_sids is set.")
        polarity_pairs = initialize_polarity_code_pairs(
            model_without_ddp, data_loader_train_list, device, args
        )

    if not args.eval:
        print("Model = %s" % str(model_without_ddp))

    for part in ["encoder", "decoder"]:
        model_part = eval(f"model.{part}")
        n_learnable_parameters = sum(
            p.numel() for p in model_part.parameters() if p.requires_grad
        )
        n_fix_parameters = sum(
            p.numel() for p in model_part.parameters() if not p.requires_grad
        )
        print(
            f"number of learnable params in model.{part}: {n_learnable_parameters / 1e6} M"
        )
        print(f"number of fixed params in model.{part}: {n_fix_parameters / 1e6} M")

    n_learnable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    n_fix_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"total number of learnable params: {n_learnable_parameters / 1e6} M")
    print(f"total number of fixed params in : {n_fix_parameters / 1e6} M")

    total_batch_size = args.batch_size * utils.get_world_size()
    # 学习率按总 batch size 线性缩放。
    args.lr = total_batch_size / 128 * args.lr
    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    print("Weigth Decay = %.8f" % args.weight_decay)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print(
        "Number of training examples per epoch = %d"
        % (total_batch_size * num_training_steps_per_epoch)
    )

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    print("Use step level LR & WD scheduler!")
    # VQ tokenizer 训练使用按 step 更新的 cosine 学习率调度。
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        test_stats = evaluate(
            data_loader_val_list, model, device, log_writer, 0, args=args
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch - 1)
        if log_writer is not None:
            log_writer.set_step((epoch - 1) * num_training_steps_per_epoch)

        if polarity_pairs:
            use_polarity_warmup = epoch <= args.polarity_warmup_epochs
            model_without_ddp.set_polarity_force_assignment(use_polarity_warmup)
            print(
                f"Polarity-aware training epoch {epoch}: "
                f"force_assignment={use_polarity_warmup}, "
                f"lambda={args.polarity_loss_weight}, margin={args.polarity_margin}"
            )

        # 核心训练逻辑位于 engine_for_vqhbr.train_one_epoch：
        # encode -> quantize -> decode -> 重建损失 + 量化损失。
        train_stats = train_one_epoch(
            model,
            data_loader_train_list,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=(epoch - 1) * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            args=args,
        )
        if args.output_dir:
            if epoch % args.save_ckpt_freq == 0:
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    save_ckpt_freq=args.save_ckpt_freq,
                )

        if data_loader_val_list is not None:
            if epoch % args.val_freq == 0:
                # 验证阶段额外可以统计 codebook 使用率，并可选做可视化。
                if polarity_pairs:
                    model_without_ddp.set_polarity_force_assignment(False)
                test_stats = evaluate(
                    data_loader_val_list,
                    model,
                    device,
                    log_writer,
                    epoch,
                    args=args,
                )
                print(
                    f"Validation loss of the network on the {sum([len(dataset) for dataset in dataset_val_list])} test ECG: {test_stats['loss']:.4f}"
                )

                if log_writer is not None:
                    log_writer.update(**test_stats, head="val/loss")

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_learnable_parameters,
                }

        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_learnable_parameters,
            }

        if args.output_dir and utils.is_main_process():
            if epoch % args.val_freq == 0:
                if log_writer is not None:
                    log_writer.flush()
                with open(
                    os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

        if (
            args.restart_unused_codes
            and epoch < args.epochs
            and epoch >= args.restart_unused_warmup_epochs
            and epoch % args.restart_unused_freq == 0
        ):
            max_codes = args.restart_unused_max_codes if args.restart_unused_max_codes > 0 else None
            restarted = model_without_ddp.quantize.restart_unused_codes(
                min_cluster_size=args.restart_unused_min_cluster_size,
                max_codes=max_codes,
            )
            print(f"Restarted unused codebook entries after epoch {epoch}: {restarted}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
