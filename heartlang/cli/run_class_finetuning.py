# --------------------------------------------------------
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model
# By Jiarui Jin and Haoyu Wang
# Based on BEiT-v2, timm, DeiT, DINO and LaBraM code bases
# ---------------------------------------------------------

"""HeartLang 微调入口：支持端到端 MLP 与基于特征的 sklearn 分类器。"""

import argparse
import datetime
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from timm.models import create_model
from timm.utils import ModelEma
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heartlang.engines.engine_for_finetuning import (
    build_regression_criterion,
    evaluate,
    train_one_epoch,
)
from heartlang.project_paths import RESULTS_METRICS_DIR
from heartlang.utils.QRSDataset import prepare_finetune_dataset
from heartlang.utils.optim_factory import LayerDecayValueAssigner, create_optimizer, get_parameter_groups
from heartlang.utils.utils import (
    NativeScalerWithGradNormCount as NativeScaler,
    freeze_except_prefix,
    freeze_specific_layers,
    get_trainable_layers,
    save_results_as_csv,
    set_seed,
)
import heartlang.utils.utils as utils
from heartlang.models import modeling_finetune as modeling_finetune  # noqa: F401
from heartlang.cli.feature_models_sklearn import SKLEARN_FEATURE_MODEL_TYPES, build_sklearn_feature_model

FEATURE_MODEL_TYPES = [*SKLEARN_FEATURE_MODEL_TYPES]


# =========================
# 运行模式 / 参数相关
# =========================
def resolve_runtime_mode(args):
    """根据 head_type 判断当前是端到端训练还是特征分类模式。"""
    if args.head_type == "mlp" and args.mlp_input_type == "end_to_end":
        return "end_to_end", None
    
    if args.head_type == "feature_model":
        return "feature_based", args.feature_model_type
    raise ValueError(f"Unsupported mode combination: head_type={args.head_type}, mlp_input_type={args.mlp_input_type}")


def is_feature_based_mode(args):
    return resolve_runtime_mode(args)[0] == "feature_based"


def is_regression_task(args):
    return getattr(args, "task_type", "classification") == "regression"


def get_active_feature_model_type(args):
    return resolve_runtime_mode(args)[1]


def get_args():
    """命令行参数。按功能分组但写法尽量紧凑。"""
    parser = argparse.ArgumentParser("HeartLang fine-tuning script for ECG classification", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="show this help message and exit")

    # 基础训练参数
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification", "regression"])
    parser.add_argument("--is_binary", action="store_true", default=True)
    parser.add_argument(
        "--multi_class",
        "--no_is_binary",
        action="store_false",
        dest="is_binary",
        help="Use multiclass softmax/CrossEntropyLoss instead of binary BCE.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--update_freq", type=int, default=1)
    parser.add_argument("--save_ckpt_freq", type=int, default=5)
    parser.add_argument("--robust_test", type=str, default=None)
    parser.add_argument("--model", type=str, default="HeartLang_finetune_base", metavar="MODEL")

    # 头部 / 运行模式
    parser.add_argument("--head_type", type=str, default="mlp", choices=["mlp", "decision_tree", "feature_model"])
    parser.add_argument("--mlp_input_type", type=str, default="end_to_end", choices=["end_to_end"])
    parser.add_argument("--task_head_type", type=str, default="linear", choices=["linear", "mlp2"])
    parser.add_argument("--task_head_hidden_dim", type=int, default=256)
    parser.add_argument("--task_head_dropout", type=float, default=0.1)
    parser.add_argument("--share_aux_task_head", action="store_true", default=False)
    parser.add_argument("--pooling_type", type=str, default="cls", choices=["cls", "mean", "cls_mean"])
    parser.add_argument("--regression_loss_type", type=str, default="mse", choices=["mse", "smooth_l1", "huber"])
    parser.add_argument("--regression_loss_beta", type=float, default=1.0)
    parser.add_argument("--aux_regression_metadata_dir", type=str, default="")
    parser.add_argument("--aux_regression_label_cols", nargs="*", default=[])
    parser.add_argument("--aux_regression_loss_weight", type=float, default=1.0)
    parser.add_argument("--aux_regression_target_scale", type=float, default=100.0)
    parser.add_argument("--feature_model_type", type=str, default="decision_tree", choices=FEATURE_MODEL_TYPES)
    parser.add_argument("--save_features", action="store_true", default=False)
    parser.add_argument("--load_features_if_exist", action="store_true", default=False)
    parser.add_argument("--feature_dir", type=str, default="")
    parser.add_argument("--force_reextract_features", action="store_true", default=False)

    # 模型超参数
    parser.add_argument("--layer_scale_init_value", type=float, default=0.1)
    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT")
    parser.add_argument("--disable_eval_during_finetuning", action="store_true", default=False)

    # Decision Tree
    parser.add_argument("--dt_max_depth", type=int, default=None)
    parser.add_argument("--dt_random_state", type=int, default=0)
    parser.add_argument("--dt_criterion", type=str, default="gini", choices=["gini", "entropy", "log_loss"])

    # Random Forest
    parser.add_argument("--rf_n_estimators", type=int, default=300)
    parser.add_argument("--rf_max_depth", type=int, default=None)
    parser.add_argument("--rf_random_state", type=int, default=0)
    parser.add_argument("--rf_n_jobs", type=int, default=-1)

    # SVM
    parser.add_argument("--svm_kernel", type=str, default="rbf")
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--svm_gamma", type=str, default="scale")
    parser.add_argument("--svm_random_state", type=int, default=0)

    # Logistic Regression
    parser.add_argument("--lr_max_iter", type=int, default=1000)
    parser.add_argument("--lr_c", type=float, default=1.0)
    parser.add_argument("--lr_solver", type=str, default="lbfgs")
    parser.add_argument("--lr_random_state", type=int, default=0)

    # XGBoost
    parser.add_argument("--xgb_n_estimators", type=int, default=300)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_random_state", type=int, default=0)
    parser.add_argument("--xgb_n_jobs", type=int, default=-1)

    # LightGBM
    parser.add_argument("--lgbm_n_estimators", type=int, default=300)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.05)
    parser.add_argument("--lgbm_num_leaves", type=int, default=31)
    parser.add_argument("--lgbm_max_depth", type=int, default=-1)
    parser.add_argument("--lgbm_random_state", type=int, default=0)
    parser.add_argument("--lgbm_n_jobs", type=int, default=-1)

    # EMA / 优化器
    parser.add_argument("--model_ema", action="store_true", default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999)
    parser.add_argument("--model_ema_force_cpu", action="store_true", default=False)
    parser.add_argument("--opt", type=str, default="adamw", metavar="OPTIMIZER")
    parser.add_argument("--opt_eps", type=float, default=1e-8, metavar="EPSILON")
    parser.add_argument("--opt_betas", type=float, nargs="+", default=None, metavar="BETA")
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--weight_decay_end", type=float, default=None)
    parser.add_argument("--lr", type=float, default=5e-4, metavar="LR")
    parser.add_argument("--layer_decay", type=float, default=0.9)
    parser.add_argument("--warmup_lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--min_lr", type=float, default=1e-5, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=5, metavar="N")
    parser.add_argument("--warmup_steps", type=int, default=-1, metavar="N")

    # 数据增强 / 正则
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT")
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--resplit", action="store_true", default=False)

    # 微调设置
    parser.add_argument("--finetune", default=False)
    parser.add_argument("--trainable", default="all")
    parser.add_argument(
        "--trainable_last_n_layers",
        type=int,
        default=1,
        help="When --trainable last_n, unfreeze this many final backbone transformer layers.",
    )
    parser.add_argument("--model_key", type=str, default="model|module")
    parser.add_argument("--model_prefix", type=str, default="")
    parser.add_argument("--model_filter_name", type=str, default="gzp")
    parser.add_argument("--init_scale", type=float, default=0.001)
    parser.add_argument("--use_mean_pooling", action="store_true")
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")

    # 数据 / 路径
    parser.add_argument("--sampling_method", default="random")
    parser.add_argument("--split_ratio", type=float, default=1.0)
    parser.add_argument("--dataset_dir", default="")
    parser.add_argument("--nb_classes", "--num_classes", dest="nb_classes", type=int, default=0)
    parser.add_argument("--regression_dim", type=int, default=1)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--log_dir", default=None)

    # 设备 / 恢复 / 分布式
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    parser.set_defaults(save_ckpt=True)
    parser.add_argument("--start_epoch", type=int, default=1, metavar="N")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--disable_amp", action="store_true", default=False)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--enable_deepspeed", action="store_true", default=False)

    known_args, _ = parser.parse_known_args()
    ds_init = None
    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig  # noqa: F401
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except Exception:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    return parser.parse_args(), ds_init


# =========================
# 数据 / 模型构建
# =========================
def get_models(args):
    """创建模型。"""
    output_dim = args.regression_dim if is_regression_task(args) else args.nb_classes
    return create_model(
        args.model,
        pretrained=False,
        num_classes=output_dim,
        task_type=args.task_type,
        task_head_type=args.task_head_type,
        task_head_hidden_dim=args.task_head_hidden_dim,
        task_head_dropout=args.task_head_dropout,
        pooling_type=args.pooling_type,
        aux_regression_dim=len(args.aux_regression_label_cols),
        share_aux_task_head=args.share_aux_task_head,
    )


def get_dataset(args):
    """构建 train / val / test 数据集，并返回默认评估指标。"""
    train_dataset, val_dataset, test_dataset = prepare_finetune_dataset(
        args.dataset_dir,
        args.split_ratio,
        args.sampling_method,
        task_type=args.task_type,
        aux_regression_metadata_dir=args.aux_regression_metadata_dir,
        aux_regression_label_cols=args.aux_regression_label_cols,
        aux_regression_target_scale=args.aux_regression_target_scale,
    )
    if is_regression_task(args):
        return train_dataset, val_dataset, test_dataset, ["r2", "mae", "mse", "rmse"]
    return train_dataset, val_dataset, test_dataset, ["accuracy", "f1", "recall", "precision"]


def infer_regression_dim_from_dataset(dataset):
    labels = dataset.labels
    if labels.ndim <= 1:
        return 1
    return int(labels.shape[-1])


def load_finetune_checkpoint(model, args):
    """只从本地加载 checkpoint；未提供或不存在时直接报错。"""
    if not args.finetune:
        raise ValueError("必须通过 --finetune 提供本地 checkpoint 路径，禁止随机初始化。")

    finetune_path = Path(args.finetune).expanduser()
    if str(args.finetune).startswith(("http://", "https://")):
        raise ValueError(
            f"--finetune 只允许本地 checkpoint，不支持网络下载: {args.finetune}"
        )
    if not finetune_path.is_file():
        raise FileNotFoundError(f"未找到本地 checkpoint: {finetune_path}")

    checkpoint = torch.load(finetune_path, map_location="cpu", weights_only=False)
    print("checkpoint.keys():", checkpoint.keys())
    print("=" * 25 + f"Load ckpt from {finetune_path}" + "=" * 25)

    checkpoint_model = next((checkpoint[k] for k in args.model_key.split("|") if k in checkpoint), checkpoint)
    if checkpoint_model is not checkpoint:
        print(f"Load state_dict by model_key = {next(k for k in args.model_key.split('|') if k in checkpoint)}")
    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)


# =========================
# 特征缓存 / 读写
# =========================
def resolve_feature_dir(args):
    """确定特征缓存目录优先级：feature_dir > output_dir/saved_features > dataset_dir/saved_features。"""
    if args.feature_dir:
        return Path(args.feature_dir)
    if args.output_dir:
        return Path(args.output_dir) / "saved_features"
    if args.dataset_dir:
        return Path(args.dataset_dir) / "saved_features"
    return Path("saved_features")


def feature_file_paths(feature_dir, split):
    return feature_dir / f"{split}_features.npy", feature_dir / f"{split}_targets.npy"


def feature_cache_exists(feature_dir, split):
    fp, tp = feature_file_paths(feature_dir, split)
    return fp.exists() and tp.exists()


def print_feature_shapes(split, features, targets, prefix):
    print(f"{prefix} {split} features shape: {features.shape}")
    print(f"{prefix} {split} targets shape: {targets.shape}")


def save_feature_split(feature_dir, split, features, targets):
    """保存单个数据划分的特征与标签。"""
    feature_dir.mkdir(parents=True, exist_ok=True)
    fp, tp = feature_file_paths(feature_dir, split)
    np.save(fp, features)
    np.save(tp, targets)
    print_feature_shapes(split, features, targets, "Saved")


def load_feature_split(feature_dir, split):
    """加载单个数据划分的缓存特征。"""
    fp, tp = feature_file_paths(feature_dir, split)
    features, targets = np.load(fp, allow_pickle=False), np.load(tp, allow_pickle=False)
    print_feature_shapes(split, features, targets, "Loaded")
    return {"features": features, "targets": targets}


# =========================
# 特征提取 / 任务推断
# =========================
@torch.no_grad()
def extract_features(model, data_loader, device, header="Extract features:"):
    """从 backbone 提取特征：
    这里提取的是模型内部的 CLS 特征（或聚合后的全局表征），
    不走最终分类头，目的是把 ECG 样本转成固定维度特征向量，
    供后续 decision tree / svm / xgboost 等传统模型使用。
    """
    if data_loader is None:
        return None, None

    # 若模型被 DDP 包装过，utils.get_model(model) 会取出真正的原始模型
    feature_model = utils.get_model(model)
    feature_model.eval()  # 只做前向提特征，不训练参数
    features, targets = [], []
    metric_logger = utils.MetricLogger(delimiter="  ")

    for batch in metric_logger.log_every(data_loader, 10, header):
        # batch 中除了 ECG 信号本身，还有标签、通道位置矩阵、时间位置矩阵
        ECG = batch[0].float().to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        in_chan_matrix = batch[2].to(device, non_blocking=True)
        in_time_matrix = batch[3].to(device, non_blocking=True)

        # 有 CUDA 时用 AMP 混合精度加速提特征；CPU 环境则什么都不做
        amp_context = torch.amp.autocast("cuda") if torch.cuda.is_available() else nullcontext()
        with amp_context:
            # extract_features 是 backbone 暴露出的“取中间表示”接口，不是最终分类输出
            cls_features = feature_model.extract_features(
                ECG, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix
            )

        # 先把每个 batch 的结果存起来，最后再拼接成完整矩阵
        features.append(cls_features.float().cpu())
        targets.append(target.cpu())

    # 按样本维拼接，得到 [N, D] 的特征矩阵和 [N] / [N, C] 的标签矩阵
    features, targets = torch.cat(features, dim=0), torch.cat(targets, dim=0)

    # 分布式场景下，每张卡只拿到局部数据，所以要把各卡结果 gather 回来再统一评估/保存
    if utils.is_dist_avail_and_initialized() and device.type == "cuda":
        features = utils.gather_tensor(features.to(device, non_blocking=True))
        targets = utils.gather_tensor(targets.to(device, non_blocking=True))

    return features.cpu().numpy(), targets.cpu().numpy()


def prepare_feature_targets(targets, is_binary):
    """将标签整理成 sklearn 更容易消费的格式。"""
    targets = np.asarray(targets)
    if is_binary:
        return targets.reshape(-1).astype(np.int64) if targets.ndim > 1 and targets.shape[1] == 1 else targets.astype(np.int64)
    if targets.ndim > 1:
        return targets.reshape(-1) if targets.shape[1] == 1 else np.argmax(targets, axis=1)
    return targets


def should_persist_features(args):
    return args.save_features or is_feature_based_mode(args)


def prepare_feature_splits(args, model, device, feature_loaders):
    """准备 train/val/test 三个划分的特征：
    - 若磁盘已有缓存且允许复用：直接加载；
    - 否则从 backbone 重新提取；
    - 必要时把提取结果保存到磁盘，避免下次重复跑神经网络前向。
    """
    feature_dir = resolve_feature_dir(args)
    required_splits = [split for split, loader in feature_loaders.items() if loader is not None]

    # 满足以下条件时直接走缓存：
    # 1) 用户允许 load_features_if_exist
    # 2) 没要求强制重提取
    # 3) train/val/test 对应缓存文件都存在
    use_cache = (
        args.load_features_if_exist
        and not args.force_reextract_features
        and all(feature_cache_exists(feature_dir, split) for split in required_splits)
    )

    if use_cache:
        # 分布式下只让主进程真正读缓存，其它 worker 直接退出该流程，避免重复操作
        if utils.is_dist_avail_and_initialized() and not utils.is_main_process():
            print("Using cached features on rank 0. Worker rank exits feature pipeline.")
            return None, feature_dir
        return {split: load_feature_split(feature_dir, split) for split in required_splits}, feature_dir

    # 没缓存可用：逐个 split 提特征
    feature_splits = {}
    for split, loader in feature_loaders.items():
        if loader is None:
            continue
        features, targets = extract_features(model, loader, device, header=f"{split.capitalize()} features:")
        feature_splits[split] = {"features": features, "targets": targets}

        if utils.is_main_process():
            print_feature_shapes(split, features, targets, "Extracted")

    # 若当前需要持久化特征（例如 feature-based 模式），就保存成 .npy，后续可直接复用
    if utils.is_main_process() and should_persist_features(args):
        for split, split_data in feature_splits.items():
            save_feature_split(feature_dir, split, split_data["features"], split_data["targets"])

    # 分布式下非主进程完成提取后直接退出，主进程继续后续训练/评估
    if utils.is_dist_avail_and_initialized() and not utils.is_main_process():
        print("Feature extraction finished on worker rank. Worker rank exits feature pipeline.")
        return None, feature_dir

    return feature_splits, feature_dir


def load_cached_feature_splits_only(args):
    """仅从磁盘缓存读取特征，不再访问原始数据集。"""
    feature_dir = resolve_feature_dir(args)
    available_splits = [split for split in ("train", "val", "test") if feature_cache_exists(feature_dir, split)]
    if "train" not in available_splits:
        return None, feature_dir
    if utils.is_dist_avail_and_initialized() and not utils.is_main_process():
        print("Using cached features on rank 0. Worker rank exits feature pipeline.")
        return None, feature_dir
    return {split: load_feature_split(feature_dir, split) for split in available_splits}, feature_dir


def infer_task_config_from_targets(targets):
    """从标签形状自动推断：二分类 / 多分类，以及类别数。"""
    targets = np.asarray(targets)
    if targets.ndim == 1:
        unique_labels = np.unique(targets)
        return (True, 1) if unique_labels.size <= 2 else (False, int(unique_labels.size))

    if targets.ndim == 2:
        if targets.shape[1] == 1:
            return infer_task_config_from_targets(targets.reshape(-1))
        unique_values, row_sums = np.unique(targets), np.sum(targets, axis=1)
        is_binary_matrix = np.all(np.isin(unique_values, [0, 1]))
        if is_binary_matrix and np.all(np.isclose(row_sums, 1.0)):
            return False, int(targets.shape[1])
        if is_binary_matrix and np.all(row_sums <= 1.0):
            return False, int(targets.shape[1])
        return True, int(targets.shape[1])

    raise ValueError(f"Unsupported target shape for inference: {targets.shape}")


def apply_inferred_task_config(args, feature_splits):
    """纯缓存特征模式下，用训练标签自动回填任务配置。"""
    inferred_is_binary, inferred_nb_classes = infer_task_config_from_targets(feature_splits["train"]["targets"])
    prev_binary, prev_classes = args.is_binary, args.nb_classes
    args.is_binary, args.nb_classes = inferred_is_binary, inferred_nb_classes

    print(f"Inferred task config from cached train targets: is_binary={args.is_binary}, nb_classes={args.nb_classes}")
    if prev_binary != args.is_binary or prev_classes not in (0, args.nb_classes):
        print(
            "Overriding CLI task config in pure cached-feature mode: "
            f"is_binary {prev_binary} -> {args.is_binary}, nb_classes {prev_classes} -> {args.nb_classes}"
        )


# =========================
# 特征模型训练 / 评估
# =========================
def create_feature_model(args, input_dim, device, log_writer=None):
    """构建 sklearn 风格分类器；实际 input_dim/device 在工厂函数里不一定用到。"""
    del input_dim, device, log_writer
    return build_sklearn_feature_model(args, get_active_feature_model_type(args), prepare_feature_targets)


def evaluate_feature_model(feature_model, features, targets, args, header):
    """评估特征分类器，并计算 loss、ROC AUC 等指标。"""
    if features is None or targets is None:
        return None

    metric_targets = np.asarray(targets)
    prepared_targets = prepare_feature_targets(metric_targets, args.is_binary)
    predictions = feature_model.predict(features)
    probabilities = feature_model.predict_proba(features, metric_targets)
    eval_targets = metric_targets if args.is_binary else prepared_targets
    eval_outputs = probabilities if args.is_binary else predictions

    stats = utils.analyze_ecg_classification(eval_outputs, eval_targets, args.is_binary, threshold=0.5)
    loss_value = feature_model.compute_loss(features, targets)
    if loss_value is not None:
        stats["loss"] = loss_value

    try:
        roc_targets = metric_targets if args.is_binary else np.eye(args.nb_classes)[prepared_targets]
        stats["roc_auc"] = utils.compute_roc_auc(probabilities, roc_targets)
    except Exception as exc:
        print(f"{header} skip ROC AUC calculation: {exc}")
        stats["roc_auc"] = float("nan")
    return stats


def save_feature_model_metrics(metrics_by_split, args, feature_dir, feature_model_type):
    """保存特征分类器结果到 csv / json / log.txt。"""
    if utils.is_main_process() and "test" in metrics_by_split and "roc_auc" in metrics_by_split["test"]:
        save_results_as_csv(metrics_by_split["test"]["roc_auc"], RESULTS_METRICS_DIR, feature_model_type, args)
    if not args.output_dir or not utils.is_main_process():
        return

    payload = {
        "head_type": args.head_type,
        "runtime_mode": "feature_based",
        "mlp_input_type": args.mlp_input_type,
        "feature_model_type": feature_model_type,
        "feature_dir": str(feature_dir),
        "metrics": metrics_by_split,
    }
    metrics_path = Path(args.output_dir) / f"{feature_model_type}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(Path(args.output_dir) / "log.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def run_feature_model_pipeline(args, feature_splits, feature_dir, device, log_writer=None):
    """完整的 feature-based 流程：训练、验证、测试、保存。"""
    train_data, val_data, test_data = feature_splits["train"], feature_splits.get("val"), feature_splits.get("test")
    input_dim, feature_model_type = train_data["features"].shape[1], get_active_feature_model_type(args)

    print(f"Feature model type: {feature_model_type}")
    print(f"Feature input dim: {input_dim}")

    feature_model = create_feature_model(args, input_dim, device, log_writer=log_writer)
    feature_model.fit(train_data, val_data=val_data)
    metrics_by_split = {}

    if val_data is not None:
        metrics_by_split["val"] = evaluate_feature_model(feature_model, val_data["features"], val_data["targets"], args, "Val")
        print(f"{feature_model_type} val metrics:", metrics_by_split["val"])
    if test_data is not None:
        metrics_by_split["test"] = evaluate_feature_model(feature_model, test_data["features"], test_data["targets"], args, "Test")
        print(f"{feature_model_type} test metrics:", metrics_by_split["test"])

    if args.output_dir and utils.is_main_process():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_model.save(output_dir)

    save_feature_model_metrics(metrics_by_split, args, feature_dir, feature_model_type)
    return metrics_by_split


# =========================
# Sampler / DataLoader
# =========================
def build_eval_sampler(dataset, args, num_tasks, global_rank):
    """验证 / 测试采样器：dist_eval 时分布式，否则顺序采样。"""
    if dataset is None:
        return None
    if args.dist_eval:
        if len(dataset) % num_tasks != 0:
            print("Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.")
        return DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    return SequentialSampler(dataset)


def build_feature_sampler(dataset, args, num_tasks, global_rank):
    """特征提取阶段的采样器：分布式下不打乱，保证提取稳定。"""
    if dataset is None:
        return None
    return DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) if args.distributed else SequentialSampler(dataset)


def build_dataset_loader(dataset, sampler, batch_size, args, drop_last):
    """统一 DataLoader 构建函数。"""
    if dataset is None:
        return None
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=drop_last)


# =========================
# 主流程
# =========================
def main(args, ds_init):
    """主入口：
    整体流程分两大类：
    1) feature-based：数据 -> backbone提特征 -> sklearn分类器；
    2) end-to-end：数据 -> 神经网络前向 -> 反向传播更新参数。
    """
    # 初始化分布式环境；单卡时这里也会做兼容处理
    utils.init_distributed_mode(args)
    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)
    device = torch.device(args.device)
    set_seed(args.seed)

    if is_regression_task(args) and is_feature_based_mode(args):
        raise ValueError("Regression task is currently supported only in end-to-end MLP fine-tuning mode.")
    if args.aux_regression_label_cols and is_feature_based_mode(args):
        raise ValueError("Auxiliary regression labels are currently supported only in end-to-end MLP fine-tuning mode.")

    # ========= 路径1：只有缓存特征，没有原始数据集 =========
    # 适用于：以前已经把 train/val/test 特征提取好并存盘了，
    # 这次只想直接训练 decision tree / svm 等传统模型。
    if is_feature_based_mode(args) and not args.dataset_dir:
        if not args.load_features_if_exist:
            raise ValueError(
                "Feature-based mode without --dataset_dir requires --load_features_if_exist and cached features under --feature_dir."
            )

        feature_splits, feature_dir = load_cached_feature_splits_only(args)
        if feature_splits is None:
            raise FileNotFoundError(
                f"No cached train features found in {feature_dir}. Provide --dataset_dir to extract features first, or point --feature_dir to an existing cache."
            )

        # 纯缓存模式下，原始数据集没有参与，因此需要从缓存标签自动推断任务类型
        # 例如：二分类还是多分类，类别数是多少
        apply_inferred_task_config(args, feature_splits)

        log_writer = utils.TensorboardLogger(log_dir=args.log_dir) if utils.get_rank() == 0 and args.log_dir is not None else None
        run_feature_model_pipeline(args, feature_splits, feature_dir, device, log_writer=log_writer)
        return

    # ========= 路径2：正常读取数据集 =========
    dataset_train, dataset_val, dataset_test, metrics = get_dataset(args)

    if is_regression_task(args):
        inferred_regression_dim = infer_regression_dim_from_dataset(dataset_train)
        if args.regression_dim != inferred_regression_dim:
            print(
                f"Adjust regression_dim from {args.regression_dim} to {inferred_regression_dim} "
                "to match dataset labels."
            )
            args.regression_dim = inferred_regression_dim
    else:
        if dataset_val is not None:
            print("val dataset has invalid columns: ", utils.check_dataset_labels(dataset_val))
        if dataset_test is not None:
            print("test dataset has invalid columns: ", utils.check_dataset_labels(dataset_test))
        if args.aux_regression_label_cols:
            print("Auxiliary regression targets:", args.aux_regression_label_cols)
            print("Auxiliary regression target scale:", args.aux_regression_target_scale)

    # 若禁用微调过程中的评估，则直接把 val/test 置空，只做训练
    if args.disable_eval_during_finetuning:
        dataset_val = dataset_test = None

    num_tasks, global_rank = utils.get_world_size(), utils.get_rank()

    # 训练集 sampler：训练时通常需要 shuffle，避免每轮顺序固定
    sampler_train = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    # 验证/测试 sampler：评估时不应打乱，保证结果稳定可复现
    sampler_val = build_eval_sampler(dataset_val, args, num_tasks, global_rank)
    sampler_test = build_eval_sampler(dataset_test, args, num_tasks, global_rank)

    # 特征提取阶段也单独准备 sampler：这里通常不 shuffle，保证缓存结果稳定一致
    feature_sampler_train = build_feature_sampler(dataset_train, args, num_tasks, global_rank)
    feature_sampler_val = build_feature_sampler(dataset_val, args, num_tasks, global_rank)
    feature_sampler_test = build_feature_sampler(dataset_test, args, num_tasks, global_rank)

    log_writer = utils.TensorboardLogger(log_dir=args.log_dir) if global_rank == 0 and args.log_dir is not None else None

    # 自动调整 batch_size，避免“总 batch 大于训练集大小”这种不合理情况
    # total_batch_size = 单卡 batch * 梯度累积步数 * GPU数
    num_gpus = utils.get_world_size()
    total_batch_size = args.batch_size * args.update_freq * num_gpus
    while total_batch_size > len(dataset_train) and args.batch_size > 1:
        args.batch_size //= 2
        total_batch_size = args.batch_size * args.update_freq * num_gpus
    args.batch_size = max(args.batch_size, 1)

    # 构建 DataLoader：训练集 drop_last=True，避免最后一个不完整 batch 影响训练稳定性
    data_loader_train = build_dataset_loader(dataset_train, sampler_train, args.batch_size, args, True)
    data_loader_val = build_dataset_loader(dataset_val, sampler_val, int(1.5 * args.batch_size), args, False)
    data_loader_test = build_dataset_loader(dataset_test, sampler_test, int(1.5 * args.batch_size), args, False)

    # 若走 feature-based，还需要额外一套“提特征专用”的 loader
    feature_loaders = {
        "train": build_dataset_loader(dataset_train, feature_sampler_train, int(1.5 * args.batch_size), args, False),
        "val": build_dataset_loader(dataset_val, feature_sampler_val, int(1.5 * args.batch_size), args, False),
        "test": build_dataset_loader(dataset_test, feature_sampler_test, int(1.5 * args.batch_size), args, False),
    }

    # 创建模型并加载预训练权重
    model = get_models(args)
    load_finetune_checkpoint(model, args)
    model.to(device)

    # ========= 路径2A：feature-based =========
    # 神经网络只负责把 ECG 转成特征，真正分类交给传统模型
    if is_feature_based_mode(args):
        feature_splits, feature_dir = prepare_feature_splits(args, model, device, feature_loaders)
        if feature_splits is None:  # 分布式下非主进程会直接返回
            return
        run_feature_model_pipeline(args, feature_splits, feature_dir, device, log_writer=log_writer)
        return

    # ========= 路径2B：end-to-end =========
    # 这里才是真正的“神经网络整体微调”
    if args.trainable == "linear":
        # 只训练最后线性头，相当于冻结 backbone，只让分类头适配当前任务
        trainable_prefixes = ["mlp_head"]
        if args.aux_regression_label_cols:
            trainable_prefixes.append("aux_regression_head")
        if args.share_aux_task_head:
            trainable_prefixes.append("shared_task_head")
        freeze_except_prefix(model, trainable_prefixes)
    elif args.trainable == "adapter":
        # 只训练 adapter 层和分类头，backbone 主体冻结
        freeze_specific_layers(model, ["adapter", "mlp_head"])
    elif args.trainable == "last_n":
        # 只训练任务头和最后 N 层 Transformer，让表征轻微适配当前任务。
        last_n = max(0, int(args.trainable_last_n_layers))
        depth = model.get_num_layers()
        first_trainable_layer = max(0, depth - last_n)
        trainable_prefixes = ["mlp_head"]
        if args.aux_regression_label_cols:
            trainable_prefixes.append("aux_regression_head")
        if args.share_aux_task_head:
            trainable_prefixes.append("shared_task_head")
        trainable_prefixes.extend(
            f"backbone.transformer.layers.{layer_idx}"
            for layer_idx in range(first_trainable_layer, depth)
        )
        freeze_except_prefix(model, trainable_prefixes)
        print(
            "Unfrozen final backbone transformer layers: "
            f"{list(range(first_trainable_layer, depth))}"
        )

    print("Trainable layers:", get_trainable_layers(model))


    # 可选 EMA
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(model, decay=args.model_ema_decay, device="cpu" if args.model_ema_force_cpu else "", resume="")
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    print("LR = %.8f" % args.lr)
    print("Batch size on one GPU = %d, %d GPUs total batch size = %d" % (args.batch_size, args.world_size, total_batch_size))
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    # layer-wise lr decay越靠近输入层，学习率越小；越靠近输出层，学习率越大
    num_layers = model_without_ddp.get_num_layers()
    assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))) if args.layer_decay < 1.0 else None
    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()

    # 创建优化器 / DDP / DeepSpeed
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None,
        )
        model, optimizer, _, _ = ds_init(args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed)
        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )
        loss_scaler = NativeScaler()

    # 学习率 / weight decay 调度器
    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
    args.weight_decay_end = args.weight_decay if args.weight_decay_end is None else args.weight_decay_end
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # 损失函数：回归用 MSE，分类保持原逻辑
    if is_regression_task(args):
        criterion = build_regression_criterion(
            args.regression_loss_type,
            args.regression_loss_beta,
        )
    else:
        criterion = torch.nn.BCEWithLogitsLoss() if args.is_binary else torch.nn.CrossEntropyLoss()
    aux_regression_criterion = (
        build_regression_criterion(
            args.regression_loss_type,
            args.regression_loss_beta,
        )
        if args.aux_regression_label_cols
        else None
    )
    print("criterion = %s" % str(criterion))
    if aux_regression_criterion is not None:
        print("aux_regression_criterion = %s" % str(aux_regression_criterion))

    # 自动恢复训练状态
    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    # 纯评估模式
    if args.eval:
        if data_loader_test is None:
            print("No test dataloader available for evaluation.")
            return
        print("Test batch size = %d" % data_loader_test.batch_size)
        print("Number of test examples = %d" % len(dataset_test))
        print("Number of testing steps in one epoch = %d" % len(data_loader_test))

        test_stats = evaluate(
            data_loader_test,
            model,
            device,
            header="Test:",
            metrics=metrics,
            is_binary=args.is_binary,
            dataset_dir=args.dataset_dir,
            task_type=args.task_type,
            regression_loss_type=args.regression_loss_type,
            regression_loss_beta=args.regression_loss_beta,
            use_amp=not args.disable_amp,
            aux_regression_loss_weight=args.aux_regression_loss_weight,
        )
        print(f"====== {args.log_dir} ======")
        if is_regression_task(args):
            print(f"====== R2: {test_stats['r2']} ======")
        else:
            print(f"====== AUC: {test_stats['roc_auc']} ======")
        if utils.is_main_process():
            os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
            if is_regression_task(args):
                save_results_as_csv(test_stats["r2"], RESULTS_METRICS_DIR, "finetune_r2", args, metric_name="R2")
            else:
                save_results_as_csv(test_stats["roc_auc"], RESULTS_METRICS_DIR, "finetune", args)
        return

    # 正式训练
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_metric = float("-inf")
    best_metric_name = "r2" if is_regression_task(args) else "roc_auc"

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch - 1)
        if log_writer is not None:
            log_writer.set_step((epoch - 1) * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model, criterion, aux_regression_criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, log_writer=log_writer,
            start_steps=(epoch - 1) * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq, is_binary=args.is_binary, task_type=args.task_type,
            use_amp=not args.disable_amp,
            aux_regression_loss_weight=args.aux_regression_loss_weight,
        )

        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq,
            )

        if data_loader_val is not None:
            print("Eval batch size = %d" % data_loader_val.batch_size)
            print("Number of eval examples = %d" % len(dataset_val))
            print("Number of eval steps in one epoch = %d" % len(data_loader_val))

            val_stats = evaluate(
                data_loader_val,
                model,
                device,
                header="Val:",
                metrics=metrics,
                is_binary=args.is_binary,
                dataset_dir=args.dataset_dir,
                task_type=args.task_type,
                regression_loss_type=args.regression_loss_type,
                regression_loss_beta=args.regression_loss_beta,
                use_amp=not args.disable_amp,
                aux_regression_loss_weight=args.aux_regression_loss_weight,
            )
            metric_value = val_stats[best_metric_name]
            if is_regression_task(args):
                print(f"R2 of the network on the {len(dataset_val)} val ECG: {metric_value:.4f}")
            else:
                print(f"AUC of the network on the {len(dataset_val)} val ECG: {metric_value*100:.2f}%")

            if best_metric < metric_value:
                best_metric = metric_value
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema,
                    )

            if is_regression_task(args):
                print(f"Best val r2: {best_metric:.4f}")
            else:
                print(f"Max auc val: {best_metric*100:.2f}%")
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key in {"accuracy", "balanced_accuracy", "f1_weighted", "pr_auc", "roc_auc", "cohen_kappa", "loss", "r2", "mae", "mse", "rmse"}:
                        log_writer.update(**{key: value}, head="val", step=epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(Path(args.output_dir) / "log.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == "__main__":
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
