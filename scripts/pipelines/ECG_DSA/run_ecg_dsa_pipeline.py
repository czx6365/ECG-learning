import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from heartlang.project_paths import (
    ECG_DSA_QRS_BASE_DIR,
    ECG_DSA_RAW_SOURCE_DIR,
    ECG_DSA_WAVEFORM_DIR,
)

# 项目根目录
ROOT = Path(__file__).resolve().parents[3]

# 默认训练标签列
DEFAULT_LABEL_COL = "解剖学狭窄"
DEFAULT_TASK_TYPE = "regression"

# 常用标签名到英文 slug 的映射，便于生成 run_name
LABEL_SLUGS = {
    "解剖学狭窄": "anatomical_stenosis",
    "心肌缺血": "myocardial_ischemia",
    "心肌桥患者": "myocardial_bridge",
    "llm_has_ge90": "llm_ge90",
    "是否接受了PCI治疗": "pci_treatment",
    "是否接受了PCI治疗.1": "pci_treatment_numeric",
}


# ---------- 工具函数 ----------
def run_step(cmd, env=None):
    # 打印将要执行的命令，方便调试
    print(f"\n>>> {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def build_env():
    # 构造子进程环境变量，确保能正确找到项目代码与依赖
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")

    # Windows 下常见 site-packages 路径，加入 PYTHONPATH
    site_packages = Path(sys.executable).resolve().parent / "Lib" / "site-packages"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(site_packages), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return env


def slugify_label(label_col: str) -> str:
    # 先查预定义映射
    if label_col in LABEL_SLUGS:
        return LABEL_SLUGS[label_col]

    # 否则自动转成安全文件名风格
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label_col).strip("_").lower()
    return slug or "label"


def require_exists(path: Path, desc: str):
    # 检查关键输入文件是否存在
    if not path.exists():
        raise FileNotFoundError(f"{desc} does not exist: {path}")


# ---------- 参数解析 ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and fine-tune HeartLang on the ECG-DSA dataset."
    )

    # 数据相关参数
    parser.add_argument(
        "--zip_path",
        type=Path,
        default=ECG_DSA_RAW_SOURCE_DIR / "data.zip",
        help="ZIP file containing ECG CSV waveforms.",
    )
    parser.add_argument(
        "--label_xlsx",
        type=Path,
        default=ECG_DSA_RAW_SOURCE_DIR / "ECG-DSA-target.xlsx",
        help="Excel file containing labels.",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default=DEFAULT_LABEL_COL,
        help="Label column used for training.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=DEFAULT_TASK_TYPE,
        choices=["classification", "regression"],
        help="Run ECG-DSA fine-tuning as classification or regression.",
    )
    parser.add_argument("--max_records", type=int, default=None)

    # checkpoint 与训练超参数
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/heartlang_base/checkpoint-200.pth"),
        help="HeartLang pretrained checkpoint used for fine-tuning.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--split_ratio", type=float, default=1.0)
    parser.add_argument("--sampling_method", type=str, default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    # 数据集划分策略
    parser.add_argument(
        "--dataset_split_strategy",
        type=str,
        default="patient_stratified",
        choices=["patient_stratified", "group_kfold"],
    )
    parser.add_argument("--dataset_train_ratio", type=float, default=0.7)
    parser.add_argument("--dataset_val_ratio", type=float, default=0.15)
    parser.add_argument("--dataset_test_ratio", type=float, default=0.15)

    # 运行模式
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--run_eval", action="store_true")

    return parser.parse_args()


# ---------- 数据准备 ----------
def prepare_dataset(args, env):
    # 1) 从原始 ZIP + 标签表生成 waveform 数据集，并完成 train/val/test 划分
    cmd = [
        sys.executable,
        "scripts/data_prep/ECG_DSA/make_ecg_dsa_dataset.py",
        "--zip_path", str(args.zip_path),
        "--label_xlsx", str(args.label_xlsx),
        "--output_dir", str(ECG_DSA_WAVEFORM_DIR),
        "--label_col", args.label_col,
        "--task_type", args.task_type,
        "--split_strategy", args.dataset_split_strategy,
        "--train_ratio", str(args.dataset_train_ratio),
        "--val_ratio", str(args.dataset_val_ratio),
        "--test_ratio", str(args.dataset_test_ratio),
        "--seed", str(args.seed),
    ]
    if args.max_records is not None:
        cmd += ["--max_records", str(args.max_records)]
    run_step(cmd, env)

    # 2) 重新生成 QRS 数据目录，避免旧结果残留
    qrs_root = ROOT / ECG_DSA_QRS_BASE_DIR
    if qrs_root.exists():
        shutil.rmtree(qrs_root)

    # 3) 对 waveform 数据做 QRS tokenizer 预处理
    run_step(
        [sys.executable, "heartlang/preprocessing/qrs_tokenizer.py", "--dataset_name", "ECG_DSA"],
        env,
    )

    # 4) 把标签文件复制到 QRS 数据目录，供下游 finetune 使用
    run_step(
        [
            sys.executable,
            "scripts/data_prep/ECG_DSA/copy_ecg_dsa_labels.py",
            "--source_dir", str(ECG_DSA_WAVEFORM_DIR),
            "--target_dir", str(ECG_DSA_QRS_BASE_DIR),
        ],
        env,
    )


# ---------- 训练 / 评估 ----------
def build_run_info(args):
    # 生成实验名、模型输出目录、日志目录
    label_slug = slugify_label(args.label_col)
    run_name = (
        f"finetune_ecg_dsa_{label_slug}_{args.task_type}_base_linear_"
        f"{args.split_ratio}_{args.sampling_method}"
    )
    output_dir = Path("checkpoints/finetune/ecg_dsa") / run_name
    log_dir = Path("log/finetune") / run_name
    return run_name, output_dir, log_dir


def train_model(args, env, output_dir, log_dir):
    # 线性探测 / 线性微调：冻结 backbone，只训练分类头
    cmd = [
        sys.executable,
        "heartlang/cli/run_class_finetuning.py",
        "--dataset_dir", str(ECG_DSA_QRS_BASE_DIR),
        "--output_dir", str(output_dir),
        "--log_dir", str(log_dir),
        "--model", "HeartLang_finetune_base",
        "--finetune", str(args.checkpoint),
        "--trainable", "linear",
        "--split_ratio", str(args.split_ratio),
        "--sampling_method", args.sampling_method,
        "--weight_decay", "0.05",
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--update_freq", "1",
        "--warmup_epochs", str(min(10, args.epochs)),
        "--epochs", str(args.epochs),
        "--layer_decay", "0.9",
        "--save_ckpt_freq", "10",
        "--seed", str(args.seed),
        "--task_type", args.task_type,
        "--world_size", "1",
        "--device", args.device,
    ]
    if args.task_type == "regression":
        cmd += ["--regression_dim", "1"]
    else:
        cmd += ["--is_binary", "--nb_classes", "1"]
    run_step(cmd, env)


def eval_model(args, env, output_dir, run_name):
    # 在 test / eval 模式下加载输出目录中的模型进行评估
    cmd = [
        sys.executable,
        "heartlang/cli/run_class_finetuning.py",
        "--dataset_dir", str(ECG_DSA_QRS_BASE_DIR),
        "--output_dir", str(output_dir),
        "--log_dir", str(Path("log/finetune_test") / run_name),
        "--model", "HeartLang_finetune_base",
        "--eval",
        "--trainable", "linear",
        "--split_ratio", str(args.split_ratio),
        "--sampling_method", args.sampling_method,
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--task_type", args.task_type,
        "--world_size", "1",
        "--device", args.device,
    ]
    if args.task_type == "regression":
        cmd += ["--regression_dim", "1"]
    else:
        cmd += ["--is_binary", "--nb_classes", "1"]
    run_step(cmd, env)


# ---------- 主流程 ----------
def main():
    args = parse_args()

    # 统一转成绝对路径，避免 cwd 变化导致路径解析错误
    args.zip_path = args.zip_path.resolve()
    args.label_xlsx = args.label_xlsx.resolve()
    args.checkpoint = args.checkpoint.resolve()

    # 检查必要输入
    require_exists(args.zip_path, "ZIP file")
    require_exists(args.label_xlsx, "Label Excel")
    require_exists(args.checkpoint, "Checkpoint")

    # 准备运行环境
    env = build_env()

    # 先做 ECG-DSA 数据准备与 QRS 预处理
    prepare_dataset(args, env)
    if args.prepare_only:
        return

    # 再执行下游训练
    run_name, output_dir, log_dir = build_run_info(args)
    train_model(args, env, output_dir, log_dir)

    # 如果指定了 --run_eval，则继续做评估
    if args.run_eval:
        eval_model(args, env, output_dir, run_name)


if __name__ == "__main__":
    main()
