# ECG-learning · HeartLang ECG 表示学习与四分类研究

本项目基于 ICLR 2025 HeartLang，研究心拍形态表示、Masked SID 预训练与多层特征读取在 **OMI、NOMI、Clinical Control、Healthy** 四分类中的作用。当前以 **Overall R²** 为主要指标，配合逐类 R²、ROC-AUC、F1 和患者分组评估。

截至 2026-11-18，已完成固定 Test 对比的模型中，**E29 Class Query Ensemble3 的 Overall R² 最高，为 0.403234**。项目包含 E12-A、E25、E29 及后续实验代码，训练数据和部分运行产物需要按原实验目录自行准备。

## 当前模型流程

```text
ECG → QRS 心拍序列 [256, 96]
    → 冻结 E3 Morphology Encoder → 连续 z_e [256, 64]
    → 冻结 E12-A Transformer → 12 层 hidden states
        ├─ 第 6 / 9 / 12 层有效 token Mean+Max → 可学习凸组合
        │   → E11 全局与导联区域融合 → 基础四分类 logits
        └─ 第 9 层有效 token → 768→32 维投影
            → 四个类别专属 Query → 四类残差 logits
    → 三个 downstream seed 的四分类概率等权平均
```

- **E12-A** 使用连续 `z_e` 输入，通过遮挡约 50% 的有效心拍并预测冻结 Codebook 分配的 SID 进行预训练。主干为 12 层、768 维 Transformer，Codebook 为 1024×64。
- **E25** 将第 6、9、12 层各自的 1536 维 Mean+Max 表示做 Softmax 凸组合，增加 3 个可学习标量。
- **E29 Class Query** 在 E25 基础上，用四个类别专属 Query 读取第 9 层有效 token，向原分类 logits 加入受零初始化门控控制的残差。此分支增加 24,833 个参数，连同 E25 共增加 24,836 个可训练参数。
- **E29 Shared Query** 使用一个共享 Query。它在原固定 Test 上的 OMI/NOMI 条件 AUC 为 0.811807，高于 Class Query 的 0.807602。

## 原固定划分结果

下游 Train / Validation / Test 为 **6,640 / 1,344 / 1,366** 条 ECG。患者与 Healthy entity 按固定分组划分。下表两边均使用 seed 0/1/2 的概率等权平均，然后计算指标。

### Overall R² 优先的模型对比

| 模型 | 固定 Test Overall R² | Clinical-3 R² | Macro ROC-AUC | Macro F1 | OMI/NOMI 条件 AUC |
|---|---:|---:|---:|---:|---:|
| E12-A Ensemble3 | 0.332666 | 0.239439 | 0.874677 | 0.586636 | 0.777533 |
| E25 Convex Mix Ensemble3 | 0.351997 | 0.256699 | 0.879675 | 0.607273 | 0.786972 |
| E29 Shared Query Ensemble3 | 0.393321 | **0.293855** | 0.894497 | 0.622628 | **0.811807** |
| **E29 Class Query Ensemble3** | **0.403234** | 0.292575 | **0.899589** | **0.642763** | 0.807602 |

### Test 逐类对比

| 标签 | ECG 数 | E12-A R² | E29 Class R² | ΔR² | ROC-AUC，E12-A → Class | F1，E12-A → Class |
|---|---:|---:|---:|---:|---:|---:|
| OMI | 247 | 0.342437 | 0.391096 | +0.048659 | 0.8756 → 0.8928 | 0.6053 → 0.6366 |
| NOMI | 335 | 0.238383 | 0.328171 | +0.089788 | 0.8298 → 0.8699 | 0.5731 → 0.6399 |
| Clinical Control | 101 | 0.137497 | 0.158458 | +0.020961 | 0.8516 → 0.8640 | 0.3077 → 0.3886 |
| Healthy | 683 | 0.612348 | 0.735212 | +0.122864 | 0.9418 → 0.9718 | 0.8604 → 0.9060 |
| **四类等权平均** | **1,366** | **0.332666** | **0.403234** | **+0.070568** | **0.8747 → 0.8996** | **0.5866 → 0.6428** |

### Validation 逐类对比

| 标签 | ECG 数 | E12-A R² | E29 Class R² | ΔR² | ROC-AUC，E12-A → Class | F1，E12-A → Class |
|---|---:|---:|---:|---:|---:|---:|
| OMI | 270 | 0.399401 | 0.415687 | +0.016285 | 0.8925 → 0.9026 | 0.6247 → 0.6367 |
| NOMI | 291 | 0.274595 | 0.336280 | +0.061686 | 0.8498 → 0.8775 | 0.5695 → 0.6035 |
| Clinical Control | 111 | 0.262570 | 0.312807 | +0.050236 | 0.9042 → 0.9221 | 0.4747 → 0.5076 |
| Healthy | 672 | 0.613245 | 0.707550 | +0.094306 | 0.9406 → 0.9635 | 0.8658 → 0.9012 |
| **四类等权平均** | **1,344** | **0.387453** | **0.443081** | **+0.055628** | **0.8968 → 0.9164** | **0.6337 → 0.6623** |

差值按未舍入数值计算。各类 ROC-AUC 为 one-vs-rest，F1 使用四分类 argmax。Overall R² 为四个 one-hot 目标列各自的 `1 − SSE/SST` 的等权平均；Clinical-3 在同一批全部记录上取三个临床输出列的 R² 均值。OMI/NOMI 条件 AUC 仅取真实 OMI/NOMI 记录，分数为 `p(OMI)/(p(OMI)+p(NOMI))`。

历史 E12-A 的单模型 Test R² 均值为 **0.328268 ± 0.002596**，与上表三 seed 集成的 **0.332666** 属于不同汇总口径。Validation 参与过多轮开发，Test 也已用于多个固定模型的对比。以上是原固定划分上的成绩，后续泛化确认需要结合患者留出和独立数据证据。

## 代码与运行入口

```bash
git clone https://github.com/czx6365/ECG-learning.git
cd ECG-learning
pip install -r requirements.txt
```

运行研究实验需要匹配的 PyTorch/CUDA 环境、冻结 E3/E12-A checkpoint、原 split manifest 和特征缓存。现有脚本保留训练工作区的路径约定，运行前应核对脚本中的输入及输出路径。

| 内容 | 入口 |
|---|---|
| 已发布 E12-A 与匹配 E3 权重、加载说明 | [checkpoint 说明](checkpoints/pretrain/E12-A_continuous_ze_sid_mlm/README.md) |
| E12-A 加载与结构 smoke 示例 | [e12a_quickstart.py](scripts/curriculum_v1/e12a_quickstart.py) |
| E12-A 连续输入 Masked SID 预训练与原下游协议 | [run_e12_continuous_input.py](scripts/curriculum_v1/run_e12_continuous_input.py) |
| E25 多层融合 | [run_e25.py](scripts/curriculum_v1/run_e25.py) |
| E29 Class / Shared Query | [run_e29_pooling.py](scripts/curriculum_v1/run_e29_pooling.py) |
| E29-R 重复性复核、E29-S 患者分组 CV | [run_e29_r.py](scripts/curriculum_v1/run_e29_r.py)、[run_e29_s.py](scripts/curriculum_v1/run_e29_s.py) |
| 固定 E25/E29 Test 评估 | [run_locked_test.py](scripts/curriculum_v1/run_locked_test.py) |
| E30 SID Utility 的目标对照实现 | [scripts/e30_sid_utility/](scripts/e30_sid_utility/) |
| ECG_DSA 早期数据准备与 Morphology-V2 流程 | [ECG_DSA 运行指南](docs/ECG_DSA_README.md) |
| 当前代码快照说明 | [CODE_SNAPSHOT_20261118.md](CODE_SNAPSHOT_20261118.md) |

E30 SID Utility 比较 SID CE、标准化连续 `z_e` 重建及联合目标，支持 smoke 检查、断点续训和两套下游 Probe。实验结果应以完成后的报告为准；本 README 的性能表来自已完成的 E12-A/E25/E29 固定划分评估。仓库中的历史 `run_e30.py` 对应早期生理信息增量实验。

```bash
# 检查 E12-A 权重加载示例参数
python scripts/curriculum_v1/e12a_quickstart.py --help

# 在原实验数据和缓存准备好后，执行 E30 SID Utility 结构检查
PYTHONPATH=scripts/curriculum_v1 python scripts/e30_sid_utility/run_e30_sid.py smoke
```

数据与大文件的仓库收录情况见 [REPOSITORY_CONTENTS.md](REPOSITORY_CONTENTS.md)。

## 上游 HeartLang 与引用

本研究代码基于原 HeartLang 实现。论文与作者信息如下，后续保留原始 MIMIC-IV 训练流程供参考。

### [ICLR 2025] Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model

Original HeartLang implementation and ICLR 2025 paper "[Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model](https://openreview.net/pdf?id=6Hz1Ko087B)".

For the current ECG_DSA Morphology-V2 / HeartLangSID experiments, see the [complete run guide](docs/ECG_DSA_README.md). It covers data inputs, safe reproduction commands, outputs, and downstream evaluation separately from the original MIMIC-IV workflow below.

> Authors: Jiarui Jin, Haoyu Wang, Hongyan Li, Jun Li, Jiahui Pan, Shenda Hong.

## 🚀 Getting Started

🚩 **News** (Jan 2025): The pre-training checkpoint is now available on [🤗 Hugging Face](https://huggingface.co/PKUDigitalHealth/HeartLang/tree/main)!

🚩 **News** (Jan 2025): The VQ-HBR checkpoint is now available on [🤗 Hugging Face](https://huggingface.co/PKUDigitalHealth/HeartLang/tree/main)!

### Installation

To clone this repository:

```
git clone https://github.com/czx6365/ECG-learning.git
```

### Environment Set Up

Install required packages:

```
conda create -n heartlang python=3.9
conda activate heartlang
pip install -r requirements.txt
```

### Published E12-A checkpoint

The validation-selected E12-A continuous-`z_e` checkpoint, its matching frozen
E3 morphology encoder/codebook, and a data-independent loading example are
included through Git LFS. See
[`checkpoints/pretrain/E12-A_continuous_ze_sid_mlm/README.md`](checkpoints/pretrain/E12-A_continuous_ze_sid_mlm/README.md).

### Prepare Pre-training Data

You can download the electrocardiogram data required for pretraining from [MIMIC-IV Dataset](https://physionet.org/content/mimic-iv-ecg/1.0/).
To preprocess the MIMIC-IV ECG dataset, use the following command:

```bash
python scripts/data_prep/MIMIC-IV/mimic_preprocess.py --dataset_path <path_to_MIMIC_data> --output_path datasets/ecg_datasets/MIMIC-IV
```

Next, generate ECG sentences from the MIMIC-IV dataset:

```bash
python heartlang/preprocessing/qrs_tokenizer.py --dataset_name MIMIC-IV
```

### Vector-Quantized Heartbeat Reconstruction Training

The purpose of VQ-HBR is to encode all individual ECG words into collective ECG words. It is recommended to use a platform with 8 NVIDIA GeForce RTX 4090 GPUs.

You can start the VQ-HBR training using the following command:

```bash
bash scripts/train/vqhbr/MIMIC-IV.sh
```

Alternatively, you can manually execute the pre-training process with the following command:

```
export OMP_NUM_THREADS=48

# Uncomment and modify the line below to specify visible GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes=1 --nproc_per_node=8 heartlang/cli/run_vqhbr_training.py \
    --output_dir checkpoints/vqhbr/MIMIC-IV/ \
    --log_dir log/vqhbr/MIMIC-IV/ \
    --model vqhbr \
    --codebook_n_emd 8192 \
    --codebook_emd_dim 128 \
    --quantize_kmeans_init \
    --batch_size 64 \
    --opt adamw \
    --opt_betas 0.9 0.99 \
    --weight_decay 1e-4 \
    --warmup_epochs 10 \
    --epochs 100 \
    --save_ckpt_freq 10 \
    --world_size 8 \
    --val_freq 10 \
    --lr 5e-5 \
    --min_lr 1e-5
```

### Masked ECG Sentence Pre-training

In this stage, HeartLang learns generalized rhythm-level representations by masking parts of individual ECG words within an ECG sentence and predicting the corresponding collective ECG words based on the unmasked context.
You can initiate pre-training by running the following script:

```bash
bash scripts/train/pretrain/MIMIC-IV.sh
```

Alternatively, you can manually execute the pre-training process with the following command:

```
export OMP_NUM_THREADS=32

# Uncomment and modify the line below to specify visible GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes=1 --master_port 49207 --nproc_per_node=8 heartlang/cli/run_heartlang_pretraining.py \
    --output_dir checkpoints/pretrain/MIMIC-IV \
    --log_dir log/pretrain/MIMIC-IV \
    --model HeartLang \
    --tokenizer_model vqhbr \
    --tokenizer_weight checkpoints/vqhbr/MIMIC-IV/checkpoint-100.pth \
    --batch_size 64 \
    --lr 5e-4 \
    --warmup_epochs 5 \
    --clip_grad 3.0 \
    --layer_scale_init_value 0.1 \
    --opt_betas 0.9 0.98 \
    --opt_eps 1e-8 \
    --epochs 200 \
    --save_ckpt_freq 10 \
    --codebook_size 8192 \
    --codebook_dim 128 \
    --gradient_accumulation_steps 1 \
    --world_size 8
```

In addition to training from scratch, you can also fine-tune downstream tasks using the checkpoints we provide in huggingface. Simply place the downloaded checkpoints into the `checkpoints` folder.

### Fine-tune on Downstream Tasks

In our paper, downstream datasets we used are as follows:

* **PTB-XL**: We downloaded the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset which consisting four Subsets, Superclass, Subclass, Form, Rhythm.
* **CPSC2018**: We used the [CPSC2018](http://2018.icbeb.org/Challenge.html) dataset which consisting three training sets. Since the original link website appears to be no longer available for download, we downloaded the dataset from [OpenDataLab](https://opendatalab.com/OpenDataLab/The_China_Physiological_Signal_etc).
* **CSN(Chapman-Shaoxing-Ningbo)**: We downloaded the [CSN](https://physionet.org/content/ecg-arrhythmia/1.0.0/) dataset.

The data processing code for each downstream dataset is located in the `scripts/data_prep` directory. Training launchers are grouped under `scripts/train`, and dataset-specific pipelines or analysis scripts live under `scripts/pipelines` and `scripts/analysis`. Specifically, for the PTB-XL dataset, we adopt the official preprocessing [code](https://github.com/helme/ecg_ptbxl_benchmarking) to obtain the training, validation, and test files.

The following example demonstrates the process using the PTB-XL dataset:

First, run `scripts/data_prep/PTBXL/ecg_ptbxl_benchmarking/code/reproduce_results.py` to obtain all subset files. Then, run `scripts/data_prep/PTBXL/move_ptbxl_files.py` to move PTB-XL dataset's files. At this point, the folder structure will be similar to the following:

```
datasets/
└── ecg_datasets/
    └── PTBXL/
        ├── all/
            ├── train_data.npy
            └── ...
        ├── diagnostic/
        ├── form/
	├── ...
	├── superdiagnostic/
```

Then, generate ECG sentences from the PTB-XL dataset:

```bash
python heartlang/preprocessing/qrs_tokenizer.py --dataset_name PTBXL
```

Next, you need to manually copy and paste the label files into the corresponding folders under `PTBXL_QRS`. (It would be even better if we could add this feature to the QRSTokenizer 😀)

```
cp datasets/ecg_datasets/PTBXL/form/*labels* datasets/ecg_datasets/PTBXL_QRS/form
```

You can manually execute the fine-tuning training process with the following command:

```bash
torchrun --nnodes=1 --master_port=2025 --nproc_per_node=1 heartlang/cli/run_class_finetuning.py \
            --dataset_dir datasets/ecg_datasets/PTBXL_QRS/form \
            --output_dir checkpoints/finetune/ptbxl/finetune_form_base_linear_1_random/ \
            --log_dir log/finetune/finetune_form_base_linear_1_random \
            --model HeartLang_finetune_base \
            --finetune checkpoints/heartlang_base/checkpoint-200.pth \
            --trainable linear \
            --split_ratio 1 \
            --sampling_method random \
            --weight_decay 0.05 \
            --batch_size 256 \
            --lr 5e-3 \
            --update_freq 1 \
            --warmup_epochs 10 \
            --epochs 100 \
            --layer_decay 0.9 \
            --save_ckpt_freq 100 \
            --seed 0 \
            --is_binary \
            --nb_classes 19 \
            --world_size 1
```

You can also manually execute the fine-tuning testing process with the following command:

```bash
torchrun --nnodes=1 --master_port=2025 --nproc_per_node=1 heartlang/cli/run_class_finetuning.py \
            --dataset_dir datasets/ecg_datasets/PTBXL_QRS/form \
            --output_dir checkpoints/finetune/ptbxl/finetune_form_base_linear_1_random/ \
            --log_dir log/finetune_test/finetune_form_base_linear_1_random \
            --model HeartLang_finetune_base \
            --eval \
            --trainable linear \
            --split_ratio 1 \
            --sampling_method random \
            --batch_size 256 \
            --seed 0 \
            --is_binary \
            --nb_classes 19 \
            --world_size 1
```

For the CPSC2018 and CSN datasets, the fine-tuning training and testing processes are similar and will not be elaborated here.

## References

If you found our work useful in your research, please consider citing our works at:
> ```
> @inproceedings{
> jin2025reading,
> title={Reading Your Heart: Learning {ECG} Words and Sentences via Pre-training {ECG} Language Model},
> author={Jiarui Jin and Haoyu Wang and Hongyan Li and Jun Li and Jiahui Pan and Shenda Hong},
> booktitle={The Thirteenth International Conference on Learning Representations},
> year={2025},
> url={https://openreview.net/forum?id=6Hz1Ko087B},
> }
> ```
