# ECG_DSA / HeartLang 实验运行指南

本文记录当前远程项目 `/home/ubuntu/HeartLang-main` 中已实现的 ECG_DSA 主流程。原项目根 `README.md` 介绍的是论文的 MIMIC-IV、8192 x 128 codebook 流程；**不要把那套命令与本文的 1024 x 64 Morphology-V2 实验混用**。

## 0. 先看流程与约定

```text
远程已有的 9 导联、100 Hz、10 秒波形 [N, 9, 1000]
  -> QRS 切分，heartbeat sentence [N, 256, 96]
  -> clean 数据 (仅供 codebook 训练)
  -> Morphology-V2 encoder + EMA VQ, 1024 x 64
  -> 用冻结 tokenizer 给原始 V2 和四类下游数据生成 SID [N, 256]
  -> HeartLangSID: SID + lead + time + position -> MLM 预训练
  -> 冻结 backbone + Linear(768, 4): 四分类 / 四输出回归
```

**关键区别：**codebook 在 `patient_healthy_raw_v2_clean_le5pct` 上训练；后续预训练 SID 则由**未 clean 的** `patient_healthy_raw_v2` 生成。下游四类数据也单独生成 SID。SID 取值 `0..1023`，padding ID 是 `1024`；它不是第 1025 个 code。

所有命令均在远程 Ubuntu 执行。先连接：

```bash
ssh liaoheL20
cd /home/ubuntu/HeartLang-main
nvidia-smi
```

已有数据和 checkpoint 体积较大，**不要在原实验目录重跑**。下文的 `repro_...` 目录是新实验输出；**每步运行前，先在 VS Code 文件管理器中确认该命令写入的输出目录不存在。如果已经存在，停下并换一个全新目录名，不要直接重跑。** GPU 训练可能耗时数小时，命令默认前台运行，不要重复启动同一任务。

## 1. 数据与输入文件

| 数据 | 路径（相对项目根） | 内容 |
| --- | --- | --- |
| V2 原始 QRS | `datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2` | 98,773 条 ECG；train 88,896 / val 9,877 |
| V2 clean QRS | `datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct` | 与原始 V2 相同 record/split，异常 token 置零并标无效 |
| 患者三类 | `datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label3` | OMI / NOMI / 对照组 |
| 下游四类 | `datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675` | 患者记录 4,675 + 健康 ECG 4,675 |
| 下游四类 ECG 级 | `datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_ecg8303_healthy8303` | 每条患者 ECG 都保留：8,303 患者 ECG + 8,303 健康 ECG |
| V2 预训练 SID | `datasets/ecg_datasets/ECG_DSA/pretrain_sid/morphology_v2_physaware_raw_v2` | `train_sid.npy`, `val_sid.npy` |
| V2 下游 SID | `datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_patient4675_healthy4675_morphology_v2` | `train/val/test_sid.npy` 与四类标签 |
| V2 下游 ECG 级 SID | `datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_ecg8303_healthy8303_morphology_v2` | 新四类数据对应的 `train/val/test_sid.npy` |

QRS 数据每个 split 至少包括 `<split>_data.npy` (`float32`, `[N,256,96]`)、`<split>_data_in_chans.npy`、`<split>_data_in_times.npy` (`[N,256]`) 和 metadata。`0` channel/time 为 padding；下游额外有 `<split>_labels.npy` (`[N,4]` one-hot)。当前 ECG_DSA 原始 CSV 使用 9 个实际导联 I、II、III、V1–V6；不要把模型中 12 导联 embedding 容量误读成此数据真的包含 12 条实测导联。

先检查现有输入：

```bash
ls -lh datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2/train_data.npy
ls -lh datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct/val_data.npy
ls -lh datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675/test_labels.npy
.venv-heartlang/bin/python -c 'import json; p="datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675/prepare_summary.json"; print(json.dumps(json.load(open(p)), ensure_ascii=False, indent=2))'
```

### 1.1 从远程已有波形生成 QRS（可选）

本文把 `/home/ubuntu/HeartLang-main/datasets/ecg_datasets/ECG_DSA/` 下的波形数组作为**远程流程的起点**。无需上传其他原始文件，也不用重新划分 train/val/test。

| 用途 | 现有波形目录 | 数组形状与划分 |
| --- | --- | --- |
| 无标签 V2 预训练 | `pretrain_waveforms/patient_healthy_raw_v2/` | train `(88896,9,1000)`；val `(9877,9,1000)` |
| OMI/NOMI/对照组三类 | `waveforms/data_0521_label3/` | train `(5897,9,1000)`；val `(1176,9,1000)`；test `(1230,9,1000)`，已有标签和 metadata |

现有 `pretrain_qrs/patient_healthy_raw_v2/` 和 `qrs/data_0521_label3/` 已经生成好了，**正常使用时跳过本节**。只有要从上述波形重新切分 heartbeat 时，才执行以下命令；输出放在全新目录，不碰现有 QRS 数据。

**无标签 V2：波形 -> QRS**

```bash
.venv-heartlang/bin/python scripts/data_prep/ECG_DSA/tokenize_pretrain_qrs.py \
  --input_dir datasets/ecg_datasets/ECG_DSA/pretrain_waveforms/patient_healthy_raw_v2 \
  --output_dir datasets/ecg_datasets/ECG_DSA/pretrain_qrs/repro_raw_v2 \
  --splits train val --fs 100 --max_len 256 --token_len 96
cp datasets/ecg_datasets/ECG_DSA/pretrain_waveforms/patient_healthy_raw_v2/{train,val}_metadata.csv \
  datasets/ecg_datasets/ECG_DSA/pretrain_qrs/repro_raw_v2/
```

输入 `pretrain_waveforms/patient_healthy_raw_v2/{train,val}_data.npy`；输出 `pretrain_qrs/repro_raw_v2/{train,val}_data.npy`、`*_data_in_chans.npy`、`*_data_in_times.npy` 及复制的 metadata。每条 ECG 被编码为最多 256 个长度 96 的 heartbeat token。

**已有三类：波形 -> QRS，保留标签**

```bash
.venv-heartlang/bin/python scripts/data_prep/ECG_DSA/tokenize_pretrain_qrs.py \
  --input_dir datasets/ecg_datasets/ECG_DSA/waveforms/data_0521_label3 \
  --output_dir datasets/ecg_datasets/ECG_DSA/qrs/repro_label3 \
  --splits train val test --fs 100 --max_len 256 --token_len 96
.venv-heartlang/bin/python scripts/data_prep/ECG_DSA/copy_ecg_dsa_labels.py \
  --source_dir datasets/ecg_datasets/ECG_DSA/waveforms/data_0521_label3 \
  --target_dir datasets/ecg_datasets/ECG_DSA/qrs/repro_label3
cp datasets/ecg_datasets/ECG_DSA/waveforms/data_0521_label3/{train,val,test}_metadata.csv \
  datasets/ecg_datasets/ECG_DSA/qrs/repro_label3/
```

输入 `waveforms/data_0521_label3/{train,val,test}_data.npy`、`*_labels.npy`、`*_metadata.csv`；输出 `qrs/repro_label3/` 中对应的 QRS token、channel/time、标签和 metadata。第 1.3 节的四类数据构建脚本仍固定读取现有 `qrs/data_0521_label3/`，不会自动改用这里的 `qrs/repro_label3/`。若要将新切分结果接入四类构建，需先修改那个脚本中的源目录并核验样本数。

### 1.2 生成 clean QRS（可选）

```bash
.venv-heartlang/bin/python scripts/data_prep/ECG_DSA/make_clean_qrs_dataset.py \
  --src datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2 \
  --dst datasets/ecg_datasets/ECG_DSA/pretrain_qrs/repro_clean_le5pct \
  --splits train val \
  --block_size 256 --eps 1e-4 --max_flat_run 50 \
  --peak_idx_min 40 --peak_idx_max 65
```

输入为 V2 原始 QRS；输出同形状的 clean `*_data.npy`、channel/time 数组和 `clean_filter_summary.json`。命中任一规则的 heartbeat **只在该位置置零、channel/time 改为 0**，不删除整条 ECG。当前阈值是本项目实验规则，**不能表述为通用医学诊断标准**；对 ST/T 异常等病理形态的潜在误删应参照 `scripts/eval/codebook/audit_clean_filter_bias.py` 的审计结果。

### 1.3 构建四类下游数据（已有时跳过）

旧版构建脚本 `scripts/build_ecg_dsa_label4_patient_healthy_dataset.py` 内写死了三类、健康源及输出路径，直接运行会重写现有 `data_0521_label4_patient4675_healthy4675`。**不要直接运行旧脚本；这里只记录旧数据的来源。** 如需复现，先在脚本副本中改 `OUT_DIR` 为新的完整目录，并确认输入版本。

它按 `original_row_index` 为每条患者病例记录保留第一份 ECG，得到 4,675 条，再从 V2 metadata 中 `cohort=healthy` 随机取 4,675 条，合计 9,350。标签 ID：`0=OMI, 1=NOMI, 2=对照组, 3=健康`。输出 train/val/test 的 `*_data.npy`、`*_data_in_chans.npy`、`*_data_in_times.npy`、`*_labels.npy`、`*_metadata.csv`，另有 `prepare_summary.json`、`label_mapping.json`。这里的 4,675 指**病例记录**，不是 4,675 个互异患者；评估患者级独立性需另查 metadata。四类 train/val/test 分别 6,640/1,344/1,366 条。

### 1.4 每条 ECG 一个标签的四类数据（当前新增版本）

三类源数据 `qrs/data_0521_label3/` 本来就有 **8,303 条不同 ECG**，由 4,675 条标签记录展开。例如一条 OMI 标签记录中的三个 XML 文件，在 metadata 中分别成为三行 OMI。第 1.3 节旧四类数据只保留了每条记录的第一份 ECG；新脚本 `scripts/build_ecg_dsa_label4_ecg_level_dataset.py` 不再按 `original_row_index` 去重，保留全部 8,303 条患者 ECG，并从健康池无放回抽取 8,303 条不同 ECG。

新数据已生成，**无需重复执行**。若输出目录尚不存在，要重新构建时使用：

```bash
.venv-heartlang/bin/python scripts/build_ecg_dsa_label4_ecg_level_dataset.py --dry_run
.venv-heartlang/bin/python scripts/build_ecg_dsa_label4_ecg_level_dataset.py \
  --output_dir datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_ecg8303_healthy8303
```

脚本拒绝覆盖已有输出。输入为三类 QRS 及其 metadata/labels、V2 健康 QRS；输出为新目录的 `train/val/test_data.npy`、`*_data_in_chans.npy`、`*_data_in_times.npy`、`*_labels.npy`、`*_metadata.csv`、`prepare_summary.json`。train/val/test 分别 **11,794 / 2,352 / 2,460** 条；原三类患者 ECG 沿用已按患者分组的 split，验证无患者跨组。健康源 metadata 没有患者 ID，因此只能保证健康 ECG 文件不重复，无法核实健康者是否跨组。

对应 Morphology-V2 SID 数据也已生成，**无需重复执行**。下列命令仅用于新目录重建：

```bash
.venv-heartlang/bin/python scripts/experiments/generate_morphology_v2_sid_dataset.py \
  --source_data_dir datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_ecg8303_healthy8303 \
  --output_dir datasets/ecg_datasets/ECG_DSA/qrs_sid/data_0521_label4_ecg8303_healthy8303_morphology_v2 \
  --tokenizer_checkpoint checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth \
  --splits train val test --batch_size 512 --device cuda \
  --codebook_size 1024 --codebook_dim 64 --pad_sid 1024
```

新 SID 目录含 `train/val/test_sid.npy`、对应 channel/time、四类标签与 metadata，可直接作为下游分类/回归脚本的 `--sid_data_dir`。原预训练数据和旧四类结果均未改动；新数据上的结果须另设输出目录重新训练下游头，不应与旧测试集分数直接混为同一次实验。

## 2. 训练 Morphology-V2 codebook

输入为 clean QRS、现有或新建 physiology cache。此实现的 SID assignment 用单个 heartbeat 波形；lead/time 在生成 SID 时只用于判定 padding，不作为 VQ 特征。模型同时训练 waveform reconstruction、VQ、correlation/derivative 与 physiology 辅助损失。当前参考 checkpoint：`checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth`。

以下是**新实验**的完整配置，避免调用 `run_morphology_v2_train.sh`，因为那个旧脚本开头会 `rm -rf` 原 checkpoint 和 log：

```bash
.venv-heartlang/bin/python scripts/experiments/train_morphology_v2_physaware_vq.py \
  --pretrain_data_dir datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --phys_cache_dir results/morphology_v2_phys_cache/patient_healthy_raw_v2_clean_le5pct \
  --output_dir checkpoints/vqhbr/repro_morphology_v2_physaware_1024_dim64_bs256_e50 \
  --log_dir log/vqhbr/repro_morphology_v2_physaware_1024_dim64_bs256_e50 \
  --batch_size 256 --epochs 50 --save_ckpt_freq 50 --val_freq 10 \
  --codebook_n_emd 1024 --codebook_emd_dim 64 --token_len 96 --fs 100 \
  --phys_head_source z_q --lambda_waveform 1.0 --lambda_corr 0.2 \
  --lambda_deriv 0.1 --lambda_vq 1.0 --lambda_phys 0.2 \
  --continuous_alpha 0.25 --ema_decay 0.99 \
  --opt adamw --lr 5e-5 --min_lr 1e-5 --warmup_epochs 5 \
  --weight_decay 1e-4 --weight_decay_end 1e-4 \
  --device cuda --seed 0 --num_workers 10
```

输出 `checkpoint-best.pth`、`checkpoint-50.pth`、`checkpoint.pth`、`codebook_vectors.npy`、`val_sid_assignments.npy`、`training_history.csv`、`config.json`、`log.txt`。`checkpoint-best.pth` 是后续 SID 生成所用权重。若选择刚重建的 clean 数据，把 `--pretrain_data_dir` 与 cache 路径同时改为对应**新**目录，不能无意复用旧 cache。

## 3. 生成 Morphology-V2 SID 数据集

下面使用第 2 步新训练的 tokenizer，形成连贯的端到端流程。若要直接使用已验证的原实验权重，将两条命令中的 checkpoint 路径都换成 `checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth`。输出用新目录，避免 `run_morphology_v2_sid_dataset.sh` 的 `--overwrite` 覆盖已有 SID。

```bash
.venv-heartlang/bin/python scripts/experiments/generate_morphology_v2_sid_dataset.py \
  --source_data_dir datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2 \
  --output_dir datasets/ecg_datasets/ECG_DSA/pretrain_sid/repro_morphology_v2_raw_v2 \
  --tokenizer_checkpoint checkpoints/vqhbr/repro_morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth \
  --splits train val --batch_size 512 --device cuda \
  --codebook_size 1024 --codebook_dim 64 --pad_sid 1024
.venv-heartlang/bin/python scripts/experiments/generate_morphology_v2_sid_dataset.py \
  --source_data_dir datasets/ecg_datasets/ECG_DSA/qrs/data_0521_label4_patient4675_healthy4675 \
  --output_dir datasets/ecg_datasets/ECG_DSA/qrs_sid/repro_label4_morphology_v2 \
  --tokenizer_checkpoint checkpoints/vqhbr/repro_morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth \
  --splits train val test --batch_size 512 --device cuda \
  --codebook_size 1024 --codebook_dim 64 --pad_sid 1024
```

输入为 QRS waveform、channel/time 数组和 VQ checkpoint；输出 `train_sid.npy`, `val_sid.npy`（下游还有 `test_sid.npy`），形状 `[N,256]`；默认复制 `*_data_in_chans.npy`、`*_data_in_times.npy`、已有 labels/metadata，并生成 `sid_dataset_config.json`。这里只是**冻结 tokenizer 推理**，不会继续训练 codebook。不要把 clean QRS SID 当作当前 raw-V2 MLM 输入。

## 4. SID + context 的 MLM 预训练

输入为第 3 步的预训练 SID 与 channel/time；模型结构 `SID embedding + lead/spatial + temporal + position -> Transformer -> MLM head`。SID 在进入 MLM 前已固定，context 不会反向进入 VQ。mask ratio 0.5，预测被 mask 的真实 SID，CrossEntropy；输出词表 1024。

```bash
.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_pretraining.py \
  --sid_data_dir datasets/ecg_datasets/ECG_DSA/pretrain_sid/repro_morphology_v2_raw_v2 \
  --output_dir checkpoints/pretrain/repro_HeartLangSID_MorphologyV2Context_rawv2_e100 \
  --log_dir log/pretrain/repro_HeartLangSID_MorphologyV2Context_rawv2_e100 \
  --batch_size 256 --epochs 100 --save_ckpt_freq 50 --val_freq 10 \
  --vocab_size 1024 --pad_sid 1024 --seq_len 256 \
  --embed_dim 768 --depth 12 --heads 8 --mlp_dim 1024 \
  --mask_ratio 0.5 --lr 5e-4 --min_lr 1e-5 --warmup_epochs 5 \
  --weight_decay 0.05 --clip_grad 3.0 --num_workers 8 --device cuda --seed 0
```

输出 `checkpoint-best.pth`（按最低 validation MLM loss 选）、`checkpoint-50.pth`、`checkpoint-100.pth`、`checkpoint.pth`、`training_history.csv`、`config.json`。历史记录含 train/val MLM loss 和 masked SID accuracy。现有参考 checkpoint：`checkpoints/pretrain/HeartLangSID_MorphologyV2Context_rawv2_e100/checkpoint-best.pth`。

## 5. 四分类：冻结 backbone，只训练 Linear(768, 4)

输入为下游 SID train/val/test、四类 one-hot 标签和第 4 步的 MLM checkpoint。模型取 backbone 的 record-level 表征 `z`，线性头输出 4 个 logits，CrossEntropy 训练。`normalized_inverse_freq` 根据 train 类别频率加权；**logit 不是概率**，需 softmax 才是四类概率。验证集优先按 macro ROC-AUC 选 best，测试集只在选模后评估。

```bash
.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_classification.py \
  --sid_data_dir datasets/ecg_datasets/ECG_DSA/qrs_sid/repro_label4_morphology_v2 \
  --pretrained_checkpoint checkpoints/pretrain/repro_HeartLangSID_MorphologyV2Context_rawv2_e100/checkpoint-best.pth \
  --output_dir checkpoints/finetune/ecg_dsa/repro_classification_label4_MorphologyV2Context_weighted_ce \
  --batch_size 64 --epochs 30 --save_ckpt_freq 10 \
  --vocab_size 1024 --pad_sid 1024 --seq_len 256 \
  --embed_dim 768 --depth 12 --heads 8 --mlp_dim 1024 \
  --class_weight normalized_inverse_freq \
  --lr 5e-4 --min_lr 1e-5 --warmup_epochs 5 \
  --weight_decay 0.05 --clip_grad 3.0 --num_workers 8 --device cuda --seed 0
```

输出 `checkpoint-best.pth`、每 10 epoch checkpoint、`training_history.csv`、`test_classification_summary.json`、`test_predictions.csv`。测试 summary 含 macro/per-class ROC-AUC、F1、balanced accuracy、accuracy、OMI recall 和 confusion matrix。现有参考结果：`checkpoints/finetune/ecg_dsa/classification_label4_HeartLangSID_MorphologyV2Context_e100_linear_weighted_ce/test_classification_summary.json`。

## 6. 四输出回归：冻结 backbone，只训练 Linear(768, 4)

这里目标**仍是四类 one-hot 向量**，损失为 MSE；线性头输出 4 个连续值，**不是类别概率，也没有要求和为 1**。R² 是这四个 0/1 指示目标的拟合优度，不是狭窄率等临床连续指标的 R²，不应拿它代替四分类 AUC。

```bash
.venv-heartlang/bin/python scripts/experiments/train_heartlang_sid_regression.py \
  --sid_data_dir datasets/ecg_datasets/ECG_DSA/qrs_sid/repro_label4_morphology_v2 \
  --pretrained_checkpoint checkpoints/pretrain/repro_HeartLangSID_MorphologyV2Context_rawv2_e100/checkpoint-best.pth \
  --output_dir checkpoints/finetune/ecg_dsa/repro_regression_label4_MorphologyV2Context_linear \
  --batch_size 64 --epochs 30 --save_ckpt_freq 10 \
  --vocab_size 1024 --pad_sid 1024 --seq_len 256 \
  --embed_dim 768 --depth 12 --heads 8 --mlp_dim 1024 --task_head_type linear \
  --lr 5e-4 --min_lr 1e-5 --warmup_epochs 5 \
  --weight_decay 0.05 --clip_grad 3.0 --num_workers 8 --device cuda --seed 0
```

验证集按 R² 选 best。输出 `checkpoint-best.pth`、`training_history.csv`、`test_regression_summary.json`、`test_outputs.csv`、`test_targets.csv`。现有参考结果：`checkpoints/finetune/ecg_dsa/regression_label4_HeartLangSID_MorphologyV2Context_e100_linear/test_regression_summary.json`。

快速读取测试指标：

```bash
.venv-heartlang/bin/python -c 'import json,sys; [print("\n",p,"\n",json.dumps(json.load(open(p)),ensure_ascii=False,indent=2)) for p in sys.argv[1:]]' \
  checkpoints/finetune/ecg_dsa/repro_classification_label4_MorphologyV2Context_weighted_ce/test_classification_summary.json \
  checkpoints/finetune/ecg_dsa/repro_regression_label4_MorphologyV2Context_linear/test_regression_summary.json
```

## 7. Codebook 评估与对照

当前已有的主要 codebook 对照：H-clean contextual VQ、Morphology-V1、Morphology-V2、V1+Phys、NoContinuous、Curriculum-V2。**这些是不同实验，不能只比较 active SID，必须同时看 reconstruction、使用率、词内形态/生理纯度与 temporal/alignment 指标。**

用相同 clean validation 数据比较 H-clean 和 Morphology-V2，输出目录另建：

```bash
.venv-heartlang/bin/python scripts/eval/codebook/evaluate_morphology_only_comparison.py \
  --run H-clean,contextual_vqhbr,checkpoints/vqhbr/ECG_DSA_patient_healthy_raw_v2_clean_le5pct_codebook1024_dim64_bs256_e50_finalonly/checkpoint-50.pth \
  --run Morphology-V2,morphology_v2,checkpoints/vqhbr/morphology_v2_physaware_1024_dim64_bs256_e50/checkpoint-best.pth \
  --data_path datasets/ecg_datasets/ECG_DSA/pretrain_qrs/patient_healthy_raw_v2_clean_le5pct \
  --split val --output_dir results/morphology_v2_codebook/repro_Hclean_vs_V2_val_shift4 \
  --codebook_size 1024 \
  --codebook_dim 64 --batch_size 256 --max_shift 4 \
  --corr_threshold 0.9 --device cuda
```

输入为同一 val heartbeat 与两个 checkpoint；统一输出到 `results/morphology_v2_codebook/repro_Hclean_vs_V2_val_shift4/`，其中有 `comparison_summary.json`、`comparison_table.csv`、`comparison_delta_table.csv`，每个 run 另有自己的 summary/逐 SID 统计文件。`max_shift=4` 表示左右各最多 4 个**采样点**，100 Hz 下约 40 ms，不是 4 秒。另有 `scripts/eval/codebook/evaluate_codebook_alignment_robustness.py` 与 `evaluate_sid_clinical_semantics.py`，参数模板见同目录的 `run_codebook_alignment_robustness.sh`、`run_sid_clinical_semantics.sh`；换独立 `--output_dir` 后运行，不覆盖现有 `results/`。

## 8. 消融：NoContinuous 与 SID-only

`Morphology-V2-NoContinuous` 仅把 codebook 训练中的 continuous physiology supervision 系数设为 `0`；对应参考权重在 `checkpoints/vqhbr/morphology_v2_no_continuous_1024_dim64_bs256_e50/checkpoint-best.pth`。其独立训练入口 `scripts/experiments/run_morphology_v2_no_continuous_pretrain.sh` 有存在性保护，但目录已存在时会拒绝运行。若需重做，把第 2 节命令中的 `--output_dir`、`--log_dir` 改成**新的完整目录路径**，并把 `--continuous_alpha 0.25` 改为 `--continuous_alpha 0.0`，其余参数保持一致。随后把第 3 节两条 SID 生成命令中的 `--tokenizer_checkpoint` 改成新 checkpoint 的完整路径，并指定新的 SID 输出目录。

`SID-only` 消融在第 4–6 节同样的命令里分别追加 `--no_spatial_embedding --no_temporal_embedding`，训练与下游两个阶段**都必须加**；否则 checkpoint 结构/实验定义不一致。位置 embedding 仍保留。现有参考目录分别是 `checkpoints/pretrain/HeartLangSID_MorphologyV2_SIDOnly_rawv2_e100` 与 `checkpoints/pretrain/HeartLangSID_MorphologyV2NoContinuous_SIDOnly_rawv2_e100`。四类下游 SID-only 的分类/回归结果在 `checkpoints/finetune/ecg_dsa/` 下的 `...MorphologyV2_SIDOnly...` 目录；具体完整参数见 `scripts/experiments/run_morphology_v2_sid_only_downstream_pipeline.sh`。**不要把 Context 模型与 SID-only 模型的 checkpoint 交叉加载。**

## 9. 可复现性与边界

- 环境以服务器 `.venv-heartlang` 为准。根目录 `requirements.txt` 是原论文依赖，不保证单独安装它即可重建当前所有新增 ECG_DSA 实验。先运行 `import torch, numpy, pandas, sklearn, wfdb` 做依赖检查。
- 每次记录 `config.json`、数据 `prepare_summary.json`、checkpoint 路径、seed 和代码版本。服务器当前不是 Git 工作树；若需要精确复现，另行保存代码快照或 commit hash。
- 当前四类数据的患者部分按病例记录去重，**不是完整严格 patient-level 4,675 人**。历史 split 与原始 V2 自监督语料是否共享 ECG，需要在论文/汇报中说明；自监督预训练不等于独立外部测试。
- `H-clean` 是 contextual VQ；`Morphology-V2` 的 SID 生成不带 lead/time 邻域 context，但下游 Context HeartLang 会重新加入 lead/time/position。两者不是“完全无 context vs 有 context”的下游比较。
- 训练脚本的 `checkpoint-best.pth` 选择标准因任务而异：VQ/MLM 按验证损失，分类按验证 macro AUC，回归按验证 R²。测试集不应用于挑 checkpoint。
- 直接运行 `scripts/experiments/run_morphology_v2_train.sh` 会删除原输出；`run_morphology_v2_sid_dataset.sh` 带 `--overwrite`；四类数据构建脚本也会重写固定目录。使用本文独立目录命令。

## 10. 代码索引

| 阶段 | 主要实现 |
| --- | --- |
| 远程波形数据 | `datasets/ecg_datasets/ECG_DSA/pretrain_waveforms/patient_healthy_raw_v2/`、`waveforms/data_0521_label3/` |
| QRS | `scripts/data_prep/ECG_DSA/tokenize_pretrain_qrs.py` |
| clean | `scripts/data_prep/ECG_DSA/make_clean_qrs_dataset.py` |
| 四类数据 | `scripts/build_ecg_dsa_label4_patient_healthy_dataset.py` |
| 四类 ECG 级数据 | `scripts/build_ecg_dsa_label4_ecg_level_dataset.py` |
| Morphology-V2 VQ | `scripts/experiments/train_morphology_v2_physaware_vq.py` |
| SID 生成 | `scripts/experiments/generate_morphology_v2_sid_dataset.py` |
| SID Transformer / MLM | `heartlang/models/modeling_heartlang_sid.py`、`scripts/experiments/train_heartlang_sid_pretraining.py` |
| 四分类 / 四输出回归 | `scripts/experiments/train_heartlang_sid_classification.py`、`train_heartlang_sid_regression.py` |
| codebook 质量 | `scripts/eval/codebook/evaluate_morphology_only_comparison.py` 及同目录其他审计脚本 |
