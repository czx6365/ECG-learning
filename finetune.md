# HeartLang Finetune 完整流程

本文档只讲 **HeartLang 的下游微调阶段**。如果把整个项目分成 3 个阶段，那么它位于：

1. `VQ-HBR` 训练：学到 ECG word 的离散词表。
2. `HeartLang` 预训练：做 masked ECG sentence pretraining。
3. `Finetune`：把预训练好的 backbone 迁移到具体分类任务。

这份仓库里的微调入口是：

```bash
heartlang/cli/run_class_finetuning.py
```

它实际上支持两条路径：

- **端到端微调**：`HeartLang_finetune_base + mlp_head`，直接训练神经网络分类头。
- **特征式微调**：先用 HeartLang backbone 抽特征，再用 `decision_tree / random_forest / svm / logistic_regression / xgboost / lightgbm` 做分类。

如果你只是想复现论文里常规下游实验，优先看“端到端微调”这条路径。

---

## 1. 微调阶段到底在做什么

微调阶段的目标，不再是自监督预测 token，而是做有监督的下游分类。

整体逻辑是：

1. 读取已经 token 化好的 ECG 数据。
2. 读取对应标签。
3. 构造 `HeartLang_finetune_base` 模型。
4. 把预训练权重加载到 `backbone`。
5. 根据 `--trainable` 决定训练全部层，还是只训练线性头。
6. 用训练集训练。
7. 用验证集选最好 checkpoint。
8. 用测试集评估，输出 AUC、accuracy、f1 等指标。

其中模型结构是：

- `backbone`：复用预训练阶段的 `ST_ECGFormer`
- `mlp_head`：一个线性分类头 `nn.Linear(embed_dim, num_classes)`

也就是说，微调本质上是：

- 输入：`QRS token 序列 + 导联位置索引 + 时间位置索引`
- 输出：分类 logits

---

## 2. 微调前需要准备什么

### 预训练 checkpoint

现在这份仓库已经改成：**必须显式提供本地 checkpoint**。

也就是说：

- 必须传 `--finetune <本地路径>`
- 不再允许不传 `--finetune`
- 不再允许通过 URL 自动下载 checkpoint
- 不再允许随机初始化后直接开始微调

代码里 `load_finetune_checkpoint()` 的行为是：

- 没传 `--finetune`：直接报错
- 如果 `--finetune` 是 `http/https`：直接报错
- 如果 `--finetune` 指向的本地文件不存在：直接报错
- 只有本地 checkpoint 文件存在时，才会执行 `torch.load(...)`

另外，这个仓库在加载预训练权重时，会自动给 key 加上 `backbone.` 前缀，再灌进微调模型，所以 **预训练阶段导出的主干权重可以直接迁移到微调模型的 backbone**。

### 下游数据

微调阶段不是直接吃原始 ECG 波形目录，而是吃 **已经经过 QRS tokenizer 处理后的数据目录**。

---

## 3. 微调数据目录必须长什么样

`run_class_finetuning.py` 通过 `prepare_finetune_dataset()` 读取一个目录，要求这个目录下面至少有下面 12 个文件：

```text
dataset_dir/
├── train_data.npy
├── train_labels.npy
├── train_data_in_chans.npy
├── train_data_in_times.npy
├── val_data.npy
├── val_labels.npy
├── val_data_in_chans.npy
├── val_data_in_times.npy
├── test_data.npy
├── test_labels.npy
├── test_data_in_chans.npy
└── test_data_in_times.npy
```

###  每个文件的含义

- `*_data.npy`
  - 每条样本的 token 化 ECG 片段
  - 常见形状：`[N, 256, 96]`
- `*_data_in_chans.npy`
  - 每个 token 对应的导联索引
  - 常见形状：`[N, 256]`
- `*_data_in_times.npy`
  - 每个 token 对应的时间块索引
  - 常见形状：`[N, 256]`
- `*_labels.npy`
  - 标签
  - 论文主路径更偏向 **多标签 / multi-hot** 形式

---

## 4. 微调入口的完整代码流程

主入口函数是：

```python
main(args, ds_init)
```

执行顺序可以概括为下面 10 步。

### 第 1 步：初始化运行环境

代码会先做：

- 分布式初始化 `utils.init_distributed_mode(args)`
- 设置随机种子 `set_seed(args.seed)`
- 创建 TensorBoard writer

如果加了 `--enable_deepspeed`，还会切换到 DeepSpeed 分支。

### 第 2 步：判断当前走哪条微调路径

`resolve_runtime_mode(args)` 会根据参数决定模式：

- `--head_type mlp` -> 端到端神经网络微调
- `--head_type decision_tree` -> 特征式微调，模型固定为决策树
- `--head_type feature_model --feature_model_type xxx` -> 特征式微调，模型为指定 sklearn 分类器

### 第 3 步：构建数据集

调用：

```python
prepare_finetune_dataset(args.dataset_dir, args.split_ratio, args.sampling_method)
```

得到：

- `dataset_train`
- `dataset_val`
- `dataset_test`

其中只有训练集会受 `split_ratio` 影响。

### 第 4 步：可选的训练集抽样

如果你传了：

- `--split_ratio < 1.0`

训练集会缩小；验证集和测试集不会缩小。

支持两种抽样方式：

- `--sampling_method random`
- `--sampling_method stratified`

其中 `stratified` 是按标签主类做近似分层抽样。

### 第 5 步：构建 DataLoader

代码会分别构建：

- 训练用 DataLoader
- 验证用 DataLoader
- 测试用 DataLoader
- 特征提取专用 DataLoader

几个细节：

- 训练采样器默认是 `DistributedSampler(..., shuffle=True)`
- 特征提取采样器在分布式下是 `shuffle=False`，保证特征提取稳定
- 如果 `total_batch_size > len(dataset_train)`，代码会自动把 `batch_size` 反复减半，避免一个 epoch 连一步都跑不动

### 第 6 步：构建模型并加载 checkpoint

模型构建入口：

```python
create_model(args.model, pretrained=False, num_classes=args.nb_classes)
```

默认模型名是：

```bash
--model HeartLang_finetune_base
```

这个模型内部是：

- `ST_ECGFormer` backbone
- 线性分类头 `mlp_head`

然后执行：

```python
load_finetune_checkpoint(model, args)
```

逻辑是：

- 如果 `--finetune` 为空：直接报错
- 如果 `--finetune` 是 URL：直接报错
- 如果 `--finetune` 是存在的本地路径：`torch.load`

### 第 7 步：分叉

从这里开始，流程会分成两大类：

- **A. 端到端微调**
- **B. 特征式微调**

## A 路径：端到端微调

这是默认路径，也是最接近论文主实验的路径。

### 前向传播发生了什么

输入一个 batch 时，每条样本会带 4 个张量：

- `samples`
- `targets`
- `in_chan_matrix`
- `in_time_matrix`

前向过程大致是：

1. `samples` 进入 `backbone`
2. `backbone` 同时接收导联位置编码 `in_chan_matrix`
3. `backbone` 同时接收时间位置编码 `in_time_matrix`
4. backbone 输出所有 token 表示
5. 取第 0 个位置的 `CLS token`
6. 把 `CLS token` 送进 `mlp_head`
7. 得到最终 logits

在代码里对应：

- `modeling_finetune.py::extract_features()`
- `modeling_finetune.py::forward()`

### 哪些层会被训练

由 `--trainable` 决定。

支持的值主要有：

- `all`
  - 全部参数参与训练
- `linear`
  - 只训练 `mlp_head`
- `adapter`
  - 只训练参数名中包含 `adapter` 或 `mlp_head` 的层

但要注意一个实现细节：

- 当前仓库主模型里 **没有实际定义 adapter 模块**
- 所以 `--trainable adapter` 在这份代码里通常会退化成“基本只训练 `mlp_head`”

### 损失函数如何选

代码里是：

- `args.is_binary=True` -> `BCEWithLogitsLoss`
- `args.is_binary=False` -> `CrossEntropyLoss`

训练和评估时：

- 二值/多标签任务会对输出做 `sigmoid`
- 多分类任务会对输出做 `softmax`

### 优化器和调度器

默认是：

- 优化器：`AdamW`
- 学习率调度：step-level cosine scheduler
- weight decay 调度：step-level cosine scheduler

另外还支持：

- 梯度累积：`--update_freq`
- 梯度裁剪：`--clip_grad`
- EMA：`--model_ema`
- layer-wise lr decay：`--layer_decay`
- DDP / DeepSpeed

###  一个 epoch 

`engine_for_finetuning.train_one_epoch()` 的核心步骤是：

1. 从 DataLoader 取一个 batch
2. 把样本搬到 GPU
3. 混合精度前向
4. 计算 loss
5. 如果是多标签任务，对 logits 做 `sigmoid`
6. 反向传播
7. 梯度累积到 `update_freq` 后才执行一次参数更新
8. 统计：loss `class_acc`lr `min_lr` weight_decay` `grad_norm`

### 验证与最优模型保存

每个 epoch 结束后，如果存在验证集，就会：

1. 调用 `evaluate(..., header="Val:")`
2. 计算：metric
3. 用 `roc_auc` 判断是否刷新最佳结果
4. 如果刷新，则额外保存一个 `best` checkpoint

### 测试阶段

如果你传 `--eval`，流程不会训练，而是直接：

1. 加载模型
2. 跑测试集
3. 计算指标
4. 保存结果

测试时除了返回指标，还会把预测值和真实标签存到 CSV。

---

##  B 路径：特征式微调

这条路径不是直接训练神经网络分类头，而是把 HeartLang 当成一个特征提取器。

### 流程分两段

#### 第一段：提取特征

`extract_features()` 会：

1. 调 `feature_model.extract_features(...)`
2. 从 backbone 里取 `CLS feature`
3. 收集每个样本的特征和标签
4. 合并成：
   - `features`
   - `targets`

最终得到的 `features` 形状通常是：

```text
[N, feature_dim]
```

#### 第二段：训练传统分类器

`run_feature_model_pipeline()` 会根据参数创建 sklearn 风格模型：

- `decision_tree`
- `random_forest`
- `svm`
- `logistic_regression`
- `xgboost`
- `lightgbm`

然后执行：

1. `fit(train_features, train_targets)`
2. 在 `val/test` 上预测
3. 计算指标
4. 保存模型和指标

### 特征缓存机制

这部分是这份代码比较实用的地方。

缓存目录优先级：

1. `--feature_dir`
2. `--output_dir/saved_features`
3. `--dataset_dir/saved_features`
4. 默认 `saved_features`

缓存文件格式：

```text
feature_dir/
├── train_features.npy
├── train_targets.npy
├── val_features.npy
├── val_targets.npy
├── test_features.npy
└── test_targets.npy
```

你可以：

- 第一次从原始 `dataset_dir` 抽特征并保存
- 以后不再传 `dataset_dir`，直接从缓存特征训练传统分类器

### 纯缓存特征模式

如果你不想再次访问原始 ECG 数据，可以：

- 不传 `--dataset_dir`
- 传 `--load_features_if_exist`
- 指定 `--feature_dir`

代码会直接读取缓存特征，并从 `train_targets` 自动推断：

- `is_binary`
- `nb_classes`

---

## 7. 常用命令模板

下面给你几种最实用的命令模板。

### 7.1 端到端微调：只训练线性头

```bash
torchrun --nnodes=1 --master_port=2025 --nproc_per_node=1 heartlang/cli/run_class_finetuning.py \
  --dataset_dir datasets/ecg_datasets/PTBXL_QRS/form \
  --output_dir checkpoints/finetune/ptbxl/form_linear \
  --log_dir log/finetune/ptbxl_form_linear \
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
  --save_ckpt_freq 10 \
  --seed 0 \
  --is_binary \
  --nb_classes 19 \
  --world_size 1
```

### 7.2 端到端微调：训练全部层

```bash
torchrun --nnodes=1 --master_port=2025 --nproc_per_node=1 heartlang/cli/run_class_finetuning.py \
  --dataset_dir datasets/ecg_datasets/PTBXL_QRS/form \
  --output_dir checkpoints/finetune/ptbxl/form_full \
  --log_dir log/finetune/ptbxl_form_full \
  --model HeartLang_finetune_base \
  --finetune checkpoints/heartlang_base/checkpoint-200.pth \
  --trainable all \
  --batch_size 128 \
  --lr 5e-4 \
  --epochs 50 \
  --weight_decay 0.05 \
  --seed 0 \
  --is_binary \
  --nb_classes 19 \
  --world_size 1
```

### 7.3 只做测试

```bash
torchrun --nnodes=1 --master_port=2025 --nproc_per_node=1 heartlang/cli/run_class_finetuning.py \
  --dataset_dir datasets/ecg_datasets/PTBXL_QRS/form \
  --output_dir checkpoints/finetune/ptbxl/form_linear \
  --log_dir log/finetune_test/ptbxl_form_linear \
  --model HeartLang_finetune_base \
  --eval \
  --trainable linear \
  --batch_size 256 \
  --seed 0 \
  --is_binary \
  --nb_classes 19 \
  --world_size 1
```

注意：

- `--eval` 只是让流程进入测试模式
- 实际加载哪个 checkpoint，取决于你是否同时配置了 `--resume` / 自动恢复目录中的 checkpoint

### 7.4 特征式微调：随机森林

```bash
torchrun --nnodes=1 --master_port=2025 --nproc_per_node=1 heartlang/cli/run_class_finetuning.py \
  --dataset_dir datasets/ecg_datasets/PTBXL_QRS/form \
  --output_dir runs/feature_rf \
  --log_dir log/feature_rf \
  --model HeartLang_finetune_base \
  --finetune checkpoints/heartlang_base/checkpoint-200.pth \
  --head_type feature_model \
  --feature_model_type random_forest \
  --feature_dir runs/shared_features/form \
  --load_features_if_exist \
  --nb_classes 19 \
  --is_binary
```

这条命令会：

1. 如果 `runs/shared_features/form` 已经有缓存特征，就直接读取
2. 否则先用 HeartLang 抽特征，再训练随机森林

### 7.5 纯缓存特征模式

```bash
torchrun --nnodes=1 --master_port=2025 --nproc_per_node=1 heartlang/cli/run_class_finetuning.py \
  --output_dir runs/feature_svm \
  --log_dir log/feature_svm \
  --head_type feature_model \
  --feature_model_type svm \
  --feature_dir runs/shared_features/form \
  --load_features_if_exist
```

这时候不会访问 `dataset_dir`，而是直接消费已经缓存好的：

- `train_features.npy`
- `train_targets.npy`
- `val_features.npy`
- `val_targets.npy`
- `test_features.npy`
- `test_targets.npy`

---

## 8. 重要参数解释

### 8.1 训练相关

- `--batch_size`
  - 单卡 micro-batch size
- `--update_freq`
  - 梯度累积步数
- `--epochs`
  - 训练 epoch 数
- `--lr`
  - 初始学习率
- `--weight_decay`
  - 权重衰减
- `--warmup_epochs`
  - warmup 周期
- `--clip_grad`
  - 梯度裁剪阈值

总 batch size 约等于：

```text
batch_size * update_freq * world_size
```

### 8.2 权重加载相关

- `--finetune`
  - 预训练或已有微调 checkpoint
- `--resume`
  - 从中断训练恢复
- `--auto_resume`
  - 自动寻找恢复点，默认开启

### 8.3 训练范围相关

- `--trainable all`
  - 全参数微调
- `--trainable linear`
  - 线性探针
- `--trainable adapter`
  - 仅训练 adapter 和头部；当前仓库里 adapter 主要是预留能力

### 8.4 数据相关

- `--dataset_dir`
  - 微调数据目录
- `--split_ratio`
  - 只作用于训练集抽样比例
- `--sampling_method`
  - `random` 或 `stratified`
- `--nb_classes`
  - 类别数
- `--is_binary`
  - 是否走 BCE / sigmoid 路径

### 8.5 特征式微调相关

- `--head_type`
  - `mlp` / `decision_tree` / `feature_model`
- `--feature_model_type`
  - `decision_tree` / `random_forest` / `svm` / `logistic_regression` / `xgboost` / `lightgbm`
- `--feature_dir`
  - 特征缓存目录
- `--load_features_if_exist`
  - 优先使用已有缓存
- `--force_reextract_features`
  - 即使缓存存在，也强制重新抽特征

---

## 9. 微调输出会落到哪里

### 9.1 端到端微调

如果设置了 `--output_dir`，通常会得到：

- `checkpoint-*.pth`
- `checkpoint-best.pth`
- `log.txt`

如果设置了 `--log_dir`，还会写 TensorBoard 日志。

### 9.2 测试预测结果

测试阶段会把输出和标签保存成 CSV，目录在 `heartlang.project_paths` 里定义，按数据集名生成文件名。

你可以理解为会有两类文件：

- `xxx_outputs.csv`
- `xxx_targets.csv`

用于后续分析。

### 9.3 特征式微调

如果是 feature-based 模式，还会额外生成：

- `decision_tree.pkl` 或 `random_forest.pkl` 等分类器文件
- `decision_tree_metrics.json` 或 `random_forest_metrics.json`
- `saved_features/*.npy` 或你指定 `feature_dir` 下的特征缓存

---

## 10. 这个仓库当前版本的几个关键注意事项

这部分很重要，属于“看代码后才能知道”的实现细节。

### 10.1 现在必须提供本地 `--finetune`

当前代码已经改成严格模式：

- 必须提供 `--finetune`
- 只能是本地文件
- 文件不存在就报错
- 禁止随机初始化作为兜底

所以如果你要跑微调，先确保本地已经有类似下面的文件：

```text
checkpoints/heartlang_base/checkpoint-200.pth
```

### 10.2 `--is_binary` 当前 CLI 基本被固定成了 `True`

参数定义是：

```python
parser.add_argument("--is_binary", action="store_true", default=True)
```

这意味着：

- 默认就是 `True`
- 命令行里没有对应的 `--no_is_binary`

结果就是：

- 从纯 CLI 角度看，端到端路径几乎总会走 `BCEWithLogitsLoss`
- 如果你要做严格的单标签 softmax 多分类，这个参数定义目前不够干净，最好先改代码

也就是说，这份仓库当前最稳的使用方式仍然是：

- **多标签 ECG 分类**

### 10.3 `--save_features` 单独使用没有完全打通

虽然参数里有：

```bash
--save_features
```

但当前主流程里，`prepare_feature_splits()` 只会在 **feature-based 模式** 下调用。

所以如果你是：

- `--head_type mlp`
- 只想顺手保存特征

当前代码并不会自动进入特征提取流程。

更稳妥的做法是：

- 显式使用 `--head_type feature_model`
- 配合 `--feature_dir` / `--load_features_if_exist`

### 10.4 `adapter` 训练模式目前更像“预留接口”

代码支持：

```bash
--trainable adapter
```

但仓库主模型里没有真正的 adapter 模块实现。

因此当前版本下，这个选项通常不适合作为主要实验设置。

### 10.5 数据集必须包含 train / val / test 三套文件

`prepare_finetune_dataset()` 是直接构建：

- `train`
- `val`
- `test`

如果缺任何一套，当前代码都很容易在读文件阶段直接报错。

### 10.6 训练集过小会自动缩 batch

如果你用 `split_ratio` 把训练集抽得很小，代码会自动减小 `batch_size`。这不是 bug，是为了避免：

- `total_batch_size > len(dataset_train)`
