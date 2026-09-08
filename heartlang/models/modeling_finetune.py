# --------------------------------------------------------
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model
# By Jiarui Jin and Haoyu Wang
# Based on BEiT-v2, timm, DeiT, DINO and LaBraM code bases
# ---------------------------------------------------------

import torch
from timm.models import register_model
from torch import nn

from heartlang.models.modeling_pretrain import ST_ECGFormer


class ST_ECGFormerClassifier(nn.Module):
    def __init__(
        self,
        seq_len,
        time_window,
        depth,
        embed_dim,
        heads,
        mlp_dim,
        dim_head=64,
        num_classes=None,
        dropout=0.0,
        emb_dropout=0.0,
        freeze_backbone=False,
        task_type="classification",
        task_head_type="linear",
        task_head_hidden_dim=256,
        task_head_dropout=0.1,
    ):
        super().__init__()

        # 主干网络 backbone：
        # 复用预训练阶段的 ST_ECGFormer，用来从 ECG 输入中提取高层特征
        self.backbone = ST_ECGFormer(
            seq_len=seq_len,
            time_window=time_window,
            depth=depth,
            embed_dim=embed_dim,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        # 记录层数，后续可用于 layer-wise lr decay 等策略
        self.depth = depth
        self.task_type = task_type

        # 下游任务头：
        # 分类时输出类别 logits，回归时输出目标值
        if task_head_type == "linear":
            self.mlp_head = nn.Linear(embed_dim, num_classes)
        elif task_head_type == "mlp2":
            self.mlp_head = nn.Sequential(
                nn.Linear(embed_dim, task_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(task_head_dropout),
                nn.Linear(task_head_hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unsupported task_head_type: {task_head_type}")
        self.task_head_type = task_head_type

        # 如果指定冻结 backbone，则只训练分类头
        if freeze_backbone:
            self.freeze_backbone()

    def extract_features(
        self,
        x,
        in_chan_matrix=None,
        in_time_matrix=None,
        return_all_tokens=True,
    ):
        # 提取 backbone 输出中的 CLS token 特征
        # backbone 输出通常形状类似 [B, seq_len+1, embed_dim]
        # 其中第 0 个位置通常是 CLS token
        return self.backbone(
            x,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
            return_all_tokens=return_all_tokens,
        )[:, 0]

    def forward(
        self,
        x,
        in_chan_matrix=None,
        in_time_matrix=None,
        return_all_tokens=True,
    ):
        # 前向传播逻辑：
        # 1. 先从 backbone 中提取 CLS token 特征
        cls_token = self.extract_features(
            x,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
            return_all_tokens=return_all_tokens,
        )

        # 2. 再送入线性分类头，输出 logits
        return self.mlp_head(cls_token)

    def get_num_layers(self):
        # 返回 backbone 层数，供训练策略使用
        return self.depth

    @property
    def feature_dim(self):
        # 返回提取出的 CLS token 特征维度
        # 这里等于分类头输入维度
        if isinstance(self.mlp_head, nn.Sequential):
            return self.mlp_head[0].in_features
        return self.mlp_head.in_features

    @torch.jit.ignore
    def no_weight_decay(self):
        # 告诉优化器：哪些参数不参与 weight decay
        # 当前返回空，表示没有特殊跳过项
        return {}

    def freeze_backbone(self):
        # 冻结 backbone 的所有参数
        # 常用于 linear probing：只训练最后的分类头
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen. Only training task head.")

    def unfreeze_backbone(self):
        # 解冻 backbone 的所有参数
        # 常用于完整 fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen.")

    def randomly_initialize_weights(self):
        # 当不加载预训练权重时，对当前模型参数做显式随机初始化
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.1, b=0.1)


def get_model_default_params():
    # 返回模型默认配置
    # 这些参数通常与预训练 backbone 的设置保持一致
    return dict(
        seq_len=256,
        time_window=96,
        num_classes=1000,
        embed_dim=768,
        depth=8,
        heads=4,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )


@register_model
def HeartLang_finetune_base(pretrained=False, **kwargs):
    # 注册一个 timm 风格的模型构建入口
    # 这样可以通过模型名 HeartLang_finetune_base 来创建模型

    config = get_model_default_params()

    # 用下游任务传入的类别数覆盖默认值
    config["num_classes"] = kwargs["num_classes"]
    config["task_type"] = kwargs.get("task_type", "classification")
    config["task_head_type"] = kwargs.get("task_head_type", "linear")
    config["task_head_hidden_dim"] = kwargs.get("task_head_hidden_dim", 256)
    config["task_head_dropout"] = kwargs.get("task_head_dropout", 0.1)

    # base 版本使用更大的配置
    config["depth"] = 12
    config["heads"] = 8
    config["mlp_dim"] = 1024

    # 返回分类模型
    return ST_ECGFormerClassifier(**config)


if __name__ == "__main__":
    # 简单测试：构建一个 12 分类模型并打印结构
    model = HeartLang_finetune_base(num_classes=12)
    print(model)
