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

from typing import Tuple
import math
import torch
from timm import create_model
from timm.models import register_model
from torch import nn
from einops import rearrange, repeat, pack
from heartlang.models import modeling_vqhbr as modeling_vqhbr


def random_masking(x, mask_ratio):
    """
    输入 x: [bs, seq_len, embed_dim]
    batch 有 4 个样本
    每个样本有 256 个 token
    每个 token 是长度为 96 的一小段心电波形；
    mask_ratio: float
    输出 mask: (bool)[bs, seq_len]

    为每个样本随机生成一个 mask 矩阵，决定哪些 token 被遮住
    """
    bs, seq_len, dim = x.shape
    len_keep = int(seq_len * (1 - mask_ratio))#保留token
    noise = torch.rand(bs, seq_len, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.ones([bs, seq_len], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return mask.to(torch.bool)


class FeedForward(nn.Module):#前馈层
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # 标准 Transformer MLP block。输入维度 dim，先升维到 hidden_dim，再降回 dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),#归一化
            nn.Linear(dim, hidden_dim),#把特征维度扩大，增强表达能力
            nn.GELU(),#非线性激活函数
            nn.Dropout(dropout),#防止过拟合
            nn.Linear(hidden_dim, dim),#投影回原维度
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, heads=8, head_dim=64, dropout=0.0):
        #heads：多头注意力头数  head_dim：每个头的维度
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == embed_dim)

        self.heads = heads
        self.scale = head_dim**-0.5#缩放

        self.norm = nn.LayerNorm(embed_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()#什么都不做
        )

    def forward(self, x):
        # Pre-LN self-attention。
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)#沿最后一个维度 切成 3 份
        #得到的就是：第一份 q第二份 k第三份 v
        
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
#map(函数, 可迭代对象) 的意思是：把“函数”作用到“可迭代对象”的每个元素上
# 原来：b = batch size  n = token 数  (h d) = 最后一维实际上由 h 个头、每头 d 维组成
# 变成：b h n d
# 也就是把最后一维拆成：头数 h  每头维度 d

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#k.transpose(-1, -2)交换倒数第一和倒数第二的维度
#torch.matmul(q, ...) 矩阵乘法
#计算每个 token 和其他 token 的相关性  得到“谁该关注谁”的分数矩阵

        attn = self.attend(dots)#softmax 把分数变成概率/权重 表示“当前 token 对其他 token 各关注多少”
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)#按照注意力权重，把别的位置的信息加权汇总过来
        out = rearrange(out, "b h n d -> b n (h d)")#把多个 head 的结果拼接起来
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])#创建一个空的模块列表 用来存很多层网络
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            embed_dim=embed_dim,
                            heads=heads,
                            head_dim=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(embed_dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        # 每层都采用残差连接：Attention + FFN。
        #x -> Attention -> 残差相加 -> FeedForward -> 残差相加
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        
        # 输入本来是每个时间位置一个原始特征向量
        # 这层用 1D 卷积（Conv1d）
        # 把它变成 Transformer 能处理的 embedding 向量
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(#给权重一个合理的随机初值
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        """输入格式 [b, seq_len, c_in]
            改成 Conv1d 需要的 [b, c_in, seq_len]
            做 1D 卷积
            再改回 Transformer 常用的 [b, seq_len, d_model]
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class SpatialEmbedding(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(SpatialEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x, in_chan_matrix):
        # 加入导联/空间位置信息。
        spatial_embeddings = self.embed(
            in_chan_matrix
        )  # [batch_size, seq_len, embed_dim]
        return x + spatial_embeddings


class TemporalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(TemporalEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x, time_index_matrix):
        # 加入 token 在整段 ECG 中的时间位置信息。
        temporal_embeddings = self.embed(
            time_index_matrix
        )  # [batch_size, seq_len, embed_dim]
        return x + temporal_embeddings


class ST_ECGFormer(nn.Module):
    """把每个波形窗口映射成向量
    对部分位置做 mask
    加入导联信息（spatial）和时间信息（temporal）
    加入 cls token 和位置编码
    用 Transformer 建模 token 之间关系
    对 mask 掉的位置做分类预测，预测它原本属于哪个 codebook 词"""
    def __init__(
        self,seq_len,time_window,depth,embed_dim, heads, mlp_dim, dim_head=64, vocab_size=1024, dropout=0.1,
        emb_dropout=0.1, aux_regression_dim=0,
    ):
        super().__init__()
        # 输入是 [bs, seq_len, time_window] 的 ECG token 序列，
        # 输出是每个 token 的上下文表示，或被 mask 位置的词表预测。
        self.token_embed = TokenEmbedding(c_in=time_window, d_model=embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(embed_dim))

        # learnable position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

        self.spa_embed = SpatialEmbedding(num_embeddings=16, embed_dim=embed_dim)
        self.tem_embed = TemporalEmbedding(num_embeddings=16, embed_dim=embed_dim)

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.norm_layer = nn.LayerNorm(embed_dim)

        # 把 Transformer 输出映射到 codebook 词表大小。
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.aux_regression_dim = int(aux_regression_dim)
        self.aux_regression_head = (
            nn.Linear(embed_dim, self.aux_regression_dim)
            if self.aux_regression_dim > 0
            else None
        )

    def forward_feature(
        self, x, mask_bool_matrix=None, in_chan_matrix=None, in_time_matrix=None
    ):
        # 1. 波形 token -> 连续 embedding。
        x = self.token_embed(x)
        b, seq_len, embed_dim = x.shape

        if mask_bool_matrix is None:
            mask_bool_matrix = torch.zeros((b, seq_len), dtype=torch.bool).to(x.device)
        # 2. 被 mask 的位置统一替换成可学习的 mask token。
        mask_tokens = self.mask_token.expand(b, seq_len, -1)
        w = mask_bool_matrix.unsqueeze(-1).type_as(mask_tokens)
        x = x * (1 - w) + mask_tokens * w

        if in_chan_matrix is not None:
            # 3. 注入导联信息。
            x = self.spa_embed(x, in_chan_matrix)

        if in_time_matrix is not None:
            # 4. 注入时间位置信息。
            x = self.tem_embed(x, in_time_matrix)

        # 5. 在序列前面拼上 cls token，并加入可学习绝对位置编码。
        cls_tokens = repeat(self.cls_token, "d -> b d", b=b)
        x, ps = pack([cls_tokens, x], "b * d")
        x += self.pos_embed

        # 6. 经过 Transformer 编码上下文关系。
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm_layer(x)

        return x

    def forward(
        self,
        x,
        mask_bool_matrix=None,
        in_chan_matrix=None,
        in_time_matrix=None,
        return_all_tokens=False,
        return_qrs_tokens=False,
        return_aux_regression=False,
        return_lead_region_tokens=False,
    ):
        x = self.forward_feature(x, mask_bool_matrix, in_chan_matrix, in_time_matrix)
        if return_all_tokens:
            # 返回包含 cls token 在内的完整序列表示。
            return x
        cls_token = x[:, 0]
        x = x[:, 1:]
        if return_qrs_tokens:
            # 返回去掉 cls 之后的全部 ECG token 表示。
            return x
        # 预训练默认只对被 mask 的位置做词表分类。
        mlm_logits = self.lm_head(x[mask_bool_matrix])
        if not return_aux_regression and not return_lead_region_tokens:
            return mlm_logits
        outputs = {"mlm_logits": mlm_logits}
        if self.aux_regression_head is not None:
            outputs["aux_regression"] = self.aux_regression_head(cls_token)
        if return_lead_region_tokens:
            outputs["qrs_tokens"] = x
        return outputs


def get_model_default_params():
    return dict(
        seq_len=256,
        time_window=96,
        embed_dim=768,
        depth=8,
        heads=4,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )


@register_model
def HeartLang(pretrained=False, **kwargs):
    # 注册到 timm 后，可以通过 create_model("HeartLang") 直接构建。
    config = get_model_default_params()
    if "vocab_size" in kwargs:
        config["vocab_size"] = kwargs["vocab_size"]
        _ = kwargs.pop("vocab_size")
    else:
        config["vocab_size"] = 8192
    config["aux_regression_dim"] = kwargs.get("aux_regression_dim", 0)

    config["depth"] = 12
    config["heads"] = 8

    config["mlp_dim"] = 1024

    model = ST_ECGFormer(**config)
    if pretrained:
        # 这里用于下游任务加载已经训练好的 HeartLang 权重。
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == "__main__":
    sample = torch.randn(4, 256, 96)
    mask_bool_matrix = random_masking(sample, 0.5)
    in_chan_matrix = torch.randint(0, 13, (4, 256))
    in_time_matrix = torch.randint(0, 11, (4, 256))
    print("mask_bool_matrix: ", mask_bool_matrix.shape)

    model = create_model("HeartLang", pretrained=False)
    out = model(sample, mask_bool_matrix, in_chan_matrix, in_time_matrix)
    print("output: ", out.shape)

    vqhbr = create_model("vqhbr", pretrained=False, n_code=1024)
    input_ids = vqhbr.get_codebook_indices(sample, in_chan_matrix, in_time_matrix)
    labels = input_ids[mask_bool_matrix]
    print(labels)
    print("vqhbr output shape:", labels.shape)
    loss = nn.CrossEntropyLoss()(out, labels)
    print("Loss:", loss.item())
