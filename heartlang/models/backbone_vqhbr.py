"""VQHBR 编码器/解码器共享的 Transformer backbone。"""

import torch
import torch.nn as nn
from einops import rearrange, repeat, pack


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # 标准 Transformer FFN。
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == embed_dim)

        self.heads = heads
        self.scale = head_dim**-0.5

        self.norm = nn.LayerNorm(embed_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        # Pre-LN 多头自注意力。
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
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
        # 每层都执行 Attention 残差和 FFN 残差。
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        # 通过 1D 卷积把波形片段映射到隐藏维度。
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
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class SpatialEmbedding(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(SpatialEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x, in_chan_matrix):
        # 导联编号对应的空间 embedding。
        spatial_embeddings = self.embed(
            in_chan_matrix
        )  # [batch_size, seq_len, embed_dim]
        return x + spatial_embeddings


class TemporalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(TemporalEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x, time_index_matrix):
        # token 在整段 ECG 中的时间分块 embedding。
        temporal_embeddings = self.embed(
            time_index_matrix
        )  # [batch_size, seq_len, embed_dim]
        return x + temporal_embeddings


# `Encoder=True` 时输入是原始 ECG token，`Encoder=False` 时输入是量化后的 code 向量。
class VqhbrBackbone(nn.Module):

    def __init__(
        self,
        seq_len,
        time_window,
        embed_dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        code_dim=32,
        Encoder=True,
    ):
        super().__init__()
        # 编码器与解码器共用同一套 backbone 结构，仅输入维度不同。
        self.token_embed = TokenEmbedding(
            c_in=time_window if Encoder else code_dim, d_model=embed_dim
        )
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

    def forward_feature(
        self,
        x,
        mask_bool_matrix=None,
        in_chan_matrix=None,
        in_time_matrix=None,
    ):
        # 1. token -> embedding。
        x = self.token_embed(x)
        b, seq_len, embed_dim = x.shape

        if mask_bool_matrix is None:
            mask_bool_matrix = torch.zeros((b, seq_len), dtype=torch.bool).to(x.device)
        # 2. 可选的 mask token 替换。
        mask_tokens = self.mask_token.expand(b, seq_len, -1)
        w = mask_bool_matrix.unsqueeze(-1).type_as(mask_tokens)
        x = x * (1 - w) + mask_tokens * w

        if in_chan_matrix is not None:
            # 3. 注入导联位置信息。
            x = self.spa_embed(x, in_chan_matrix)

        if in_time_matrix is not None:
            # 4. 注入时间位置信息。
            x = self.tem_embed(x, in_time_matrix)

        # 5. 拼接 cls token 并加入绝对位置编码。
        cls_tokens = repeat(self.cls_token, "d -> b d", b=b)
        x, ps = pack([cls_tokens, x], "b * d")
        x += self.pos_embed

        # 6. Transformer 提取上下文特征。
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
        return_qrs_tokens=False,
        return_all_tokens=False,
    ):
        # 纯零 token 是 padding，后面会把这些位置重新置零，避免污染下游统计。
        zero_mask = (x == 0).all(dim=2)
        x = self.forward_feature(
            x,
            mask_bool_matrix,
            in_chan_matrix,
            in_time_matrix,
        )
        if return_all_tokens:
            return x
        x = x[:, 1:, :]
        x[zero_mask] = 0
        if return_qrs_tokens:
            # 返回每个 QRS token 的上下文表示。
            return x
        else:
            # 默认返回全句 token 的平均表示，供重建或分类头使用。
            return x.mean(1)


if __name__ == "__main__":
    model = VqhbrBackbone(
        seq_len=256,
        time_window=96,
        embed_dim=768,
        depth=12,
        heads=10,
        mlp_dim=2048,
        dropout=0.01,
        emb_dropout=0.01,
        Encoder=True,
    )

    input_data = torch.randn(4, 256, 96)
    inchans = torch.randint(0, 13, (4, 256))
    intimes = torch.randint(0, 13, (4, 256))

    out = model(input_data, in_chan_matrix=inchans, in_time_matrix=intimes)
    print(out.shape)
