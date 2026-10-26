"""HeartLang over discrete ECG SID tokens.

This module keeps SID assignment outside the language model. The model only
receives already-generated morphology SID ids, then adds lead/time/position
context before the Transformer.
"""

from __future__ import annotations

import torch
from einops import pack, repeat
from timm.models import register_model
from torch import nn

from heartlang.models.modeling_pretrain import SpatialEmbedding, TemporalEmbedding, Transformer


class HeartLangSIDBackbone(nn.Module):
    def __init__(
        self,
        seq_len: int = 256,
        vocab_size: int = 1024,
        embed_dim: int = 768,
        depth: int = 12,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        pad_sid: int | None = None,
        use_spatial_embedding: bool = True,
        use_temporal_embedding: bool = True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.pad_sid = int(vocab_size if pad_sid is None else pad_sid)
        self.use_spatial_embedding = bool(use_spatial_embedding)
        self.use_temporal_embedding = bool(use_temporal_embedding)
        if self.pad_sid < self.vocab_size:
            raise ValueError("pad_sid must be outside the normal vocabulary range.")

        self.sid_embed = nn.Embedding(self.pad_sid + 1, embed_dim, padding_idx=self.pad_sid)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len + 1, embed_dim))
        self.spa_embed = (
            SpatialEmbedding(num_embeddings=16, embed_dim=embed_dim)
            if self.use_spatial_embedding else None
        )
        self.tem_embed = (
            TemporalEmbedding(num_embeddings=16, embed_dim=embed_dim)
            if self.use_temporal_embedding else None
        )
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
        self.depth = int(depth)
        self.embed_dim = int(embed_dim)

    def valid_sid_mask(self, sid_ids, in_chan_matrix=None, in_time_matrix=None):
        valid = (sid_ids >= 0) & (sid_ids < self.vocab_size)
        if self.use_spatial_embedding and in_chan_matrix is not None:
            valid = valid & (in_chan_matrix > 0)
        if self.use_temporal_embedding and in_time_matrix is not None:
            valid = valid & (in_time_matrix > 0)
        return valid

    def forward_feature(
        self,
        sid_ids,
        mask_bool_matrix=None,
        in_chan_matrix=None,
        in_time_matrix=None,
    ):
        sid_ids = sid_ids.long()
        valid = self.valid_sid_mask(sid_ids, in_chan_matrix, in_time_matrix)
        safe_sid_ids = torch.where(valid, sid_ids, torch.full_like(sid_ids, self.pad_sid))
        x = self.sid_embed(safe_sid_ids)
        b, seq_len, _ = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}.")

        if mask_bool_matrix is None:
            mask_bool_matrix = torch.zeros((b, seq_len), dtype=torch.bool, device=x.device)
        else:
            mask_bool_matrix = mask_bool_matrix.bool() & valid
        mask_tokens = self.mask_token.expand(b, seq_len, -1)
        w = mask_bool_matrix.unsqueeze(-1).type_as(mask_tokens)
        x = x * (1 - w) + mask_tokens * w

        if self.spa_embed is not None and in_chan_matrix is not None:
            x = self.spa_embed(x, in_chan_matrix.long().clamp_min(0).clamp_max(15))
        if self.tem_embed is not None and in_time_matrix is not None:
            x = self.tem_embed(x, in_time_matrix.long().clamp_min(0).clamp_max(15))

        cls_tokens = repeat(self.cls_token, "d -> b d", b=b)
        x, _ = pack([cls_tokens, x], "b * d")
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm_layer(x)
        return x, valid

    def forward(
        self,
        sid_ids,
        mask_bool_matrix=None,
        in_chan_matrix=None,
        in_time_matrix=None,
        return_all_tokens=False,
        return_qrs_tokens=False,
    ):
        x, valid = self.forward_feature(
            sid_ids,
            mask_bool_matrix=mask_bool_matrix,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
        )
        if return_all_tokens:
            return x
        tokens = x[:, 1:, :]
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)
        if return_qrs_tokens:
            return tokens
        return x[:, 0]


class HeartLangSIDForMLM(nn.Module):
    def __init__(self, vocab_size: int = 1024, pad_sid: int | None = None, **kwargs):
        super().__init__()
        self.backbone = HeartLangSIDBackbone(
            vocab_size=vocab_size,
            pad_sid=pad_sid,
            **kwargs,
        )
        self.lm_head = nn.Linear(self.backbone.embed_dim, vocab_size)
        self.vocab_size = int(vocab_size)

    def forward(
        self,
        sid_ids,
        mask_bool_matrix=None,
        in_chan_matrix=None,
        in_time_matrix=None,
        return_all_tokens=False,
        return_qrs_tokens=False,
    ):
        if return_all_tokens or return_qrs_tokens:
            return self.backbone(
                sid_ids,
                mask_bool_matrix=mask_bool_matrix,
                in_chan_matrix=in_chan_matrix,
                in_time_matrix=in_time_matrix,
                return_all_tokens=return_all_tokens,
                return_qrs_tokens=return_qrs_tokens,
            )
        x, valid = self.backbone.forward_feature(
            sid_ids,
            mask_bool_matrix=mask_bool_matrix,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
        )
        tokens = x[:, 1:, :]
        if mask_bool_matrix is None:
            mask_bool_matrix = valid
        else:
            mask_bool_matrix = mask_bool_matrix.bool() & valid
        return self.lm_head(tokens[mask_bool_matrix])

    def extract_features(self, sid_ids, in_chan_matrix=None, in_time_matrix=None):
        return self.backbone(
            sid_ids,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
        )

    def get_num_layers(self):
        return self.backbone.depth

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"backbone.pos_embed", "backbone.cls_token", "backbone.mask_token"}


class HeartLangSIDForRegression(nn.Module):
    def __init__(
        self,
        num_outputs: int = 4,
        task_head_type: str = "linear",
        task_head_hidden_dim: int = 256,
        task_head_dropout: float = 0.1,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.backbone = HeartLangSIDBackbone(**kwargs)
        if task_head_type == "linear":
            self.mlp_head = nn.Linear(self.backbone.embed_dim, num_outputs)
        elif task_head_type == "mlp2":
            self.mlp_head = nn.Sequential(
                nn.Linear(self.backbone.embed_dim, task_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(task_head_dropout),
                nn.Linear(task_head_hidden_dim, num_outputs),
            )
        else:
            raise ValueError(f"Unsupported task_head_type: {task_head_type}")
        if freeze_backbone:
            self.freeze_backbone()

    def extract_features(self, sid_ids, in_chan_matrix=None, in_time_matrix=None):
        return self.backbone(
            sid_ids,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
        )

    def forward(self, sid_ids, in_chan_matrix=None, in_time_matrix=None):
        return self.mlp_head(self.extract_features(sid_ids, in_chan_matrix, in_time_matrix))

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_num_layers(self):
        return self.backbone.depth

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"backbone.pos_embed", "backbone.cls_token", "backbone.mask_token"}


def default_sid_params():
    return dict(
        seq_len=256,
        vocab_size=1024,
        embed_dim=768,
        depth=12,
        heads=8,
        dim_head=64,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
        pad_sid=1024,
        use_spatial_embedding=True,
        use_temporal_embedding=True,
    )


@register_model
def HeartLangSID(pretrained=False, **kwargs):
    del pretrained
    config = default_sid_params()
    config.update({k: v for k, v in kwargs.items() if k in config})
    return HeartLangSIDForMLM(**config)


@register_model
def HeartLangSID_finetune_base(pretrained=False, **kwargs):
    del pretrained
    config = default_sid_params()
    config.update({k: v for k, v in kwargs.items() if k in config})
    num_outputs = kwargs.get("num_classes", kwargs.get("num_outputs", 4))
    return HeartLangSIDForRegression(
        num_outputs=num_outputs,
        task_head_type=kwargs.get("task_head_type", "linear"),
        task_head_hidden_dim=kwargs.get("task_head_hidden_dim", 256),
        task_head_dropout=kwargs.get("task_head_dropout", 0.1),
        **config,
    )
