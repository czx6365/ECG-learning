"""Portable E12-A continuous-heartbeat sentence encoder.

This module intentionally contains no ECG_DSA dataset paths.  It loads the
published E12-A checkpoint and consumes the continuous ``z_e`` features made
by the matching frozen E3 morphology encoder.
"""

from __future__ import annotations

from pathlib import Path

import torch
from einops import rearrange
from torch import nn

from heartlang.models.modeling_pretrain import Transformer


DEFAULT_CONFIG = {
    "seq_len": 256,
    "code_dim": 64,
    "vocab_size": 1024,
    "embed_dim": 768,
    "depth": 12,
    "heads": 8,
    "dim_head": 64,
    "mlp_dim": 1024,
    "dropout": 0.1,
    "emb_dropout": 0.1,
}


def _masked_transformer_forward(
    transformer: Transformer, x: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Apply the E12-A Transformer while excluding padding keys and queries."""
    key_valid = valid[:, None, None, :]
    query_valid = valid.unsqueeze(-1).to(x.dtype)
    for attn, ff in transformer.layers:
        y = attn.norm(x)
        qkv = attn.to_qkv(y).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=attn.heads), qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * attn.scale
        dots = dots.masked_fill(~key_valid, torch.finfo(dots.dtype).min)
        weights = attn.dropout(attn.attend(dots))
        out = torch.matmul(weights, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        x = (attn.to_out(out) + x) * query_valid
        x = (ff(x) + x) * query_valid
    return x


class E12ABackbone(nn.Module):
    """Continuous ``z_e`` to contextual heartbeat-token representations."""

    def __init__(self, **cfg):
        super().__init__()
        self.seq_len = int(cfg["seq_len"])
        self.code_dim = int(cfg["code_dim"])
        self.embed_dim = int(cfg["embed_dim"])
        self.input_proj = nn.Linear(self.code_dim, self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.randn(self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.seq_len + 1, self.embed_dim)
        )
        self.dropout = nn.Dropout(float(cfg["emb_dropout"]))
        self.transformer = Transformer(
            embed_dim=self.embed_dim,
            depth=int(cfg["depth"]),
            heads=int(cfg["heads"]),
            dim_head=int(cfg["dim_head"]),
            mlp_dim=int(cfg["mlp_dim"]),
            dropout=float(cfg["dropout"]),
        )
        self.norm_layer = nn.LayerNorm(self.embed_dim)

    def forward_feature(
        self,
        ze: torch.Tensor,
        valid: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ze.ndim != 3 or tuple(ze.shape[1:]) != (self.seq_len, self.code_dim):
            raise ValueError(
                f"Expected z_e [B,{self.seq_len},{self.code_dim}], got {tuple(ze.shape)}"
            )
        valid = valid.bool()
        x = self.input_proj(ze.float()) * valid.unsqueeze(-1).to(ze.dtype)
        masked = torch.zeros_like(valid) if mask is None else mask.bool() & valid
        x = torch.where(masked.unsqueeze(-1), self.mask_token.expand_as(x), x)
        cls = self.cls_token.view(1, 1, -1).expand(len(x), 1, -1)
        x = torch.cat([cls, x], dim=1)
        full_valid = torch.cat(
            [torch.ones(len(x), 1, dtype=torch.bool, device=x.device), valid], dim=1
        )
        x = self.dropout(x + self.pos_embed)
        x = _masked_transformer_forward(self.transformer, x, full_valid)
        return self.norm_layer(x) * full_valid.unsqueeze(-1).to(x.dtype)


class E12AForMaskedSID(nn.Module):
    """Published E12-A backbone plus its 1024-class masked-SID head."""

    def __init__(self, **cfg):
        super().__init__()
        self.config = dict(cfg)
        self.backbone = E12ABackbone(**cfg)
        self.lm_head = nn.Linear(int(cfg["embed_dim"]), int(cfg["vocab_size"]))

    def forward(
        self, ze: torch.Tensor, valid: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        tokens = self.backbone.forward_feature(ze, valid, mask)[:, 1:]
        return self.lm_head(tokens[mask.bool() & valid.bool()])

    def extract_tokens(self, ze: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_feature(ze, valid)[:, 1:]

    def pooled_mean_max(self, ze: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        tokens = self.extract_tokens(ze, valid)
        valid = valid.bool()
        count = valid.sum(1).clamp_min(1).unsqueeze(-1)
        mean = (tokens * valid.unsqueeze(-1)).sum(1) / count
        maximum = tokens.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
        maximum[~valid.any(1)] = 0
        return torch.cat([mean, maximum], dim=-1)


def load_e12a(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    *,
    train: bool = False,
) -> E12AForMaskedSID:
    """Load the published checkpoint with strict architecture validation."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise KeyError("E12-A checkpoint is missing the 'model' state dictionary")
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(payload.get("config", {}))
    model = E12AForMaskedSID(**cfg)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.train(train)
    return model
