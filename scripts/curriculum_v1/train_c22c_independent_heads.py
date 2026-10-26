#!/usr/bin/env python3
"""C2.2 independent per-output nonlinear heads on a frozen C2.1 SID backbone."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "heartlang").exists())
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import train_heartlang_sid_regression as base


class IndependentRegressionHeads(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_outputs: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_outputs)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features)
        return torch.cat([head(normalized) for head in self.heads], dim=-1)


def build_model(args, num_outputs: int):
    model = base.HeartLangSIDForRegression(
        num_outputs=num_outputs,
        task_head_type="linear",
        freeze_backbone=True,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        pad_sid=args.pad_sid,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        use_spatial_embedding=args.use_spatial_embedding,
        use_temporal_embedding=args.use_temporal_embedding,
    )
    model.mlp_head = IndependentRegressionHeads(
        embed_dim=args.embed_dim,
        hidden_dim=args.task_head_hidden_dim,
        num_outputs=num_outputs,
        dropout=args.task_head_dropout,
    )
    return model


_original_parse_args = base.parse_args


def parse_args():
    args = _original_parse_args()
    args.task_head_type = "independent_mlp2"
    return args


base.parse_args = parse_args
base.build_model = build_model


if __name__ == "__main__":
    base.main()
