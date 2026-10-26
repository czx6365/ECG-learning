"""四种严格防泄漏的 ECG Sentence 输入，供 E2 公平预训练使用。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from heartlang.models.modeling_pretrain import Transformer


class PaddingSafeTransformer(Transformer):
    """沿用原权重键名，在注意力 softmax 前排除 padding key。"""

    def forward(self, x: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        if key_valid.shape != (b, n) or not key_valid[:, 0].all():
            raise ValueError("CLS 必须有效，key_valid 应为 [B,L+1]")
        for attn, ff in self.layers:
            q, k, v = attn.to_qkv(attn.norm(x)).chunk(3, dim=-1)
            h = attn.heads
            q, k, v = (t.reshape(b, n, h, -1).transpose(1, 2) for t in (q, k, v))
            logits = (q @ k.transpose(-1, -2)) * attn.scale
            logits = logits.masked_fill(~key_valid[:, None, None, :], torch.finfo(logits.dtype).min)
            weights = attn.dropout(attn.attend(logits))
            attended = (weights @ v).transpose(1, 2).reshape(b, n, -1)
            x = x + attn.to_out(attended)
            x = x + ff(x)
        return x


class ECGRepresentationMLM(nn.Module):
    """A 波形、B SID、C SID+z_e、D z_e 使用相同句子主干。"""

    MODES = {"waveform", "sid_only", "hybrid", "ze_only"}

    def __init__(self, mode: str, *, seq_len: int = 256, token_len: int = 96,
                 vocab_size: int = 1024, code_dim: int = 64, embed_dim: int = 768,
                 depth: int = 12, heads: int = 8, dim_head: int = 64,
                 mlp_dim: int = 1024, dropout: float = 0.1,
                 spatial: bool = True, temporal: bool = True):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode = mode
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_sid = vocab_size
        self.embed_dim = embed_dim
        self.wave_embed = nn.Linear(token_len, embed_dim) if mode == "waveform" else None
        self.sid_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size) if mode in {"sid_only", "hybrid"} else None
        self.ze_embed = nn.Linear(code_dim, embed_dim, bias=False) if mode in {"hybrid", "ze_only"} else None
        self.input_norm = nn.LayerNorm(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))
        self.spa_embed = nn.Embedding(16, embed_dim) if spatial else None
        self.tem_embed = nn.Embedding(16, embed_dim) if temporal else None
        self.dropout = nn.Dropout(dropout)
        self.transformer = PaddingSafeTransformer(embed_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.norm_layer = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def student_embeddings(self, wave: torch.Tensor, sid: torch.Tensor, ze: torch.Tensor,
                           valid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """先隐藏目标心拍的全部模态，再构建学生输入。"""
        if mask.shape != valid.shape or (mask & ~valid).any():
            raise ValueError("mask 只能选择有效心拍")
        hidden = mask | ~valid
        if self.mode == "waveform":
            # 每个心拍独立投影；隐藏波形在进入投影层前即清零。
            tokens = self.wave_embed(wave.masked_fill(hidden[..., None], 0.0))
        elif self.mode == "sid_only":
            safe_sid = sid.masked_fill(hidden, self.pad_sid)
            tokens = self.sid_embed(safe_sid)
        elif self.mode == "hybrid":
            safe_sid = sid.masked_fill(hidden, self.pad_sid)
            safe_ze = ze.masked_fill(hidden[..., None], 0.0)
            tokens = self.sid_embed(safe_sid) + self.ze_embed(safe_ze)
        else:
            tokens = self.ze_embed(ze.masked_fill(hidden[..., None], 0.0))
        tokens = self.input_norm(tokens)
        tokens = torch.where(mask[..., None], self.mask_token.expand_as(tokens), tokens)
        return tokens.masked_fill((~valid)[..., None], 0.0)

    def forward(self, wave: torch.Tensor, sid: torch.Tensor, ze: torch.Tensor,
                valid: torch.Tensor, mask: torch.Tensor, chans: torch.Tensor,
                times: torch.Tensor, *, return_features: bool = False):
        if wave.shape[:2] != (sid.shape[0], self.seq_len) or sid.shape != valid.shape:
            raise ValueError("期望波形 [B,L,96] 与 SID/valid [B,L]")
        tokens = self.student_embeddings(wave, sid, ze, valid, mask)
        if self.spa_embed is not None:
            tokens = tokens + self.spa_embed(chans.long().clamp(0, 15))
        if self.tem_embed is not None:
            tokens = tokens + self.tem_embed(times.long().clamp(0, 15))
        cls = self.cls_token.view(1, 1, -1).expand(wave.shape[0], 1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, : self.seq_len + 1]
        key_valid = torch.cat([torch.ones(wave.shape[0], 1, dtype=torch.bool, device=wave.device), valid], dim=1)
        hidden = self.norm_layer(self.transformer(self.dropout(x), key_valid))
        if return_features:
            return hidden, key_valid
        return self.lm_head(hidden[:, 1:][mask])

    def mlm_loss(self, wave, sid, ze, valid, mask, chans, times):
        if not mask.any():
            raise ValueError("本 batch 没有被遮住的有效心拍")
        logits = self(wave, sid, ze, valid, mask, chans, times)
        return F.cross_entropy(logits.float(), sid[mask].long())


def random_valid_mask(valid: torch.Tensor, ratio: float, generator: torch.Generator | None = None):
    """逐条样本对有效心拍采样；padding 永不成为 MLM 标签。"""
    out = torch.zeros_like(valid)
    for row in range(len(valid)):
        indices = torch.nonzero(valid[row], as_tuple=False).flatten()
        if len(indices):
            count = max(1, round(len(indices) * ratio))
            chosen = torch.randperm(len(indices), device=valid.device, generator=generator)[:count]
            out[row, indices[chosen]] = True
    return out


@torch.no_grad()
def encode_independent_teacher(tokenizer, wave, chans, times, code_dim=64):
    """仅支持不读取其他心拍的 MorphologyEncoder tokenizer。"""
    valid, _beats, ze, zq, ids, _loss = tokenizer.encode_valid(wave, chans, times)
    ze_full = wave.new_zeros(*wave.shape[:2], code_dim)
    zq_full = wave.new_zeros(*wave.shape[:2], code_dim)
    sid_full = torch.full(wave.shape[:2], tokenizer.quantize.num_tokens, device=wave.device, dtype=torch.long)
    ze_full[valid], zq_full[valid], sid_full[valid] = ze, zq, ids
    return valid, ze_full, zq_full, sid_full


def warmstart_from_sid_mlm(model: ECGRepresentationMLM, checkpoint):
    """仅复制形状兼容的 E3 句子权重，保留新输入层随机初始化。"""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload["model"]
    current = model.state_dict()
    mapped = {}
    for name, value in source.items():
        candidate = name.removeprefix("backbone.")
        if candidate in current and current[candidate].shape == value.shape:
            mapped[candidate] = value
    required = ["cls_token", "pos_embed", "norm_layer.weight", "norm_layer.bias", "lm_head.weight", "lm_head.bias"]
    required += [name for name in current if name.startswith("transformer.")]
    if model.sid_embed is not None:
        required.append("sid_embed.weight")
    missing = sorted(set(required) - set(mapped))
    if missing:
        raise ValueError(f"E3 checkpoint 缺少兼容权重: {missing[:5]}")
    current.update(mapped)
    model.load_state_dict(current, strict=True)
    return {"loaded_tensors": len(mapped), "new_input_tensors": sorted(set(current) - set(mapped))}


def initialize_matched_core(model: ECGRepresentationMLM, seed: int):
    """同种子四架构使用逐张量一致的句子主干随机初始权重。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = ECGRepresentationMLM("sid_only", seq_len=model.seq_len,
                                         vocab_size=model.vocab_size, embed_dim=model.embed_dim,
                                         depth=len(model.transformer.layers),
                                         heads=model.transformer.layers[0][0].heads,
                                         dim_head=model.transformer.layers[0][0].to_qkv.out_features // (3 * model.transformer.layers[0][0].heads),
                                         mlp_dim=model.transformer.layers[0][1].net[1].out_features,
                                         spatial=model.spa_embed is not None,
                                         temporal=model.tem_embed is not None)
    current, template = model.state_dict(), reference.state_dict()
    shared = {name: tensor for name, tensor in template.items()
              if name in current and current[name].shape == tensor.shape
              and not name.startswith(("wave_embed.", "ze_embed."))}
    required = [name for name in current if name.startswith("transformer.")]
    if not set(required) <= set(shared):
        raise RuntimeError("共享 Transformer 初始化不完整")
    current.update(shared)
    model.load_state_dict(current, strict=True)
    return {"matched_core_seed": seed, "shared_tensors": len(shared)}
