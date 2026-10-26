"""E2 形状、梯度、padding 与隐藏心拍信息泄漏测试。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ecg_representation_e2 import ECGRepresentationMLM, PaddingSafeTransformer, initialize_matched_core, random_valid_mask, warmstart_from_sid_mlm
from heartlang.models.modeling_pretrain import TokenEmbedding, Transformer


def test_tiny():
    torch.manual_seed(17)
    b, n, v, d = 2, 8, 32, 16
    wave = torch.randn(b, n, 96)
    sid = torch.randint(0, v, (b, n))
    ze = torch.randn(b, n, d)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    sid[~valid] = v
    chans = torch.randint(1, 10, (b, n))
    times = torch.randint(1, 10, (b, n))
    chans[~valid], times[~valid] = 0, 0
    mask = random_valid_mask(valid, 0.5, torch.Generator().manual_seed(6))
    assert not (mask & ~valid).any() and int(mask.sum()) == 4
    matched = []
    for candidate in ("waveform", "sid_only", "hybrid", "ze_only"):
        small = ECGRepresentationMLM(candidate, seq_len=n, vocab_size=v, code_dim=d,
                                     embed_dim=32, depth=2, heads=4, dim_head=8,
                                     mlp_dim=64, dropout=0.0)
        initialize_matched_core(small, 31)
        matched.append(small.transformer.layers[0][0].to_qkv.weight.detach().clone())
    assert all(torch.equal(matched[0], item) for item in matched[1:])
    old = Transformer(32, 2, 4, 8, 64, 0.0).eval()
    safe = PaddingSafeTransformer(32, 2, 4, 8, 64, 0.0).eval()
    safe.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(2, 9, 32)
    assert torch.allclose(old(x), safe(x, torch.ones(2, 9, dtype=torch.bool)), atol=2e-6)
    legacy = TokenEmbedding(96, 32).eval()
    with torch.no_grad():
        legacy.tokenConv.weight.fill_(1.0)
        source = torch.zeros(1, 8, 96)
        mutated = source.clone()
        mutated[:, 3] = 1.0
        # 旧版先跨 token 卷积，隐藏第 3 个心拍后，第 2 个可见位置仍已改变。
        assert not torch.equal(legacy(source)[:, 2], legacy(mutated)[:, 2])
    for mode in sorted(ECGRepresentationMLM.MODES):
        model = ECGRepresentationMLM(mode, seq_len=n, vocab_size=v, code_dim=d,
                                     embed_dim=32, depth=2, heads=4, dim_head=8,
                                     mlp_dim=64, dropout=0.0).eval()
        logits = model(wave, sid, ze, valid, mask, chans, times)
        assert logits.shape == (int(mask.sum()), v)
        loss = model.mlm_loss(wave, sid, ze, valid, mask, chans, times)
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        # 在同一 mask 下只改变被隐藏和 padding 波形/特征，不得改变学生预测。
        wave2, sid2, ze2 = wave.clone(), sid.clone(), ze.clone()
        wave2[mask | ~valid] += 100.0
        sid2[mask] = (sid2[mask] + 7) % v
        ze2[mask | ~valid] += 100.0
        chans2, times2 = chans.clone(), times.clone()
        chans2[~valid], times2[~valid] = 15, 15
        before = model.student_embeddings(wave, sid, ze, valid, mask)
        after = model.student_embeddings(wave2, sid2, ze2, valid, mask)
        assert torch.equal(before, after), mode
        with torch.no_grad():
            a = model(wave, sid, ze, valid, mask, chans, times)
            b_ = model(wave2, sid2, ze2, valid, mask, chans2, times2)
        assert torch.allclose(a, b_, atol=1e-5), mode
        print(f"{mode}: shape/backward/mask/padding/leakage passed", flush=True)


def test_checkpoint(path: Path):
    model = ECGRepresentationMLM("sid_only", spatial=False, temporal=False)
    result = warmstart_from_sid_mlm(model, path)
    assert result["loaded_tensors"] > 100
    print(f"checkpoint loading passed: {result['loaded_tensors']} tensors", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    test_tiny()
    if args.checkpoint:
        test_checkpoint(args.checkpoint)
