"""Codebook strategies — review Q2.1 / Q7."""

from __future__ import annotations

import torch

from lmf.models.rhca.codebook import GeometricCodebook, LowRankCodebook, build_codebook


def test_lowrank_param_savings():
    v, d, e = 32768, 1664, 256
    cb = LowRankCodebook(v, d, e)
    params = sum(p.numel() for p in cb.parameters())
    flat = v * d
    # Review target: ~8.8M vs 54.6M — comfortably under half.
    assert params < 0.25 * flat


def test_lowrank_embed_and_logits_shapes():
    cb = LowRankCodebook(64, 32, 8)
    ids = torch.randint(0, 64, (2, 5))
    emb = cb.embed(ids)
    assert emb.shape == (2, 5, 32)
    logits = cb.logits(emb)
    assert logits.shape == (2, 5, 64)
    assert torch.isfinite(logits).all()


def test_geometric_roundtrip_recall():
    torch.manual_seed(0)
    cb = GeometricCodebook(48, 32)
    ids = torch.arange(48)
    # Decoding a token's own embedding should rank that token highly.
    recovered = cb.logits(cb.embed(ids)).argmax(-1)
    assert (recovered == ids).float().mean() > 0.5


def test_build_codebook_dispatch():
    assert isinstance(build_codebook("lowrank", 10, 8, 4), LowRankCodebook)
    assert isinstance(build_codebook("geometric", 10, 8, 4), GeometricCodebook)
