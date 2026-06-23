"""Correctness gates for the Spectral Memory (SM-LM) family — the spec's S15.

Covers: chunk == recurrent equivalence (the crux), strict causality, gradient
finiteness, shapes, tiny overfit, parallel == incremental-decode, and the
banded-decay invariant.
"""

from __future__ import annotations

import torch

from lmf.core.registry import MODELS, TRAINERS
from lmf.models.spectral_memory import (
    SpectralMemoryConfig,
    SpectralMemoryLM,
    delta_rule_chunked,
    delta_rule_recurrent,
)


def _model(**overrides) -> SpectralMemoryLM:
    cfg = dict(vocab_size=64, dim=32, layers=4, banks=3, head_dim=16, attn_heads=4,
               window=8, chunk=8, mlp_ratio=2)
    cfg.update(overrides)
    return SpectralMemoryLM(SpectralMemoryConfig(**cfg))


# --- S15.2: the chunk-parallel path must equal the recurrent reference --------

def test_chunked_matches_recurrent_no_segments():
    torch.manual_seed(0)
    n, t, d = 6, 37, 16
    q = torch.randn(n, t, d)
    k = torch.nn.functional.normalize(torch.randn(n, t, d), dim=-1)
    v = torch.randn(n, t, d)
    log_a = -torch.rand(n, t) * 0.5          # decays in a stable range
    beta = torch.sigmoid(torch.randn(n, t))
    for chunk in (1, 4, 8, 64):
        o_c, s_c = delta_rule_chunked(q, k, v, log_a, beta, chunk=chunk)
        o_r, s_r = delta_rule_recurrent(q, k, v, log_a, beta)
        assert torch.allclose(o_c, o_r, atol=1e-4), f"output mismatch at chunk={chunk}"
        assert torch.allclose(s_c, s_r, atol=1e-4), f"state mismatch at chunk={chunk}"


def test_chunked_matches_recurrent_with_segments_and_state():
    torch.manual_seed(1)
    n, t, d = 4, 25, 12
    q = torch.randn(n, t, d)
    k = torch.nn.functional.normalize(torch.randn(n, t, d), dim=-1)
    v = torch.randn(n, t, d)
    log_a = -torch.rand(n, t) * 0.3
    beta = torch.sigmoid(torch.randn(n, t))
    # two contiguous segments per row, with a non-trivial incoming state
    segs = torch.zeros(n, t, dtype=torch.long)
    segs[:, t // 2:] = 1
    state = torch.randn(n, d, d) * 0.1
    o_c, s_c = delta_rule_chunked(q, k, v, log_a, beta, segs, chunk=7, state=state)
    o_r, s_r = delta_rule_recurrent(q, k, v, log_a, beta, segs, state=state)
    assert torch.allclose(o_c, o_r, atol=1e-4)
    assert torch.allclose(s_c, s_r, atol=1e-4)


# --- S15.1: strict causality --------------------------------------------------

def test_causality_future_does_not_leak():
    torch.manual_seed(0)
    model = _model().eval()
    tokens = torch.randint(0, 64, (2, 24))
    cut = 10
    logits1, _ = model(tokens)
    perturbed = tokens.clone()
    perturbed[:, cut + 1:] = torch.randint(0, 64, perturbed[:, cut + 1:].shape)
    logits2, _ = model(perturbed)
    assert torch.allclose(logits1[:, : cut + 1], logits2[:, : cut + 1], atol=1e-4)


# --- S15.3 / S15.4: gradients + shapes ---------------------------------------

def test_training_step_shapes_and_finite_grads():
    model = _model()
    tokens = torch.randint(0, 64, (3, 20))
    logits, _ = model(tokens)
    assert logits.shape == (3, 20, 64)
    losses = model.training_step(tokens)
    assert "total" in losses and torch.isfinite(losses["total"]).all()
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads)


# --- S15.5: tiny overfit ------------------------------------------------------

def test_tiny_overfit():
    torch.manual_seed(0)
    model = _model(layers=3)
    tokens = torch.randint(0, 64, (2, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = model.training_step(tokens)["total"].item()
    for _ in range(200):
        opt.zero_grad()
        loss = model.training_step(tokens)["total"]
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first, f"failed to overfit: {first:.3f} -> {loss.item():.3f}"


# --- parallel path must equal incremental decode (cache correctness) ----------

def test_parallel_matches_incremental_decode():
    torch.manual_seed(0)
    model = _model().eval()
    tokens = torch.randint(0, 64, (2, 13))
    full, _ = model(tokens)
    # feed one token at a time through the KV/state cache
    logits, caches = model(tokens[:, :1], use_cache=True)
    step_logits = [logits[:, -1]]
    for t in range(1, tokens.shape[1]):
        logits, caches = model(tokens[:, t : t + 1], caches=caches, use_cache=True)
        step_logits.append(logits[:, -1])
    incremental = torch.stack(step_logits, dim=1)
    assert torch.allclose(full, incremental, atol=1e-3)


def test_generate_shape():
    model = _model().eval()
    out = model.generate(torch.randint(0, 64, (2, 5)), 7)
    assert out.shape == (2, 7)


# --- banded-decay invariant: each bank stays inside its half-life band --------

def test_decay_bands_are_ordered_and_disjoint():
    model = _model(banks=4)
    mtdm = next(b.mixer for b in model.blocks if not b.is_attention)
    lo, hi = mtdm.hl_lo, mtdm.hl_hi
    assert torch.all(lo < hi)                       # each band non-empty
    assert torch.all(lo[1:] >= hi[:-1] - 1e-3)      # bands tile without overlap
    assert lo[0].item() <= 4.0 + 1e-3 and hi[-1].item() >= 2048.0 - 1e-3


# --- ablation variants all build, run, and learn ------------------------------

def test_ablation_variants_forward_and_backward():
    import pytest

    variants = [
        dict(router="sigmoid"), dict(router="softmax"), dict(router="none"),
        dict(decay_mode="banded"), dict(decay_mode="free"),
        dict(attention_layers=[]),          # 0 attention layers (pure-linear)
        dict(attention_layers=[2]),         # 1 attention layer
        dict(attention_layers=[1, 3]),      # 2 attention layers
    ]
    for overrides in variants:
        model = _model(**overrides)
        tokens = torch.randint(0, 64, (2, 18))
        losses = model.training_step(tokens)
        assert torch.isfinite(losses["total"]).all(), overrides
        losses["total"].backward()


def test_router_none_has_no_router_params():
    gated = _model(router="sigmoid")
    summed = _model(router="none")
    assert sum(p.numel() for p in gated.parameters()) > sum(p.numel() for p in summed.parameters())


def test_free_decay_spans_full_range():
    model = _model(banks=4, decay_mode="free")
    mtdm = next(b.mixer for b in model.blocks if not b.is_attention)
    assert torch.allclose(mtdm.hl_lo, torch.full_like(mtdm.hl_lo, 4.0))
    assert torch.allclose(mtdm.hl_hi, torch.full_like(mtdm.hl_hi, 2048.0))


# --- registry wiring ----------------------------------------------------------

def test_registered_in_model_and_trainer_registries():
    assert "spectral_memory" in MODELS
    assert "spectral_memory" in TRAINERS
    model = MODELS.create("spectral_memory", {"dim": 32, "layers": 2}, 50)
    assert model.config.vocab_size == 50
    manifest = model.architecture_manifest()
    assert manifest["name"] == "SpectralMemoryLM"
    assert manifest["invariants"]["delta_memory"] is True
