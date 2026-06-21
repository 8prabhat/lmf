"""Diagnostics: profiling, health, sensitivity, and merged verdicts.

Generic over model family -- exercised against a tiny ``CachedTransformerLM``
and a tiny ``OPETTransformerLM``, with no model-specific code in
``lmf.diagnostics``.
"""

from __future__ import annotations

import torch

from lmf.data import ProceduralCorpus
from lmf.diagnostics import (
    cache_bytes,
    component_report,
    diagnose,
    health_report,
    parameter_count,
    profile_model,
    sensitivity_report,
)
from lmf.models.opet import OPETTransformerConfig, OPETTransformerLM
from lmf.models.transformer import CachedTransformerLM, TransformerConfig


def _transformer() -> CachedTransformerLM:
    torch.manual_seed(0)
    return CachedTransformerLM(TransformerConfig(vocab_size=64, dim=32, layers=2, heads=4))


def _opet() -> OPETTransformerLM:
    torch.manual_seed(0)
    return OPETTransformerLM(OPETTransformerConfig(
        vocab_size=64, dim=32, layers=2, heads=4, context_window=2, n_freq_bands=4, dropout=0.0))


def test_profile_model_transformer():
    model = _transformer()
    batch = torch.randint(3, 64, (2, 16))
    profile = profile_model(model, batch, n_warmup=1, n_iters=2)
    assert "blocks.0.qkv" in profile
    entry = profile["blocks.0.qkv"]
    assert entry["n_params"] > 0
    assert entry["fwd_ms"] >= 0.0
    assert entry["bwd_ms"] >= 0.0
    # Percentages are shares of the total across profiled modules.
    assert 0.0 <= entry["fwd_pct"] <= 100.0


def test_health_report_transformer():
    model = _transformer()
    batch = torch.randint(3, 64, (2, 16))
    health = health_report(model, batch)
    assert "blocks.0.qkv" in health
    entry = health["blocks.0.qkv"]
    assert entry["weight_norm"] > 0.0
    assert entry["grad_norm"] >= 0.0
    assert "act_mean" in entry
    assert "act_near_zero_frac" in entry


def test_sensitivity_report_transformer():
    model = _transformer()
    corpus = ProceduralCorpus(vocab_size=64)
    report = sensitivity_report(model, corpus, batch_size=2, seq_len=16, n_batches=1)
    assert report["baseline_bpt"] > 0.0
    points = {r["point"]: r for r in report["points"]}
    assert "blocks.skip[0]" in points
    assert points["blocks.skip[0]"]["status"] == "ok"
    assert "delta_bpt" in points["blocks.skip[0]"]
    # Shape-mismatched bypass points are reported, not crashes.
    not_ablatable = [r for r in report["points"] if r["status"] == "not_ablatable"]
    assert any("output tensor shape" in r["error"] for r in not_ablatable)


def test_component_report_merges_and_assigns_verdicts():
    model = _transformer()
    corpus = ProceduralCorpus(vocab_size=64)
    report = component_report(model, corpus, batch_size=2, seq_len=16, n_batches=1,
                                n_warmup=1, n_iters=2)
    assert report["summary"]["n_components"] > 0
    assert set(report["summary"]["verdict_counts"]) <= {
        "bottleneck", "degenerate", "likely-useless", "healthy"}
    entry = report["components"]["blocks.0.qkv"]
    assert "profile" in entry
    assert "health" in entry
    assert entry["verdict"] in {"bottleneck", "degenerate", "likely-useless", "healthy"}


def test_diagnose_opet_smoke():
    model = _opet()
    corpus = ProceduralCorpus(vocab_size=64)
    report = diagnose(model, corpus, batch_size=2, seq_len=16, n_batches=1, n_warmup=1, n_iters=2)
    assert report["summary"]["n_components"] > 0
    assert report["summary"]["baseline_bpt"] > 0.0
    # OPET's embedding-driven blocks should still show up as profiled components.
    assert any(path.startswith("blocks.") for path in report["components"])


def test_diagnose_rhca_smoke(tiny_model, tiny_config):
    corpus = ProceduralCorpus(vocab_size=tiny_config.vocab_size)
    report = diagnose(tiny_model, corpus, batch_size=2, seq_len=tiny_config.frontier_size * 4,
                       n_batches=1, n_warmup=1, n_iters=1)
    assert report["summary"]["n_components"] > 0
    assert any(path.startswith("settle_ssm.blocks.") for path in report["components"])


def test_parameter_count_matches_manual_sum():
    model = _transformer()
    assert parameter_count(model) == sum(p.numel() for p in model.parameters())


def test_cache_bytes_tensor():
    tensor = torch.zeros(4, 8, dtype=torch.float32)
    assert cache_bytes(tensor) == 4 * 8 * 4  # float32 == 4 bytes/element


def test_cache_bytes_nested_containers():
    cache = {
        "a": torch.zeros(2, dtype=torch.float32),
        "b": [torch.zeros(3, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)],
    }
    assert cache_bytes(cache) == (2 + 3 + 1) * 4


def test_cache_bytes_non_tensor_is_zero():
    assert cache_bytes("not a tensor") == 0
    assert cache_bytes(42) == 0
