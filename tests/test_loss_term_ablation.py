"""``loss_term_scales`` threading: backward-compatible (None == current behavior),
and each family scales the requested term while leaving the others alone.

Generic across families -- no model-specific ablation code paths, just the
optional ``training_step(..., loss_term_scales=...)`` kwarg from
``core.interfaces.Trainable`` and the matching ``BaseTrainer`` plumbing.
"""

from __future__ import annotations

import torch

from lmf.ablation.loss_terms import discover_loss_terms
from lmf.models.opet import OPETTransformerConfig, OPETTransformerLM
from lmf.models.transformer import CachedTransformerLM, TransformerConfig


def _transformer() -> CachedTransformerLM:
    torch.manual_seed(0)
    return CachedTransformerLM(TransformerConfig(vocab_size=64, dim=32, layers=2, heads=4))


def _opet() -> OPETTransformerLM:
    torch.manual_seed(0)
    return OPETTransformerLM(OPETTransformerConfig(
        vocab_size=64, dim=32, layers=2, heads=4, context_window=2, n_freq_bands=4, dropout=0.0))


def test_transformer_loss_term_scales_none_matches_default():
    model = _transformer()
    tokens = torch.randint(3, 64, (2, 16))
    a = model.training_step(tokens)
    b = model.training_step(tokens, loss_term_scales=None)
    assert torch.equal(a["total"], b["total"])


def test_transformer_loss_term_scale_zero_zeroes_total():
    model = _transformer()
    tokens = torch.randint(3, 64, (2, 16))
    scaled = model.training_step(tokens, loss_term_scales={"language_modeling": 0.0})
    assert torch.equal(scaled["total"], torch.zeros_like(scaled["total"]))
    # The raw (unscaled) per-term value is still reported for diagnostics.
    assert scaled["language_modeling"].item() > 0.0


def test_opet_loss_term_scales_none_matches_default():
    model = _opet()
    tokens = torch.randint(3, 64, (2, 16))
    a = model.training_step(tokens)
    b = model.training_step(tokens, loss_term_scales=None)
    assert torch.equal(a["total"], b["total"])


def test_opet_loss_term_scale_zero_drops_term_from_total():
    model = _opet()
    tokens = torch.randint(3, 64, (2, 16))
    full = model.training_step(tokens)
    zeroed = model.training_step(tokens, loss_term_scales={"phase_coherence": 0.0})
    expected = full["total"] - model.config.lambda_coherence * full["phase_coherence"]
    assert torch.allclose(zeroed["total"], expected)
    # Raw per-term value unaffected.
    assert torch.equal(zeroed["phase_coherence"], full["phase_coherence"])


def test_opet_discover_loss_terms_includes_phase_terms():
    model = _opet()
    tokens = torch.randint(3, 64, (2, 16))
    terms = discover_loss_terms(model, tokens)
    assert "language_modeling" in terms
    assert "phase_coherence" in terms
    assert "phase_sharpness" in terms
    assert "phase_orthogonality" in terms
    assert "phase_amplitude" in terms


def test_rhca_loss_term_scales_none_matches_default(tiny_model, tiny_config, tiny_tokens):
    a = tiny_model.training_step(tiny_tokens)
    b = tiny_model.training_step(tiny_tokens, loss_term_scales=None)
    assert torch.equal(a["total"], b["total"])


def test_rhca_loss_term_scale_zero_drops_routing_balance(tiny_model, tiny_config, tiny_tokens):
    full = tiny_model.training_step(tiny_tokens)
    zeroed = tiny_model.training_step(tiny_tokens, loss_term_scales={"routing_balance": 0.0})
    expected = full["total"] - tiny_config.routing_balance_weight * full["routing_balance"]
    assert torch.allclose(zeroed["total"], expected)
    assert torch.equal(zeroed["routing_balance"], full["routing_balance"])


def test_rhca_discover_loss_terms(tiny_model, tiny_config, tiny_tokens):
    terms = discover_loss_terms(tiny_model, tiny_tokens)
    assert "commit_token" in terms
    assert "routing_balance" in terms
