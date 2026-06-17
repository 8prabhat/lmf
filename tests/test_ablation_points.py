"""Generic structural ablation points: discovery + bypass across model families.

These tests are the genericity evidence for ``lmf.ablation.points`` -- the same
``discover_points``/``bypass_module``/``apply_point`` machinery is exercised
against a tiny ``CachedTransformerLM``, a tiny ``OPETTransformerLM``, and the
``SettleSSM.blocks`` stack inside RHCA, with zero model-specific code paths.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from lmf.ablation.points import (
    BypassError,
    apply_point,
    bypass_module,
    discover_points,
    skip_listed_module,
)
from lmf.models.opet import OPETTransformerConfig, OPETTransformerLM
from lmf.models.transformer import CachedTransformerLM, TransformerConfig


def _transformer() -> CachedTransformerLM:
    return CachedTransformerLM(TransformerConfig(vocab_size=64, dim=32, layers=2, heads=4))


def _opet() -> OPETTransformerLM:
    return OPETTransformerLM(OPETTransformerConfig(
        vocab_size=64, dim=32, layers=2, heads=4, context_window=2, n_freq_bands=4, dropout=0.0))


def test_discover_points_transformer_blocks():
    model = _transformer()
    points = discover_points(model)
    assert "blocks.skip[0]" in points
    assert "blocks.skip[1]" in points
    assert points["blocks.skip[0]"].path == "blocks.0"
    # Leaf parameterized submodules (e.g. the QKV projection) get bypass/zero points;
    # `blocks.0` itself has no *direct* parameters (they live in its children).
    assert "blocks.0.qkv.bypass" in points
    assert "blocks.0.qkv.zero" in points


def test_skip_listed_block_identity_preserves_shape():
    model = _transformer()
    ids = torch.randint(3, 64, (2, 8))
    with skip_listed_module(model, "blocks", 0):
        logits, _ = model(ids)
    assert logits.shape == (2, 8, 64)


def test_bypass_module_zero_mode_preserves_shape():
    model = _transformer()
    ids = torch.randint(3, 64, (2, 8))
    points = discover_points(model)
    with apply_point(model, points["blocks.0.qkv.zero"]):
        logits, _ = model(ids)
    assert logits.shape == (2, 8, 64)


def test_identity_bypass_changes_block_output():
    """Identity-bypassing block 0 changes the input seen by block 1 (vs. baseline)."""
    model = _transformer()
    ids = torch.randint(3, 64, (2, 8))
    with torch.no_grad():
        baseline, _ = model(ids)
        with skip_listed_module(model, "blocks", 0):
            bypassed, _ = model(ids)
    assert not torch.allclose(baseline, bypassed)


def test_discover_points_opet_blocks_and_embedding():
    model = _opet()
    points = discover_points(model)
    assert "blocks.skip[0]" in points
    assert "blocks.skip[1]" in points
    # OPET's embedding returns a dict from forward(); identity bypass on a
    # transformer block (tensor in, (tensor, None) out) still works generically.
    ids = torch.randint(3, 64, (2, 6))
    with apply_point(model, points["blocks.skip[0]"]):
        logits, opet_out = model(ids)
    assert logits.shape == (2, 6, 64)
    assert isinstance(opet_out, dict)


def test_discover_points_rhca_settle_ssm(tiny_model, tiny_config):
    points = discover_points(tiny_model)
    assert "settle_ssm.blocks.skip[0]" in points
    assert "settle_ssm.blocks.skip[1]" in points
    assert points["settle_ssm.blocks.skip[0]"].path == "settle_ssm.blocks.0"


def test_rhca_settle_ssm_identity_bypass_runs(tiny_model, tiny_config, tiny_tokens):
    state = tiny_model.prefill(tiny_tokens[:, : tiny_config.frontier_size])
    points = discover_points(tiny_model)
    with apply_point(tiny_model, points["settle_ssm.blocks.skip[0]"]):
        frontier, intermediates = tiny_model.settle_ssm(state.frontier, state.memory, state.tail)
    assert frontier.shape == state.frontier.shape
    assert len(intermediates) >= 1


def test_bypass_module_shape_mismatch_raises_bypass_error():
    class Mismatched(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 8)

        def forward(self, x):
            return self.proj(x)

    model = Mismatched()
    x = torch.randn(2, 4)
    with bypass_module(model, "proj", mode="identity"):
        try:
            model(x)
        except BypassError as e:
            assert "shape" in str(e)
        else:
            raise AssertionError("expected BypassError")


def test_unknown_bypass_mode_raises_value_error():
    model = _transformer()
    try:
        with bypass_module(model, "blocks.0", mode="bogus"):
            pass
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_custom_ablation_points_merge():
    class WithCustomPoints(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.calls: list[str] = []

        def forward(self, x):
            return self.proj(x)

        def ablation_points(self):
            @contextlib.contextmanager
            def _custom(model):
                model.calls.append("entered")
                try:
                    yield
                finally:
                    model.calls.append("exited")

            return {"custom:my_point": _custom}

    model = WithCustomPoints()
    points = discover_points(model)
    assert "custom:my_point" in points
    assert points["custom:my_point"].mode == "custom"

    with apply_point(model, points["custom:my_point"]):
        assert model.calls == ["entered"]
    assert model.calls == ["entered", "exited"]
