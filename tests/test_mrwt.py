from __future__ import annotations

import torch

from lmf.ablation.points import discover_points
from lmf.models.mrwt import MRWTConfig, MultiGearResidualWorkbenchTransformerLM


def _exercise_model(model):
    tokens = torch.randint(0, 64, (3, 20))
    logits, _ = model(tokens)
    assert logits.shape == (3, 20, 64)
    losses = model.training_step(tokens)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    out = model.generate(tokens[:, :5], 7)
    assert out.shape == (3, 7)


def test_mrwt_forward_backward_generate_and_anchor_fallback():
    model = MultiGearResidualWorkbenchTransformerLM(
        MRWTConfig(vocab_size=64, dim=32, layers=2, heads=4, max_seq_len=64)
    )
    tokens = torch.randint(0, 64, (2, 16))
    logits, _ = model(tokens)
    anchor = model.anchor_logits(tokens)
    assert torch.allclose(logits, anchor, atol=1e-6)
    _exercise_model(model)


def test_full_mrwt_exposes_workbench_points_and_preserves_anchor_at_zero_gate():
    model = MultiGearResidualWorkbenchTransformerLM(
        MRWTConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            heads=4,
            max_seq_len=64,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            workbench_rounds=2,
            budget_aux_weight=0.01,
            draft_aux_weight=0.01,
        )
    )
    tokens = torch.randint(0, 64, (2, 20))
    logits, _ = model(tokens)
    assert torch.allclose(logits, model.anchor_logits(tokens), atol=1e-6)
    losses = model.training_step(tokens)
    assert {"budget_controller", "draft_tree"}.issubset(losses)
    assert torch.isfinite(losses["total"])
    points = discover_points(model)
    assert "atlas.scales.skip[0]" in points
    assert "budget_controller.bypass" in points
    assert "workbench_rounds.skip[0]" in points


def test_full_mrwt_training_reuses_single_hidden_pass(monkeypatch):
    model = MultiGearResidualWorkbenchTransformerLM(
        MRWTConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            heads=4,
            max_seq_len=64,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            workbench_rounds=2,
            budget_aux_weight=0.01,
            draft_aux_weight=0.01,
        )
    )
    calls = 0
    original = model._forward_hidden

    def counted_forward(ids, attention_mask=None):
        nonlocal calls
        calls += 1
        return original(ids, attention_mask=attention_mask)

    monkeypatch.setattr(model, "_forward_hidden", counted_forward)
    tokens = torch.randint(0, 64, (2, 20))
    losses = model.training_step(tokens)
    assert torch.isfinite(losses["total"])
    assert calls == 1


def test_mrwt_zero_gate_generation_uses_anchor_cache(monkeypatch):
    model = MultiGearResidualWorkbenchTransformerLM(
        MRWTConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            heads=4,
            max_seq_len=64,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            workbench_rounds=2,
        )
    )
    called = False

    def fake_generate(prompt_tokens, max_new_tokens, sampling_config=None):
        nonlocal called
        called = True
        return torch.zeros(
            prompt_tokens.shape[0],
            max_new_tokens,
            dtype=torch.long,
            device=prompt_tokens.device,
        )

    monkeypatch.setattr(model.anchor, "generate", fake_generate)
    out = model.generate(torch.randint(0, 64, (2, 5)), 7)
    assert called
    assert out.shape == (2, 7)
