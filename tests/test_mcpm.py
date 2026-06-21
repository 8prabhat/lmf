from __future__ import annotations

import torch

from lmf.ablation.points import discover_points
from lmf.models.mcpm import MultiGearConstructiveProgramMachineLM
from lmf.models._shared.causal_mesh_base import NativeLMConfig


def _exercise_model(model):
    tokens = torch.randint(0, 64, (3, 20))
    logits, _ = model(tokens)
    assert logits.shape == (3, 20, 64)
    losses = model.training_step(tokens)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    out = model.generate(tokens[:, :5], 7)
    assert out.shape == (3, 7)


def test_mcpm_forward_backward_generate():
    model = MultiGearConstructiveProgramMachineLM(
        NativeLMConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=64,
            execution_residual=True,
        )
    )
    _exercise_model(model)


def test_full_mcpm_exposes_program_execution_and_verifier_points():
    model = MultiGearConstructiveProgramMachineLM(
        NativeLMConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=64,
            execution_residual=True,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            mesh_layers=2,
            route_aux_weight=0.01,
            draft_aux_weight=0.01,
            program_aux_weight=0.01,
            verifier_aux_weight=0.01,
        )
    )
    tokens = torch.randint(0, 64, (2, 20))
    losses = model.training_step(tokens)
    assert {"program_controller", "contract_verifier", "draft_tree"}.issubset(losses)
    assert torch.isfinite(losses["total"])
    points = discover_points(model)
    assert "program_controller.bypass" in points
    assert "execution_workbench.rounds.skip[0]" in points
    assert "contract_verifier.bypass" in points


def test_full_mcpm_training_reuses_single_hidden_pass(monkeypatch):
    model = MultiGearConstructiveProgramMachineLM(
        NativeLMConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=64,
            execution_residual=True,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            mesh_layers=2,
            route_aux_weight=0.01,
            draft_aux_weight=0.01,
            program_aux_weight=0.01,
            verifier_aux_weight=0.01,
        )
    )
    calls = 0
    original = model._forward_hidden

    def counted_forward(ids):
        nonlocal calls
        calls += 1
        return original(ids)

    monkeypatch.setattr(model, "_forward_hidden", counted_forward)
    tokens = torch.randint(0, 64, (2, 20))
    losses = model.training_step(tokens)
    assert torch.isfinite(losses["total"])
    assert calls == 1
