from __future__ import annotations

import torch

from lmf.ablation.points import discover_points
from lmf.data import MultiGearTokenizer, ProceduralCorpus, SpecialTokenTokenizer
from lmf.evaluation import bits_per_token
from lmf.models.mecm import MultiGearElasticCausalMeshLM
from lmf.models._shared.causal_mesh_base import NativeLMConfig
from lmf.models._shared.trainer import NativeLMTrainer


def _exercise_model(model):
    tokens = torch.randint(0, 64, (3, 20))
    logits, _ = model(tokens)
    assert logits.shape == (3, 20, 64)
    losses = model.training_step(tokens)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    out = model.generate(tokens[:, :5], 7)
    assert out.shape == (3, 7)


def test_mecm_forward_backward_generate():
    model = MultiGearElasticCausalMeshLM(
        NativeLMConfig(vocab_size=64, dim=32, layers=2, kernel_size=5, max_seq_len=64)
    )
    _exercise_model(model)


def test_full_mecm_exposes_research_losses_and_ablation_points():
    model = MultiGearElasticCausalMeshLM(
        NativeLMConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=64,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            mesh_layers=2,
            route_aux_weight=0.01,
            draft_aux_weight=0.01,
        )
    )
    tokens = torch.randint(0, 64, (2, 20))
    losses = model.training_step(tokens)
    assert {"route_balance", "draft_tree"}.issubset(losses)
    assert torch.isfinite(losses["total"])
    points = discover_points(model)
    assert "span_atlas.scales.skip[0]" in points
    assert "active_cover.bypass" in points
    assert "reasoning_mesh.layers.skip[0]" in points


def test_mecm_gear_aware_output_uses_token_hierarchy():
    base = MultiGearTokenizer(max_vocab=340, max_token_bytes=16)
    text = "alpha beta gamma delta conference Australia India Morgan " * 20
    base.train([text])
    tokenizer = SpecialTokenTokenizer(base)
    model = MultiGearElasticCausalMeshLM(
        NativeLMConfig(
            vocab_size=tokenizer.vocab_size,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=96,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            mesh_layers=1,
            draft_aux_weight=0.0,
            gear_aware_output=True,
            gear_aux_weight=0.01,
        )
    )
    model.configure_token_hierarchy(**tokenizer.token_hierarchy())
    ids = tokenizer.encode(text)[:48]
    tokens = torch.tensor([ids, list(reversed(ids))], dtype=torch.long)
    losses = model.training_step(tokens)
    assert {"gear_prediction", "within_gear", "gear_aux"}.issubset(losses)
    assert torch.isfinite(losses["total"])
    logits, _ = model(tokens)
    assert logits.shape == (2, len(ids), tokenizer.vocab_size)
    generated = model.generate(tokens[:, :8], 3)
    assert generated.shape == (2, 3)


def test_mecm_gear_bias_output_uses_auxiliary_gear_loss():
    base = MultiGearTokenizer(max_vocab=340, max_token_bytes=16)
    text = "alpha beta gamma delta conference Australia India Morgan " * 20
    base.train([text])
    tokenizer = SpecialTokenTokenizer(base)
    model = MultiGearElasticCausalMeshLM(
        NativeLMConfig(
            vocab_size=tokenizer.vocab_size,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=96,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            mesh_layers=1,
            draft_aux_weight=0.0,
            gear_aware_output=True,
            gear_output_mode="bias",
            gear_aux_weight=0.01,
        )
    )
    model.configure_token_hierarchy(**tokenizer.token_hierarchy())
    ids = tokenizer.encode(text)[:48]
    tokens = torch.tensor([ids, list(reversed(ids))], dtype=torch.long)
    losses = model.training_step(tokens)
    assert "gear_aux" in losses
    assert torch.isfinite(losses["total"])
    logits, _ = model(tokens)
    assert logits.shape == (2, len(ids), tokenizer.vocab_size)


def test_mecm_trainer_and_bpt_smoke():
    corpus = ProceduralCorpus(vocab_size=64)
    model = MultiGearElasticCausalMeshLM(
        NativeLMConfig(vocab_size=64, dim=32, layers=2, kernel_size=5, max_seq_len=64)
    )
    trainer = NativeLMTrainer(
        model,
        corpus,
        device="cpu",
        precision="fp32",
        warmup_steps=1,
        total_steps=3,
        lr=1e-3,
    )
    records = trainer.train_steps(2, batch_size=2, seq_len=24, log_every=0)
    assert len(records) == 2
    assert bits_per_token(model, corpus, batch_size=2, seq_len=24, n_batches=1) > 0
