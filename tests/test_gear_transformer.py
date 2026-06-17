"""Multi-Rate Latent Gear Transformer smoke tests."""

from __future__ import annotations

import torch

from lmf.core.build import build
from lmf.core.config import load_config
from lmf.core.registry import MODELS, TRAINERS
from lmf.data import ProceduralCorpus
from lmf.evaluation import bits_per_token
from lmf.models.gear_transformer import GearTransformerConfig, MHGTransformerLM
from lmf.models.gear_transformer.trainer import build_gear_transformer_trainer


def _model(update_mode: str = "dilated", num_gears: int = 4) -> MHGTransformerLM:
    return MHGTransformerLM(
        GearTransformerConfig(
            vocab_size=64,
            dim=32,
            layers=3,
            heads=4,
            max_seq_len=128,
            num_gears=num_gears,
            gear_speeds=[1, 2, 4, 8][:num_gears],
            gear_slots=[8, 8, 8, 8][:num_gears],
            gear_receptive_fields=[4, 8, 16, 32][:num_gears],
            gear_update_mode=update_mode,
            gear_layer_strategy="upper_alternate",
            future_horizons=[2, 4],
            future_loss_weight=0.05,
            diversity_loss_weight=0.001,
            alignment_loss_weight=0.01,
            consistency_loss_weight=0.01,
            agreement_dim=16,
        )
    )


def test_registry_entries_exist():
    assert "gear_transformer" in MODELS
    assert "mlgt" in MODELS
    assert "gear_only" in MODELS
    assert "gear_transformer" in TRAINERS
    assert "mlgt" in TRAINERS
    assert "gear_only" in TRAINERS


def test_forward_training_step_and_backward():
    model = _model()
    tokens = torch.randint(0, 64, (3, 20))
    losses = model.training_step(tokens)
    for key in (
        "language_modeling",
        "future_latent",
        "gear_diversity",
        "alignment_calibration",
        "consistency",
        "gear_write_activity",
        "gear_read_activity",
        "gear_coupling_entropy",
        "gear_coupling_gate",
        "gear_coupling_offdiag",
        "total",
    ):
        assert key in losses
        assert torch.isfinite(losses[key]).all()
    losses["total"].backward()


def test_update_modes_train():
    tokens = torch.randint(0, 64, (2, 18))
    for mode in ("chunked", "dilated", "scan"):
        model = _model(update_mode=mode)
        losses = model.training_step(tokens)
        assert torch.isfinite(losses["total"]).all()


def test_cached_generate_and_alignment_scores():
    model = _model()
    prompt = torch.randint(0, 64, (2, 7))
    out = model.generate(prompt, 5)
    assert out.shape == (2, 5)
    scores = model.alignment_scores(prompt)
    assert set(scores) == {"conflict", "risk", "trigger"}


def test_future_only_variant_without_gears():
    model = _model(num_gears=0)
    tokens = torch.randint(0, 64, (2, 18))
    losses = model.training_step(tokens)
    assert torch.isfinite(losses["total"]).all()
    assert losses["gear_diversity"].item() == 0.0


def test_gear_only_removes_attention_and_trains():
    model = MODELS.create(
        "gear_only",
        {
            "vocab_size": 64,
            "dim": 32,
            "layers": 2,
            "heads": 4,
            "max_seq_len": 128,
            "gear_speeds": [1, 2, 4, 8],
            "gear_slots": [8, 8, 8, 8],
            "gear_receptive_fields": [4, 8, 16, 32],
            "future_horizons": [2],
            "agreement_dim": 16,
        },
        None,
    )
    assert isinstance(model, MHGTransformerLM)
    assert not model.config.use_attention
    assert all(block.qkv is None and block.proj is None and block.norm1 is None for block in model.blocks)
    tokens = torch.randint(0, 64, (2, 20))
    losses = model.training_step(tokens)
    assert torch.isfinite(losses["total"]).all()
    out = model.generate(tokens[:, :5], 4)
    assert out.shape == (2, 4)


def test_trainer_and_bpt():
    corpus = ProceduralCorpus(vocab_size=64)
    model = _model()
    trainer = build_gear_transformer_trainer(
        model,
        corpus,
        device="cpu",
        precision="fp32",
        warmup_steps=2,
        total_steps=5,
        lr=3e-3,
    )
    trainer.train_steps(2, batch_size=2, seq_len=24, log_every=0)
    bpt = bits_per_token(model, corpus, batch_size=2, seq_len=24, n_batches=1)
    assert bpt > 0


def test_config_builds_from_registry():
    cfg = load_config(
        "configs/gear_transformer.yaml",
        "smoke",
        overrides=[
            "model.dim=32",
            "model.layers=2",
            "model.heads=4",
            "model.gear_slots=[8,8,8,8]",
            "model.gear_receptive_fields=[4,8,16,32]",
            "model.future_horizons=[2]",
            "run.batch_size=2",
            "run.seq_len=24",
            "run.steps=1",
            "trainer.total_steps=1",
        ],
    )
    corpus, model, trainer, run = build(cfg)
    assert corpus.vocab_size == 512
    assert isinstance(model, MHGTransformerLM)
    assert run["steps"] == 1
    trainer.train_steps(1, batch_size=2, seq_len=24, log_every=0)
