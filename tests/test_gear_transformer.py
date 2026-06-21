"""Multi-Rate Latent Gear Transformer smoke tests."""

from __future__ import annotations

import torch
import pytest

from lmf.core.build import build
from lmf.core.config import load_config
from lmf.core.registry import MODELS, TRAINERS
from lmf.data import ProceduralCorpus
from lmf.evaluation import bits_per_token
from lmf.models.gear_transformer import (
    GearTransformerConfig,
    MHGTransformerLM,
    ParallelGearSystem,
    SimplifiedGearTransformerLM,
)
from lmf.models.gear_transformer.trainer import build_gear_transformer_trainer
from lmf.models.transformer import CachedTransformerLM, TransformerConfig


def _model(num_gears: int = 9) -> MHGTransformerLM:
    speeds = [1.4, 1.0, 0.7, 0.45, 0.3, 0.18, 0.11, 0.065, 0.035]
    slots = [8, 8, 8, 8, 6, 6, 4, 4, 4]
    fields = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    return MHGTransformerLM(
        GearTransformerConfig(
            vocab_size=64,
            dim=32,
            layers=3,
            heads=4,
            max_seq_len=128,
            num_gears=num_gears,
            gear_speeds=speeds[:num_gears],
            gear_slots=slots[:num_gears],
            gear_receptive_fields=fields[:num_gears],
            gear_update_mode="parallel",
            gear_rotation_dims=8,
            gear_layer_strategy="upper_alternate",
            future_horizons=[2, 4],
            future_loss_weight=0.05,
            diversity_loss_weight=0.001,
            alignment_loss_weight=0.01,
            consistency_loss_weight=0.01,
            agreement_dim=16,
        )
    )


def _gear_system(model: MHGTransformerLM) -> ParallelGearSystem:
    for block in model.blocks:
        if isinstance(block.gears, ParallelGearSystem):
            return block.gears
    assert isinstance(model.shared_gears, ParallelGearSystem)
    return model.shared_gears


def _v5_model() -> MHGTransformerLM:
    return MHGTransformerLM(
        GearTransformerConfig(
            vocab_size=64,
            dim=32,
            layers=4,
            heads=4,
            max_seq_len=128,
            num_gears=9,
            gear_system="parallel_v5",
            gear_lane_sizes=[3, 2, 2, 2],
            gear_speeds=[
                1.4, 1.0, 0.7, 0.45, 0.3, 0.18, 0.11, 0.065, 0.035
            ],
            gear_slots=[8, 8, 8, 8, 6, 6, 4, 4, 4],
            gear_receptive_fields=[
                4, 8, 16, 32, 64, 128, 256, 512, 1024
            ],
            gear_rotation_dims=8,
            gear_layer_strategy="stacked_parallel",
            phase_coupling_init=0.12,
            phase_coupling_max=0.35,
            phase_lock_loss_weight=0.002,
            future_horizons=[2, 4],
            future_token_loss_weight=0.01,
            lane_token_loss_weight=0.005,
            prediction_loss_stride=4,
            lane_dropout=0.05,
            agreement_dim=16,
        )
    )


def test_registry_entries_exist():
    assert "gear_transformer" in MODELS
    assert "mlgt" in MODELS
    assert "gear_only" in MODELS
    assert "simplified_gear_transformer" in MODELS
    assert "gear_transformer" in TRAINERS
    assert "mlgt" in TRAINERS
    assert "gear_only" in TRAINERS
    assert "simplified_gear_transformer" in TRAINERS


def test_simplified_architecture_keeps_only_one_uncoupled_fast_bank():
    model = MODELS.create(
        "simplified_gear_transformer",
        {
            "vocab_size": 64,
            "dim": 32,
            "layers": 3,
            "heads": 4,
            "max_seq_len": 128,
            "num_gears": 5,
            "gear_dim": 16,
            "gear_lane_sizes": [2, 1, 1, 1],
            "gear_speeds": [1.4, 0.7, 0.3, 0.11, 0.035],
            "gear_slots": [8, 8, 6, 4, 4],
            "gear_receptive_fields": [4, 16, 64, 256, 1024],
            "gear_layers": [0],
            "gear_bank_temporal_strides": [1],
            "future_horizons": [2, 4],
            "future_dim": 16,
        },
        None,
    )
    assert isinstance(model, SimplifiedGearTransformerLM)
    systems = [
        block.gears
        for block in model.blocks
        if isinstance(block.gears, ParallelGearSystem)
    ]
    assert len(systems) == 1
    system = systems[0]
    assert system.temporal_stride == 1
    assert system.clock.coupling_logits.numel() == 0
    assert system.context_bank is None
    assert system.carrier_out is None

    tokens = torch.randint(0, 64, (3, 24))
    losses = model.training_step(tokens, {"training_step": 10_000})
    losses["total"].backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_forward_training_step_and_backward():
    model = _model()
    tokens = torch.randint(0, 64, (3, 20))
    losses = model.training_step(tokens)
    for key in (
        "language_modeling",
        "future_latent",
        "gear_diversity",
        "slot_usage",
        "lane_prediction",
        "alignment_calibration",
        "consistency",
        "gear_write_activity",
        "gear_read_activity",
        "gear_coupling_entropy",
        "gear_coupling_gate",
        "gear_coupling_offdiag",
        "gear_rotation_activity",
        "gear_minimum_phase_advance",
        "gear_lane_balance",
        "total",
    ):
        assert key in losses
        assert torch.isfinite(losses[key]).all()
    losses["total"].backward()
    gears = _gear_system(model)
    assert gears.clock.token_drive.weight.grad is not None
    assert gears.clock.coupling_logits.grad is not None
    assert gears.preferred_phase.grad is not None
    assert gears.memory_candidate.grad is not None
    assert gears.lane_value_score.grad is not None
    assert model.future_to_hidden is not None
    assert model.future_to_hidden.weight.grad is not None


def test_parallel_gear_system_has_four_lanes():
    gears = _gear_system(_model())
    assert gears.num_gears == 9
    assert gears.lane_sizes == (3, 2, 2, 2)
    assert gears.lane_mask.shape == (4, 9)


def test_cached_generate_and_alignment_scores():
    model = _model()
    prompt = torch.randint(0, 64, (2, 7))
    out = model.generate(prompt, 5)
    assert out.shape == (2, 5)
    scores = model.alignment_scores(prompt)
    assert set(scores) == {"conflict", "risk", "trigger"}
    diagnostics = model.gear_diagnostics(prompt)
    assert diagnostics["active"]
    assert diagnostics["minimum_phase_advance"] > 0.0
    assert len(diagnostics["lane_usage"]) == 4


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
            "num_gears": 9,
            "gear_speeds": [1.4, 1.0, 0.7, 0.45, 0.3, 0.18, 0.11, 0.065, 0.035],
            "gear_slots": [8, 8, 8, 8, 6, 6, 4, 4, 4],
            "gear_receptive_fields": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "gear_update_mode": "parallel",
            "gear_rotation_dims": 8,
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
            "model.gear_layers=[0,1]",
            "model.gear_slots=[8,8,6,4,4]",
            "model.gear_receptive_fields=[4,16,64,256,1024]",
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


def test_requires_definition_aligned_gear_count_and_speed_order():
    with pytest.raises(ValueError, match="at least 5 gears"):
        GearTransformerConfig(
            vocab_size=64,
            dim=32,
            heads=4,
            num_gears=4,
            gear_speeds=[1.0, 0.5, 0.25, 0.125],
            gear_slots=[8] * 4,
            gear_receptive_fields=[4, 8, 16, 32],
        )
    with pytest.raises(ValueError, match="strictly decreasing"):
        GearTransformerConfig(
            vocab_size=64,
            dim=32,
            heads=4,
            num_gears=5,
            gear_speeds=[1.0, 0.5, 0.5, 0.125, 0.0625],
            gear_slots=[8] * 5,
            gear_receptive_fields=[4, 8, 16, 32, 64],
        )


def test_monotonic_clocks_and_positive_context_driven_rotation():
    model = _model()
    gears = _gear_system(model)
    with torch.no_grad():
        gears.clock.first_speed_offset.fill_(10.0)
        gears.clock.gap_offsets.copy_(torch.linspace(-10.0, 10.0, 8))
    speeds = gears.clock.speeds()
    assert torch.all(speeds[:-1] > speeds[1:])

    h = torch.randn(2, model.config.gear_dim)
    context = torch.randn_like(h)
    phase = gears.clock.initial_phase(2, h.device)
    phases_a, delta_a, _ = gears.clock.step(h, context, phase)
    phases_b, delta_b, _ = gears.clock.step(h + 0.5, context, phase)
    assert torch.all(delta_a > 0)
    assert torch.all(delta_b > 0)
    assert not torch.allclose(phases_a, phases_b)


def test_cached_forward_matches_full_forward():
    model = _model().eval()
    tokens = torch.randint(0, 64, (2, 12))
    full, _ = model(tokens)
    first, cache = model(tokens[:, :7], use_cache=True)
    pieces = [first]
    for index in range(7, tokens.shape[1]):
        logits, cache = model(tokens[:, index:index + 1], caches=cache, use_cache=True)
        pieces.append(logits)
    cached = torch.cat(pieces, dim=1)
    assert torch.allclose(full, cached, atol=2e-5, rtol=2e-4)


def test_future_prediction_changes_generation_logits():
    model = _model().eval()
    tokens = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        model.future_residual_gate_logit.fill_(-20.0)
        without_future, _ = model(tokens)
        model.future_residual_gate_logit.fill_(20.0)
        with_future, _ = model(tokens)
    assert not torch.allclose(without_future, with_future)


def test_rotation_is_geometric_and_preserves_pair_norm():
    gears = _gear_system(_model())
    memory = torch.randn(2, gears.num_gears, gears.dim)
    advance = torch.rand(2, gears.num_gears) + 0.1
    rotated, activity = gears._rotate_memory(memory, advance, 1.0)
    before = memory[..., : gears.rotation_dims].reshape(
        2, gears.num_gears, -1, 2
    ).norm(dim=-1)
    after = rotated[..., : gears.rotation_dims].reshape(
        2, gears.num_gears, -1, 2
    ).norm(dim=-1)
    assert activity > 0
    assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)


def test_phase_rotation_and_parallel_lanes_materially_change_logits():
    model = _model().eval()
    tokens = torch.randint(0, 64, (2, 24))
    full_hidden, _, full_aux = model._forward_hidden(
        tokens,
        return_aux=True,
        gear_scale=1.0,
        phase_scale=1.0,
    )
    no_phase_hidden, _, _ = model._forward_hidden(
        tokens,
        return_aux=True,
        gear_scale=1.0,
        phase_scale=0.0,
    )
    assert not torch.allclose(full_hidden, no_phase_hidden)
    lane_weights = full_aux[0]["lane_weights"]
    assert lane_weights.shape[-1] == 4
    assert torch.all(lane_weights > 0)


def test_staged_training_scales_and_gear_lr_multiplier():
    model = _model()
    early = model._training_scales({"training_step": 0})
    late = model._training_scales({"training_step": 10_000})
    assert early == {"gear": 0.0, "phase": 0.0, "auxiliary": 0.0, "future": 0.0}
    assert late == {"gear": 1.0, "phase": 1.0, "auxiliary": 1.0, "future": 1.0}

    corpus = ProceduralCorpus(vocab_size=64)
    trainer = build_gear_transformer_trainer(
        model,
        corpus,
        device="cpu",
        precision="fp32",
        total_steps=10,
        warmup_steps=1,
        lr=1e-3,
    )
    multipliers = sorted(
        float(group.get("lr_multiplier", 1.0))
        for group in trainer.optimizer.param_groups
    )
    assert multipliers == [1.0, model.config.gear_lr_multiplier]


def test_can_warm_start_transformer_trunk_without_overwriting_gears():
    model = _model()
    baseline = CachedTransformerLM(
        TransformerConfig(
            vocab_size=64,
            dim=32,
            layers=3,
            heads=4,
            max_seq_len=128,
        )
    )
    gears = _gear_system(model)
    original_slots = gears.slots.detach().clone()
    copied = model.initialize_trunk_from_transformer(baseline)
    assert copied > 0
    assert torch.equal(model.token.weight, baseline.token.weight)
    assert torch.equal(gears.slots, original_slots)


def test_v5_builds_three_specialized_parallel_banks():
    model = _v5_model()
    assert model.config.selected_gear_layers() == (0, 2, 3)
    systems = [
        block.gears
        for block in model.blocks
        if isinstance(block.gears, ParallelGearSystem)
    ]
    assert len(systems) == 3
    assert [system.bank_index for system in systems] == [0, 1, 2]
    assert systems[0].speeds()[0] > systems[1].speeds()[0]
    assert systems[1].speeds()[0] > systems[2].speeds()[0]
    assert systems[0].horizon_scale < systems[2].horizon_scale


def test_v5_sparse_mechanical_coupling_and_predictor_are_active():
    model = _v5_model()
    gears = _gear_system(model)
    mask = gears.clock.coupling_mask
    dense_edges = gears.num_gears * (gears.num_gears - 1) // 2
    assert 0 < int(mask.sum()) < dense_edges
    assert torch.all(mask.diagonal() == 0)
    assert mask[1, 0] and mask[2, 1]
    tokens = torch.randint(0, 64, (2, 32))
    diagnostics = model.gear_diagnostics(tokens)
    assert diagnostics["layers"] == 3
    assert diagnostics["bank_temporal_strides"] == [1, 2, 4]
    assert diagnostics["bank_active_fraction"] == [1.0, 0.5, 0.25]
    assert diagnostics["phase_coupling_activity"] > 0.0
    assert diagnostics["phase_lock_error"] >= 0.0


def test_v5_cached_forward_matches_full_with_temporal_and_interbank_carriers():
    model = _v5_model().eval()
    tokens = torch.randint(0, 64, (2, 24))
    full, _ = model(tokens)
    first, cache = model(tokens[:, :7], use_cache=True)
    pieces = [first]
    for index in range(7, tokens.shape[1]):
        logits, cache = model(
            tokens[:, index:index + 1],
            caches=cache,
            use_cache=True,
        )
        pieces.append(logits)
    cached = torch.cat(pieces, dim=1)
    assert torch.allclose(full, cached, atol=2e-5, rtol=2e-4)


def test_v5_is_causal_and_each_bank_changes_predictions():
    model = _v5_model().eval()
    tokens = torch.randint(0, 64, (2, 24))
    changed = tokens.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 64
    full = model.component_logits(tokens)
    changed_logits = model.component_logits(changed)
    assert torch.allclose(full[:, :-1], changed_logits[:, :-1], atol=2e-5)
    for bank_index in range(3):
        without_bank = model.component_logits(
            tokens,
            (f"bank_{bank_index}",),
        )
        assert not torch.allclose(full, without_bank)


def test_v5_component_ablation_api_and_predictive_losses():
    model = _v5_model()
    tokens = torch.randint(0, 64, (2, 24))
    losses = model.training_step(tokens, {"training_step": 10_000})
    for key in (
        "future_token",
        "lane_token",
        "phase_lock",
        "gear_phase_lock_error",
        "gear_interbank_activity",
        "gear_temporal_context_gate",
    ):
        assert key in losses
        assert torch.isfinite(losses[key])
    losses["total"].backward()
    systems = [
        block.gears
        for block in model.blocks
        if isinstance(block.gears, ParallelGearSystem)
    ]
    assert systems[1].context_bank.weight.grad is not None
    assert systems[1].interbank_gate_logit.grad is not None
    metrics = model.component_ablation_metrics(
        tokens,
        (
            "phase_coupling",
            "rotation",
            "temporal_context",
            "interbank_coupling",
            "lane_mixing",
        ),
    )
    assert set(metrics) >= {
        "full",
        "phase_coupling",
        "rotation",
        "temporal_context",
        "interbank_coupling",
        "lane_mixing",
        "bank_0",
        "bank_1",
        "bank_2",
    }


def test_sequence_length_curriculum_is_monotonic_and_bounded():
    corpus = ProceduralCorpus(vocab_size=64)
    trainer = build_gear_transformer_trainer(
        _v5_model(),
        corpus,
        device="cpu",
        precision="fp32",
        total_steps=20,
        warmup_steps=1,
        lr=1e-3,
        seq_len_curriculum=[16, 32, 64],
        seq_len_curriculum_steps=[0, 5, 10],
    )
    assert trainer.effective_seq_len(48, 0) == 16
    assert trainer.effective_seq_len(48, 7) == 32
    assert trainer.effective_seq_len(48, 12) == 48
