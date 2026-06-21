from __future__ import annotations

import inspect

import pytest
import torch

from lmf.core.registry import MODELS, TRAINERS
from lmf.data import TrainingBatch
from lmf.models.bounded_hybrid_gear import (
    BoundedTransformerConfig,
    BoundedTransformerLM,
    BlockHybridGearV4Config,
    BlockHybridGearV4LM,
    HybridParallelGearConfig,
    HybridParallelGearLM,
    PureParallelGearV3Config,
    PureParallelGearV3LM,
    PureParallelGearV3Trainer,
    chunked_affine_scan,
    chunked_rotor_scan,
    complex_mul,
    mps_affine_scan,
)
from lmf.diagnostics import cache_bytes
from lmf.training.checkpoints import load_checkpoint, save_checkpoint


def strict_config(**overrides):
    values = {
        "vocab_size": 97,
        "dim": 32,
        "layers": 2,
        "ffn_dim": 64,
        "num_banks": 4,
        "gears_per_bank": 4,
        "rotor_channels": 2,
        "cell_dim": 8,
        "bank_rank": 8,
        "scan_chunk_tokens": 8,
    }
    values.update(overrides)
    return PureParallelGearV3Config(**values)


def hybrid_config(**overrides):
    values = strict_config(**overrides).to_dict()
    values.update(
        {
            "attention_window": 8,
            "attention_heads": 4,
            "attention_kv_heads": 2,
            "attention_every": 2,
        }
    )
    return HybridParallelGearConfig(**values)


def bounded_config(**overrides):
    values = {
        "vocab_size": 97,
        "dim": 32,
        "layers": 2,
        "ffn_dim": 64,
        "heads": 4,
        "kv_heads": 2,
        "attention_window": 8,
    }
    values.update(overrides)
    return BoundedTransformerConfig(**values)


def block_additive_config(**overrides):
    values = {
        "vocab_size": 97,
        "dim": 32,
        "layers": 2,
        "ffn_dim": 64,
        "heads": 4,
        "kv_heads": 2,
        "attention_window": 8,
        "block_tokens": 4,
        "gears_per_bank": 4,
        "rotor_channels": 1,
        "cell_dim": 8,
        "bank_rank": 8,
    }
    values.update(overrides)
    return BlockHybridGearV4Config(**values)


def block_selective_film_config(**overrides):
    values = {
        "fusion_mode": "selective_film",
        "fusion_rank": 8,
    }
    values.update(overrides)
    return block_additive_config(**values)


def block_bank_router_config(**overrides):
    values = {
        "fusion_mode": "bank_router",
        "fusion_rank": 8,
    }
    values.update(overrides)
    return block_additive_config(**values)


def masks(tokens):
    segment = torch.zeros_like(tokens)
    segment[:, tokens.shape[1] // 2 :] = 1
    ends = torch.zeros_like(tokens, dtype=torch.bool)
    ends[:, tokens.shape[1] // 2 - 1] = True
    return segment, ends


def _sequential(multiplier, bias, initial):
    state = initial
    rows = []
    for position in range(multiplier.shape[1]):
        state = complex_mul(multiplier[:, position], state) + bias[:, position]
        rows.append(state)
    return torch.stack(rows, dim=1)


def test_bounded_hybrid_gear_families_are_registered():
    for name in (
        "pure_parallel_gear_v3",
        "hybrid_parallel_gear",
        "bounded_transformer",
        "bounded_hybrid_gear_block_additive",
        "bounded_hybrid_gear_block_selective_film",
        "bounded_hybrid_gear_block_bank_router",
    ):
        assert name in MODELS
        assert name in TRAINERS


def test_bounded_hybrid_gear_checkpoint_cannot_cross_architectures(tmp_path):
    strict = PureParallelGearV3LM(strict_config(layers=1))
    optimizer = torch.optim.AdamW(strict.parameters(), lr=1e-3)
    path = tmp_path / "strict.pt"
    save_checkpoint(path, strict, optimizer, step=0)
    hybrid = HybridParallelGearLM(hybrid_config(layers=1))
    with pytest.raises(RuntimeError, match="architecture-specific"):
        load_checkpoint(path, hybrid)
    block_additive = BlockHybridGearV4LM(block_additive_config())
    with pytest.raises(RuntimeError, match="architecture-specific"):
        load_checkpoint(path, block_additive)
    block_selective_film = BlockHybridGearV4LM(block_selective_film_config())
    with pytest.raises(RuntimeError, match="architecture-specific"):
        load_checkpoint(path, block_selective_film)
    block_bank_router = BlockHybridGearV4LM(block_bank_router_config())
    with pytest.raises(RuntimeError, match="architecture-specific"):
        load_checkpoint(path, block_bank_router)


def test_associative_scan_matches_sequential_output_and_gradient():
    torch.manual_seed(3)
    phase = torch.randn(2, 19, 2, 3, 1)
    magnitude = 0.90 + 0.09 * torch.rand(2, 19, 2, 3, 1, 1)
    multiplier = (
        magnitude
        * torch.stack((phase.cos(), phase.sin()), dim=-1)
    ).requires_grad_(True)
    bias = (0.05 * torch.randn_like(multiplier)).requires_grad_(True)
    initial = torch.randn(2, 2, 3, 1, 2, requires_grad=True)
    multiplier_ref = multiplier.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    initial_ref = initial.detach().clone().requires_grad_(True)
    # A zero multiplier is an associative reset transform.
    multiplier_reset = multiplier.clone()
    multiplier_reset[:, 9] = 0.0
    multiplier_ref_reset = multiplier_ref.clone()
    multiplier_ref_reset[:, 9] = 0.0

    scanned, _, _ = chunked_affine_scan(
        multiplier_reset, bias, initial, chunk_size=4
    )
    reference = _sequential(
        multiplier_ref_reset, bias_ref, initial_ref
    )
    assert torch.allclose(scanned, reference, atol=1e-5, rtol=1e-5)
    weight = torch.randn_like(scanned)
    (scanned * weight).sum().backward()
    (reference * weight).sum().backward()
    assert torch.allclose(multiplier.grad, multiplier_ref.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(bias.grad, bias_ref.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(initial.grad, initial_ref.grad, atol=2e-5, rtol=2e-5)


def test_specialized_rotor_scan_matches_reset_reference_and_gradient():
    torch.manual_seed(4)
    batch, length, banks, gears, channels = 2, 31, 2, 3, 1
    phase = 0.2 * torch.randn(batch, length, banks, gears, channels)
    magnitude = 0.91 + 0.08 * torch.rand(
        batch, length, banks, gears, channels
    )
    multiplier = (
        magnitude[..., None]
        * torch.stack((phase.cos(), phase.sin()), dim=-1)
    ).requires_grad_(True)
    bias = (0.03 * torch.randn_like(multiplier)).requires_grad_(True)
    initial = torch.randn(batch, banks, gears, channels, 2, requires_grad=True)
    reset_initial = torch.randn_like(initial, requires_grad=True)
    reset = torch.zeros(batch, length, dtype=torch.bool)
    reset[:, 0] = True
    reset[0, 9] = True
    reset[1, 17] = True

    actual, _, _ = chunked_rotor_scan(
        multiplier,
        bias,
        initial,
        reset,
        reset_initial,
        chunk_size=8,
    )
    reference_state = initial
    reference_rows = []
    for position in range(length):
        reference_state = torch.where(
            reset[:, position, None, None, None, None],
            reset_initial,
            reference_state,
        )
        reference_state = (
            complex_mul(multiplier[:, position], reference_state)
            + bias[:, position]
        )
        reference_rows.append(reference_state)
    reference = torch.stack(reference_rows, dim=1)
    assert torch.allclose(actual, reference, atol=1e-5, rtol=1e-5)
    weight = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(
        (actual * weight).sum(),
        (multiplier, bias, initial, reset_initial),
        retain_graph=True,
    )
    reference_grad = torch.autograd.grad(
        (reference * weight).sum(),
        (multiplier, bias, initial, reset_initial),
    )
    for found, expected in zip(actual_grad, reference_grad):
        assert torch.allclose(found, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="fused Metal scan requires MPS",
)
def test_fused_mps_scan_matches_reference_output_and_gradient():
    torch.manual_seed(11)
    device = torch.device("mps")
    phase = 0.2 * torch.randn(2, 37, 3, 4, 1, device=device)
    magnitude = 0.90 + 0.09 * torch.rand(
        2, 37, 3, 4, 1, device=device
    )
    multiplier = (
        magnitude[..., None]
        * torch.stack((phase.cos(), phase.sin()), dim=-1)
    ).requires_grad_(True)
    bias = (0.03 * torch.randn_like(multiplier)).requires_grad_(True)
    initial = torch.randn(2, 3, 4, 1, 2, device=device, requires_grad=True)
    multiplier_ref = multiplier.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    initial_ref = initial.detach().clone().requires_grad_(True)

    actual = mps_affine_scan(multiplier, bias, initial)
    expected = _sequential(multiplier_ref, bias_ref, initial_ref)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    assert torch.allclose(
        multiplier.grad,
        multiplier_ref.grad,
        atol=3e-5,
        rtol=3e-5,
    )
    assert torch.allclose(bias.grad, bias_ref.grad, atol=3e-5, rtol=3e-5)
    assert torch.allclose(
        initial.grad,
        initial_ref.grad,
        atol=3e-5,
        rtol=3e-5,
    )


@pytest.mark.parametrize(
    "model",
    [
        PureParallelGearV3LM(strict_config()),
        HybridParallelGearLM(hybrid_config()),
        BoundedTransformerLM(bounded_config()),
        BlockHybridGearV4LM(block_additive_config()),
    ],
)
def test_full_and_streaming_logits_match(model):
    torch.manual_seed(5)
    model.eval()
    tokens = torch.randint(0, 97, (2, 19))
    segment, ends = masks(tokens)
    full, _ = model(
        tokens,
        segment_ids=segment,
        sentence_end_mask=ends,
    )
    cache = None
    pieces = []
    sizes = []
    for position in range(tokens.shape[1]):
        logits, cache = model(
            tokens[:, position : position + 1],
            cache=cache,
            use_cache=True,
            segment_ids=segment[:, position : position + 1],
            sentence_end_mask=ends[:, position : position + 1],
        )
        pieces.append(logits)
        sizes.append(cache_bytes(cache))
    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=2e-5, rtol=2e-5)
    assert len(set(sizes)) == 1

    cache = None
    pieces = []
    for start in range(0, tokens.shape[1], 7):
        stop = min(start + 7, tokens.shape[1])
        logits, cache = model(
            tokens[:, start:stop],
            cache=cache,
            use_cache=True,
            segment_ids=segment[:, start:stop],
            sentence_end_mask=ends[:, start:stop],
        )
        pieces.append(logits)
    assert torch.allclose(
        full,
        torch.cat(pieces, dim=1),
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize(
    "model",
    [
        PureParallelGearV3LM(strict_config()),
        HybridParallelGearLM(hybrid_config()),
        BoundedTransformerLM(bounded_config()),
        BlockHybridGearV4LM(block_additive_config()),
    ],
)
def test_future_tokens_do_not_change_past_and_segments_reset(model):
    torch.manual_seed(7)
    model.eval()
    tokens = torch.randint(0, 97, (1, 20))
    segment, ends = masks(tokens)
    original, _ = model(tokens, segment_ids=segment, sentence_end_mask=ends)
    changed = tokens.clone()
    changed[:, 14:] = (changed[:, 14:] + 13) % 97
    altered, _ = model(changed, segment_ids=segment, sentence_end_mask=ends)
    assert torch.allclose(original[:, :14], altered[:, :14], atol=1e-6)
    changed = tokens.clone()
    changed[:, :10] = (changed[:, :10] + 29) % 97
    altered, _ = model(changed, segment_ids=segment, sentence_end_mask=ends)
    assert torch.allclose(original[:, 10:], altered[:, 10:], atol=2e-5, rtol=2e-5)


def test_timescale_bands_cannot_cross_after_optimizer_step():
    model = PureParallelGearV3LM(strict_config(layers=1))
    tokens = torch.randint(0, 97, (2, 32))
    metrics = model.training_step(tokens)
    metrics["total"].backward()
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    record = model.layers[0](
        model.token(tokens),
        token_mask=torch.ones_like(tokens, dtype=torch.bool),
        segment_ids=torch.zeros_like(tokens),
        sentence_end_mask=torch.zeros_like(tokens, dtype=torch.bool),
    )[2]
    half = record["half_life"]
    period = record["period"]
    for bank, (half_band, period_band) in enumerate(
        zip(model.config.half_life_bands, model.config.period_bands)
    ):
        assert float(half[:, :, bank].detach().min()) >= half_band[0] - 1e-5
        assert float(half[:, :, bank].detach().max()) <= half_band[1] + 1e-5
        assert float(period[:, :, bank].detach().min()) >= period_band[0] - 1e-5
        assert float(period[:, :, bank].detach().max()) <= period_band[1] + 1e-5
        log_period = period[:, :, bank].detach().log()
        gear_spacing = log_period[:, :, 1:] - log_period[:, :, :-1]
        assert torch.allclose(
            gear_spacing,
            gear_spacing[:, :, :1].expand_as(gear_spacing),
            atol=1e-6,
            rtol=1e-6,
        )
        expected_direction = torch.tensor([1.0, -1.0, 1.0, -1.0])
        assert torch.equal(
            model.layers[0].rotation_direction[0, :, 0],
            expected_direction,
        )


def test_future_state_objective_reaches_every_bank_head():
    model = PureParallelGearV3LM(strict_config(layers=1))
    tokens = torch.randint(0, 97, (2, 300))
    metrics = model.training_step(tokens)
    assert float(metrics["future_state"].detach()) > 0.0
    metrics["total"].backward()
    assert all(
        head.weight.grad is not None
        and bool(torch.isfinite(head.weight.grad).all())
        and float(head.weight.grad.norm()) > 0.0
        for head in model.future_heads
    )


def test_block_additive_future_state_objective_reaches_every_bank_head():
    model = BlockHybridGearV4LM(block_additive_config())
    tokens = torch.randint(0, 97, (2, 300))
    metrics = model.training_step(tokens)
    assert float(metrics["future_state"].detach()) > 0.0
    metrics["total"].backward()
    assert all(
        head.weight.grad is not None
        and bool(torch.isfinite(head.weight.grad).all())
        and float(head.weight.grad.norm()) > 0.0
        for head in model.future_heads
    )


def test_block_memory_changes_only_after_completed_blocks():
    model = BlockHybridGearV4LM(block_additive_config()).eval()
    tokens = torch.randint(0, 97, (1, 8))
    cache = None
    contexts = []
    with torch.no_grad():
        for position in range(tokens.shape[1]):
            _, cache = model(
                tokens[:, position : position + 1],
                cache=cache,
                use_cache=True,
            )
            contexts.append(cache.gear_memory[0].context.clone())
    assert torch.equal(contexts[0], contexts[1])
    assert torch.equal(contexts[1], contexts[2])
    assert not torch.equal(contexts[2], contexts[3])
    assert torch.equal(contexts[3], contexts[4])


def test_block_selective_film_modulation_is_identity_without_context():
    model = BlockHybridGearV4LM(block_selective_film_config()).eval()
    fusion = model.gear_memories[0].fusion
    assert fusion is not None
    hidden = torch.randn(2, 11, model.config.dim)
    output, gate = fusion(hidden, torch.zeros_like(hidden))
    assert torch.allclose(output, hidden, atol=1e-7, rtol=1e-7)
    assert torch.allclose(gate, torch.full_like(gate, 0.5))


def test_block_selective_film_path_gets_language_model_gradients():
    model = BlockHybridGearV4LM(block_selective_film_config())
    tokens = torch.randint(0, 97, (2, 32))
    loss = model.training_step(
        tokens,
        loss_term_scales={"future_state": 0.0},
    )["total"]
    loss.backward()
    fusion = model.gear_memories[0].fusion
    assert fusion is not None
    assert float(fusion.projection.weight.grad.norm()) > 0.0
    assert float(
        model.gear_memories[0].memory.write_projection.weight.grad.norm()
    ) > 0.0


def test_block_bank_router_is_identity_without_bank_memory():
    model = BlockHybridGearV4LM(block_bank_router_config()).eval()
    router = model.gear_memories[0].bank_router
    assert router is not None
    hidden = torch.randn(2, 3, 4, model.config.dim)
    banks = torch.zeros(
        2,
        3,
        model.config.num_banks,
        model.config.cell_dim,
    )
    output, gate = router(hidden, banks)
    assert torch.allclose(output, hidden, atol=1e-7, rtol=1e-7)
    assert torch.allclose(gate, torch.full_like(gate, 0.5))


def test_block_bank_router_and_rotor_get_language_model_gradients():
    model = BlockHybridGearV4LM(block_bank_router_config())
    tokens = torch.randint(0, 97, (2, 32))
    loss = model.training_step(
        tokens,
        loss_term_scales={"future_state": 0.0},
    )["total"]
    loss.backward()
    memory = model.gear_memories[0]
    assert memory.bank_router is not None
    assert float(memory.bank_router.query.weight.grad.norm()) > 0.0
    assert float(memory.bank_router.value.weight.grad.norm()) > 0.0
    assert float(memory.memory.write_projection.weight.grad.norm()) > 0.0


def test_strict_manifest_and_forward_have_no_host_scalar_control():
    model = PureParallelGearV3LM(strict_config())
    manifest = model.architecture_manifest()
    assert manifest["version"] == 3
    assert manifest["invariants"]["self_attention"] is False
    assert manifest["invariants"]["history_tensor"] is False
    source = inspect.getsource(type(model.layers[0]).forward)
    assert ".item(" not in source
    assert ".cpu(" not in source


class _LaneCorpus:
    vocab_size = 97

    def __init__(self):
        self.position = 0

    def sample_batch(self, batch, seq_len, split="train"):
        del split
        values = torch.arange(
            self.position,
            self.position + batch * seq_len,
        ).reshape(batch, seq_len) % self.vocab_size
        self.position += seq_len
        mask = torch.ones_like(values, dtype=torch.bool)
        segment = torch.zeros_like(values)
        return TrainingBatch(
            values.long(),
            mask,
            mask,
            metadata={
                "segment_ids": segment,
                "sentence_end_mask": torch.zeros_like(mask),
                "contiguous_lanes": True,
            },
        )


def test_stateful_trainer_carries_and_detaches_two_chunks():
    model = PureParallelGearV3LM(strict_config(layers=1))
    trainer = PureParallelGearV3Trainer(
        model,
        _LaneCorpus(),
        device="cpu",
        precision="fp32",
        lr=1e-3,
        total_steps=2,
        stateful=True,
        tbptt_chunks=2,
    )
    records = trainer.train_steps(1, batch_size=2, seq_len=32, log_every=0)
    assert len(records) == 1
    assert trainer._stream_cache is not None
    assert trainer.supervised_tokens_seen == 2 * 2 * 31
    assert all(
        state.rotor.grad_fn is None
        for state in trainer._stream_cache.gear_states
    )


def test_block_additive_stateful_trainer_carries_aligned_block_cache():
    model = BlockHybridGearV4LM(block_additive_config())
    trainer = PureParallelGearV3Trainer(
        model,
        _LaneCorpus(),
        device="cpu",
        precision="fp32",
        lr=1e-3,
        total_steps=2,
        stateful=True,
        tbptt_chunks=2,
    )
    records = trainer.train_steps(
        1, batch_size=2, seq_len=32, log_every=0
    )
    assert len(records) == 1
    assert trainer._stream_cache is not None
    assert trainer._stream_cache.block_offset == 0
    assert all(
        memory.state.rotor.grad_fn is None
        for memory in trainer._stream_cache.gear_memory
    )


def test_long_strict_sequence_remains_finite():
    model = PureParallelGearV3LM(
        strict_config(
            dim=16,
            layers=1,
            ffn_dim=16,
            gears_per_bank=2,
            rotor_channels=1,
            cell_dim=4,
            bank_rank=4,
        )
    )
    tokens = torch.randint(0, 97, (1, 16384))
    metrics = model.training_step(tokens)
    assert torch.isfinite(metrics["total"])
    metrics["total"].backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
