"""RHCA model: prefill, settle, generation, manifest, unshared macro steps."""

from __future__ import annotations

import torch

from lmf.models.rhca.state import SamplingConfig


def test_prefill_state_shapes(tiny_model, tiny_config, tiny_tokens):
    state = tiny_model.prefill(tiny_tokens[:, : tiny_config.frontier_size])
    b = tiny_tokens.shape[0]
    assert state.memory.shape == (b, tiny_config.memory_slots, tiny_config.field_dim)
    assert state.tail.shape == (b, tiny_config.tail_size, tiny_config.field_dim)
    assert state.frontier.shape[2:] == (tiny_config.frontier_size, tiny_config.field_dim)


def test_settle_runs_without_auxiliary_energy_compute(tiny_model, tiny_config, tiny_tokens):
    state = tiny_model.prefill(tiny_tokens[:, : tiny_config.frontier_size])
    frontier, traj = tiny_model.settle(state, active_only=True)
    assert frontier.shape[1] == 1  # single active hypothesis
    assert len(traj) >= 2


def test_unshared_macro_steps_have_distinct_weights(tiny_model):
    blocks = tiny_model.settle_ssm.blocks
    assert len(blocks) >= 2
    w0 = blocks[0].correction_rule.mix_in.weight
    w1 = blocks[1].correction_rule.mix_in.weight
    assert w0.data_ptr() != w1.data_ptr()


def test_generate_block_commit(tiny_model, tiny_config):
    prompt = torch.randint(3, tiny_config.vocab_size, (2, 6))
    result = tiny_model.generate(prompt, 24, SamplingConfig(deterministic=True))
    assert result.token_ids.shape == (2, 24)
    assert result.tokens_per_settle >= 1.0
    assert result.cycles > 0


def test_manifest_reports_compute_fraction(tiny_model):
    manifest = tiny_model.architecture_manifest()
    assert manifest["name"] == "RollingFrontierRHCA"
    # Factorised codebook => the bulk of params are compute, not embedding.
    assert manifest["parameters"]["compute_fraction"] > 0.5
