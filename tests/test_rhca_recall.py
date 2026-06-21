"""Regression guard for the exact-recall (tail-attention) capability.

Direct investigation found that RHCA's tail-attention mechanism — the
architecture's signature long-range feature — was not being learned at all:
on ProceduralCorpus's deterministic echo task (a trivial copy from a fixed
distance back), echo-position loss was *worse* than the harder, genuinely
stochastic Markov-position loss, even after thousands of steps. These tests
are the permanent guard against that regression recurring.

Marked slow (excluded from the default `pytest tests/` run; pass `-m slow`
or unset the marker filter to run them) since they involve real training.
"""

from __future__ import annotations

import math

import pytest
import torch

from lmf.data import ProceduralCorpus
from lmf.evaluation.benchmarks import needle_in_tail
from lmf.experiments.rfk.kernels import rfk_echo_recovery
from lmf.models.rhca import RHCAConfig, RollingFrontierRHCA

pytestmark = pytest.mark.slow

_SMOKE_MODEL = dict(
    vocab_size=512, field_dim=128, latent_dim=64, codebook="lowrank",
    codebook_factor_dim=32, frontier_size=16, max_commit=4, memory_slots=32,
    memory_read_top_k=8, memory_write_top_k=4, memory_write_temperature=3.0,
    tail_size=64, local_kernel_size=5, max_hypotheses=1, ssm_macro_steps=2,
    ssm_scan_steps=6, commit_entropy_threshold=0.80, routing_balance_weight=0.05,
)


def test_echo_recovery_kernel_passes_at_smoke_scale():
    result = rfk_echo_recovery({"seed": 0, "device": "cpu", "model": dict(_SMOKE_MODEL)})
    assert result["pass"], result
    assert result["echo_ce"] < result["non_echo_ce"]
    assert result["echo_ce"] < 0.7 * result["uniform_floor"]


def test_needle_in_tail_recall_beats_chance_after_training():
    """Independent cross-check via the purpose-built induction/copy probe."""
    torch.manual_seed(0)
    config = RHCAConfig(**_SMOKE_MODEL, special_token_ids={"eos": 1, "pad": 0})
    model = RollingFrontierRHCA(config)
    corpus = ProceduralCorpus(vocab_size=config.vocab_size, seed=0,
                              echo_distance=24, echo_every=8)
    h = config.frontier_size

    before = needle_in_tail(model, distance=24, vocab_size=config.vocab_size)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(200):
        tokens = corpus.sample_tokenized(8, h * 8, "train")
        opt.zero_grad()
        model.carried_state_training_step(tokens)["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    chance = before["chance_level"]
    for distance in (8, 24, 56):
        after = needle_in_tail(model, distance=distance, vocab_size=config.vocab_size)
        assert after["recall_accuracy"] > 10 * chance, (distance, after)
