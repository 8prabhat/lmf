"""Fast, deterministic proof that tail-attention RoPE is actually wired in.

This is deliberately separate from tests/test_rhca_recall.py: that file asks
"did training learn to exploit relative position" (slow, needs real training,
and — per direct investigation — needs more steps than is practical for a fast
CI gate). This file asks the cheaper, structural question: "is the mechanism
that *could* learn it actually present," by checking a property pure
content-addressed attention can never have, with no training involved.

If FrontierDynamicsRule's tail attention is pure content addressing (no
position signal), moving an identical content vector to a different tail slot
must leave the output exactly unchanged — SDPA is permutation-equivariant in
its key/value pairs, so relabelling which slot holds which content is a no-op
unless something makes the result depend on slot position. With tail_rope
enabled, moving the same content to a different slot must change the output
(the whole point of adding it).
"""

from __future__ import annotations

import torch

from lmf.models.rhca.config import RHCAConfig
from lmf.models.rhca.dynamics import FrontierDynamicsRule


def _run(tail_rope: bool, needle_slot: int) -> torch.Tensor:
    cfg = RHCAConfig(
        vocab_size=64, field_dim=32, latent_dim=16, tail_size=20,
        frontier_size=4, max_commit=2, memory_slots=8, memory_read_top_k=4,
        memory_write_top_k=2, local_kernel_size=3, ssm_macro_steps=1,
        ssm_scan_steps=2, tail_rope=tail_rope,
    )
    torch.manual_seed(1)
    rule = FrontierDynamicsRule(cfg)
    b, p, h, d = 1, 1, cfg.frontier_size, cfg.field_dim
    torch.manual_seed(2)
    frontier = torch.randn(b, p, h, d)
    memory = torch.zeros(b, cfg.memory_slots, d)
    torch.manual_seed(3)
    needle = torch.randn(d) * 3.0
    tail = torch.zeros(b, cfg.tail_size, d)
    tail[:, needle_slot] = needle
    with torch.no_grad():
        return rule(frontier, memory, tail)


def test_tail_rope_disabled_is_position_invariant():
    base, moved = _run(False, needle_slot=5), _run(False, needle_slot=15)
    assert torch.equal(base, moved)


def test_tail_rope_enabled_is_position_sensitive():
    base, moved = _run(True, needle_slot=5), _run(True, needle_slot=15)
    assert (base - moved).abs().max() > 1e-4
