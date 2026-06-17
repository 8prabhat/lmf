"""Falsification kernels run and pass at smoke scale."""

from __future__ import annotations

import torch

from lmf.experiments.rfk import KERNELS

_CFG = {
    "seed": 0,
    "device": "cpu",
    "model": {
        "vocab_size": 64, "field_dim": 32, "latent_dim": 16, "codebook": "lowrank",
        "codebook_factor_dim": 8, "frontier_size": 8, "max_commit": 3, "memory_slots": 12,
        "memory_read_top_k": 4, "memory_write_top_k": 3,
        "tail_size": 16, "local_kernel_size": 3, "max_hypotheses": 1, "ssm_macro_steps": 2,
        "ssm_scan_steps": 4,
    },
}


def test_all_kernels_pass():
    torch.manual_seed(0)   # deterministic — kernels must not be flaky (finding 8)
    failures = {}
    for name, fn in KERNELS.items():
        result = fn(_CFG)
        if not result.get("pass"):
            failures[name] = result
    assert not failures, f"kernels failed: {failures}"
