"""Shared fixtures: a tiny RHCA config/model usable on CPU in milliseconds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lmf.models.rhca import RHCAConfig, RollingFrontierRHCA


@pytest.fixture
def tiny_config() -> RHCAConfig:
    return RHCAConfig(
        vocab_size=64, field_dim=32, latent_dim=16, codebook="lowrank",
        codebook_factor_dim=8, frontier_size=8, max_commit=3, memory_slots=12,
        memory_read_top_k=4, memory_write_top_k=3,
        tail_size=16, local_kernel_size=3, max_hypotheses=1, ssm_macro_steps=2,
        ssm_scan_steps=4, special_token_ids={"eos": 1, "pad": 0})


@pytest.fixture
def tiny_model(tiny_config) -> RollingFrontierRHCA:
    torch.manual_seed(0)
    return RollingFrontierRHCA(tiny_config)


@pytest.fixture
def tiny_tokens(tiny_config) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(3, tiny_config.vocab_size, (3, tiny_config.frontier_size * 4))
