"""Spectral Memory trainer — reuses the shared native-LM loop.

SM-LM scores bits/token exactly like the Transformer baseline (one parallel
teacher-forced forward), so it needs no bespoke optimization loop: it reuses
``NativeLMTrainer`` (which evaluates via ``transformer_bits_per_token``) and only
registers a builder that strips RHCA/gear-only knobs a shared config might carry.
"""

from __future__ import annotations

from ...core.registry import TRAINERS
from .._shared.trainer import NativeLMTrainer, drop_irrelevant


@TRAINERS.register("spectral_memory")
def build_spectral_memory_trainer(model, corpus, **kwargs) -> NativeLMTrainer:
    return NativeLMTrainer(model, corpus, **drop_irrelevant(kwargs))
