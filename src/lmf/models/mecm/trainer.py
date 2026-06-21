"""Trainer registration for MECM."""

from __future__ import annotations

from ...core.registry import TRAINERS
from .._shared.trainer import NativeLMTrainer, drop_irrelevant


@TRAINERS.register("mecm")
def build_mecm_trainer(model, corpus, **kwargs) -> NativeLMTrainer:
    return NativeLMTrainer(model, corpus, **drop_irrelevant(kwargs))
