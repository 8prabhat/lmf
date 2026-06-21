"""Trainer registration for MGCF."""

from __future__ import annotations

from ...core.registry import TRAINERS
from .._shared.trainer import NativeLMTrainer, drop_irrelevant


@TRAINERS.register("mgcf")
def build_mgcf_trainer(model, corpus, **kwargs) -> NativeLMTrainer:
    return NativeLMTrainer(model, corpus, **drop_irrelevant(kwargs))
