"""Trainer registration for MRWT."""

from __future__ import annotations

from ...core.registry import TRAINERS
from .._shared.trainer import NativeLMTrainer, drop_irrelevant


@TRAINERS.register("mrwt")
def build_mrwt_trainer(model, corpus, **kwargs) -> NativeLMTrainer:
    return NativeLMTrainer(model, corpus, **drop_irrelevant(kwargs))
