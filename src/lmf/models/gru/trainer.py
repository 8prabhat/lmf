"""Trainer registration for the GRU control baseline."""

from ...core.registry import TRAINERS
from ...training.base_trainer import BaseTrainer


@TRAINERS.register("gru_lm")
def build_gru_trainer(model, corpus, **kwargs):
    return BaseTrainer(model, corpus, **kwargs)
