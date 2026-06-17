"""Trainer registration for the Gear Transformer family."""

from __future__ import annotations

from ...core.registry import TRAINERS
from ..transformer.trainer import TransformerTrainer


@TRAINERS.register("gear_transformer")
def build_gear_transformer_trainer(model, corpus, **kwargs) -> TransformerTrainer:
    return TransformerTrainer(model, corpus, **kwargs)


@TRAINERS.register("mlgt")
def build_multi_rate_latent_gear_transformer_trainer(model, corpus, **kwargs) -> TransformerTrainer:
    return build_gear_transformer_trainer(model, corpus, **kwargs)


@TRAINERS.register("gear_only")
def build_gear_only_trainer(model, corpus, **kwargs) -> TransformerTrainer:
    return build_gear_transformer_trainer(model, corpus, **kwargs)
