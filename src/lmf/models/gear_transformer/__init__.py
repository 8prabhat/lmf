"""Generative Multi-Scale Gear Transformer family."""

from __future__ import annotations

from .model import (
    GearAlignmentScorer,
    GearSlotRouter,
    GearTransformerConfig,
    HierarchicalGearClock,
    MHGTransformerLM,
    MultiRateGearModule,
    SimplifiedGearTransformerLM,
    build_gear_only_transformer,
    build_gear_transformer,
    build_multi_rate_latent_gear_transformer,
    build_simplified_gear_transformer,
)
from .parallel import ParallelGearSystem, PositiveParallelGearClock
from .trainer import (
    build_gear_only_trainer,
    build_gear_transformer_trainer,
    build_multi_rate_latent_gear_transformer_trainer,
    build_simplified_gear_transformer_trainer,
)

__all__ = [
    "GearAlignmentScorer",
    "GearSlotRouter",
    "GearTransformerConfig",
    "HierarchicalGearClock",
    "MHGTransformerLM",
    "MultiRateGearModule",
    "SimplifiedGearTransformerLM",
    "ParallelGearSystem",
    "PositiveParallelGearClock",
    "build_gear_only_transformer",
    "build_gear_transformer",
    "build_multi_rate_latent_gear_transformer",
    "build_simplified_gear_transformer",
    "build_gear_only_trainer",
    "build_gear_transformer_trainer",
    "build_multi_rate_latent_gear_transformer_trainer",
    "build_simplified_gear_transformer_trainer",
]
