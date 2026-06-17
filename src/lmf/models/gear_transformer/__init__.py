"""Multi-Rate Latent Gear Transformer family."""

from __future__ import annotations

from .model import (
    GearAlignmentScorer,
    GearSlotRouter,
    GearTransformerConfig,
    MHGTransformerLM,
    MultiRateGearModule,
    build_gear_only_transformer,
    build_gear_transformer,
    build_multi_rate_latent_gear_transformer,
)
from .trainer import (
    build_gear_only_trainer,
    build_gear_transformer_trainer,
    build_multi_rate_latent_gear_transformer_trainer,
)

__all__ = [
    "GearAlignmentScorer",
    "GearSlotRouter",
    "GearTransformerConfig",
    "MHGTransformerLM",
    "MultiRateGearModule",
    "build_gear_only_transformer",
    "build_gear_transformer",
    "build_multi_rate_latent_gear_transformer",
    "build_gear_only_trainer",
    "build_gear_transformer_trainer",
    "build_multi_rate_latent_gear_transformer_trainer",
]
