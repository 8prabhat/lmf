"""Canonical Pure Parallel Gear public API."""

from .model import (
    FastWeightMemory,
    FastWeightMemoryState,
    GearCache,
    GearState,
    PureGearLayer,
    PureParallelGearConfig,
    PureParallelGearLM,
    build_pure_parallel_gear,
)
from .trainer import PureParallelGearTrainer, build_pure_parallel_gear_trainer

__all__ = [
    "FastWeightMemory",
    "FastWeightMemoryState",
    "GearCache",
    "GearState",
    "PureGearLayer",
    "PureParallelGearConfig",
    "PureParallelGearLM",
    "PureParallelGearTrainer",
    "build_pure_parallel_gear",
    "build_pure_parallel_gear_trainer",
]
