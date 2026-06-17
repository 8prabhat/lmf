"""RHCA — the rolling-frontier model family (v4)."""

from __future__ import annotations

from .codebook import GeometricCodebook, LowRankCodebook, build_codebook
from .config import RHCAConfig
from .memory import SlotMemory
from .model import RollingFrontierRHCA, build_rhca
from .settle import SettleSSM
from .state import (
    AdvanceResult,
    GenerationResult,
    GenerationState,
    SamplingConfig,
)
from .trainer import RHCATrainer  # noqa: E402,F401  (registers the trainer)

__all__ = [
    "RHCAConfig",
    "RollingFrontierRHCA",
    "build_rhca",
    "RHCATrainer",
    "SamplingConfig",
    "GenerationState",
    "AdvanceResult",
    "GenerationResult",
    "SlotMemory",
    "SettleSSM",
    "GeometricCodebook",
    "LowRankCodebook",
    "build_codebook",
]
