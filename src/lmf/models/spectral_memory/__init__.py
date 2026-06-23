"""Spectral Memory (SM-LM) public API.

A recall-capable, Mac-fast, matmul-first language model: a bank of
error-correcting fast-weight delta memories at log-spaced timescales
(Multi-Timescale Delta Memory) fused by a cross-band router, with a thin slice
of sliding-window attention for exact local copy.
"""

from .delta_scan import delta_rule_chunked, delta_rule_recurrent
from .model import (
    GatedMLP,
    MultiTimescaleDeltaMemory,
    RMSNorm,
    SlidingWindowAttention,
    SpectralMemoryBlock,
    SpectralMemoryConfig,
    SpectralMemoryLM,
    build_spectral_memory,
)
from .trainer import build_spectral_memory_trainer

__all__ = [
    "GatedMLP",
    "MultiTimescaleDeltaMemory",
    "RMSNorm",
    "SlidingWindowAttention",
    "SpectralMemoryBlock",
    "SpectralMemoryConfig",
    "SpectralMemoryLM",
    "build_spectral_memory",
    "build_spectral_memory_trainer",
    "delta_rule_chunked",
    "delta_rule_recurrent",
]
