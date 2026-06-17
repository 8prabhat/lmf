"""Evaluation: comparable quality metrics and structural benchmarks (review §7)."""

from __future__ import annotations

from .benchmarks import long_context_throughput, needle_in_tail, tokens_per_settle
from .metrics import (
    bits_per_byte,
    bits_per_token,
    calibrate_commit_threshold,
    lm_metrics,
    repetition_rate,
    rhca_bits_per_token,
    rhca_lm_metrics,
    transformer_bits_per_token,
    transformer_lm_metrics,
)

__all__ = [
    "bits_per_byte",
    "bits_per_token",
    "lm_metrics",
    "rhca_bits_per_token",
    "rhca_lm_metrics",
    "transformer_bits_per_token",
    "transformer_lm_metrics",
    "calibrate_commit_threshold",
    "repetition_rate",
    "long_context_throughput",
    "tokens_per_settle",
    "needle_in_tail",
]
