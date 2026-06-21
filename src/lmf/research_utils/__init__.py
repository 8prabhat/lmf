"""Shared utilities for the one-off research/benchmark scripts in ``scripts/``.

Cross-architecture framework code (statistics, hashing, atomic IO, device
sync, model introspection) lives in ``lmf.ablation``, ``lmf.core``, and
``lmf.diagnostics`` instead -- this package is only for logic that is
genuinely specific to ad-hoc research scripts, not part of the trainable
framework itself: family-aware checkpoint loading for evaluation, and
zero-dependency SVG report charts.
"""

from __future__ import annotations

from .checkpoints import load_model
from .svg_charts import bar_chart_svg

__all__ = ["load_model", "bar_chart_svg"]
