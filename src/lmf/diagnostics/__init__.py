"""Per-component diagnostics: profiling, gradient/activation health, and
ablation-point sensitivity, with an automatic per-component verdict.

Generic over any ``nn.Module`` via ``lmf.ablation.points``; see
``lmf.diagnostics.report`` for the top-level ``diagnose`` entry point.
"""

from __future__ import annotations

from .health import health_report
from .profiler import profile_model
from .report import component_report, diagnose
from .sensitivity import sensitivity_report

__all__ = [
    "profile_model",
    "health_report",
    "sensitivity_report",
    "component_report",
    "diagnose",
]
