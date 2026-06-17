"""Ablation studies: config-axis sweeps + generic structural/loss-term ablation.

See ``lmf.ablation.spec`` for the spec format, ``lmf.ablation.executor`` for
running a sweep, and ``lmf.ablation.points`` for the generic structural
ablation mechanism (works across any registered model family).
"""

from __future__ import annotations

from .executor import run_ablation
from .matrix import Cell, build_matrix
from .points import BypassError, PointSpec, apply_point, bypass_module, discover_points, skip_listed_module
from .report import build_report, maybe_plot, write_report
from .runner import run_cell
from .spec import AblationSpec, AxisSpec, VariantSpec, load_ablation_spec
from .storage import CellResult, load_results, write_result

__all__ = [
    "AblationSpec",
    "AxisSpec",
    "VariantSpec",
    "load_ablation_spec",
    "Cell",
    "build_matrix",
    "CellResult",
    "run_cell",
    "run_ablation",
    "load_results",
    "write_result",
    "build_report",
    "write_report",
    "maybe_plot",
    "PointSpec",
    "BypassError",
    "discover_points",
    "bypass_module",
    "skip_listed_module",
    "apply_point",
]
