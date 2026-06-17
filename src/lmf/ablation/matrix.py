"""Expand an :class:`AblationSpec` into a concrete, deterministic list of Cells.

This module is purely config/structure-level — it never imports a model
module, so it works identically for any registered model family.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import combinations, product
from typing import Any

from ..core.config import apply_overrides, deep_merge
from .spec import AblationSpec, AxisSpec


@dataclass(frozen=True)
class Cell:
    cell_id: str
    axis_values: dict[str, Any]
    overrides: dict[str, Any]
    dotted_overrides: list[str]
    seed: int
    is_baseline: bool = False
    variant_name: str | None = None
    structural_points: dict[str, bool] = field(default_factory=dict)
    loss_term_scales: dict[str, float] = field(default_factory=dict)
    aliases: list[dict[str, Any]] = field(default_factory=list)


def _slugify(value: Any) -> str:
    """Path-safe token for a cell id, e.g. ``True -> "true"``, ``0.1 -> "0p1"``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        s = repr(value)
        return s.replace("-", "neg").replace(".", "p")
    if isinstance(value, int):
        return str(value)
    # Dotted config paths (e.g. "model.dim") are filesystem-safe as-is — keep
    # "." but normalize the genuinely unsafe characters.
    s = str(value)
    for ch in " :/\\[]{}()<>,=\"'":
        s = s.replace(ch, "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "x"


def cell_id_for(axis_values: dict[str, Any], variant_name: str | None, seed: int) -> str:
    """Deterministic, filesystem-safe id, e.g. ``"dim=256__layers=8__seed1"``."""
    parts: list[str] = []
    if variant_name is not None:
        parts.append(f"variant={_slugify(variant_name)}")
    parts.extend(f"{_slugify(k)}={_slugify(v)}" for k, v in axis_values.items())
    if not parts:
        parts.append("baseline")
    return "__".join(parts) + f"__seed{seed}"


def _flatten_overrides(d: dict[str, Any], prefix: str = "") -> list[str]:
    """``dict -> ["a.b.c=value", ...]`` (mirrors ``apply_overrides`` input)."""
    out: list[str] = []
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.extend(_flatten_overrides(value, path))
        else:
            out.append(f"{path}={value!r}")
    return out


def _apply_axis(axis: AxisSpec, value: Any, *, dotted_overrides: list[str],
                structural_points: dict[str, bool], loss_term_scales: dict[str, float]) -> None:
    if axis.kind == "config":
        dotted_overrides.append(f"{axis.path}={value!r}")
    elif axis.kind == "structural":
        structural_points[axis.target] = bool(value)
    elif axis.kind == "loss_term":
        loss_term_scales[axis.target] = float(value)
    else:  # pragma: no cover - exhaustive by AxisSpec.kind
        raise ValueError(f"unknown axis kind {axis.kind!r}")


def _build_cell(axis_value_pairs: list[tuple[AxisSpec, Any]], *, base_overrides: dict[str, Any],
                seed: int, variant_name: str | None = None, is_baseline: bool = False) -> Cell:
    axis_values: dict[str, Any] = {}
    dotted_overrides = _flatten_overrides(base_overrides)
    structural_points: dict[str, bool] = {}
    loss_term_scales: dict[str, float] = {}
    for axis, value in axis_value_pairs:
        axis_values[axis.display_name] = value
        _apply_axis(axis, value, dotted_overrides=dotted_overrides,
                    structural_points=structural_points, loss_term_scales=loss_term_scales)
    return Cell(
        cell_id=cell_id_for(axis_values, variant_name, seed),
        axis_values=axis_values,
        overrides=deepcopy(base_overrides),
        dotted_overrides=dotted_overrides,
        seed=seed,
        is_baseline=is_baseline,
        variant_name=variant_name,
        structural_points=structural_points,
        loss_term_scales=loss_term_scales,
    )


def _axis_changes_config(axis: AxisSpec, base_resolved: dict[str, Any]) -> bool:
    """True iff at least one of ``axis.values`` actually changes the resolved config."""
    for value in axis.values:
        candidate = apply_overrides(deepcopy(base_resolved), [f"{axis.path}={value!r}"])
        if candidate != base_resolved:
            return True
    return False


def _validate_axes(axes: list[AxisSpec], base_resolved: dict[str, Any]) -> None:
    for axis in axes:
        if axis.kind != "config":
            continue
        if not _axis_changes_config(axis, base_resolved):
            raise ValueError(
                f"ablation axis {axis.path!r} does not change the resolved config for "
                f"any of its values {axis.values!r} — likely a typo'd path")


def _resolved_signature(cell: Cell, base_resolved: dict[str, Any]) -> str:
    merged = deep_merge(base_resolved, cell.overrides)
    merged = apply_overrides(merged, cell.dotted_overrides)
    payload = {
        "config": merged,
        "structural": cell.structural_points,
        "loss_term": cell.loss_term_scales,
        "seed": cell.seed,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _dedup(cells: list[Cell], base_resolved: dict[str, Any]) -> list[Cell]:
    """Alias cells whose resolved config+structural+loss_term+seed is identical
    to an already-emitted cell — kept once, with aliased axis labels recorded."""
    seen: dict[str, int] = {}
    result: list[Cell] = []
    for cell in cells:
        sig = _resolved_signature(cell, base_resolved)
        if sig in seen:
            idx = seen[sig]
            existing = result[idx]
            if cell.axis_values and cell.axis_values != existing.axis_values:
                result[idx] = replace(existing, aliases=existing.aliases + [cell.axis_values])
            continue
        seen[sig] = len(result)
        result.append(cell)
    return result


def _baseline_cells(spec: AblationSpec) -> list[Cell]:
    return [_build_cell([], base_overrides={}, seed=seed, is_baseline=True) for seed in spec.seeds]


def build_matrix(spec: AblationSpec, base_resolved: dict[str, Any]) -> list[Cell]:
    """Expand ``spec`` into a deterministic list of :class:`Cell`.

    ``base_resolved`` is the fully merged base config dict (``load_config(...).raw``),
    used to validate that each config-kind axis actually changes something, and
    to dedup cells whose resolved configuration is identical.
    """
    _validate_axes(spec.axes, base_resolved)

    cells: list[Cell] = _baseline_cells(spec)

    if spec.mode == "grid":
        for seed in spec.seeds:
            for combo in product(*(axis.values for axis in spec.axes)):
                pairs = list(zip(spec.axes, combo))
                if not pairs:
                    continue
                cells.append(_build_cell(pairs, base_overrides={}, seed=seed))

    elif spec.mode == "one_at_a_time":
        for axis in spec.axes:
            for value in axis.values:
                for seed in spec.seeds:
                    cells.append(_build_cell([(axis, value)], base_overrides={}, seed=seed))

    elif spec.mode == "named_variants":
        for variant in spec.variants:
            for seed in spec.seeds:
                if spec.axes:
                    for axis in spec.axes:
                        for value in axis.values:
                            cells.append(_build_cell(
                                [(axis, value)], base_overrides=variant.overrides,
                                seed=seed, variant_name=variant.name))
                else:
                    cells.append(_build_cell(
                        [], base_overrides=variant.overrides, seed=seed, variant_name=variant.name))

    elif spec.mode == "pairwise":
        for i, j in combinations(range(len(spec.axes)), 2):
            axis_i, axis_j = spec.axes[i], spec.axes[j]
            for value_i, value_j in product(axis_i.values, axis_j.values):
                for seed in spec.seeds:
                    cells.append(_build_cell(
                        [(axis_i, value_i), (axis_j, value_j)], base_overrides={}, seed=seed))

    else:
        raise ValueError(f"unknown ablation mode {spec.mode!r}")

    return _dedup(cells, base_resolved)
