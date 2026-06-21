"""Atomic per-cell JSON result storage with resume support."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.io import atomic_write_json


@dataclass
class CellResult:
    cell_id: str
    seed: int
    status: str  # "ok" | "diverged" | "failed" | "not_ablatable" | "skipped"
    axis_values: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    resolved_config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    curve: list[dict[str, float]] = field(default_factory=list)
    train_seconds: float = 0.0
    params_total: int = 0
    architecture_fingerprint: str = ""
    config_hash: str = ""
    error: str | None = None
    started_at: str = ""
    finished_at: str = ""


def result_path(results_dir: str | Path, cell_id: str, seed: int) -> Path:
    return Path(results_dir) / "cells" / f"{cell_id}__seed{seed}.json"


def write_result(results_dir: str | Path, result: CellResult) -> Path:
    """Atomically write ``result`` to its JSON path (write to ``.tmp`` then replace)."""
    path = result_path(results_dir, result.cell_id, result.seed)
    atomic_write_json(path, asdict(result))
    return path


def load_result(results_dir: str | Path, cell_id: str, seed: int) -> CellResult | None:
    """Return ``None`` if missing or corrupt (corrupt is treated as not-yet-run)."""
    path = result_path(results_dir, cell_id, seed)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return CellResult(**data)
    except (json.JSONDecodeError, TypeError, OSError):
        return None


def load_results(results_dir: str | Path) -> list[CellResult]:
    """Load all ``cells/*.json`` results, skipping ``*.tmp`` and corrupt files."""
    cells_dir = Path(results_dir) / "cells"
    if not cells_dir.exists():
        return []
    results: list[CellResult] = []
    for path in sorted(cells_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            results.append(CellResult(**data))
        except (json.JSONDecodeError, TypeError, OSError):
            continue
    return results


def has_result(results_dir: str | Path, cell_id: str, seed: int) -> bool:
    """For resume/skip: True iff a non-``"failed"`` result already exists.

    Failed cells are retried on resume (call with ``force=True`` to retry
    everything, including ``"ok"``/``"diverged"`` cells).
    """
    result = load_result(results_dir, cell_id, seed)
    return result is not None and result.status != "failed"
