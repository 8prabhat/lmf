"""Run one Cell end-to-end: resolve config -> build -> train -> eval -> CellResult.

Family-agnostic by construction: this module never imports a model module. It
goes through ``lmf.core.build.build`` (registry-based), exactly like the CLI.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any, Callable

from ..core.build import build
from ..core.config import ExperimentConfig, apply_overrides, deep_merge
from ..training.callbacks import PeriodicEval
from ..training.checkpoints import architecture_fingerprint
from .matrix import Cell
from .points import BypassError
from .spec import AblationSpec
from .storage import CellResult


def resolve_cell_config(base_raw: dict[str, Any], cell: Cell) -> dict[str, Any]:
    """``deep_merge(base_raw, cell.overrides)`` then ``apply_overrides`` the
    cell's dotted config-axis overrides, and stamp ``cfg["seed"] = cell.seed``."""
    merged = deep_merge(base_raw, cell.overrides)
    merged = apply_overrides(merged, cell.dotted_overrides)
    merged["seed"] = cell.seed
    return merged


def config_hash(resolved_raw: dict[str, Any]) -> str:
    payload = json.dumps(resolved_raw, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# Open registry of (predicate, metric_fn) pairs — any family can append at
# import time (Open/Closed) instead of the framework special-casing names.
# This mirrors the existing `hasattr(model, "prefill")` dispatch in cmd_eval —
# not a new RHCA coupling, same genericity level as before.
EXTRA_METRICS: list[tuple[Callable[[Any], bool], Callable[..., dict[str, float]]]] = []


def _register_default_extra_metrics() -> None:
    from ..evaluation.benchmarks import tokens_per_settle

    def _rhca_tokens_per_settle(model, **_: Any) -> dict[str, float]:
        result = tokens_per_settle(model)
        return {"tokens_per_settle": float(result.get("tokens_per_settle", 1.0))}

    EXTRA_METRICS.append((lambda m: hasattr(m, "prefill"), _rhca_tokens_per_settle))


_register_default_extra_metrics()


def gather_extra_metrics(model, corpus, trainer, run: dict[str, Any],
                          eval_cfg: dict[str, Any]) -> dict[str, float]:
    extra: dict[str, float] = {}
    for predicate, metric_fn in EXTRA_METRICS:
        if predicate(model):
            extra.update(metric_fn(model, corpus=corpus, trainer=trainer, run=run, eval_cfg=eval_cfg))
    return extra


class _CurveCallback(PeriodicEval):
    """``PeriodicEval`` that also appends ``{"step", "eval_bpt"}`` to ``curve``."""

    def __init__(self, curve: list[dict[str, float]], *, every: int, batch_size: int,
                 seq_len: int, n_batches: int = 5, split: str = "valid") -> None:
        super().__init__(every, batch_size, seq_len, n_batches, split)
        self.curve = curve

    def on_step_end(self, trainer, step, record) -> None:
        super().on_step_end(trainer, step, record)
        if "eval_bpt" in record:
            self.curve.append({"step": step, "eval_bpt": record["eval_bpt"]})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cell(cell: Cell, base_raw: dict[str, Any], spec: AblationSpec) -> CellResult:
    resolved = resolve_cell_config(base_raw, cell)
    started_at = _now()
    status, error = "ok", None
    metrics: dict[str, float] = {}
    curve: list[dict[str, float]] = []
    train_seconds = 0.0
    model = None

    try:
        cfg = ExperimentConfig(resolved, block=base_raw.get("__block__", "ablation"))
        corpus, model, trainer, run = build(cfg, run_overrides=spec.run)

        steps = int(run.get("steps", 200))
        batch_size = int(run.get("batch_size", 8))
        seq_len = int(run.get("seq_len", 256))
        n_batches = int(spec.eval.get("n_batches", 5))
        split = spec.eval.get("split", "valid")

        callbacks = []
        if steps > 0:
            curve_cb = _CurveCallback(
                curve, every=max(1, steps // 10), batch_size=batch_size, seq_len=seq_len,
                n_batches=n_batches, split=split)
            callbacks.append(curve_cb)

        with ExitStack() as stack:
            _enter_structural_points(stack, model, cell.structural_points)
            _set_loss_term_scales(trainer, cell.loss_term_scales)

            t0 = time.perf_counter()
            if steps > 0:
                trainer.train_steps(steps, batch_size, seq_len, log_every=0, callbacks=callbacks)
            train_seconds = time.perf_counter() - t0

            bpt = trainer.evaluate_bpt(
                batch_size, seq_len, n_batches=int(spec.eval.get("n_batches", 10)), split=split)
            metrics = {"bits_per_token": bpt,
                       **gather_extra_metrics(model, corpus, trainer, run, spec.eval)}
    except FloatingPointError as exc:
        status, error = "diverged", str(exc)
    except BypassError as exc:
        status, error = "not_ablatable", str(exc)
    except Exception:
        status, error = "failed", traceback.format_exc()

    params_total = sum(p.numel() for p in model.parameters()) if model is not None else 0
    fingerprint = architecture_fingerprint(model) if model is not None else ""

    return CellResult(
        cell_id=cell.cell_id, seed=cell.seed, status=status,
        axis_values=cell.axis_values, overrides=cell.overrides,
        resolved_config=resolved, metrics=metrics, curve=curve,
        train_seconds=train_seconds, params_total=params_total,
        architecture_fingerprint=fingerprint, config_hash=config_hash(resolved),
        error=error, started_at=started_at, finished_at=_now())


def _enter_structural_points(stack: ExitStack, model: Any, structural_points: dict[str, bool]) -> None:
    if not structural_points:
        return
    from .points import apply_point, discover_points

    active = {name for name, on in structural_points.items() if on}
    if not active:
        return
    points = discover_points(model)
    for name in active:
        if name not in points:
            raise KeyError(f"unknown ablation point {name!r}; available: {sorted(points)}")
        stack.enter_context(apply_point(model, points[name]))


def _set_loss_term_scales(trainer: Any, loss_term_scales: dict[str, float]) -> None:
    if not loss_term_scales:
        return
    if getattr(trainer, "_supports_loss_term_scales", False):
        trainer.loss_term_scales = dict(loss_term_scales)
