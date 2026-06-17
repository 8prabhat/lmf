"""Merge profiling + health + sensitivity into a per-component diagnosis.

Generic over any ``nn.Module``: components are keyed by the dotted module
path produced by ``named_modules()``. Sensitivity results (keyed by ablation
*point* names, e.g. ``"<path>.bypass"``/``"<path>.zero"``/``"<list>.skip[i]"``)
are folded back onto their owning module path where possible.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .health import health_report
from .profiler import profile_model
from .sensitivity import sensitivity_report

# Verdict thresholds -- documented constants, not magic numbers.
BOTTLENECK_PCT_THRESHOLD = 25.0          # fwd_pct + bwd_pct above this -> "bottleneck"
DEGENERATE_GRAD_RATIO_THRESHOLD = 1e-6   # grad_norm/weight_norm below this -> "degenerate"
DEGENERATE_ACT_NEAR_ZERO_THRESHOLD = 0.95  # act_near_zero_frac above this -> "degenerate"
USELESS_DELTA_BPT_THRESHOLD = 0.01       # max |delta_bpt| below this -> "likely-useless"


def _point_owner_path(point_name: str) -> str | None:
    if point_name.endswith(".bypass") or point_name.endswith(".zero"):
        return point_name.rsplit(".", 1)[0]
    return None


def _verdict(entry: dict[str, Any]) -> str:
    profile = entry.get("profile")
    health = entry.get("health")
    sensitivity = entry.get("sensitivity")

    if profile and (profile["fwd_pct"] + profile["bwd_pct"]) > BOTTLENECK_PCT_THRESHOLD:
        return "bottleneck"
    if health is not None:
        if health["grad_to_weight_ratio"] < DEGENERATE_GRAD_RATIO_THRESHOLD:
            return "degenerate"
        if health.get("act_near_zero_frac", 0.0) > DEGENERATE_ACT_NEAR_ZERO_THRESHOLD:
            return "degenerate"
    if sensitivity:
        ok = [r for r in sensitivity if r["status"] == "ok"]
        if ok and max(abs(r["delta_bpt"]) for r in ok) < USELESS_DELTA_BPT_THRESHOLD:
            return "likely-useless"
    return "healthy"


def component_report(model: nn.Module, corpus: Any, *, batch_size: int = 4, seq_len: int = 64,
                     n_batches: int = 3, split: str = "valid", n_warmup: int = 2,
                     n_iters: int = 5) -> dict[str, Any]:
    """Per-component {profile, health, sensitivity, verdict}, plus the raw sensitivity report."""
    sample = corpus.sample_tokenized(batch_size, seq_len, "train")
    if isinstance(sample, torch.Tensor):
        sample = sample.to(next(model.parameters()).device)

    profiling = profile_model(model, sample, n_warmup=n_warmup, n_iters=n_iters)
    health = health_report(model, sample)
    sensitivity = sensitivity_report(model, corpus, batch_size, seq_len, n_batches, split)

    sensitivity_by_path: dict[str, list[dict[str, Any]]] = {}
    for r in sensitivity["points"]:
        owner = _point_owner_path(r["point"])
        if owner is not None:
            sensitivity_by_path.setdefault(owner, []).append(r)

    components: dict[str, dict[str, Any]] = {}
    for path in set(profiling) | set(health):
        entry: dict[str, Any] = {}
        if path in profiling:
            entry["profile"] = profiling[path]
        if path in health:
            entry["health"] = health[path]
        if path in sensitivity_by_path:
            entry["sensitivity"] = sensitivity_by_path[path]
        entry["verdict"] = _verdict(entry)
        components[path] = entry

    verdict_counts: dict[str, int] = {}
    for entry in components.values():
        verdict_counts[entry["verdict"]] = verdict_counts.get(entry["verdict"], 0) + 1

    return {
        "components": components,
        "sensitivity": sensitivity,
        "summary": {
            "n_components": len(components),
            "verdict_counts": verdict_counts,
            "baseline_bpt": sensitivity["baseline_bpt"],
        },
    }


def diagnose(model: nn.Module, corpus: Any, *, batch_size: int = 4, seq_len: int = 64,
             n_batches: int = 3, split: str = "valid", n_warmup: int = 2,
             n_iters: int = 5) -> dict[str, Any]:
    """Top-level entry point used by ``lmf diagnose``."""
    return component_report(model, corpus, batch_size=batch_size, seq_len=seq_len,
                            n_batches=n_batches, split=split, n_warmup=n_warmup, n_iters=n_iters)
