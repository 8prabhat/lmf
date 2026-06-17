"""Per-ablation-point sensitivity: how much does bypassing/zeroing each
generically-discovered point move ``bits_per_token``?

Built entirely on ``lmf.ablation.points.discover_points`` -- no model-specific
code. Points that can't be safely bypassed (``BypassError``) are reported with
``status="not_ablatable"`` rather than failing the whole report.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..ablation.points import BypassError, apply_point, discover_points
from ..evaluation.metrics import bits_per_token


def sensitivity_report(model: nn.Module, corpus: Any, batch_size: int = 4, seq_len: int = 64,
                       n_batches: int = 3, split: str = "valid") -> dict[str, Any]:
    """Baseline ``bits_per_token``, then re-measure with each point active.

    Returns ``{"baseline_bpt": ..., "points": [...]}`` where ``points`` is
    sorted by ``|delta_bpt|`` descending (``not_ablatable`` points last).

    The baseline and every ablation point are measured on the *same* sampled
    batches: each ``bits_per_token`` call advances the corpus's RNG, so without
    resetting it, deltas would be confounded with sampling noise rather than
    isolating the effect of the ablation.
    """
    has_sampler = hasattr(corpus, "sampler_state") and hasattr(corpus, "load_sampler_state")
    snapshot = corpus.sampler_state() if has_sampler else None

    def _measure() -> float:
        if has_sampler:
            corpus.load_sampler_state(snapshot)
        return bits_per_token(model, corpus, batch_size, seq_len, n_batches, split)

    baseline_bpt = _measure()

    points = discover_points(model)
    results: list[dict[str, Any]] = []
    for name, point in points.items():
        try:
            with apply_point(model, point):
                bpt = _measure()
            results.append({
                "point": name, "status": "ok", "bpt": bpt,
                "delta_bpt": bpt - baseline_bpt,
            })
        except BypassError as exc:
            results.append({"point": name, "status": "not_ablatable", "error": str(exc)})

    if has_sampler:
        corpus.load_sampler_state(snapshot)

    results.sort(key=lambda r: abs(r.get("delta_bpt", 0.0)), reverse=True)
    return {"baseline_bpt": baseline_bpt, "points": results}
