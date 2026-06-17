"""Forward/backward timing profile, per parameterized submodule.

Generic over any ``nn.Module``: walks ``named_modules()`` for submodules with
*direct* parameters and times their forward/backward contribution via hooks.
Backward timing is a heuristic (elapsed time between consecutive
``register_full_backward_hook`` firings, which run in roughly reverse
topological order) -- adequate for spotting relative bottlenecks, not a
substitute for ``torch.profiler``.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


def _parameterized_leaves(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(name, m) for name, m in model.named_modules()
            if name and any(True for _ in m.parameters(recurse=False))]


def profile_model(model: nn.Module, batch: torch.Tensor, n_warmup: int = 2,
                  n_iters: int = 5) -> dict[str, dict[str, Any]]:
    """Run ``n_warmup + n_iters`` training steps, returning per-module timings.

    Each entry has ``fwd_ms``/``bwd_ms`` (averaged over ``n_iters``) and
    ``fwd_pct``/``bwd_pct`` (share of total forward/backward time across all
    profiled modules), plus ``n_params`` (direct parameter count).
    """
    leaves = _parameterized_leaves(model)
    fwd_times = {name: 0.0 for name, _ in leaves}
    bwd_times = {name: 0.0 for name, _ in leaves}
    n_params = {name: sum(p.numel() for p in m.parameters(recurse=False)) for name, m in leaves}
    bwd_clock = {"t": 0.0}

    def fwd_pre(module: nn.Module, _inp: Any) -> None:
        module._diag_fwd_t0 = time.perf_counter()

    def make_fwd_post(name: str):
        def hook(module: nn.Module, _inp: Any, _out: Any) -> None:
            fwd_times[name] += time.perf_counter() - module._diag_fwd_t0
        return hook

    def make_bwd_hook(name: str):
        def hook(_module: nn.Module, _grad_input: Any, _grad_output: Any) -> None:
            now = time.perf_counter()
            bwd_times[name] += now - bwd_clock["t"]
            bwd_clock["t"] = now
        return hook

    handles = []
    for name, m in leaves:
        handles.append(m.register_forward_pre_hook(fwd_pre))
        handles.append(m.register_forward_hook(make_fwd_post(name)))
        handles.append(m.register_full_backward_hook(make_bwd_hook(name)))

    try:
        for i in range(n_warmup + n_iters):
            if i == n_warmup:
                for name in fwd_times:
                    fwd_times[name] = 0.0
                    bwd_times[name] = 0.0
            model.zero_grad(set_to_none=True)
            losses = model.training_step(batch)
            bwd_clock["t"] = time.perf_counter()
            losses["total"].backward()
    finally:
        for h in handles:
            h.remove()
        for _, m in leaves:
            if hasattr(m, "_diag_fwd_t0"):
                del m._diag_fwd_t0

    n_iters = max(1, n_iters)
    fwd_total = sum(fwd_times.values()) or 1.0
    bwd_total = sum(bwd_times.values()) or 1.0
    return {
        name: {
            "fwd_ms": 1000.0 * fwd_times[name] / n_iters,
            "bwd_ms": 1000.0 * bwd_times[name] / n_iters,
            "fwd_pct": 100.0 * fwd_times[name] / fwd_total,
            "bwd_pct": 100.0 * bwd_times[name] / bwd_total,
            "n_params": n_params[name],
        }
        for name, _ in leaves
    }
