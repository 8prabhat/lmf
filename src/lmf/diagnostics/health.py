"""Gradient/weight/activation health, per parameterized submodule.

Generic over any ``nn.Module``: one backward pass, forward hooks capture the
first tensor leaf of each parameterized submodule's output for activation
stats, and parameter ``.grad``/``.data`` give weight/gradient norms.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _first_tensor(out: Any) -> torch.Tensor | None:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            t = _first_tensor(o)
            if t is not None:
                return t
        return None
    if isinstance(out, dict):
        for o in out.values():
            t = _first_tensor(o)
            if t is not None:
                return t
        return None
    return None


def health_report(model: nn.Module, batch: torch.Tensor) -> dict[str, dict[str, Any]]:
    """Run one training step and return per-module gradient/weight/activation stats."""
    leaves = [(name, m) for name, m in model.named_modules()
              if name and any(True for _ in m.parameters(recurse=False))]

    activations: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module: nn.Module, _inp: Any, out: Any) -> None:
            t = _first_tensor(out)
            if t is not None:
                activations[name] = t.detach()
        return hook

    handles = [m.register_forward_hook(make_hook(name)) for name, m in leaves]
    try:
        model.zero_grad(set_to_none=True)
        losses = model.training_step(batch)
        losses["total"].backward()
    finally:
        for h in handles:
            h.remove()

    report: dict[str, dict[str, Any]] = {}
    for name, m in leaves:
        params = list(m.parameters(recurse=False))
        weight_norm = sum(p.detach().float().norm().item() ** 2 for p in params) ** 0.5
        grads = [p.grad for p in params if p.grad is not None]
        grad_norm = sum(g.float().norm().item() ** 2 for g in grads) ** 0.5 if grads else 0.0
        entry: dict[str, Any] = {
            "weight_norm": weight_norm,
            "grad_norm": grad_norm,
            "grad_to_weight_ratio": grad_norm / weight_norm if weight_norm > 0 else 0.0,
        }
        if name in activations:
            act = activations[name].float()
            entry["act_mean"] = act.mean().item()
            entry["act_std"] = act.std().item()
            entry["act_near_zero_frac"] = (act.abs() < 1e-6).float().mean().item()
        report[name] = entry
    return report
