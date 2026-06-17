"""Generic loss-term axis discovery.

Any family that returns extra named scalars from ``training_step`` (besides
``"total"``) automatically gets those names as candidate ``loss_term:`` axis
targets — no per-family registration needed.
"""

from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def discover_loss_terms(model: Any, sample_tokens: torch.Tensor,
                        task_metadata: dict[str, Any] | None = None) -> list[str]:
    """Run one no-grad ``training_step`` and return scalar-tensor keys != "total"."""
    was_training = model.training
    model.eval()
    try:
        losses = model.training_step(sample_tokens, task_metadata)
    finally:
        model.train(was_training)
    return sorted(
        key for key, value in losses.items()
        if key != "total" and torch.is_tensor(value) and value.dim() == 0
    )
