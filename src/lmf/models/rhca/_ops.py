"""Small shared tensor ops for the RHCA family."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS-normalise the last dimension."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)
