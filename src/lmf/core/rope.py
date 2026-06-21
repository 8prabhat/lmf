"""Rotary position embedding, shared across model families.

Originally lived only in the transformer family (which needs simple sequential
positions). RHCA's tail-attention reuses the same math but with two non-trivial
position streams (frontier draft offset for queries, tail recency for keys) —
factored out here so both families apply one tested implementation rather than
each carrying their own copy.
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotate the last dimension of ``x`` by a per-position angle.

    ``x`` is ``(..., seq, dim)``; ``positions`` is ``(seq,)`` and may be
    negative (e.g. "distance into the past") — only relative phase differences
    between query and key positions affect the resulting attention scores, so
    queries and keys may use independent position conventions (e.g. a query at
    "+p" and a key at "-d") as long as each is internally consistent.
    """
    dim = x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    ang = positions[:, None].float() * inv_freq[None, :]
    ang = torch.cat([ang, ang], dim=-1).to(x.dtype)
    view_shape = (1,) * (x.dim() - 2) + ang.shape
    cos, sin = ang.cos().view(view_shape), ang.sin().view(view_shape)
    return x * cos + rotate_half(x) * sin
