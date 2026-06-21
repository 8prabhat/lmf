"""Swappable token codebooks (Strategy pattern) — review Q2.1 / Q7.

Both implement one interface so the model never cares which is in use:

    embed(ids)      -> (..., D)   rms-normalised token hypervector
    logits(field)   -> (..., V)   tied cosine logits over the whole vocabulary

``LowRankCodebook`` is the v4 default. It factorises the V x D embedding into a
per-token latent ``code`` (V x e) and a shared up-projection ``up`` (e x D):

    embedding[v] = up(code[v])

so the parameter count drops from V*D (54.6M at V=32768, D=1664) to V*e + e*D
(~8.8M at e=256), freeing the budget the review wants to spend on depth. Decoding
is still tied cosine over the reconstructed vectors, so its FLOP cost matches a
normal output projection — only the parameter count shrinks.

``GeometricCodebook`` is the flat tied-embedding fallback for ablations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._ops import rms


class _BaseCodebook(nn.Module):
    def embed(self, ids: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def logits(self, field: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def decode_weight(self) -> torch.Tensor:  # pragma: no cover - interface
        """Reconstruct the V x D decode matrix (the part that's V*e+e*D for the
        low-rank codebook, not free). Callers that decode the same field shape
        repeatedly in one pass (e.g. advance()'s per-token commit loop) should
        call this once and reuse it via logits_from_weight, instead of paying
        the reconstruction cost on every call to logits()."""
        raise NotImplementedError

    def logits_from_weight(self, field: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (F.normalize(field, dim=-1) @ F.normalize(weight, dim=-1).t()) + self.bias


class GeometricCodebook(_BaseCodebook):
    """Flat tied input/output geometry with a learned logit temperature."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, dim) / math.sqrt(dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return rms(F.embedding(ids, self.weight))

    def decode_weight(self) -> torch.Tensor:
        return self.weight

    def logits(self, field: torch.Tensor) -> torch.Tensor:
        return self.logits_from_weight(field, self.weight)


class LowRankCodebook(_BaseCodebook):
    """Low-rank factorised tied embedding/decoder (review Q2.1)."""

    def __init__(self, vocab_size: int, dim: int, factor_dim: int = 256) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.factor_dim = int(factor_dim)
        self.code = nn.Parameter(torch.randn(vocab_size, factor_dim) / math.sqrt(factor_dim))
        self.up = nn.Linear(factor_dim, dim, bias=False)
        # Per-token learned decode scale (review Q7): gives the decode side more
        # freedom than a single shared temperature without re-pinning the clamp.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def decode_weight(self) -> torch.Tensor:
        """Reconstruct the full V x D embedding (used for tied decoding)."""
        return self.up(self.code)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        # Gather code rows first, then up-project — cheap (B*N*e*D), avoids
        # materialising the full V x D matrix on the embed path.
        return rms(self.up(F.embedding(ids, self.code)))

    def logits(self, field: torch.Tensor) -> torch.Tensor:
        return self.logits_from_weight(field, self.decode_weight())


def build_codebook(kind: str, vocab_size: int, dim: int, factor_dim: int) -> _BaseCodebook:
    if kind == "geometric":
        return GeometricCodebook(vocab_size, dim)
    if kind == "lowrank":
        return LowRankCodebook(vocab_size, dim, factor_dim)
    raise ValueError(f"unknown codebook kind {kind!r}")
