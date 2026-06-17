"""Settle-interior modules: the correction rule and the multi-query reader.

The exact-recall **tail** read uses ``F.scaled_dot_product_attention``. The
**memory** read stays a cosine top-k gather, because only top-k of S slots is
wanted and dense attention over S=128 slots would waste work. The derived plan
read duplicated memory context and was removed from this hot path.

The correction rule's mixing MLP is widened to ``Linear(3r->2r) -> GELU ->
Linear(2r->r)`` (review Q2.3): the single per-macro-step nonlinearity was the
expressivity bottleneck.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._ops import rms


def _read_topk(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
               top_k: int) -> torch.Tensor:
    """Cosine top-k retrieval. query (bp, Q, r); keys/values (bp, S, r)."""
    s = keys.shape[-2]
    top_k = min(top_k, s)
    scores = F.normalize(query, dim=-1) @ F.normalize(keys, dim=-1).transpose(-1, -2)
    if top_k < s:
        top_scores, idx = scores.topk(top_k, dim=-1)
        batch_idx = torch.arange(query.shape[0], device=values.device).view(-1, 1, 1).expand_as(idx)
        chosen = values[batch_idx, idx]                       # (bp, Q, top_k, r)
        return (top_scores.softmax(-1).unsqueeze(-1) * chosen).sum(-2)
    return scores.softmax(-1) @ values


class FrontierDynamicsRule(nn.Module):
    """One nonlinear gated-residual settle update with memory and tail reads."""

    def __init__(self, cfg) -> None:
        super().__init__()
        d, r = cfg.field_dim, cfg.latent_dim
        self.cfg = cfg
        self.down = nn.Linear(d, r, bias=False)
        self.local_kernel = nn.Parameter(
            torch.randn(cfg.local_kernel_size, r) / math.sqrt(cfg.local_kernel_size))
        self.memory_key = nn.Linear(d, r, bias=False)
        self.memory_value = nn.Linear(d, r, bias=False)
        self.tail_q = nn.Linear(r, r, bias=False)
        self.tail_k = nn.Linear(d, r, bias=False)
        self.tail_v = nn.Linear(d, r, bias=False)
        # Widened mixing MLP (review Q2.3).
        self.mix_in = nn.Linear(3 * r, 2 * r, bias=False)
        self.mix_out = nn.Linear(2 * r, r, bias=False)
        self.up = nn.Linear(r, d, bias=False)
        self.gate = nn.Linear(r, 1)
        nn.init.constant_(self.gate.bias, -1.0)

    def _mix(self, latent_local, mem_ctx, tail_ctx) -> torch.Tensor:
        cat = torch.cat([latent_local, mem_ctx, tail_ctx], dim=-1)
        return F.gelu(self.mix_out(F.gelu(self.mix_in(cat))))

    def forward(self, frontier, memory, tail, context_only: bool = False):
        b, p, h, d = frontier.shape
        bp = b * p
        flat = frontier.reshape(bp, h, d)
        latent = F.gelu(self.down(flat))
        padded = F.pad(latent, (0, 0, self.cfg.local_kernel_size - 1, 0))
        local = sum(padded[:, off:off + h] * self.local_kernel[off]
                    for off in range(self.cfg.local_kernel_size))
        mem = memory[:, None].expand(-1, p, -1, -1).reshape(bp, memory.shape[1], d)
        tl = tail[:, None].expand(-1, p, -1, -1).reshape(bp, tail.shape[1], d)
        mem_ctx = _read_topk(latent, self.memory_key(mem), self.memory_value(mem),
                             self.cfg.memory_read_top_k)
        tail_ctx = F.scaled_dot_product_attention(self.tail_q(latent), self.tail_k(tl), self.tail_v(tl))
        mixed = self._mix(latent + local, mem_ctx, tail_ctx)
        if context_only:
            return mixed.reshape(b, p, h, self.cfg.latent_dim)
        update = self.up(mixed)
        gate = torch.sigmoid(self.gate(mixed))
        return rms(flat + gate * update).reshape(b, p, h, d)


class MultiQueryContextReader(nn.Module):
    """Reads L diverse contexts (one per scan step) in one batched pass.

    Each scan step gets its own query = base_latent + learned step offset, so the
    L parallel reads probe different regions of memory/tail instead of one
    stale context. Returns (bp, H, L, r).
    """

    def __init__(self, cfg, num_queries: int) -> None:
        super().__init__()
        d, r = cfg.field_dim, cfg.latent_dim
        self.cfg = cfg
        self.L = num_queries
        self.step_embs = nn.Parameter(torch.randn(num_queries, r) / math.sqrt(r))
        self.down = nn.Linear(d, r, bias=False)
        self.local_kernel = nn.Parameter(
            torch.randn(cfg.local_kernel_size, r) / math.sqrt(cfg.local_kernel_size))
        self.memory_key = nn.Linear(d, r, bias=False)
        self.memory_value = nn.Linear(d, r, bias=False)
        self.tail_q = nn.Linear(r, r, bias=False)
        self.tail_k = nn.Linear(d, r, bias=False)
        self.tail_v = nn.Linear(d, r, bias=False)
        self.mix_in = nn.Linear(3 * r, 2 * r, bias=False)
        self.mix_out = nn.Linear(2 * r, r, bias=False)

    def forward(self, frontier, memory, tail):
        b, p, h, d = frontier.shape
        bp = b * p
        L, r = self.L, self.cfg.latent_dim
        flat = frontier.reshape(bp, h, d)
        latent = F.gelu(self.down(flat))                       # (bp, H, r)
        padded = F.pad(latent, (0, 0, self.cfg.local_kernel_size - 1, 0))
        local = sum(padded[:, off:off + h] * self.local_kernel[off]
                    for off in range(self.cfg.local_kernel_size))
        q_all = (latent.unsqueeze(-2) + self.step_embs).reshape(bp, h * L, r)   # (bp, H*L, r)
        mem = memory[:, None].expand(-1, p, -1, -1).reshape(bp, memory.shape[1], d)
        tl = tail[:, None].expand(-1, p, -1, -1).reshape(bp, tail.shape[1], d)
        mem_ctx = _read_topk(q_all, self.memory_key(mem), self.memory_value(mem),
                             self.cfg.memory_read_top_k)
        tail_ctx = F.scaled_dot_product_attention(self.tail_q(q_all), self.tail_k(tl), self.tail_v(tl))
        mem_ctx = mem_ctx.reshape(bp, h, L, r)
        tail_ctx = tail_ctx.reshape(bp, h, L, r)
        base = (latent + local).unsqueeze(-2).expand(-1, -1, L, -1)
        cat = torch.cat([base, mem_ctx, tail_ctx], dim=-1)
        return F.gelu(self.mix_out(F.gelu(self.mix_in(cat))))   # (bp, H, L, r)
