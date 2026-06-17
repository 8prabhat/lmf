"""Parallel-scan settle with UNSHARED deep macro steps (review Q2.2).

v3 weight-tied a single reader + correction rule across all macro steps, making
the compute path a 2-module recurrent net applied K times — depth-equivalent of
roughly 2, expressivity much lower. Here each of K macro steps owns its own
``MacroBlock`` (reader + scan parameters + correction rule), turning the same
wall-clock structure into a genuinely deep K-layer network. With the embedding
budget freed by the factorised codebook, K defaults to 4.

Each macro block:
  a. L diverse context reads (MultiQueryContextReader)
  b. input-dependent SSM gates  delta, A, B
  c. closed-form scan state h_L (no Python loop; O(log L) algebra)
  d. scan residual into the frontier
  e. a full nonlinear correction-rule application
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._ops import rms
from .dynamics import FrontierDynamicsRule, MultiQueryContextReader


class MacroBlock(nn.Module):
    """One unshared macro step."""

    def __init__(self, cfg, scan_steps: int) -> None:
        super().__init__()
        d, r = cfg.field_dim, cfg.latent_dim
        self.reader = MultiQueryContextReader(cfg, num_queries=scan_steps)
        self.A_log = nn.Parameter(torch.full((scan_steps, r), -0.5))
        self.delta_proj = nn.Linear(r, r, bias=True)
        nn.init.constant_(self.delta_proj.bias, -2.0)
        self.B_proj = nn.Linear(r, r, bias=False)
        self.h_down = nn.Linear(d, r, bias=False)
        self.h_up = nn.Linear(r, d, bias=False)
        self.correction_rule = FrontierDynamicsRule(cfg)

    def forward(self, frontier, memory, tail):
        b, p, h, d = frontier.shape
        bp = b * p
        ctx = self.reader(frontier, memory, tail)              # (bp, H, L, r)
        delta = F.softplus(self.delta_proj(ctx))               # (bp, H, L, r)
        A = torch.exp(-self.A_log.exp() * delta)
        B_t = self.B_proj(ctx) * delta
        h0 = self.h_down(frontier.reshape(bp, h, d))           # (bp, H, r)
        # Closed-form h_L = P_L * h0 + sum_l suffix_prod_l * B_l
        A_shifted = torch.cat([A[..., 1:, :], torch.ones_like(A[..., :1, :])], dim=-2)
        suffix_prod = A_shifted.flip(-2).cumprod(dim=-2).flip(-2)
        P_L = A.prod(dim=-2)
        h_final = P_L * h0 + (suffix_prod * B_t).sum(dim=-2)    # (bp, H, r)
        frontier = rms(frontier + self.h_up(h_final).reshape(b, p, h, d))
        return self.correction_rule(frontier, memory, tail)


class SettleSSM(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MacroBlock(cfg, cfg.ssm_scan_steps) for _ in range(cfg.ssm_macro_steps)]
        )

    def forward(self, frontier, memory, tail, n_macro: int | None = None):
        n = len(self.blocks) if n_macro is None else max(1, min(n_macro, len(self.blocks)))
        intermediates: list[torch.Tensor] = []
        for i in range(n):
            frontier = self.blocks[i](frontier, memory, tail)
            intermediates.append(frontier)
        return frontier, intermediates
