"""Content-routed slot memory (the compressed long-range state).

The review's evidence (uniform routing entropy, RFK17/18 inside the noise floor)
retired the resonance-routing machinery. What remains is the part that works: a
bank of S slots, each token writing sparsely to its top-k content-matched slots
through a learned retain gate. Crucially, the *same* incremental write path is
used in training (carried-state) and inference, so the memory-state distribution
the model sees at decode time is the one it was trained on (review Q1, the
train/inference write mismatch).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._ops import rms


class SlotMemory(nn.Module):
    def __init__(self, slots: int, dim: int, write_top_k: int,
                 write_temperature: float = 3.0) -> None:
        super().__init__()
        self.slots = int(slots)
        self.dim = int(dim)
        self.write_top_k = int(write_top_k)
        self.write_temperature = float(write_temperature)
        self.roles = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.seed = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.write_query = nn.Linear(dim, slots, bias=False)   # content routing
        self.write_value = nn.Linear(dim, dim, bias=False)
        self.retain = nn.Linear(2 * dim, 1)

    def initial(self, batch: int, device, dtype) -> torch.Tensor:
        return rms(self.seed + self.roles).unsqueeze(0).expand(batch, -1, -1).to(
            device=device, dtype=dtype).clone()

    def routing_scores(self, values: torch.Tensor) -> torch.Tensor:
        return self.write_query(values)

    def write(self, memory: torch.Tensor, values: torch.Tensor,
              mask: torch.Tensor | None = None) -> torch.Tensor:
        """Sparse top-k write with per-slot retain gate.

        ``values`` is (B, N, D) token embeddings; each token writes to its top-k
        content-matched slots. Silent slots retain their content unchanged.
        """
        s, k = self.slots, self.write_top_k
        scores = self.routing_scores(values)                  # B x N x S
        if mask is not None:
            values = values * mask.unsqueeze(-1).to(values.dtype)
            scores = scores.masked_fill(~mask[:, :, None], float("-inf"))
        if k < s:
            topk_vals, topk_idx = scores.topk(k, dim=-1)
            routed = torch.full_like(scores, float("-inf"))
            routed.scatter_(-1, topk_idx, topk_vals)
        else:
            routed = scores
        slot_scores = routed.transpose(1, 2)                  # B x S x N
        has_write = (slot_scores > float("-inf") / 2).any(dim=-1)   # B x S
        safe = slot_scores.masked_fill(~has_write.unsqueeze(-1), 0.0)
        weights = (safe * self.write_temperature).softmax(dim=-1)
        weights = weights * has_write.unsqueeze(-1).to(weights.dtype)
        writes = weights @ self.write_value(values)           # B x S x D
        retain_gate = torch.sigmoid(self.retain(torch.cat([memory, writes], dim=-1)))
        retain = torch.where(has_write.unsqueeze(-1), retain_gate, torch.ones_like(retain_gate))
        updated = rms(retain * memory + (1.0 - retain) * writes + self.roles)
        return torch.where(has_write.unsqueeze(-1), updated, memory)

    @torch.no_grad()
    def utilization(self, values: torch.Tensor) -> dict[str, float]:
        """Slot-usage diagnostics on a batch of token embeddings."""
        import math
        s, k = self.slots, self.write_top_k
        scores = self.routing_scores(values)
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        routed = torch.full_like(scores, float("-inf"))
        routed.scatter_(-1, topk_idx, topk_vals)
        has_write = (routed.transpose(1, 2) > float("-inf") / 2).any(dim=-1)
        probs = scores.softmax(dim=-1)
        entropy = (-(probs * probs.clamp_min(1e-9).log()).sum(-1) / math.log(s)).mean().item()
        return {"slot_utilization": round(has_write.float().mean().item(), 4),
                "routing_entropy": round(entropy, 4)}
