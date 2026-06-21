"""RHCA v4 rolling-frontier model.

This is the centrepiece. Relative to the v3 line it folds in the architecture
review's fixes:

* **Bounded carried-state training** — each optimizer step supervises only the
  final few carried windows, preventing activation memory from growing with the
  full sequence length.
* **Factorised codebook + unshared deep macro steps + widened mix** (Q2).
* **All-position chain supervision with a minimal CE + routing loss**.
* **Entropy-based commit confidence** with no learned auxiliary energy network.
* **Hypothesis width 4, adaptive expansion off for LM** (Q5).
* **512-token SDPA exact-recall tail** (Q6).
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS
from ._ops import rms
from .codebook import build_codebook
from .config import RHCAConfig
from .dynamics import FrontierDynamicsRule  # noqa: F401  (re-exported for tests)
from .memory import SlotMemory
from .settle import SettleSSM
from .state import AdvanceResult, GenerationResult, GenerationState, SamplingConfig


class RollingFrontierRHCA(nn.Module):
    def __init__(self, config: RHCAConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        c, d = config, config.field_dim
        self.codebook = build_codebook(c.codebook, c.vocab_size, d, c.codebook_factor_dim)
        self.frontier_roles = nn.Parameter(torch.randn(c.frontier_size, d) * 0.02)
        self.tail_roles = nn.Parameter(torch.randn(c.tail_size, d) * 0.02)
        self.hypothesis_roles = nn.Parameter(torch.randn(c.max_hypotheses, d) * 0.02)
        self.free_state = nn.Parameter(torch.randn(d) * 0.02)
        self.commit_cond_proj = nn.Linear(d, d, bias=False)
        self.memory = SlotMemory(c.memory_slots, d, c.memory_write_top_k, c.memory_write_temperature)
        self.settle_ssm = SettleSSM(c)
        self._runtime_commit_threshold: float | None = None

    def fp32_modules(self) -> list[nn.Module]:
        return []

    # ------------------------------------------------------------------ helpers
    def _init_frontier(self, batch: int, device, dtype) -> torch.Tensor:
        base = self.free_state.view(1, 1, 1, -1) + self.frontier_roles.view(1, 1, -1, self.config.field_dim)
        hyp = self.hypothesis_roles.view(1, -1, 1, self.config.field_dim)
        return rms(base + hyp).expand(batch, -1, -1, -1).to(device=device, dtype=dtype).clone()

    def prefill(self, prompt_tokens: torch.Tensor,
                prompt_mask: torch.Tensor | None = None) -> GenerationState:
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        if prompt_tokens.dim() != 2 or prompt_tokens.shape[1] == 0:
            raise ValueError("prompt_tokens must be a non-empty B x N tensor")
        if prompt_tokens.min() < 0 or prompt_tokens.max() >= self.config.vocab_size:
            raise ValueError("prompt token id outside the configured vocabulary")
        b, n = prompt_tokens.shape
        device = prompt_tokens.device
        if prompt_mask is not None and prompt_mask.shape != prompt_tokens.shape:
            raise ValueError("prompt_mask must match prompt_tokens")
        embedded = self.codebook.embed(prompt_tokens)
        memory = self.memory.initial(b, device, embedded.dtype)
        # Padded prompt positions must not write to memory (review finding 9).
        memory = self.memory.write(memory, embedded, prompt_mask)
        t = self.config.tail_size
        pad_id = self.config.special_token_ids.get("pad", 0)
        if n >= t:
            tail = embedded[:, -t:]
            tail_ids = prompt_tokens[:, -t:]
        else:
            pad = self.free_state.view(1, 1, -1).expand(b, t - n, -1)
            tail = torch.cat([pad, embedded], dim=1)
            tail_ids = F.pad(prompt_tokens, (t - n, 0), value=pad_id)
        active = torch.zeros(b, self.config.max_hypotheses, dtype=torch.bool, device=device)
        active[:, 0] = True
        committed = (torch.full((b,), n, dtype=torch.long, device=device)
                     if prompt_mask is None else prompt_mask.sum(dim=1).long())
        return GenerationState(
            memory=memory, frontier=self._init_frontier(b, device, embedded.dtype),
            tail=rms(tail), tail_ids=tail_ids, active_hypotheses=active,
            finished=torch.zeros(b, dtype=torch.bool, device=device),
            committed_count=committed,
        )

    def settle(self, state: GenerationState, active_only: bool = False):
        """Run the unshared deep settle and return its trajectory."""
        if active_only and bool((state.active_hypotheses.sum(dim=1) == 1).all()):
            selected = state.active_hypotheses.to(torch.int64).argmax(dim=1)
            frontier = state.frontier.gather(
                1, selected[:, None, None, None].expand(
                    -1, 1, self.config.frontier_size, self.config.field_dim))
        else:
            frontier = state.frontier
        tail_normed = rms(state.tail + self.tail_roles)
        frontier_init = frontier
        frontier, intermediates = self.settle_ssm(frontier, state.memory, tail_normed)
        return frontier, [frontier_init, *intermediates]

    def _chain_conditioned_fields_and_logits(self, frontier, committed_tokens):
        """Teacher-force a drafted block with the generation chain rule."""
        c = committed_tokens.shape[1]
        p = frontier.shape[1]
        shifted = F.pad(committed_tokens[:, :-1], (1, 0), value=0)
        cond = self.commit_cond_proj(self.codebook.embed(shifted))[:, None].expand(-1, p, -1, -1)
        has_cond = (torch.arange(c, device=frontier.device) > 0).view(1, 1, -1, 1)
        drafted = frontier[:, :, :c]
        fields = torch.where(has_cond, rms(drafted + cond), drafted)
        return fields, self.codebook.logits(fields)

    def _shift_frontier(self, frontier, count: torch.Tensor) -> torch.Tensor:
        b, p, h, d = frontier.shape
        positions = torch.arange(h, device=frontier.device).view(1, 1, h)
        source = (positions + count.view(b, 1, 1)).clamp(max=h - 1)
        shifted = frontier.gather(2, source.unsqueeze(-1).expand(-1, p, -1, d))
        appended = positions >= (h - count.view(b, 1, 1))
        fresh = self._init_frontier(b, frontier.device, frontier.dtype)
        return torch.where(appended.unsqueeze(-1), fresh, shifted)

    def _advance_state(self, state: GenerationState, settled_frontier: torch.Tensor,
                       committed: torch.Tensor, write_mask: torch.Tensor | None = None
                       ) -> GenerationState:
        """Advance the rolling state by committing tokens — the SINGLE state
        transition shared by training, evaluation and inference (review finding 1).

        It carries the *settled* frontier forward (shifting it by the number of
        committed tokens, keeping the already-settled but uncommitted positions),
        exactly as ``advance`` does at inference. Training and eval call this so all
        three paths use identical state dynamics; only ``advance`` adds the entropy
        threshold / EOS / finished bookkeeping on top.
        """
        b, count = committed.shape
        counts = torch.full((b,), count, dtype=torch.long, device=committed.device)
        embedded = self.codebook.embed(committed)
        memory = self.memory.write(state.memory, embedded, write_mask)
        tail = torch.cat([state.tail, embedded], dim=1)[:, -self.config.tail_size:]
        tail_ids = torch.cat([state.tail_ids, committed], dim=1)[:, -self.config.tail_size:]
        frontier = self._shift_frontier(settled_frontier[:, :1], counts).expand(
            -1, self.config.max_hypotheses, -1, -1).clone()
        active = torch.zeros_like(state.active_hypotheses)
        active[:, 0] = True
        return GenerationState(
            memory=memory, frontier=frontier, tail=tail, tail_ids=tail_ids,
            active_hypotheses=active, finished=state.finished,
            committed_count=state.committed_count + counts)

    # ------------------------------------------------------------------ training
    def training_step(self, tokens: torch.Tensor,
                      task_metadata: dict[str, Any] | None = None,
                      loss_term_scales: dict[str, float] | None = None) -> dict[str, torch.Tensor]:
        """Trainable interface — delegates to carried-state training (review Q1)."""
        meta = task_metadata or {}
        return self.carried_state_training_step(
            tokens,
            segment=meta.get("segment_len"),
            max_train_windows=int(meta.get("max_train_windows", 2)),
            loss_mask=meta.get("loss_mask"),
            attention_mask=meta.get("attention_mask"),
            loss_term_scales=loss_term_scales,
        )

    def carried_state_training_step(self, tokens: torch.Tensor, segment: int | None = None,
                                    max_train_windows: int = 2,
                                    loss_mask: torch.Tensor | None = None,
                                    attention_mask: torch.Tensor | None = None,
                                    loss_term_scales: dict[str, float] | None = None,
                                    ) -> dict[str, torch.Tensor]:
        """Bounded carried-state training with inference-matched state dynamics.

        The final ``max_train_windows`` are supervised. Earlier tokens are folded
        into the parallel prefill, bounding both settle compute and retained
        activation memory independently of sequence length.
        """
        h = self.config.frontier_size
        stride = self.config.max_commit
        segment = segment or h
        n = tokens.shape[1]
        if n < segment + stride:
            raise ValueError(f"sequence length {n} too short for segment {segment} + max_commit {stride}")
        max_train_windows = max(1, int(max_train_windows))
        segment = max(segment, n - max_train_windows * stride)
        seg_mask = None if attention_mask is None else attention_mask[:, :segment]
        state = self.prefill(tokens[:, :segment], seg_mask)
        acc: dict[str, list[torch.Tensor]] = {}
        pos = segment
        while pos + stride <= n:
            settled, _ = self.settle(state, active_only=True)
            # Score only the positions advance() actually commits this cycle
            # (the first `max_commit` chain-conditioned frontier slots) — the
            # rest of the settled frontier is carried forward and re-settled
            # before it is ever committed, so supervising it here under an
            # oracle-shifted condition it will never see again at commit time
            # only teaches a context pattern that's never replayed (review
            # finding: train/inference objective mismatch).
            targets = tokens[:, pos:pos + stride]
            cmask = None if attention_mask is None else attention_mask[:, pos:pos + stride]
            wmask = None if loss_mask is None else loss_mask[:, pos:pos + stride]
            wmask = cmask if wmask is None else (wmask if cmask is None else (wmask & cmask))
            losses = self._window_losses(settled, targets, wmask)
            for k, v in losses.items():
                acc.setdefault(k, []).append(v)
            state = self._advance_state(state, settled, targets, cmask)
            state = state.detach()            # TBPTT boundary (review Q1)
            pos += stride
        reduced = {k: torch.stack(v).mean() for k, v in acc.items()}
        scales = loss_term_scales or {}
        reduced["total"] = (
            1.00 * scales.get("commit_token", 1.0) * reduced["commit_token"]
            + self.config.routing_balance_weight * scales.get("routing_balance", 1.0)
              * reduced["routing_balance"]
        )
        return reduced

    def _window_losses(self, frontier, targets, mask) -> dict[str, torch.Tensor]:
        """Minimal losses for one settled, inference-matched frontier window."""
        c = targets.shape[1]
        fields, logits = self._chain_conditioned_fields_and_logits(frontier, targets)  # (b,1,c,*)
        logits1 = logits[:, 0]                               # (b, c, V)
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)

        def masked_mean(value, m):
            return (value * m).sum() / m.sum().clamp_min(1)

        # (1) all-position chain-conditioned CE, uniform weight (review Q3.1)
        ce = F.cross_entropy(logits1.reshape(-1, self.config.vocab_size),
                             targets.reshape(-1), reduction="none").reshape_as(targets)
        commit_token = masked_mean(ce, mask)

        # Routing balance is the only auxiliary structural regulariser.
        emb = self.codebook.embed(targets)
        load = self.memory.routing_scores(emb).softmax(dim=-1).mean(dim=(0, 1))
        routing_balance = (load * self.config.memory_slots - 1.0).pow(2).mean()

        return {
            "commit_token": commit_token,
            "routing_balance": routing_balance,
        }

    # ------------------------------------------------------------------ generation
    def _sample(self, logits, cfg: SamplingConfig) -> torch.Tensor:
        logits = logits / max(cfg.temperature, 1e-5)
        if cfg.top_k > 0:
            thresh = logits.topk(min(cfg.top_k, logits.shape[-1]), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        if cfg.top_p < 1.0:
            sl, idx = logits.sort(dim=-1, descending=True)
            remove = sl.softmax(dim=-1).cumsum(dim=-1) > cfg.top_p
            remove[..., 0] = False
            sl = sl.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sl)
        if cfg.deterministic:
            return logits.argmax(dim=-1)
        return torch.multinomial(logits.softmax(dim=-1).reshape(-1, logits.shape[-1]), 1
                                 ).reshape(logits.shape[:-1])

    @torch.no_grad()
    def advance(self, state: GenerationState, cfg: SamplingConfig | None = None) -> AdvanceResult:
        cfg = cfg or SamplingConfig()
        cfg.validate()
        started = time.perf_counter()
        frontier, _ = self.settle(state, active_only=True)  # (b,1,h,d)
        max_c = self.config.max_commit
        committed_steps, logit_steps = [], []
        prev_embed = None
        # Reconstruct the low-rank decode weight once for this whole commit
        # block instead of once per iteration — logits() rebuilds it from
        # scratch on every call, so without this the V x D matmul was repeated
        # up to max_commit times per settle cycle for identical output.
        decode_weight = self.codebook.decode_weight()
        for i in range(max_c):
            hidden = frontier[:, 0, i]
            if prev_embed is not None:
                hidden = rms(hidden + self.commit_cond_proj(prev_embed))
            logits_i = self.codebook.logits_from_weight(hidden, decode_weight)
            if cfg.repetition_penalty != 1.0:
                # Penalise tokens already in the verbatim tail or committed this block.
                repeated = torch.zeros_like(logits_i, dtype=torch.bool)
                repeated.scatter_(1, state.tail_ids, True)
                for prev_token in committed_steps:
                    repeated.scatter_(1, prev_token[:, None], True)
                penalised = torch.where(logits_i >= 0, logits_i / cfg.repetition_penalty,
                                        logits_i * cfg.repetition_penalty)
                logits_i = torch.where(repeated, penalised, logits_i)
            token_i = self._sample(logits_i, cfg)
            committed_steps.append(token_i)
            logit_steps.append(logits_i)
            prev_embed = self.codebook.embed(token_i)
        committed = torch.stack(committed_steps, dim=1)               # (b, max_c)
        logits = torch.stack(logit_steps, dim=1)[:, None]
        probs = logits[:, 0].softmax(dim=-1)
        commit_entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1) / math.log(
            self.config.vocab_size)

        threshold = (cfg.commit_entropy_threshold if cfg.commit_entropy_threshold is not None
                     else (self._runtime_commit_threshold if self._runtime_commit_threshold is not None
                           else self.config.commit_entropy_threshold))
        stable = commit_entropy < threshold
        prefix_stable = stable.to(torch.int64).cumprod(dim=1).bool()
        count = prefix_stable.sum(dim=1).clamp(min=1, max=max_c).masked_fill(state.finished, 0)
        commit_mask = torch.arange(max_c, device=frontier.device)[None] < count[:, None]
        committed = committed.masked_fill(~commit_mask, 0)
        eos_id = self.config.special_token_ids.get("eos")
        finished = state.finished.clone()
        if eos_id is not None:
            positions = torch.arange(max_c, device=frontier.device)[None]
            eos_pos = torch.where((committed == eos_id) & commit_mask, positions, max_c)
            first_eos = eos_pos.min(dim=1).values
            commit_mask &= positions <= first_eos[:, None]
            count = commit_mask.sum(dim=1)
            committed = committed.masked_fill(~commit_mask, 0)
            finished |= ((committed == eos_id) & commit_mask).any(dim=1)

        embedded = self.codebook.embed(committed)
        memory = self.memory.write(state.memory, embedded, commit_mask)
        memory = torch.where(state.finished[:, None, None], state.memory, memory)
        tail = torch.cat([state.tail, embedded], dim=1)
        tail = tail.gather(1, (torch.arange(self.config.tail_size, device=tail.device)[None]
                               + count[:, None]).unsqueeze(-1).expand(-1, -1, tail.shape[-1]))
        tail_ids = torch.cat([state.tail_ids, committed], dim=1)
        tail_ids = tail_ids.gather(1, torch.arange(self.config.tail_size, device=tail_ids.device)[None]
                                   + count[:, None])
        next_frontier = self._shift_frontier(
            frontier[:, :1].expand(-1, self.config.max_hypotheses, -1, -1), count)
        next_frontier = torch.where(state.finished[:, None, None, None], state.frontier, next_frontier)
        active = torch.zeros_like(state.active_hypotheses)
        active[:, 0] = True
        next_state = GenerationState(
            memory=memory, frontier=next_frontier, tail=tail, tail_ids=tail_ids,
            active_hypotheses=active, finished=finished,
            committed_count=state.committed_count + count)
        return AdvanceResult(next_state, committed, commit_mask, commit_entropy,
                             torch.zeros_like(count), time.perf_counter() - started)

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int,
                 sampling_config: SamplingConfig | None = None) -> GenerationResult:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        cfg = sampling_config or SamplingConfig()
        cfg.validate()
        state = self.prefill(prompt_tokens)
        batch = state.finished.shape[0]
        pad_id = self.config.special_token_ids.get("pad", 0)
        device = state.tail_ids.device
        output = torch.full((batch, max_new_tokens), pad_id, dtype=torch.long, device=device)
        generated = torch.zeros(batch, dtype=torch.long, device=device)
        diagnostics: list[dict[str, Any]] = []
        settles = 0
        started = time.perf_counter()
        while bool(((generated < max_new_tokens) & ~state.finished).any()):
            result = self.advance(state, cfg)
            settles += 1
            width = result.committed_token_ids.shape[1]
            positions = generated[:, None] + torch.arange(width, device=device)[None]
            valid = result.commit_mask & (positions < max_new_tokens)
            rows = torch.arange(batch, device=device)[:, None].expand_as(valid)[valid]
            output[rows, positions[valid]] = result.committed_token_ids[valid]
            generated = generated + valid.sum(dim=1)
            diagnostics.append({"committed": int(result.commit_mask.sum().item())})
            state = result.state
        elapsed = time.perf_counter() - started
        total_gen = float(generated.sum())
        return GenerationResult(
            token_ids=output, generated_lengths=generated, cycles=len(diagnostics),
            tokens_per_second=total_gen / max(elapsed, 1e-9),
            tokens_per_settle=total_gen / max(settles, 1),
            diagnostics=diagnostics)

    # ------------------------------------------------------------------ manifest
    def architecture_manifest(self) -> dict[str, Any]:
        c = self.config
        params = {n: sum(p.numel() for p in m.parameters())
                  for n, m in [("codebook", self.codebook), ("settle", self.settle_ssm),
                               ("memory", self.memory)]}
        total = sum(p.numel() for p in self.parameters())
        return {
            "name": "RollingFrontierRHCA",
            "config": c.to_dict(),
            "parameters": {"total": total, "compute_fraction": round(1 - params["codebook"] / total, 4),
                           **params},
            "state_shapes": {
                "memory": ["B", c.memory_slots, c.field_dim],
                "frontier": ["B", c.max_hypotheses, c.frontier_size, c.field_dim],
                "tail": ["B", c.tail_size, c.field_dim],
            },
        }


@MODELS.register("rhca")
def build_rhca(model_cfg: dict, vocab_size: int | None = None) -> RollingFrontierRHCA:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return RollingFrontierRHCA(RHCAConfig(**cfg))
