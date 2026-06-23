"""Spectral Memory Language Model (SM-LM).

A clean-slate, recall-capable, Mac-fast architecture (see
``docs/spectral_memory/spectral_memory.md`` and the technical spec). The core
mixer is :class:`MultiTimescaleDeltaMemory` (MTDM): a bank of error-correcting
fast-weight memories, each *band-limited* to a distinct log-spaced decay
timescale (the "spectrum"), fused by an input-dependent Cross-Band Interference
Router (CBIR). A thin slice of sliding-window attention (:class:`SlidingWindowAttention`)
on 1-2 designated layers supplies exact local copy / induction.

All heavy compute is dense matmul + elementwise (the fast path on Apple Silicon
MPS): MTDM uses the chunk-parallel delta rule in :mod:`.delta_scan`, local mixing
is RWKV-style token-shift (not depthwise conv), and attention is windowed.
Inference is O(1)/token: each bank carries a fixed ``d_v x d_k`` state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS
from ...core.rope import apply_rope
from .delta_scan import delta_rule_chunked


@dataclass(frozen=True)
class SpectralMemoryConfig:
    vocab_size: int
    dim: int = 512
    layers: int = 8
    banks: int = 4
    head_dim: int = 64
    attn_heads: int = 8
    attention_layers: tuple[int, ...] | None = None
    window: int = 256
    chunk: int = 64
    mlp_ratio: int = 2
    router: str = "sigmoid"  # "sigmoid" (independent), "softmax" (competitive), "none" (sum)
    decay_mode: str = "banded"  # "banded" (log-spaced bands) or "free" (full range per bank)
    min_half_life: float = 4.0
    max_half_life: float = 2048.0
    max_seq_len: int = 4096

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.dim < 8:
            raise ValueError("dim must be at least 8")
        if self.layers < 1:
            raise ValueError("layers must be positive")
        if self.banks < 1:
            raise ValueError("banks must be positive")
        if self.head_dim < 2:
            raise ValueError("head_dim must be at least 2")
        if self.attn_heads < 1 or self.dim % self.attn_heads:
            raise ValueError("dim must be divisible by attn_heads")
        if (self.dim // self.attn_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.window < 1:
            raise ValueError("window must be positive")
        if self.chunk < 1:
            raise ValueError("chunk must be positive")
        if self.router not in {"sigmoid", "softmax", "none"}:
            raise ValueError("router must be 'sigmoid', 'softmax', or 'none'")
        if self.decay_mode not in {"banded", "free"}:
            raise ValueError("decay_mode must be 'banded' or 'free'")
        if not 0 < self.min_half_life < self.max_half_life:
            raise ValueError("require 0 < min_half_life < max_half_life")
        layers = self._resolved_attention_layers()
        if any(not 0 <= i < self.layers for i in layers):
            raise ValueError("attention_layers out of range")
        object.__setattr__(self, "attention_layers", layers)

    def _resolved_attention_layers(self) -> tuple[int, ...]:
        if self.attention_layers is not None:
            return tuple(sorted(set(int(i) for i in self.attention_layers)))
        if self.layers < 4:
            return (self.layers - 1,)
        return tuple(sorted({self.layers // 2, self.layers - 1}))

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GatedMLP(nn.Module):
    def __init__(self, dim: int, ratio: int) -> None:
        super().__init__()
        hidden = ratio * dim
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MultiTimescaleDeltaMemory(nn.Module):
    """MTDM: a spectrum of error-correcting fast-weight memories + CBIR read."""

    def __init__(self, config: SpectralMemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.banks = config.banks
        self.head_dim = config.head_dim
        self.chunk = config.chunk
        inner = config.banks * config.head_dim

        self.shift_mix = nn.Parameter(torch.ones(config.dim))
        self.q = nn.Linear(config.dim, inner, bias=False)
        self.k = nn.Linear(config.dim, inner, bias=False)
        self.v = nn.Linear(config.dim, inner, bias=False)
        self.beta = nn.Linear(config.dim, config.banks)
        self.tau = nn.Linear(config.dim, config.banks)
        self.router = (
            None if config.router == "none" else nn.Linear(config.dim, config.banks)
        )
        self.out = nn.Linear(inner, config.dim, bias=False)

        # Log-spaced half-life bands: in "banded" mode bank h is hard-restricted
        # to [lo_h, hi_h], so the bank set tiles the timescale axis and cannot
        # collapse to one scale. This banding is SM-LM's point of departure from
        # free per-head decay (Gated DeltaNet) — the "free" mode reproduces that
        # baseline by letting every bank range over the full [min, max] window
        # (the key ablation for whether structure earns its keep).
        if config.decay_mode == "free":
            lo = torch.full((config.banks,), float(config.min_half_life))
            hi = torch.full((config.banks,), float(config.max_half_life))
        else:
            edges = torch.exp(
                torch.linspace(
                    math.log(config.min_half_life),
                    math.log(config.max_half_life),
                    config.banks + 1,
                )
            )
            lo, hi = edges[:-1].clone(), edges[1:].clone()
        self.register_buffer("hl_lo", lo, persistent=False)
        self.register_buffer("hl_hi", hi, persistent=False)

    def _shift(self, h, prev):
        shifted = torch.cat([prev, h[:, :-1]], dim=1)
        return self.shift_mix * h + (1.0 - self.shift_mix) * shifted

    def forward(self, h, cache=None, use_cache=False, segment_ids=None):
        b, t_len, _ = h.shape
        prev = (
            cache["shift"]
            if cache is not None
            else torch.zeros(b, 1, h.shape[-1], device=h.device, dtype=h.dtype)
        )
        hs = self._shift(h, prev)

        q = self.q(hs).view(b, t_len, self.banks, self.head_dim)
        k = F.normalize(self.k(hs).view(b, t_len, self.banks, self.head_dim), dim=-1)
        v = self.v(hs).view(b, t_len, self.banks, self.head_dim)

        frac = torch.sigmoid(self.tau(hs))                  # [B, T, banks]
        tau = self.hl_lo * (self.hl_hi / self.hl_lo) ** frac
        log_a = (-math.log(2.0) / tau)                      # log decay, banded
        beta = torch.sigmoid(self.beta(hs))                 # [B, T, banks]

        # fold banks into the batch dim for the scan
        def fold(x):  # [B, T, banks, d] -> [B*banks, T, d]
            return x.permute(0, 2, 1, 3).reshape(b * self.banks, t_len, self.head_dim)

        qm, km, vm = fold(q), fold(k), fold(v)
        la = log_a.permute(0, 2, 1).reshape(b * self.banks, t_len)
        bm = beta.permute(0, 2, 1).reshape(b * self.banks, t_len)
        seg = (
            None
            if segment_ids is None
            else segment_ids[:, None, :].expand(b, self.banks, t_len).reshape(b * self.banks, t_len)
        )
        state = None if cache is None else cache["state"].reshape(b * self.banks, self.head_dim, self.head_dim)

        out, new_state = delta_rule_chunked(qm, km, vm, la, bm, seg, self.chunk, state)
        out = out.reshape(b, self.banks, t_len, self.head_dim).permute(0, 2, 1, 3)

        if self.router is not None:                         # CBIR read; else plain sum
            gate = self.router(hs)                          # [B, T, banks]
            gate = (
                torch.softmax(gate, dim=-1)
                if self.config.router == "softmax"
                else torch.sigmoid(gate)
            )
            out = out * gate[..., None]
        out = out.reshape(b, t_len, self.banks * self.head_dim)
        out = self.out(out)

        next_cache = None
        if use_cache:
            next_cache = {
                "state": new_state.reshape(b, self.banks, self.head_dim, self.head_dim),
                "shift": h[:, -1:],
            }
        return out, next_cache


class SlidingWindowAttention(nn.Module):
    """Causal attention restricted to a window — exact local recall / induction."""

    def __init__(self, config: SpectralMemoryConfig) -> None:
        super().__init__()
        self.heads = config.attn_heads
        self.head_dim = config.dim // config.attn_heads
        self.window = config.window
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, h, cache=None, use_cache=False, attention_mask=None, segment_ids=None):
        b, t_len, d = h.shape
        qkv = self.qkv(h).view(b, t_len, 3, self.heads, self.head_dim)
        q, k, v = (x.transpose(1, 2) for x in qkv.unbind(dim=2))  # [B, H, T, hd]
        past = 0 if cache is None else cache[0].shape[2]
        q = apply_rope(q, torch.arange(past, past + t_len, device=h.device))
        k = apply_rope(k, torch.arange(past, past + t_len, device=h.device))
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        total_k = k.shape[2]
        q_pos = torch.arange(past, past + t_len, device=h.device)[:, None]
        k_pos = torch.arange(total_k, device=h.device)[None, :]
        mask = (k_pos <= q_pos) & ((q_pos - k_pos) < self.window)
        mask = mask[None, None]
        if cache is None and attention_mask is not None:
            mask = mask & attention_mask[:, None, None, :total_k].bool()
        if cache is None and segment_ids is not None:
            same = segment_ids[:, None, :, None] == segment_ids[:, None, None, :]
            mask = mask & same

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, t_len, d)
        out = self.proj(out)
        return out, ((k, v) if use_cache else None)


class SpectralMemoryBlock(nn.Module):
    def __init__(self, config: SpectralMemoryConfig, is_attention: bool) -> None:
        super().__init__()
        self.is_attention = is_attention
        self.norm1 = RMSNorm(config.dim)
        self.mixer = (
            SlidingWindowAttention(config) if is_attention else MultiTimescaleDeltaMemory(config)
        )
        self.norm2 = RMSNorm(config.dim)
        self.mlp = GatedMLP(config.dim, config.mlp_ratio)

    def forward(self, x, cache=None, use_cache=False, attention_mask=None, segment_ids=None):
        if self.is_attention:
            mix, next_cache = self.mixer(
                self.norm1(x), cache, use_cache, attention_mask, segment_ids
            )
        else:
            mix, next_cache = self.mixer(self.norm1(x), cache, use_cache, segment_ids)
        x = x + mix
        x = x + self.mlp(self.norm2(x))
        return x, next_cache


class SpectralMemoryLM(nn.Module):
    def __init__(self, config: SpectralMemoryConfig) -> None:
        super().__init__()
        self.config = config
        attn = set(config.attention_layers or ())
        self.token = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [SpectralMemoryBlock(config, i in attn) for i in range(config.layers)]
        )
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)

    def forward(
        self,
        ids,
        caches=None,
        use_cache=False,
        attention_mask=None,
        segment_ids=None,
        sentence_end_mask=None,
    ):
        x = self.token(ids)
        next_caches: list[Any] = []
        for i, block in enumerate(self.blocks):
            x, nc = block(
                x,
                cache=None if caches is None else caches[i],
                use_cache=use_cache,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
            )
            if use_cache:
                next_caches.append(nc)
        x = self.norm(x)
        return self.head(x), (next_caches if use_cache else None)

    @staticmethod
    def _valid_targets(tokens, loss_mask, attention_mask, segment_ids):
        valid = torch.ones_like(tokens[:, 1:], dtype=torch.bool)
        if loss_mask is not None:
            valid = valid & loss_mask[:, 1:].bool()
        if attention_mask is not None:
            valid = valid & attention_mask[:, 1:].bool()
        if segment_ids is not None:
            valid = valid & (segment_ids[:, 1:] == segment_ids[:, :-1])
        return valid

    def training_step(
        self,
        tokens,
        task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        meta = task_metadata or {}
        attention_mask = meta.get("attention_mask")
        loss_mask = meta.get("loss_mask")
        segment_ids = (
            None if bool(meta.get("single_segment_rows", False)) else meta.get("segment_ids")
        )
        logits, _ = self.forward(
            tokens, attention_mask=attention_mask, segment_ids=segment_ids
        )
        targets = tokens[:, 1:]
        pred = logits[:, :-1]
        valid = self._valid_targets(tokens, loss_mask, attention_mask, segment_ids)
        losses = F.cross_entropy(
            pred.reshape(-1, pred.shape[-1]), targets.reshape(-1), reduction="none"
        ).reshape_as(targets)
        valid_f = valid.to(losses.dtype)
        language_modeling = (losses * valid_f).sum() / valid_f.sum().clamp_min(1)
        scales = loss_term_scales or {}
        total = scales.get("language_modeling", 1.0) * language_modeling
        return {"language_modeling": language_modeling, "total": total}

    def _sample(self, logits, cfg) -> torch.Tensor:
        if cfg is None or getattr(cfg, "deterministic", True):
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(getattr(cfg, "temperature", 1.0), 1e-5)
        top_k = getattr(cfg, "top_k", 0)
        if top_k and top_k > 0:
            thresh = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        top_p = getattr(cfg, "top_p", 1.0)
        if top_p < 1.0:
            sorted_logits, idx = logits.sort(dim=-1, descending=True)
            remove = sorted_logits.softmax(-1).cumsum(-1) > top_p
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
        return torch.multinomial(logits.softmax(-1), 1)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, sampling_config=None):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return torch.empty(prompt.shape[0], 0, dtype=torch.long, device=prompt.device)
        logits, caches = self.forward(prompt, use_cache=True)
        token = self._sample(logits[:, -1], sampling_config)
        out = []
        for index in range(max_new_tokens):
            out.append(token)
            if index + 1 == max_new_tokens:
                break
            logits, caches = self.forward(token, caches=caches, use_cache=True)
            token = self._sample(logits[:, -1], sampling_config)
        return torch.cat(out, dim=1)

    def architecture_manifest(self) -> dict:
        return {
            "name": "SpectralMemoryLM",
            "version": 1,
            "config": self.config.to_dict(),
            "parameters": {"total": sum(p.numel() for p in self.parameters())},
            "invariants": {
                "self_attention": bool(self.config.attention_layers),
                "delta_memory": True,
                "kv_cache": bool(self.config.attention_layers),
                "constant_size_memory_state": True,
            },
        }


@MODELS.register("spectral_memory")
def build_spectral_memory(model_cfg: dict, vocab_size: int | None = None) -> SpectralMemoryLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return SpectralMemoryLM(SpectralMemoryConfig(**cfg))
