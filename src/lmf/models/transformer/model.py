"""Parameter-matched modern transformer baseline (RMSNorm + RoPE + SwiGLU + SDPA).

The honest comparison for RHCA is not a vanilla 2017 GPT but a transformer built
from the same current building blocks, so any quality gap reflects the core
mechanism rather than which model got the better standard tricks. Attention uses
``F.scaled_dot_product_attention`` (fused, causal) for both training and KV-cached
decoding.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    dim: int = 512
    layers: int = 8
    heads: int = 8
    max_seq_len: int = 4096
    hierarchical_output: bool = False
    hierarchy_output_mode: str = "factorized"
    input_gear_embedding: bool = False
    hierarchy_gears: int = 6
    hierarchy_aux_weight: float = 0.0
    hierarchy_aux_min_gear: int = 2
    hierarchy_aux_target: str = "bytes"
    hierarchy_aux_max_bytes: int = 16

    def __post_init__(self) -> None:
        if self.hierarchy_gears < 1:
            raise ValueError("hierarchy_gears must be positive")
        if self.hierarchy_output_mode not in {"factorized", "bias"}:
            raise ValueError("hierarchy_output_mode must be 'factorized' or 'bias'")
        if self.hierarchy_aux_weight < 0.0:
            raise ValueError("hierarchy_aux_weight must be non-negative")
        if self.hierarchy_aux_min_gear < 0:
            raise ValueError("hierarchy_aux_min_gear must be non-negative")
        if self.hierarchy_aux_target not in {"bytes", "children"}:
            raise ValueError("hierarchy_aux_target must be 'bytes' or 'children'")
        if self.hierarchy_aux_max_bytes < 1:
            raise ValueError("hierarchy_aux_max_bytes must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope(x, positions):
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    ang = positions[:, None].float() * inv_freq[None, :]
    ang = torch.cat([ang, ang], dim=-1).to(x.dtype)
    return x * ang.cos()[None, None] + _rotate_half(x) * ang.sin()[None, None]


class SwiGLU(nn.Module):
    def __init__(self, dim: int, multiple_of: int = 32) -> None:
        super().__init__()
        hidden = int(2 * (4 * dim) / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim)

    def forward(self, x, cache=None, use_cache=False, attn_mask=None):
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        past = 0 if cache is None else cache[0].shape[2]
        positions = torch.arange(past, past + n, device=x.device)
        q, k = _rope(q, positions), _rope(k, positions)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        if attn_mask is not None:
            # Explicit boolean mask (True = attend) already folds in causality +
            # key padding; don't also pass is_causal.
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None and n > 1)
        out = out.transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(out)
        x = x + self.ff(self.norm2(x))
        return x, ((k, v) if use_cache else None)


class CachedTransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([Block(config.dim, config.heads) for _ in range(config.layers)])
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.gear_head = (
            nn.Linear(config.dim, config.hierarchy_gears, bias=False)
            if config.hierarchical_output
            else None
        )
        self.gear_embedding = (
            nn.Embedding(config.hierarchy_gears, config.dim)
            if config.input_gear_embedding
            else None
        )
        self.decomposition_slots = (
            nn.Parameter(
                torch.empty(
                    2 if config.hierarchy_aux_target == "children"
                    else config.hierarchy_aux_max_bytes,
                    config.dim,
                )
            )
            if config.hierarchy_aux_weight > 0.0
            else None
        )
        if (
            config.hierarchical_output
            or config.input_gear_embedding
            or config.hierarchy_aux_weight > 0.0
        ):
            self.register_buffer(
                "_token_gears",
                torch.full((config.vocab_size,), -1, dtype=torch.long),
            )
            self.register_buffer(
                "_token_children",
                torch.full((config.vocab_size, 2), -1, dtype=torch.long),
            )
            self.register_buffer(
                "_token_to_local",
                torch.full((config.vocab_size,), -1, dtype=torch.long),
            )
            self.register_buffer(
                "_gear_active",
                torch.zeros(config.hierarchy_gears, dtype=torch.bool),
            )
            if config.hierarchy_aux_target == "bytes":
                self.register_buffer(
                    "_token_bytes",
                    torch.full(
                        (config.vocab_size, config.hierarchy_aux_max_bytes),
                        -1,
                        dtype=torch.long,
                    ),
                )
        # nn.Embedding defaults to N(0,1); tied to the head this produces
        # logits with std ~= sqrt(dim), far larger than the ~unit scale a
        # softmax over `vocab_size` classes expects (initial CE >> log(V)).
        # Use the standard GPT-2-style small-std init instead.
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)
        if self.gear_head is not None:
            nn.init.normal_(self.gear_head.weight, mean=0.0, std=0.02)
        if self.gear_embedding is not None:
            nn.init.normal_(self.gear_embedding.weight, mean=0.0, std=0.02)
        if self.decomposition_slots is not None:
            nn.init.normal_(self.decomposition_slots, mean=0.0, std=0.02)

    def configure_token_hierarchy(
        self,
        gear_count: int,
        token_gears: list[int],
        token_children: list[list[int]],
        token_bytes: list[list[int]],
    ) -> None:
        """Install tokenizer hierarchy metadata used by optional objectives."""
        if not hasattr(self, "_token_gears"):
            return
        if gear_count > self.config.hierarchy_gears:
            raise ValueError(
                f"tokenizer needs {gear_count} gears, model supports "
                f"{self.config.hierarchy_gears}"
            )
        if len(token_gears) != self.config.vocab_size:
            raise ValueError("token_gears length must equal model vocabulary size")
        if len(token_children) != self.config.vocab_size:
            raise ValueError("token_children length must equal model vocabulary size")
        if len(token_bytes) != self.config.vocab_size:
            raise ValueError("token_bytes length must equal model vocabulary size")
        gears = torch.tensor(token_gears, dtype=torch.long, device=self._token_gears.device)
        children = torch.tensor(
            token_children, dtype=torch.long, device=self._token_children.device
        )
        if bool(((gears < 0) | (gears >= gear_count)).any()):
            raise ValueError("token gear outside declared gear_count")
        if bool(((children < -1) | (children >= self.config.vocab_size)).any()):
            raise ValueError("token child outside vocabulary")
        local = torch.full_like(self._token_to_local, -1)
        active = torch.zeros_like(self._gear_active)
        for gear in range(gear_count):
            token_ids = torch.nonzero(gears == gear, as_tuple=False).flatten()
            active[gear] = bool(len(token_ids))
            local[token_ids] = torch.arange(len(token_ids), device=local.device)
        self._token_gears.copy_(gears)
        self._token_children.copy_(children)
        self._token_to_local.copy_(local)
        self._gear_active.copy_(active)
        if hasattr(self, "_token_bytes"):
            byte_targets = torch.full_like(self._token_bytes, -1)
            for token_id, values in enumerate(token_bytes):
                values = values[:self.config.hierarchy_aux_max_bytes]
                if values:
                    byte_targets[token_id, :len(values)] = torch.tensor(
                        values, dtype=torch.long, device=byte_targets.device
                    )
            if bool(((byte_targets < -1) | (byte_targets > 255)).any()):
                raise ValueError("token_bytes must contain raw byte ids")
            self._token_bytes.copy_(byte_targets)

    def _require_token_hierarchy(self) -> None:
        if not hasattr(self, "_gear_active") or not bool(self._gear_active.any()):
            raise RuntimeError(
                "token hierarchy is required; build with a MultiGear tokenizer "
                "or call configure_token_hierarchy()"
            )

    @staticmethod
    def _full_attn_mask(attention_mask, n, device):
        """Combine causal + key-padding into a boolean (b,1,n,n) mask (True = attend).

        Returns None when there is no padding, so the fast is_causal path is used.
        """
        if attention_mask is None or bool(attention_mask.all()):
            return None
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
        key_valid = attention_mask.bool()[:, None, None, :]          # (b,1,1,n)
        return causal[None, None] & key_valid                        # (b,1,n,n)

    def _forward_hidden(self, ids, caches=None, use_cache=False, attention_mask=None):
        x = self.token(ids)
        if self.gear_embedding is not None:
            self._require_token_hierarchy()
            x = x + self.gear_embedding(self._token_gears[ids])
        attn_mask = (self._full_attn_mask(attention_mask, ids.shape[1], ids.device)
                     if caches is None else None)
        next_caches = []
        for i, block in enumerate(self.blocks):
            x, nc = block(x, None if caches is None else caches[i], use_cache, attn_mask)
            if use_cache:
                next_caches.append(nc)
        return self.norm(x), next_caches

    def _gear_logits(self, hidden):
        self._require_token_hierarchy()
        logits = self.gear_head(hidden)
        return logits.masked_fill(~self._gear_active, float("-inf"))

    def _hierarchical_scores(self, hidden):
        """Return full-vocabulary log probabilities for generation/evaluation."""
        token_scores = self.head(hidden)
        gear_log_probs = self._gear_logits(hidden).log_softmax(dim=-1)
        normalizers = []
        for gear in range(self.config.hierarchy_gears):
            token_ids = torch.nonzero(self._token_gears == gear, as_tuple=False).flatten()
            if len(token_ids):
                normalizers.append(
                    token_scores.index_select(-1, token_ids).logsumexp(dim=-1)
                )
            else:
                normalizers.append(torch.zeros_like(token_scores[..., 0]))
        within_gear_normalizers = torch.stack(normalizers, dim=-1)
        return (
            token_scores
            - within_gear_normalizers.index_select(-1, self._token_gears)
            + gear_log_probs.index_select(-1, self._token_gears)
        )

    def _gear_biased_scores(self, hidden):
        """Cheaper hierarchy use: full token logits plus next-token gear bias."""
        token_scores = self.head(hidden)
        return token_scores + self._gear_logits(hidden).index_select(-1, self._token_gears)

    def _output_scores(self, hidden):
        if self.config.hierarchical_output:
            if self.config.hierarchy_output_mode == "bias":
                return self._gear_biased_scores(hidden)
            return self._hierarchical_scores(hidden)
        return self.head(hidden)

    def forward(self, ids, caches=None, use_cache=False, attention_mask=None):
        hidden, next_caches = self._forward_hidden(ids, caches, use_cache, attention_mask)
        return self._output_scores(hidden), next_caches

    @staticmethod
    def _valid_targets(tokens, loss_mask, attention_mask):
        valid = torch.ones_like(tokens[:, 1:], dtype=torch.bool)
        if loss_mask is not None:
            valid = valid & loss_mask[:, 1:].bool()
        if attention_mask is not None:
            valid = valid & attention_mask[:, 1:].bool()
        return valid

    def _flat_language_modeling_loss(self, hidden, targets, valid):
        logits = self.head(hidden)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        valid_float = valid.to(losses.dtype)
        return (losses * valid_float).sum() / valid_float.sum().clamp_min(1)

    def _hierarchical_language_modeling_loss(self, hidden, targets, valid):
        self._require_token_hierarchy()
        target_gears = self._token_gears[targets]
        gear_logits = self._gear_logits(hidden)
        gear_losses = F.cross_entropy(
            gear_logits.reshape(-1, gear_logits.shape[-1]),
            target_gears.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        valid_float = valid.to(gear_losses.dtype)
        count = valid_float.sum().clamp_min(1)
        gear_loss = (gear_losses * valid_float).sum() / count

        token_loss_sum = hidden.sum() * 0.0
        for gear in range(self.config.hierarchy_gears):
            positions = valid & (target_gears == gear)
            if not bool(positions.any()):
                continue
            token_ids = torch.nonzero(self._token_gears == gear, as_tuple=False).flatten()
            local_logits = F.linear(hidden[positions], self.token.weight[token_ids])
            local_targets = self._token_to_local[targets[positions]]
            token_loss_sum = token_loss_sum + F.cross_entropy(
                local_logits, local_targets, reduction="sum"
            )
        token_loss = token_loss_sum / count
        return gear_loss + token_loss, gear_loss, token_loss

    def _hierarchy_auxiliary_loss(self, hidden, targets, valid):
        self._require_token_hierarchy()
        target_gears = self._token_gears[targets]
        selected = valid & (target_gears >= self.config.hierarchy_aux_min_gear)
        if self.config.hierarchy_aux_target == "children":
            child_targets = self._token_children[targets]
            selected = selected & (child_targets[..., 0] >= 0)
        else:
            child_targets = self._token_bytes[targets]
            selected = selected & (child_targets[..., 0] >= 0)
        if not bool(selected.any()):
            return hidden.sum() * 0.0
        selected_hidden = hidden[selected]
        selected_children = child_targets[selected]
        total = selected_hidden.sum() * 0.0
        count = 0
        for slot in range(selected_children.shape[1]):
            slot_valid = selected_children[:, slot] >= 0
            if not bool(slot_valid.any()):
                continue
            candidate_weight = (
                self.token.weight
                if self.config.hierarchy_aux_target == "children"
                else self.token.weight[:256]
            )
            logits = F.linear(
                selected_hidden[slot_valid] + self.decomposition_slots[slot],
                candidate_weight,
            )
            total = total + F.cross_entropy(
                logits, selected_children[slot_valid, slot], reduction="sum"
            )
            count += int(slot_valid.sum())
        return total / max(count, 1)

    def training_step(self, tokens, task_metadata: dict[str, Any] | None = None,
                      loss_term_scales: dict[str, float] | None = None) -> dict[str, torch.Tensor]:
        meta = task_metadata or {}
        attention_mask = meta.get("attention_mask")
        loss_mask = meta.get("loss_mask")
        hidden, _ = self._forward_hidden(tokens, attention_mask=attention_mask)
        prediction_hidden = hidden[:, :-1]
        targets = tokens[:, 1:]
        valid = self._valid_targets(tokens, loss_mask, attention_mask)
        scales = loss_term_scales or {}
        if (
            self.config.hierarchical_output
            and self.config.hierarchy_output_mode == "factorized"
        ):
            language_modeling, gear_prediction, within_gear = (
                self._hierarchical_language_modeling_loss(
                    prediction_hidden, targets, valid
                )
            )
        else:
            language_modeling = self._flat_language_modeling_loss(
                prediction_hidden, targets, valid
            )
            gear_prediction = None
            within_gear = None

        total = scales.get("language_modeling", 1.0) * language_modeling
        result = {"language_modeling": language_modeling}
        if gear_prediction is not None:
            result["gear_prediction"] = gear_prediction
            result["within_gear"] = within_gear
        if self.config.hierarchy_aux_weight > 0.0:
            hierarchy_aux = self._hierarchy_auxiliary_loss(
                prediction_hidden, targets, valid
            )
            total = total + (
                self.config.hierarchy_aux_weight
                * scales.get("hierarchy_aux", 1.0)
                * hierarchy_aux
            )
            result["hierarchy_aux"] = hierarchy_aux
        result["total"] = total
        return result

    def _sample_token(self, logits, cfg) -> torch.Tensor:
        if cfg is None or cfg.deterministic:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(cfg.temperature, 1e-5)
        if cfg.top_k > 0:
            thresh = logits.topk(min(cfg.top_k, logits.shape[-1]), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        if cfg.top_p < 1.0:
            sl, idx = logits.sort(dim=-1, descending=True)
            remove = sl.softmax(-1).cumsum(-1) > cfg.top_p
            remove[..., 0] = False
            sl = sl.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sl)
        return torch.multinomial(logits.softmax(-1), 1)

    def _sample_hierarchical_token(self, hidden, cfg) -> torch.Tensor:
        """Select a gear, then score only that gear's token subset."""
        gears = self._sample_token(self._gear_logits(hidden), cfg).squeeze(-1)
        output = torch.empty((hidden.shape[0], 1), dtype=torch.long, device=hidden.device)
        for gear in torch.unique(gears).tolist():
            rows = torch.nonzero(gears == gear, as_tuple=False).flatten()
            token_ids = torch.nonzero(self._token_gears == gear, as_tuple=False).flatten()
            local_logits = F.linear(hidden[rows], self.token.weight[token_ids])
            local_choice = self._sample_token(local_logits, cfg).squeeze(-1)
            output[rows, 0] = token_ids[local_choice]
        return output

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, sampling_config=None):
        """KV-cached decoding honouring SamplingConfig (greedy by default)."""
        if (
            self.config.hierarchical_output
            and self.config.hierarchy_output_mode == "factorized"
        ):
            hidden, caches = self._forward_hidden(prompt, use_cache=True)
            token = self._sample_hierarchical_token(hidden[:, -1], sampling_config)
        else:
            logits, caches = self(prompt, use_cache=True)
            token = self._sample_token(logits[:, -1], sampling_config)
        out = []
        for _ in range(max_new_tokens):
            out.append(token)
            if (
                self.config.hierarchical_output
                and self.config.hierarchy_output_mode == "factorized"
            ):
                hidden, caches = self._forward_hidden(token, caches=caches, use_cache=True)
                token = self._sample_hierarchical_token(hidden[:, -1], sampling_config)
            else:
                logits, caches = self(token, caches=caches, use_cache=True)
                token = self._sample_token(logits[:, -1], sampling_config)
        return torch.cat(out, dim=1)

    def architecture_manifest(self) -> dict:
        return {"name": "CachedTransformerLM", "config": self.config.to_dict(),
                "parameters": {"total": sum(p.numel() for p in self.parameters())}}


@MODELS.register("transformer")
def build_transformer(model_cfg: dict, vocab_size: int | None = None) -> CachedTransformerLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return CachedTransformerLM(TransformerConfig(**cfg))


@MODELS.register("mght")
def build_multigear_hierarchical_transformer(
    model_cfg: dict, vocab_size: int | None = None
) -> CachedTransformerLM:
    """MultiGear Hierarchical Transformer.

    Uses the standard Transformer trunk, but consumes MultiGear hierarchy through
    input gear embeddings and a gear-aware output head. Defaults favor the fast
    gear-bias head; configs can switch to the factorized head for stricter
    gear-then-token scoring.
    """
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    cfg.setdefault("hierarchical_output", True)
    cfg.setdefault("hierarchy_output_mode", "bias")
    cfg.setdefault("input_gear_embedding", True)
    return CachedTransformerLM(TransformerConfig(**cfg))
