"""MRWT: MultiGear Residual Workbench Transformer baseline.

Transformer anchor with an exact fallback path plus zero-gated residual atlas
and workbench adapters. Unlike mecm/mcpm/mgcf, MRWT does not subclass
``NativeCausalLM`` -- it wraps a Transformer anchor directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS
from ..transformer.model import CachedTransformerLM, RMSNorm, TransformerConfig
from .._shared.causal_mesh_base import (
    CausalResidualAdapter,
    HierarchicalDraftHead,
    MultiScaleSpanAtlas,
    _gate_is_zero,
    lm_cross_entropy,
    parameter_count,
    sample_from_logits,
)


@dataclass(frozen=True)
class MRWTConfig:
    """Configuration for the Transformer-anchor residual workbench baseline."""

    vocab_size: int
    dim: int = 256
    layers: int = 6
    heads: int = 8
    max_seq_len: int = 2048
    dropout: float = 0.0
    atlas_kernel_size: int = 9
    workbench_kernel_size: int = 17
    use_atlas: bool = True
    use_workbench: bool = True
    full_architecture: bool = False
    atlas_kernel_sizes: tuple[int, ...] = (5, 17, 65)
    workbench_rounds: int = 2
    draft_horizons: tuple[int, ...] = (2, 4)
    draft_aux_stride: int = 1
    budget_aux_weight: float = 0.0
    draft_aux_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.layers < 1:
            raise ValueError("layers must be positive")
        if self.atlas_kernel_size < 2:
            raise ValueError("atlas_kernel_size must be at least 2")
        if self.workbench_kernel_size < 2:
            raise ValueError("workbench_kernel_size must be at least 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.workbench_rounds < 0:
            raise ValueError("workbench_rounds must be non-negative")
        if not self.atlas_kernel_sizes:
            raise ValueError("atlas_kernel_sizes must not be empty")
        if any(int(k) < 2 for k in self.atlas_kernel_sizes):
            raise ValueError("atlas_kernel_sizes must contain values >= 2")
        if any(int(h) < 1 for h in self.draft_horizons):
            raise ValueError("draft_horizons must contain positive offsets")
        if self.draft_aux_stride < 1:
            raise ValueError("draft_aux_stride must be positive")
        if self.budget_aux_weight < 0.0:
            raise ValueError("budget_aux_weight must be non-negative")
        if self.draft_aux_weight < 0.0:
            raise ValueError("draft_aux_weight must be non-negative")

    def anchor_config(self) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=self.vocab_size,
            dim=self.dim,
            layers=self.layers,
            heads=self.heads,
            max_seq_len=self.max_seq_len,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BudgetController(nn.Module):
    """MRWT profile selector trained with deterministic pseudo budgets."""

    profile_count = 4

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.profile_head = nn.Linear(dim, self.profile_count, bias=False)
        self.profile_embed = nn.Embedding(self.profile_count, dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.training and _gate_is_zero(self.gate):
            return hidden
        probs = self.profile_head(self.norm(hidden)).softmax(dim=-1)
        return hidden + self.gate * (probs @ self.profile_embed.weight)

    def loss(self, hidden: torch.Tensor, tokens: torch.Tensor, valid_next: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] < 2:
            return hidden.sum() * 0.0
        nxt = tokens[:, 1:]
        targets = torch.zeros_like(nxt)
        targets = torch.where(nxt < 256, torch.ones_like(targets), targets)
        targets = torch.where((nxt % 7) == 0, torch.full_like(targets, 2), targets)
        targets = torch.where((nxt % 11) == 0, torch.full_like(targets, 3), targets)
        logits = self.profile_head(self.norm(hidden[:, :-1]))
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).reshape_as(targets)
        return (losses * valid_next.to(losses.dtype)).sum() / valid_next.sum().clamp_min(1)


def transformer_anchor(config: MRWTConfig) -> CachedTransformerLM:
    anchor = CachedTransformerLM(config.anchor_config())
    # Match the non-anchor heads' initialization scale when residual modules are
    # enabled later. The anchor itself already initializes its tied embedding.
    return anchor


class MultiGearResidualWorkbenchTransformerLM(nn.Module):
    """MRWT baseline with exact anchor fallback and zero-gated residual adapters."""

    family_name = "mrwt"

    def __init__(self, config: MRWTConfig) -> None:
        super().__init__()
        self.config = config
        self.anchor = transformer_anchor(config)
        if config.full_architecture:
            self.atlas = (
                MultiScaleSpanAtlas(config.dim, tuple(config.atlas_kernel_sizes))
                if config.use_atlas
                else None
            )
            self.budget_controller = BudgetController(config.dim)
            self.workbench_rounds = (
                nn.ModuleList(
                    [
                        CausalResidualAdapter(config.dim, config.workbench_kernel_size)
                        for _ in range(config.workbench_rounds)
                    ]
                )
                if config.use_workbench
                else nn.ModuleList()
            )
            self.draft_tree = HierarchicalDraftHead(
                config.dim,
                config.vocab_size,
                tuple(config.draft_horizons),
                stride=config.draft_aux_stride,
            )
            self.workbench = None
        else:
            self.atlas = (
                CausalResidualAdapter(config.dim, config.atlas_kernel_size)
                if config.use_atlas
                else None
            )
            self.workbench = (
                CausalResidualAdapter(config.dim, config.workbench_kernel_size)
                if config.use_workbench
                else None
            )
            self.budget_controller = None
            self.workbench_rounds = nn.ModuleList()
            self.draft_tree = None

    def _forward_hidden(self, ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        hidden, _ = self.anchor._forward_hidden(ids, attention_mask=attention_mask)
        if self.atlas is not None:
            hidden = self.atlas(hidden)
        if self.budget_controller is not None:
            hidden = self.budget_controller(hidden)
        if self.workbench is not None:
            hidden = self.workbench(hidden)
        for workbench_round in self.workbench_rounds:
            hidden = workbench_round(hidden)
        return hidden

    def _logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.anchor._output_scores(hidden)

    @staticmethod
    def _valid_next(tokens: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
        valid = torch.ones_like(tokens[:, 1:], dtype=torch.bool)
        if meta.get("loss_mask") is not None:
            valid = valid & meta["loss_mask"][:, 1:].bool()
        if meta.get("attention_mask") is not None:
            valid = valid & meta["attention_mask"][:, 1:].bool()
        return valid

    def forward(self, ids: torch.Tensor, attention_mask=None, **_: Any):
        hidden = self._forward_hidden(ids, attention_mask=attention_mask)
        return self._logits_from_hidden(hidden), None

    def anchor_logits(self, ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        logits, _ = self.anchor(ids, attention_mask=attention_mask)
        return logits

    def _residual_paths_disabled(self) -> bool:
        for name, parameter in self.named_parameters():
            if name.startswith("anchor."):
                continue
            if name.endswith("gate") and bool((parameter.detach().abs() != 0).any()):
                return False
        return True

    def training_step(
        self,
        tokens: torch.Tensor,
        task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        meta = task_metadata or {}
        hidden = self._forward_hidden(tokens, attention_mask=meta.get("attention_mask"))
        logits = self._logits_from_hidden(hidden)
        language_modeling = lm_cross_entropy(
            logits,
            tokens,
            loss_mask=meta.get("loss_mask"),
            attention_mask=meta.get("attention_mask"),
        )
        scale = (loss_term_scales or {}).get("language_modeling", 1.0)
        total = scale * language_modeling
        result = {"language_modeling": language_modeling}
        valid_next = self._valid_next(tokens, meta)
        if self.budget_controller is not None and self.config.budget_aux_weight > 0.0:
            budget_loss = self.budget_controller.loss(hidden, tokens, valid_next)
            total = total + (
                self.config.budget_aux_weight
                * (loss_term_scales or {}).get("budget_controller", 1.0)
                * budget_loss
            )
            result["budget_controller"] = budget_loss
        if self.draft_tree is not None and self.config.draft_aux_weight > 0.0:
            draft_loss = self.draft_tree.loss(hidden, tokens, valid_next)
            total = total + (
                self.config.draft_aux_weight
                * (loss_term_scales or {}).get("draft_tree", 1.0)
                * draft_loss
            )
            result["draft_tree"] = draft_loss
        result["total"] = total
        return result

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, sampling_config=None):
        # Full-prefix decoding keeps residual atlas/workbench semantics exact.
        if prompt_tokens.ndim != 2:
            raise ValueError("prompt_tokens must be a rank-2 tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return torch.empty(
                prompt_tokens.shape[0], 0, dtype=torch.long, device=prompt_tokens.device
            )
        if self._residual_paths_disabled():
            return self.anchor.generate(prompt_tokens, max_new_tokens, sampling_config)
        sequence = prompt_tokens
        out = []
        for _ in range(max_new_tokens):
            logits, _ = self(sequence)
            token = sample_from_logits(logits[:, -1], sampling_config)
            out.append(token)
            sequence = torch.cat([sequence, token], dim=1)
        if not out:
            return torch.empty(
                prompt_tokens.shape[0], 0, dtype=torch.long, device=prompt_tokens.device
            )
        return torch.cat(out, dim=1)

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "family": self.family_name,
            "config": self.config.to_dict(),
            "parameters": {"total": parameter_count(self)},
        }


@MODELS.register("mrwt")
def build_mrwt(
    model_cfg: dict, vocab_size: int | None = None
) -> MultiGearResidualWorkbenchTransformerLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return MultiGearResidualWorkbenchTransformerLM(MRWTConfig(**cfg))
