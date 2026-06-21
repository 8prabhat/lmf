"""MGCF: MultiGear Fractal Causal Field baseline.

MGCF is intentionally not a Transformer/Mamba variant. Its sequence mixer is a
bank of gated dilated causal convolutions plus causal prefix-memory views.
MultiGear hierarchy enters through input gear embeddings and the same
gear-aware output contract used by MECM/MCPM (reused via ``NativeCausalLM``
in ``lmf.models._shared.causal_mesh_base`` -- MGCF replaces the trunk but
keeps the inherited gear-hierarchy, training, and generation scaffolding).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS
from ..transformer.model import RMSNorm
from .._shared.causal_mesh_base import (
    CausalDepthwiseConv,
    NativeCausalLM,
    init_embedding,
    sample_from_logits,
)


@dataclass(frozen=True)
class MGCFConfig:
    """Configuration for the MultiGear Fractal Causal Field model.

    MGCF is intentionally not a Transformer/Mamba variant. Its sequence mixer is
    a bank of gated dilated causal convolutions plus causal prefix-memory views.
    MultiGear hierarchy enters through input gear embeddings and the same
    gear-aware output contract used by MECM.
    """

    vocab_size: int
    dim: int = 256
    layers: int = 6
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    memory_scales: tuple[int, ...] = (4, 16, 64)
    max_seq_len: int = 2048
    dropout: float = 0.0
    hierarchy_gears: int = 6
    input_gear_embedding: bool = True
    hierarchy_composition: bool = True
    byte_length_embedding: bool = True
    max_token_bytes: int = 64
    composition_gate_init: float = 0.05
    composition_aux_weight: float = 0.0
    gear_aware_output: bool = True
    gear_output_mode: str = "bias"
    gear_aux_weight: float = 0.0
    memory_type: str = "learned"
    memory_gate_init: float = 0.05

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.dim < 8:
            raise ValueError("dim must be at least 8")
        if self.layers < 1:
            raise ValueError("layers must be positive")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if not self.dilations:
            raise ValueError("dilations must not be empty")
        if any(int(d) < 1 for d in self.dilations):
            raise ValueError("dilations must contain positive values")
        if any(int(s) < 1 for s in self.memory_scales):
            raise ValueError("memory_scales must contain positive values")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.hierarchy_gears < 1:
            raise ValueError("hierarchy_gears must be positive")
        if self.max_token_bytes < 1:
            raise ValueError("max_token_bytes must be positive")
        if self.composition_aux_weight < 0.0:
            raise ValueError("composition_aux_weight must be non-negative")
        if self.gear_output_mode not in {"factorized", "bias"}:
            raise ValueError("gear_output_mode must be 'factorized' or 'bias'")
        if self.gear_aux_weight < 0.0:
            raise ValueError("gear_aux_weight must be non-negative")
        if self.memory_type not in {"average", "learned"}:
            raise ValueError("memory_type must be 'average' or 'learned'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalDilatedDepthwiseConv(nn.Module):
    """Depthwise causal convolution with dilation and no future leakage."""

    def __init__(self, dim: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(
            dim,
            dim,
            self.kernel_size,
            dilation=self.dilation,
            groups=dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = F.pad(y, ((self.kernel_size - 1) * self.dilation, 0))
        y = self.conv(y)
        return y.transpose(1, 2)


class CausalPrefixMemory(nn.Module):
    """Causal multi-scale prefix views using sliding averages.

    This is deliberately deterministic and parallel under teacher forcing. It
    gives MGCF cheap long-range context without attention or recurrent
    selective-state machinery.
    """

    def __init__(self, dim: int, scales: tuple[int, ...]) -> None:
        super().__init__()
        self.scales = tuple(int(scale) for scale in scales)
        self.proj = nn.Linear(dim * len(self.scales), dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        views = []
        for scale in self.scales:
            padded = F.pad(y, (scale - 1, 0))
            pooled = F.avg_pool1d(padded, kernel_size=scale, stride=1)
            views.append(pooled.transpose(1, 2))
        return self.proj(torch.cat(views, dim=-1))


class LearnableCausalLongMemory(nn.Module):
    """Learned multi-scale causal long filters.

    Unlike fixed prefix averages, each scale owns a depthwise causal filter and
    the model learns which long-range patterns to preserve. The operation is
    still parallel under teacher forcing and contains no attention matrix.
    """

    def __init__(self, dim: int, scales: tuple[int, ...]) -> None:
        super().__init__()
        self.filters = nn.ModuleList(
            [CausalDepthwiseConv(dim, int(scale)) for scale in scales]
        )
        self.router = nn.Linear(dim, len(scales), bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.router(x).softmax(dim=-1)
        views = torch.stack([long_filter(x) for long_filter in self.filters], dim=-2)
        mixed = (views * weights.unsqueeze(-1)).sum(dim=-2)
        return self.proj(F.silu(mixed))


class FractalCausalFieldBlock(nn.Module):
    """MGCF block: routed dilated causal fields plus prefix-memory residual."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        memory_scales: tuple[int, ...],
        dropout: float = 0.0,
        memory_type: str = "learned",
        memory_gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.branches = nn.ModuleList(
            [
                CausalDilatedDepthwiseConv(dim, kernel_size, int(dilation))
                for dilation in dilations
            ]
        )
        self.router = nn.Linear(dim, len(dilations), bias=False)
        self.mix = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.memory_norm = RMSNorm(dim)
        if memory_scales and memory_type == "learned":
            self.memory = LearnableCausalLongMemory(dim, memory_scales)
        elif memory_scales:
            self.memory = CausalPrefixMemory(dim, memory_scales)
        else:
            self.memory = None
        self.memory_gate = nn.Parameter(torch.tensor(float(memory_gate_init)))
        self.norm2 = RMSNorm(dim)
        self.ff_gate = nn.Linear(dim, 4 * dim, bias=False)
        self.ff_up = nn.Linear(dim, 4 * dim, bias=False)
        self.ff_down = nn.Linear(4 * dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        weights = self.router(z).softmax(dim=-1)
        fields = torch.stack([branch(z) for branch in self.branches], dim=-2)
        mixed = (fields * weights.unsqueeze(-1)).sum(dim=-2)
        gate, value = self.mix(mixed).chunk(2, dim=-1)
        x = x + self.dropout(self.proj(F.silu(gate) * value))
        if self.memory is not None:
            x = x + self.memory_gate * self.dropout(self.memory(self.memory_norm(x)))
        z = self.norm2(x)
        x = x + self.dropout(self.ff_down(F.silu(self.ff_gate(z)) * self.ff_up(z)))
        return x


class MultiGearFractalCausalFieldLM(NativeCausalLM):
    """MGCF: non-attention, non-Mamba MultiGear-native causal field model."""

    family_name = "mgcf"

    def __init__(self, config: MGCFConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.dim)
        self.position = nn.Embedding(config.max_seq_len, config.dim)
        self.gear_embedding = (
            nn.Embedding(config.hierarchy_gears, config.dim)
            if config.input_gear_embedding
            else None
        )
        self.byte_length_embedding = (
            nn.Embedding(config.max_token_bytes + 1, config.dim)
            if config.byte_length_embedding
            else None
        )
        self.composition_proj = (
            nn.Linear(2 * config.dim, config.dim, bias=False)
            if config.hierarchy_composition
            else None
        )
        self.composition_gate = (
            nn.Parameter(torch.tensor(float(config.composition_gate_init)))
            if config.hierarchy_composition
            else None
        )
        self.composition_slots = (
            nn.Parameter(torch.empty(2, config.dim))
            if config.composition_aux_weight > 0.0
            else None
        )
        self.blocks = nn.ModuleList(
            [
                FractalCausalFieldBlock(
                    config.dim,
                    config.kernel_size,
                    tuple(config.dilations),
                    tuple(config.memory_scales),
                    config.dropout,
                    memory_type=config.memory_type,
                    memory_gate_init=config.memory_gate_init,
                )
                for _ in range(config.layers)
            ]
        )
        self.mesh = None
        self.span_atlas = None
        self.active_cover = None
        self.reasoning_mesh = None
        self.execution = None
        self.program_controller = None
        self.execution_workbench = None
        self.contract_verifier = None
        self.draft_tree = None
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.gear_head = (
            nn.Linear(config.dim, config.hierarchy_gears, bias=False)
            if (
                config.gear_aware_output
                or config.gear_aux_weight > 0.0
                or config.input_gear_embedding
            )
            else None
        )
        if self.gear_head is not None:
            self.register_buffer(
                "_token_gears",
                torch.full((config.vocab_size,), -1, dtype=torch.long),
            )
            self.register_buffer(
                "_token_to_local",
                torch.full((config.vocab_size,), -1, dtype=torch.long),
            )
            self.register_buffer(
                "_gear_active",
                torch.zeros(config.hierarchy_gears, dtype=torch.bool),
            )
        if (
            config.hierarchy_composition
            or config.byte_length_embedding
            or config.composition_aux_weight > 0.0
        ):
            self.register_buffer(
                "_token_children",
                torch.full((config.vocab_size, 2), -1, dtype=torch.long),
            )
            self.register_buffer(
                "_token_byte_lengths",
                torch.zeros(config.vocab_size, dtype=torch.long),
            )
        init_embedding(self.token)
        if self.gear_embedding is not None:
            init_embedding(self.gear_embedding)
        if self.byte_length_embedding is not None:
            init_embedding(self.byte_length_embedding)
        if self.gear_head is not None:
            nn.init.normal_(self.gear_head.weight, mean=0.0, std=0.02)
        if self.composition_slots is not None:
            nn.init.normal_(self.composition_slots, mean=0.0, std=0.02)

    def _hierarchy_input_residual(self, ids: torch.Tensor) -> torch.Tensor:
        self._require_token_hierarchy()
        residual = None
        if self.composition_proj is not None:
            children = self._token_children[ids]
            child_mask = children.ge(0).unsqueeze(-1)
            safe_children = children.clamp_min(0)
            child_embeddings = self.token(safe_children) * child_mask.to(self.token.weight.dtype)
            composed = self.composition_proj(child_embeddings.flatten(start_dim=-2))
            parent_mask = child_mask.any(dim=-2).to(composed.dtype)
            composed = self.composition_gate * composed * parent_mask
            residual = composed if residual is None else residual + composed
        if self.byte_length_embedding is not None:
            lengths = self._token_byte_lengths[ids].clamp_max(self.config.max_token_bytes)
            length_state = self.byte_length_embedding(lengths)
            residual = length_state if residual is None else residual + length_state
        if residual is None:
            return torch.zeros(
                *ids.shape,
                self.config.dim,
                dtype=self.token.weight.dtype,
                device=ids.device,
            )
        return residual

    def _forward_hidden(self, ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        pos = (torch.arange(ids.shape[1], device=ids.device) + int(position_offset)) % self.config.max_seq_len
        hidden = self.token(ids) + self.position(pos)[None]
        if self.gear_embedding is not None:
            self._require_token_hierarchy()
            gear_ids = self._token_gears[ids].clamp_min(0)
            hidden = hidden + self.gear_embedding(gear_ids)
        if self.composition_proj is not None or self.byte_length_embedding is not None:
            hidden = hidden + self._hierarchy_input_residual(ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)

    def _composition_auxiliary_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        self._require_token_hierarchy()
        if self.composition_slots is None:
            return hidden.sum() * 0.0
        children = self._token_children[targets]
        total = hidden.sum() * 0.0
        terms = 0
        for slot in range(2):
            child_targets = children[..., slot]
            mask = valid & child_targets.ge(0)
            if not bool(mask.any()):
                continue
            slot_hidden = hidden + self.composition_slots[slot]
            logits = F.linear(slot_hidden[mask], self.token.weight)
            total = total + F.cross_entropy(logits, child_targets[mask], reduction="mean")
            terms += 1
        return total / max(1, terms)

    def _generation_context(self) -> int:
        branch_reach = (self.config.kernel_size - 1) * max(self.config.dilations)
        memory_reach = max(self.config.memory_scales, default=1) - 1
        per_layer = max(branch_reach, memory_reach)
        return min(self.config.max_seq_len, 1 + self.config.layers * per_layer)

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, sampling_config=None):
        if prompt_tokens.ndim != 2:
            raise ValueError("prompt_tokens must be a rank-2 tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        sequence = prompt_tokens
        out = []
        context = self._generation_context()
        for _ in range(max_new_tokens):
            window = sequence[:, -context:]
            offset = sequence.shape[1] - window.shape[1]
            if (
                self.config.gear_aware_output
                and self.config.gear_output_mode == "factorized"
            ):
                hidden = self._forward_hidden(window, position_offset=offset)
                token = self._sample_hierarchical_token(hidden[:, -1], sampling_config)
            else:
                hidden = self._forward_hidden(window, position_offset=offset)
                token = sample_from_logits(
                    self._logits_from_hidden(hidden)[:, -1], sampling_config
                )
            out.append(token)
            sequence = torch.cat([sequence, token], dim=1)
        if not out:
            return torch.empty(
                prompt_tokens.shape[0], 0, dtype=torch.long, device=prompt_tokens.device
            )
        return torch.cat(out, dim=1)


@MODELS.register("mgcf")
def build_mgcf(model_cfg: dict, vocab_size: int | None = None) -> MultiGearFractalCausalFieldLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return MultiGearFractalCausalFieldLM(MGCFConfig(**cfg))
