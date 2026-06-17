"""Shared components for the MultiGear-native architecture baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformer.model import CachedTransformerLM, RMSNorm, TransformerConfig


def _gate_is_zero(gate: torch.Tensor) -> bool:
    """True when a zero-gated residual is exactly disabled."""

    return bool((gate.detach().abs() == 0).all())


@dataclass(frozen=True)
class NativeLMConfig:
    """Configuration shared by MECM and MCPM smoke implementations."""

    vocab_size: int
    dim: int = 256
    layers: int = 6
    kernel_size: int = 7
    max_seq_len: int = 2048
    dropout: float = 0.0
    mesh_residual: bool = True
    execution_residual: bool = False
    full_architecture: bool = False
    atlas_kernel_sizes: tuple[int, ...] = (3, 7, 15)
    mesh_layers: int = 2
    draft_horizons: tuple[int, ...] = (2, 4)
    draft_aux_stride: int = 1
    route_aux_weight: float = 0.0
    draft_aux_weight: float = 0.0
    program_aux_weight: float = 0.0
    verifier_aux_weight: float = 0.0
    gear_aware_output: bool = False
    hierarchy_gears: int = 6
    gear_output_mode: str = "factorized"
    gear_aux_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.dim < 8:
            raise ValueError("dim must be at least 8")
        if self.layers < 1:
            raise ValueError("layers must be positive")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mesh_layers < 0:
            raise ValueError("mesh_layers must be non-negative")
        if not self.atlas_kernel_sizes:
            raise ValueError("atlas_kernel_sizes must not be empty")
        if any(int(k) < 2 for k in self.atlas_kernel_sizes):
            raise ValueError("atlas_kernel_sizes must contain values >= 2")
        if any(int(h) < 1 for h in self.draft_horizons):
            raise ValueError("draft_horizons must contain positive offsets")
        if self.draft_aux_stride < 1:
            raise ValueError("draft_aux_stride must be positive")
        if self.hierarchy_gears < 1:
            raise ValueError("hierarchy_gears must be positive")
        if self.gear_output_mode not in {"factorized", "bias"}:
            raise ValueError("gear_output_mode must be 'factorized' or 'bias'")
        for name in ("route_aux_weight", "draft_aux_weight", "program_aux_weight", "verifier_aux_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.gear_aux_weight < 0.0:
            raise ValueError("gear_aux_weight must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


class CausalDepthwiseConv(nn.Module):
    """Depthwise causal convolution over sequence positions."""

    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x N x D. Left-pad only; never expose future positions.
        y = x.transpose(1, 2)
        y = F.pad(y, (self.kernel_size - 1, 0))
        y = self.conv(y)
        return y.transpose(1, 2)


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


class CausalConvBlock(nn.Module):
    """A small non-attention causal block used by MECM/MCPM."""

    def __init__(self, dim: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.conv = CausalDepthwiseConv(dim, kernel_size)
        self.mix = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.ff_gate = nn.Linear(dim, 4 * dim, bias=False)
        self.ff_up = nn.Linear(dim, 4 * dim, bias=False)
        self.ff_down = nn.Linear(4 * dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(self.norm1(x))
        gate, value = self.mix(y).chunk(2, dim=-1)
        x = x + self.dropout(self.proj(F.silu(gate) * value))
        z = self.norm2(x)
        x = x + self.dropout(self.ff_down(F.silu(self.ff_gate(z)) * self.ff_up(z)))
        return x


class ZeroGatedCausalSummary(nn.Module):
    """Causal residual adapter with exact zero-gate fallback."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, 2 * dim, bias=False),
            nn.SiLU(),
            nn.Linear(2 * dim, dim, bias=False),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.training and _gate_is_zero(self.gate):
            return hidden
        steps = torch.arange(1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype)
        summary = hidden.cumsum(dim=1) / steps.view(1, -1, 1)
        return hidden + self.gate * self.proj(self.norm(summary))


class ExecutionTraceAdapter(nn.Module):
    """Deterministic typed-token statistics for MCPM's executable-plane stub.

    This is intentionally conservative: it exposes causal exact-token features
    through a zero gate instead of inventing a verifier that does not exist yet.
    """

    def __init__(self, dim: int, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.proj = nn.Sequential(
            nn.Linear(4, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training and _gate_is_zero(self.gate):
            return hidden
        values = tokens.to(hidden.dtype) / max(1.0, float(self.vocab_size - 1))
        steps = torch.arange(1, tokens.shape[1] + 1, device=tokens.device, dtype=hidden.dtype)
        causal_mean = values.cumsum(dim=1) / steps.view(1, -1)
        repeat = torch.zeros_like(values)
        repeat[:, 1:] = (tokens[:, 1:] == tokens[:, :-1]).to(hidden.dtype)
        delta = torch.zeros_like(values)
        delta[:, 1:] = (values[:, 1:] - values[:, :-1]).abs()
        features = torch.stack([values, causal_mean, repeat, delta], dim=-1)
        return hidden + self.gate * self.proj(features)


class CausalResidualAdapter(nn.Module):
    """Bounded causal residual adapter for MRWT's atlas/workbench paths."""

    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.conv = CausalDepthwiseConv(dim, kernel_size)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def residual(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.conv(self.norm(hidden))))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.training and _gate_is_zero(self.gate):
            return hidden
        return hidden + self.gate * self.residual(hidden)


class MultiScaleSpanAtlas(nn.Module):
    """Bounded reversible-span approximation with ablation-visible scales."""

    def __init__(self, dim: int, kernel_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.scales = nn.ModuleList(
            [CausalResidualAdapter(dim, int(kernel_size)) for kernel_size in kernel_sizes]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        update = None
        for scale in self.scales:
            if not scale.training and _gate_is_zero(scale.gate):
                continue
            delta = scale.gate * scale.residual(hidden)
            update = delta if update is None else update + delta
        if update is None:
            return hidden
        return hidden + update


class ActiveCoverRouter(nn.Module):
    """Learn a causal mixture over local/mid/wide compute views."""

    def __init__(self, dim: int, kernel_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.router = nn.Linear(dim, len(kernel_sizes), bias=False)
        self.views = nn.ModuleList(
            [
                nn.Sequential(
                    CausalDepthwiseConv(dim, int(kernel_size)),
                    nn.SiLU(),
                    nn.Linear(dim, dim, bias=False),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.gate = nn.Parameter(torch.zeros(()))
        self.last_balance_loss: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.training and _gate_is_zero(self.gate):
            self.last_balance_loss = hidden.sum() * 0.0
            return hidden
        normalized = self.norm(hidden)
        probs = self.router(normalized).softmax(dim=-1)
        views = torch.stack([view(normalized) for view in self.views], dim=-2)
        mixed = (views * probs.unsqueeze(-1)).sum(dim=-2)
        mean_probs = probs.reshape(-1, probs.shape[-1]).mean(dim=0)
        uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
        self.last_balance_loss = (
            mean_probs * (mean_probs.clamp_min(1e-9) / uniform).log()
        ).sum()
        return hidden + self.gate * mixed


class ReasoningMeshLayer(nn.Module):
    """Causal evidence ledger update for MECM-style reasoning mesh."""

    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.local = CausalConvBlock(dim, kernel_size)
        self.evidence_norm = RMSNorm(dim)
        self.evidence_proj = nn.Linear(dim, dim, bias=False)
        self.merge = nn.Linear(2 * dim, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.training and _gate_is_zero(self.gate):
            return hidden
        local = self.local(hidden)
        steps = torch.arange(1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype)
        evidence = hidden.cumsum(dim=1) / steps.view(1, -1, 1)
        evidence = self.evidence_proj(self.evidence_norm(evidence))
        update = self.merge(torch.cat([local, evidence], dim=-1))
        return hidden + self.gate * update


class SparseReasoningMesh(nn.Module):
    """Append-only reasoning mesh approximation with named layers."""

    def __init__(self, dim: int, layers: int, kernel_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ReasoningMeshLayer(dim, kernel_size) for _ in range(int(layers))]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class HierarchicalDraftHead(nn.Module):
    """Auxiliary multi-horizon next-event heads for draft-tree ablations."""

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        horizons: tuple[int, ...],
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.horizons = tuple(int(h) for h in horizons)
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError("stride must be positive")
        self.heads = nn.ModuleList([nn.Linear(dim, vocab_size, bias=False) for _ in self.horizons])

    def loss(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        valid_next: torch.Tensor,
    ) -> torch.Tensor:
        total = hidden.sum() * 0.0
        count = 0
        for horizon, head in zip(self.horizons, self.heads):
            if tokens.shape[1] <= horizon:
                continue
            logits = head(hidden[:, :-horizon:self.stride])
            targets = tokens[:, horizon::self.stride]
            valid = valid_next[:, horizon - 1::self.stride]
            if not bool(valid.any()):
                continue
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            total = total + (losses * valid.to(losses.dtype)).sum() / valid.sum().clamp_min(1)
            count += 1
        return total / max(1, count)


class ProgramController(nn.Module):
    """Typed instruction proposal head for MCPM."""

    instruction_count = 8

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.instruction_head = nn.Linear(dim, self.instruction_count, bias=False)
        self.instruction_embed = nn.Embedding(self.instruction_count, dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.training and _gate_is_zero(self.gate):
            return hidden
        logits = self.instruction_head(self.norm(hidden))
        probs = logits.softmax(dim=-1)
        instruction_state = probs @ self.instruction_embed.weight
        return hidden + self.gate * instruction_state

    def loss(self, hidden: torch.Tensor, tokens: torch.Tensor, valid_next: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] < 2:
            return hidden.sum() * 0.0
        current = tokens[:, :-1]
        nxt = tokens[:, 1:]
        delta = (nxt - current).abs()
        targets = torch.zeros_like(nxt)
        targets = torch.where(nxt < 256, torch.ones_like(targets), targets)
        targets = torch.where(nxt == current, torch.full_like(targets, 2), targets)
        targets = torch.where(delta <= 2, torch.full_like(targets, 3), targets)
        targets = torch.where((nxt % 5) == 0, torch.full_like(targets, 4), targets)
        targets = torch.where((nxt % 7) == 0, torch.full_like(targets, 5), targets)
        targets = torch.where((nxt % 11) == 0, torch.full_like(targets, 6), targets)
        targets = torch.where((nxt % 13) == 0, torch.full_like(targets, 7), targets)
        logits = self.instruction_head(self.norm(hidden[:, :-1]))
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).reshape_as(targets)
        return (losses * valid_next.to(losses.dtype)).sum() / valid_next.sum().clamp_min(1)


class ProgramExecutionRound(nn.Module):
    """Copy-on-write branch-DAG proxy: typed state update plus counterexample signal."""

    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.state_update = CausalConvBlock(dim, kernel_size)
        self.counterexample = CausalResidualAdapter(dim, max(2, kernel_size // 2))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.training and _gate_is_zero(self.gate):
            return hidden
        state = self.state_update(hidden)
        update = state - hidden
        if self.training or not _gate_is_zero(self.counterexample.gate):
            update = update + self.counterexample.gate * self.counterexample.residual(state)
        return hidden + self.gate * update


class ExecutionWorkbench(nn.Module):
    """Ablation-visible MCPM execution plane with repeated rounds."""

    def __init__(self, dim: int, rounds: int, kernel_size: int) -> None:
        super().__init__()
        self.rounds = nn.ModuleList(
            [ProgramExecutionRound(dim, kernel_size) for _ in range(int(rounds))]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for execution_round in self.rounds:
            hidden = execution_round(hidden)
        return hidden


class ContractVerifier(nn.Module):
    """Verifier-cascade proxy with type/check/repair heads."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.type_head = nn.Linear(dim, 4, bias=False)
        self.check_head = nn.Linear(dim, 2, bias=False)
        self.repair_head = nn.Linear(dim, 2, bias=False)
        self.proj = nn.Linear(8, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.training and _gate_is_zero(self.gate):
            return hidden
        normalized = self.norm(hidden)
        features = torch.cat(
            [
                self.type_head(normalized).softmax(dim=-1),
                self.check_head(normalized).softmax(dim=-1),
                self.repair_head(normalized).softmax(dim=-1),
            ],
            dim=-1,
        )
        return hidden + self.gate * self.proj(features)

    def loss(self, hidden: torch.Tensor, tokens: torch.Tensor, valid_next: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] < 2:
            return hidden.sum() * 0.0
        current = tokens[:, :-1]
        nxt = tokens[:, 1:]
        normalized = self.norm(hidden[:, :-1])
        type_target = (nxt % 4).long()
        check_target = (nxt >= current).long()
        repair_target = (nxt == current).long()
        loss = hidden.sum() * 0.0
        for logits, target in (
            (self.type_head(normalized), type_target),
            (self.check_head(normalized), check_target),
            (self.repair_head(normalized), repair_target),
        ):
            terms = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
            ).reshape_as(target)
            loss = loss + (terms * valid_next.to(terms.dtype)).sum() / valid_next.sum().clamp_min(1)
        return loss / 3.0


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


def sample_from_logits(logits: torch.Tensor, cfg: Any | None = None) -> torch.Tensor:
    """Sample one token per row, honoring the repository SamplingConfig shape."""

    if cfg is None or getattr(cfg, "deterministic", False):
        return logits.argmax(dim=-1, keepdim=True)
    temperature = max(float(getattr(cfg, "temperature", 1.0)), 1e-5)
    logits = logits / temperature
    top_k = int(getattr(cfg, "top_k", 0))
    if top_k > 0:
        threshold = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    top_p = float(getattr(cfg, "top_p", 1.0))
    if top_p < 1.0:
        sorted_logits, indices = logits.sort(dim=-1, descending=True)
        remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > top_p
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, indices, sorted_logits)
    probs = logits.softmax(dim=-1)
    if not torch.isfinite(probs).all():
        raise FloatingPointError("non-finite probabilities during sampling")
    return torch.multinomial(probs, 1)


def lm_cross_entropy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked next-token cross entropy for all native models."""

    targets = tokens[:, 1:]
    losses = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    valid = torch.ones_like(targets, dtype=torch.bool)
    if loss_mask is not None:
        valid = valid & loss_mask[:, 1:].bool()
    if attention_mask is not None:
        valid = valid & attention_mask[:, 1:].bool()
    valid_float = valid.to(losses.dtype)
    return (losses * valid_float).sum() / valid_float.sum().clamp_min(1)


def init_embedding(module: nn.Embedding) -> None:
    nn.init.normal_(module.weight, mean=0.0, std=0.02)


def transformer_anchor(config: MRWTConfig) -> CachedTransformerLM:
    anchor = CachedTransformerLM(config.anchor_config())
    # Match the non-anchor heads' initialization scale when residual modules are
    # enabled later. The anchor itself already initializes its tied embedding.
    return anchor


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def positional_ids(length: int, max_seq_len: int, device: torch.device) -> torch.Tensor:
    # Modulo fallback lets generation continue past the training window without
    # an index error. It is approximate and should be replaced by RoPE/ALiBi for
    # serious large-context MECM/MCPM runs.
    return torch.arange(length, device=device) % max_seq_len
