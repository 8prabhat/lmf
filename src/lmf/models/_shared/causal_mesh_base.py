"""Shared causal-mesh scaffolding used by the mecm, mcpm, and mgcf families.

``NativeCausalLM`` and its component zoo are a single flexible implementation
controlled by config flags (mesh residual, execution residual, full
architecture, gear-aware output, ...). MECM and MCPM are thin named subclasses
that only differ in which flags their builder sets; MGCF reuses the same
gear-hierarchy and training/generation scaffolding through inheritance even
though it replaces ``__init__`` and ``_forward_hidden`` with its own
non-attention causal-field trunk. Splitting this into per-family copies would
duplicate behavior and risk divergence, so it stays here as shared
infrastructure rather than under any one family's folder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformer.model import RMSNorm


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


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def positional_ids(length: int, max_seq_len: int, device: torch.device) -> torch.Tensor:
    # Modulo fallback lets generation continue past the training window without
    # an index error. It is approximate and should be replaced by RoPE/ALiBi for
    # serious large-context MECM/MCPM runs.
    return torch.arange(length, device=device) % max_seq_len


class NativeCausalLM(nn.Module):
    """Non-Transformer causal baseline used for MECM, MCPM, and (via inheritance) MGCF."""

    family_name = "native_causal"

    def __init__(self, config: NativeLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.dim)
        self.position = nn.Embedding(config.max_seq_len, config.dim)
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(config.dim, config.kernel_size, config.dropout)
                for _ in range(config.layers)
            ]
        )
        self.mesh = ZeroGatedCausalSummary(config.dim) if config.mesh_residual else None
        if config.full_architecture:
            self.span_atlas = MultiScaleSpanAtlas(config.dim, tuple(config.atlas_kernel_sizes))
            self.active_cover = ActiveCoverRouter(config.dim, tuple(config.atlas_kernel_sizes))
            self.reasoning_mesh = SparseReasoningMesh(
                config.dim, config.mesh_layers, max(config.kernel_size, max(config.atlas_kernel_sizes))
            )
            self.draft_tree = HierarchicalDraftHead(
                config.dim,
                config.vocab_size,
                tuple(config.draft_horizons),
                stride=config.draft_aux_stride,
            )
        else:
            self.span_atlas = None
            self.active_cover = None
            self.reasoning_mesh = None
            self.draft_tree = None
        self.execution = (
            ExecutionTraceAdapter(config.dim, config.vocab_size)
            if config.execution_residual
            else None
        )
        if config.full_architecture and config.execution_residual:
            self.program_controller = ProgramController(config.dim)
            self.execution_workbench = ExecutionWorkbench(
                config.dim, max(1, config.mesh_layers), config.kernel_size
            )
            self.contract_verifier = ContractVerifier(config.dim)
        else:
            self.program_controller = None
            self.execution_workbench = None
            self.contract_verifier = None
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.gear_head = (
            nn.Linear(config.dim, config.hierarchy_gears, bias=False)
            if config.gear_aware_output or config.gear_aux_weight > 0.0
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
        init_embedding(self.token)
        if self.gear_head is not None:
            nn.init.normal_(self.gear_head.weight, mean=0.0, std=0.02)

    def _forward_hidden(self, ids: torch.Tensor) -> torch.Tensor:
        pos = positional_ids(ids.shape[1], self.config.max_seq_len, ids.device)
        hidden = self.token(ids) + self.position(pos)[None]
        for block in self.blocks:
            hidden = block(hidden)
        if self.mesh is not None:
            hidden = self.mesh(hidden)
        if self.span_atlas is not None:
            hidden = self.span_atlas(hidden)
        if self.active_cover is not None:
            hidden = self.active_cover(hidden)
        if self.reasoning_mesh is not None:
            hidden = self.reasoning_mesh(hidden)
        if self.execution is not None:
            hidden = self.execution(hidden, ids)
        if self.program_controller is not None:
            hidden = self.program_controller(hidden)
        if self.execution_workbench is not None:
            hidden = self.execution_workbench(hidden)
        if self.contract_verifier is not None:
            hidden = self.contract_verifier(hidden)
        return self.norm(hidden)

    def _logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.config.gear_aware_output:
            if self.config.gear_output_mode == "bias":
                return self._gear_biased_scores(hidden)
            return self._hierarchical_scores(hidden)
        return self.head(hidden)

    def configure_token_hierarchy(
        self,
        gear_count: int,
        token_gears: list[int],
        token_children: list[list[int]] | None = None,
        token_bytes: list[list[int]] | None = None,
    ) -> None:
        """Install MultiGear token->gear metadata for gear-aware output."""
        if self.gear_head is None:
            return
        if gear_count > self.config.hierarchy_gears:
            raise ValueError(
                f"tokenizer needs {gear_count} gears, model supports "
                f"{self.config.hierarchy_gears}"
            )
        if len(token_gears) != self.config.vocab_size:
            raise ValueError("token_gears length must equal model vocabulary size")
        if hasattr(self, "_token_children"):
            if token_children is None:
                raise ValueError("token_children are required by this model")
            if len(token_children) != self.config.vocab_size:
                raise ValueError("token_children length must equal model vocabulary size")
        if hasattr(self, "_token_byte_lengths"):
            if token_bytes is None:
                raise ValueError("token_bytes are required by this model")
            if len(token_bytes) != self.config.vocab_size:
                raise ValueError("token_bytes length must equal model vocabulary size")
        gears = torch.tensor(token_gears, dtype=torch.long, device=self._token_gears.device)
        if bool(((gears < 0) | (gears >= gear_count)).any()):
            raise ValueError("token gear outside declared gear_count")
        if hasattr(self, "_token_children"):
            children = torch.tensor(
                token_children, dtype=torch.long, device=self._token_children.device
            )
            if bool(((children < -1) | (children >= self.config.vocab_size)).any()):
                raise ValueError("token child outside vocabulary")
            self._token_children.copy_(children)
        if hasattr(self, "_token_byte_lengths"):
            lengths = torch.tensor(
                [
                    min(len(values), getattr(self.config, "max_token_bytes", 64))
                    for values in token_bytes
                ],
                dtype=torch.long,
                device=self._token_byte_lengths.device,
            )
            self._token_byte_lengths.copy_(lengths)
        local = torch.full_like(self._token_to_local, -1)
        active = torch.zeros_like(self._gear_active)
        for gear in range(gear_count):
            token_ids = torch.nonzero(gears == gear, as_tuple=False).flatten()
            active[gear] = bool(len(token_ids))
            local[token_ids] = torch.arange(len(token_ids), device=local.device)
        self._token_gears.copy_(gears)
        self._token_to_local.copy_(local)
        self._gear_active.copy_(active)

    def _require_token_hierarchy(self) -> None:
        if self.gear_head is None:
            raise RuntimeError("gear-aware output is not enabled")
        if not bool(self._gear_active.any()):
            raise RuntimeError(
                "token hierarchy is required; build with a MultiGear tokenizer "
                "or call configure_token_hierarchy()"
            )

    def _gear_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        self._require_token_hierarchy()
        logits = self.gear_head(hidden)
        return logits.masked_fill(~self._gear_active, float("-inf"))

    def _hierarchical_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return full-vocabulary log scores from gear + within-gear factors."""
        self._require_token_hierarchy()
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

    def _gear_biased_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Cheaper gear-aware scores: token logits plus token-gear bias."""
        self._require_token_hierarchy()
        return self.head(hidden) + self._gear_logits(hidden).index_select(-1, self._token_gears)

    @staticmethod
    def _valid_next(tokens: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
        valid = torch.ones_like(tokens[:, 1:], dtype=torch.bool)
        if meta.get("loss_mask") is not None:
            valid = valid & meta["loss_mask"][:, 1:].bool()
        if meta.get("attention_mask") is not None:
            valid = valid & meta["attention_mask"][:, 1:].bool()
        return valid

    def _hierarchical_language_modeling_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _gear_auxiliary_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        self._require_token_hierarchy()
        target_gears = self._token_gears[targets]
        gear_logits = self._gear_logits(hidden)
        gear_losses = F.cross_entropy(
            gear_logits.reshape(-1, gear_logits.shape[-1]),
            target_gears.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        valid_float = valid.to(gear_losses.dtype)
        return (gear_losses * valid_float).sum() / valid_float.sum().clamp_min(1)

    def forward(self, ids: torch.Tensor, attention_mask=None, **_: Any):
        # attention_mask is accepted for trainer/evaluator compatibility. These
        # non-Transformer baselines do not attend to pad tokens; loss masking
        # still prevents pad positions from contributing to training/eval loss.
        hidden = self._forward_hidden(ids)
        return self._logits_from_hidden(hidden), None

    def training_step(
        self,
        tokens: torch.Tensor,
        task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        meta = task_metadata or {}
        hidden = self._forward_hidden(tokens)
        valid_next = self._valid_next(tokens, meta)
        prediction_hidden = hidden[:, :-1]
        targets = tokens[:, 1:]
        if (
            self.config.gear_aware_output
            and self.config.gear_output_mode == "factorized"
        ):
            language_modeling, gear_prediction, within_gear = (
                self._hierarchical_language_modeling_loss(
                    prediction_hidden, targets, valid_next
                )
            )
        else:
            logits = self._logits_from_hidden(hidden)
            language_modeling = lm_cross_entropy(
                logits,
                tokens,
                loss_mask=meta.get("loss_mask"),
                attention_mask=meta.get("attention_mask"),
            )
            gear_prediction = None
            within_gear = None
        scale = (loss_term_scales or {}).get("language_modeling", 1.0)
        total = scale * language_modeling
        result = {"language_modeling": language_modeling}
        if gear_prediction is not None:
            result["gear_prediction"] = gear_prediction
            result["within_gear"] = within_gear
        if self.config.gear_aux_weight > 0.0:
            gear_aux = self._gear_auxiliary_loss(prediction_hidden, targets, valid_next)
            total = total + (
                self.config.gear_aux_weight
                * (loss_term_scales or {}).get("gear_aux", 1.0)
                * gear_aux
            )
            result["gear_aux"] = gear_aux
        composition_aux_weight = getattr(self.config, "composition_aux_weight", 0.0)
        if composition_aux_weight > 0.0 and hasattr(self, "_composition_auxiliary_loss"):
            composition_aux = self._composition_auxiliary_loss(
                prediction_hidden, targets, valid_next
            )
            total = total + (
                composition_aux_weight
                * (loss_term_scales or {}).get("composition_aux", 1.0)
                * composition_aux
            )
            result["composition_aux"] = composition_aux
        if self.active_cover is not None and self.config.route_aux_weight > 0.0:
            route_loss = self.active_cover.last_balance_loss
            if route_loss is not None:
                total = total + (
                    self.config.route_aux_weight
                    * (loss_term_scales or {}).get("route_balance", 1.0)
                    * route_loss
                )
                result["route_balance"] = route_loss
        if self.draft_tree is not None and self.config.draft_aux_weight > 0.0:
            draft_loss = self.draft_tree.loss(hidden, tokens, valid_next)
            total = total + (
                self.config.draft_aux_weight
                * (loss_term_scales or {}).get("draft_tree", 1.0)
                * draft_loss
            )
            result["draft_tree"] = draft_loss
        if self.program_controller is not None and self.config.program_aux_weight > 0.0:
            program_loss = self.program_controller.loss(hidden, tokens, valid_next)
            total = total + (
                self.config.program_aux_weight
                * (loss_term_scales or {}).get("program_controller", 1.0)
                * program_loss
            )
            result["program_controller"] = program_loss
        if self.contract_verifier is not None and self.config.verifier_aux_weight > 0.0:
            verifier_loss = self.contract_verifier.loss(hidden, tokens, valid_next)
            total = total + (
                self.config.verifier_aux_weight
                * (loss_term_scales or {}).get("contract_verifier", 1.0)
                * verifier_loss
            )
            result["contract_verifier"] = verifier_loss
        result["total"] = total
        return result

    def _sample_hierarchical_token(self, hidden: torch.Tensor, sampling_config=None) -> torch.Tensor:
        """Sample gear first, then sample only from that gear's token subset."""
        gears = sample_from_logits(self._gear_logits(hidden), sampling_config).squeeze(-1)
        output = torch.empty((hidden.shape[0], 1), dtype=torch.long, device=hidden.device)
        for gear in torch.unique(gears).tolist():
            rows = torch.nonzero(gears == gear, as_tuple=False).flatten()
            token_ids = torch.nonzero(self._token_gears == gear, as_tuple=False).flatten()
            local_logits = F.linear(hidden[rows], self.token.weight[token_ids])
            local_choice = sample_from_logits(local_logits, sampling_config).squeeze(-1)
            output[rows, 0] = token_ids[local_choice]
        return output

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, sampling_config=None):
        if prompt_tokens.ndim != 2:
            raise ValueError("prompt_tokens must be a rank-2 tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        sequence = prompt_tokens
        out = []
        for _ in range(max_new_tokens):
            if (
                self.config.gear_aware_output
                and self.config.gear_output_mode == "factorized"
            ):
                hidden = self._forward_hidden(sequence)
                token = self._sample_hierarchical_token(hidden[:, -1], sampling_config)
            else:
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
