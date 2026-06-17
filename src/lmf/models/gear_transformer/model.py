"""Multi-Rate Latent Gear Transformer.

This family keeps the existing modern GPT baseline intact and adds a residual
multi-rate latent gear module in selected layers. Attention remains responsible
for exact token retrieval. The gear module supplies parallel causal latent
filters with phase-conditioned slot routing, auxiliary future prediction, and a
calibratable gear-conflict signal.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS
from ..transformer.model import RMSNorm, SwiGLU, _rope


def _as_tuple(value: Any, cast) -> tuple:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(cast(v) for v in value)
    if isinstance(value, list):
        return tuple(cast(v) for v in value)
    return (cast(value),)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class GearTransformerConfig:
    vocab_size: int
    dim: int = 512
    layers: int = 8
    heads: int = 8
    max_seq_len: int = 4096
    dropout: float = 0.0
    use_attention: bool = True

    num_gears: int = 4
    gear_dim: int = 0
    gear_speeds: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
    gear_slots: tuple[int, ...] = (64, 64, 32, 32)
    gear_receptive_fields: tuple[int, ...] = (8, 32, 128, 256)
    gear_update_mode: str = "dilated"
    gear_layer_strategy: str = "upper_alternate"
    gear_layers: tuple[int, ...] = ()
    share_gear_modules: bool = True
    gear_write_summary: bool = True
    cross_gear_coupling: bool = True
    phase_harmonics: int = 2
    max_log_speed_offset: float = 0.15
    gear_residual_init: float = 0.02
    gear_write_gate_init: float = 0.35
    gear_read_gate_init: float = 0.35
    gear_coupling_init: float = 0.05
    agreement_dim: int = 64

    future_horizons: tuple[int, ...] = (4, 16)
    future_dim: int = 0
    future_loss_weight: float = 0.05
    future_contrastive_weight: float = 0.0
    future_contrastive_samples: int = 256
    future_contrastive_temperature: float = 0.07
    diversity_loss_weight: float = 0.001
    alignment_loss_weight: float = 0.01
    consistency_loss_weight: float = 0.01

    reasoning_trigger_threshold: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "gear_speeds", _as_tuple(self.gear_speeds, float))
        object.__setattr__(self, "gear_slots", _as_tuple(self.gear_slots, int))
        object.__setattr__(
            self,
            "gear_receptive_fields",
            _as_tuple(self.gear_receptive_fields, int),
        )
        object.__setattr__(self, "gear_layers", _as_tuple(self.gear_layers, int))
        object.__setattr__(self, "future_horizons", _as_tuple(self.future_horizons, int))
        if self.gear_dim == 0:
            object.__setattr__(self, "gear_dim", min(self.dim, max(16, (3 * self.dim) // 4)))
        if self.future_dim == 0:
            object.__setattr__(self, "future_dim", min(self.dim, max(16, (3 * self.dim) // 4)))
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.dim < 8:
            raise ValueError("dim must be at least 8")
        if self.gear_dim < 1:
            raise ValueError("gear_dim must be positive")
        if self.future_dim < 1:
            raise ValueError("future_dim must be positive")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.layers < 1:
            raise ValueError("layers must be positive")
        if self.num_gears < 0:
            raise ValueError("num_gears must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.gear_update_mode not in {"chunked", "dilated", "scan"}:
            raise ValueError("gear_update_mode must be 'chunked', 'dilated', or 'scan'")
        if self.gear_layer_strategy not in {
            "none",
            "explicit",
            "all",
            "alternate",
            "upper_half",
            "upper_alternate",
        }:
            raise ValueError("unknown gear_layer_strategy")
        if self.num_gears:
            for name, value in (
                ("gear_speeds", self.gear_speeds),
                ("gear_slots", self.gear_slots),
                ("gear_receptive_fields", self.gear_receptive_fields),
            ):
                if len(value) != self.num_gears:
                    raise ValueError(f"{name} length must equal num_gears")
            if any(v <= 0.0 for v in self.gear_speeds):
                raise ValueError("gear speeds must be positive")
            if any(v < 1 for v in self.gear_slots):
                raise ValueError("gear slots must be positive")
            if any(v < 1 for v in self.gear_receptive_fields):
                raise ValueError("gear receptive fields must be positive")
        if self.phase_harmonics < 1:
            raise ValueError("phase_harmonics must be positive")
        if self.agreement_dim < 1:
            raise ValueError("agreement_dim must be positive")
        for name in (
            "future_loss_weight",
            "future_contrastive_weight",
            "diversity_loss_weight",
            "alignment_loss_weight",
            "consistency_loss_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if any(h < 1 for h in self.future_horizons):
            raise ValueError("future horizons must be positive")
        if self.future_contrastive_samples < 1:
            raise ValueError("future_contrastive_samples must be positive")
        if self.future_contrastive_temperature <= 0.0:
            raise ValueError("future_contrastive_temperature must be positive")

    def selected_gear_layers(self) -> tuple[int, ...]:
        if self.num_gears == 0 or self.gear_layer_strategy == "none":
            return ()
        if self.gear_layers:
            layers = self.gear_layers
        elif self.gear_layer_strategy == "explicit":
            layers = ()
        elif self.gear_layer_strategy == "all":
            layers = tuple(range(self.layers))
        elif self.gear_layer_strategy == "alternate":
            layers = tuple(range(1 if self.layers > 1 else 0, self.layers, 2))
        elif self.gear_layer_strategy == "upper_half":
            layers = tuple(range(self.layers // 2, self.layers))
        else:
            start = self.layers // 2
            layers = tuple(range(start, self.layers, 2)) or (self.layers - 1,)
        for layer in layers:
            if layer < 0 or layer >= self.layers:
                raise ValueError(f"gear layer {layer} outside model depth")
        return tuple(sorted(set(layers)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalDepthwiseUpdate(nn.Module):
    """Learned causal depthwise filter with optional autoregressive cache."""

    def __init__(self, dim: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        self.dilation = max(1, int(dilation))
        self.history = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(
            dim,
            dim,
            self.kernel_size,
            dilation=self.dilation,
            groups=dim,
            bias=False,
        )
        nn.init.constant_(self.conv.weight, 1.0 / self.kernel_size)

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        y = x.transpose(1, 2)
        if cache is None:
            padded = F.pad(y, (self.history, 0))
        else:
            hist = cache.transpose(1, 2)
            if hist.shape[-1] < self.history:
                hist = F.pad(hist, (self.history - hist.shape[-1], 0))
            else:
                hist = hist[..., -self.history:]
            padded = torch.cat([hist, y], dim=-1)
        out = self.conv(padded).transpose(1, 2)
        next_cache = None
        if use_cache and self.history > 0:
            full = x if cache is None else torch.cat([cache, x], dim=1)
            next_cache = full[:, -self.history:].detach()
        return out, next_cache


def _causal_moving_average(
    x: torch.Tensor,
    window: int,
    cache: torch.Tensor | None = None,
    use_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    window = max(1, int(window))
    full = x if cache is None else torch.cat([cache, x], dim=1)
    n_total = full.shape[1]
    csum = torch.cat(
        [torch.zeros_like(full[:, :1]), torch.cumsum(full, dim=1)],
        dim=1,
    )
    end = torch.arange(1, n_total + 1, device=x.device)
    start = (end - window).clamp_min(0)
    sums = csum[:, end] - csum[:, start]
    counts = (end - start).to(dtype=x.dtype).view(1, n_total, 1)
    out = sums / counts.clamp_min(1.0)
    out = out[:, -x.shape[1]:]
    next_cache = None
    if use_cache and window > 1:
        next_cache = full[:, -(window - 1):].detach()
    return out, next_cache


class GearSlotRouter(nn.Module):
    def __init__(self, dim: int, slots: int, phase_dim: int) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.empty(slots, dim))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.phase_proj = nn.Linear(phase_dim, dim, bias=False)
        nn.init.normal_(self.slots, mean=0.0, std=0.02)

    def forward(self, h: torch.Tensor, phase_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(h) + self.phase_proj(phase_features)[None]
        scores = q @ self.slots.T / math.sqrt(h.shape[-1])
        alpha = scores.softmax(dim=-1)
        return alpha @ self.slots, alpha


class GearCell(nn.Module):
    def __init__(
        self,
        dim: int,
        slots: int,
        base_speed: float,
        receptive_field: int,
        update_mode: str,
        phase_harmonics: int,
        max_log_speed_offset: float,
        dropout: float,
        write_summary: bool,
        write_gate_init: float,
        read_gate_init: float,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.base_speed_value = float(base_speed)
        self.receptive_field = int(receptive_field)
        self.update_mode = update_mode
        self.write_summary = bool(write_summary)
        self.phase_harmonics = int(phase_harmonics)
        self.phase_dim = 2 * self.phase_harmonics
        self.max_log_speed_offset = float(max_log_speed_offset)
        self.register_buffer("base_speed", torch.tensor(float(base_speed)))
        self.log_speed_offset = nn.Parameter(torch.zeros(()))
        self.phase_offset = nn.Parameter(torch.zeros(()))
        self.mask_gain = nn.Parameter(torch.tensor(2.0))
        self.mask_bias = nn.Parameter(torch.tensor(-0.5))
        self.router = GearSlotRouter(dim, slots, self.phase_dim)
        hidden = max(16, 2 * dim)
        proposal_in = (3 * dim if self.write_summary else 2 * dim) + self.phase_dim
        self.proposal = nn.Sequential(
            nn.Linear(proposal_in, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, dim, bias=False),
        )
        self.write_gate = nn.Linear(proposal_in, dim, bias=True)
        if update_mode == "dilated":
            dilation = max(1, round(self.base_speed_value))
            kernel = max(1, math.ceil(self.receptive_field / dilation))
            self.update = CausalDepthwiseUpdate(dim, kernel, dilation)
        else:
            self.update = None
        initial_decay = 1.0 - 1.0 / max(self.base_speed_value, 1.0)
        self.scan_decay_logit = nn.Parameter(torch.tensor(_logit(initial_decay)))
        self.state_norm = RMSNorm(dim)
        read_in = (4 * dim if self.write_summary else 3 * dim) + self.phase_dim
        self.read_gate = nn.Linear(read_in, dim, bias=True)
        self.out_proj = nn.Linear(read_in, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.write_gate.bias, _logit(write_gate_init))
        nn.init.constant_(self.read_gate.bias, _logit(read_gate_init))

    def speed(self) -> torch.Tensor:
        bounded = self.max_log_speed_offset * torch.tanh(self.log_speed_offset)
        return self.base_speed.to(self.log_speed_offset.device) * torch.exp(bounded)

    def phase(self, positions: torch.Tensor) -> torch.Tensor:
        speed = self.speed().clamp_min(1e-4)
        return (2.0 * math.pi * positions.float() / speed) + self.phase_offset

    def phase_features(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        phase = self.phase(positions)
        parts = []
        for harmonic in range(1, self.phase_harmonics + 1):
            p = phase * harmonic
            parts.extend([p.sin(), p.cos()])
        return phase, torch.stack(parts, dim=-1).to(dtype=dtype)

    def _scan_update(
        self,
        proposal: torch.Tensor,
        update_mask: torch.Tensor,
        cache: torch.Tensor | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        decay = torch.sigmoid(self.scan_decay_logit).to(dtype=proposal.dtype)
        if cache is None:
            prev = torch.zeros(
                proposal.shape[0],
                proposal.shape[-1],
                dtype=proposal.dtype,
                device=proposal.device,
            )
        else:
            prev = cache.to(dtype=proposal.dtype, device=proposal.device)
        states = []
        for t in range(proposal.shape[1]):
            alpha = (1.0 - decay) * update_mask[:, t, :]
            prev = (1.0 - alpha) * prev + alpha * proposal[:, t, :]
            states.append(prev)
        state = torch.stack(states, dim=1)
        return state, (prev.detach() if use_cache else None)

    def _updated_state(
        self,
        proposal: torch.Tensor,
        update_mask: torch.Tensor,
        cache: torch.Tensor | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.update_mode == "scan":
            return self._scan_update(proposal, update_mask, cache, use_cache)
        if self.update_mode == "chunked":
            window = max(1, round(self.base_speed_value))
            state, next_cache = _causal_moving_average(proposal, window, cache, use_cache)
        else:
            assert self.update is not None
            state, next_cache = self.update(proposal, cache, use_cache)
        state = proposal + update_mask * (state - proposal)
        return state, next_cache

    def forward(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        cache: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        phase, phase_features = self.phase_features(positions, h.dtype)
        routed, alpha = self.router(h, phase_features)
        phase_b = phase_features[None].expand(h.shape[0], -1, -1)
        if isinstance(cache, dict):
            update_cache = cache.get("update")
            summary_cache = cache.get("summary")
        else:
            update_cache = cache
            summary_cache = None
        if self.write_summary:
            summary, next_summary_cache = _causal_moving_average(
                h,
                self.receptive_field,
                summary_cache,
                use_cache,
            )
            write_context = torch.cat([h, routed, summary, phase_b], dim=-1)
        else:
            summary = None
            next_summary_cache = None
            write_context = torch.cat([h, routed, phase_b], dim=-1)
        proposal = self.proposal(write_context)
        periodic_mask = torch.sigmoid(
            self.mask_gain.to(dtype=h.dtype) * phase.cos().to(dtype=h.dtype)
            + self.mask_bias.to(dtype=h.dtype)
        )[None, :, None]
        write_gate = torch.sigmoid(self.write_gate(write_context)) * periodic_mask
        state, next_update_cache = self._updated_state(
            proposal,
            write_gate,
            update_cache,
            use_cache,
        )
        state = self.state_norm(state)
        if self.write_summary:
            assert summary is not None
            read_context = torch.cat([h, routed, state, summary, phase_b], dim=-1)
        else:
            read_context = torch.cat([h, routed, state, phase_b], dim=-1)
        read_gate = torch.sigmoid(self.read_gate(read_context))
        out = read_gate * self.out_proj(read_context)
        next_cache = (
            {"update": next_update_cache, "summary": next_summary_cache}
            if use_cache
            else None
        )
        route_usage = alpha.mean(dim=(0, 1))
        route_balance = ((route_usage - (1.0 / alpha.shape[-1])) ** 2).sum() * alpha.shape[-1]
        route_entropy = (
            -(alpha * alpha.clamp_min(1e-9).log()).sum(dim=-1).mean()
            / math.log(alpha.shape[-1])
        )
        return self.dropout(out), next_cache, {
            "route_balance": route_balance,
            "route_entropy": route_entropy,
            "update_activity": periodic_mask.mean(),
            "write_activity": write_gate.mean(),
            "read_activity": read_gate.mean(),
        }


class GearFusion(nn.Module):
    def __init__(self, dim: int, num_gears: int, residual_init: float, dropout: float) -> None:
        super().__init__()
        self.router = nn.Linear(dim, num_gears, bias=False)
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.residual_gate_logit = nn.Parameter(torch.tensor(_logit(residual_init)))

    def forward(self, h: torch.Tensor, gear_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.router(h).softmax(dim=-1)
        fused = (gear_outputs * weights.unsqueeze(-1)).sum(dim=-2)
        gate = torch.sigmoid(self.residual_gate_logit).to(dtype=h.dtype)
        return gate * self.dropout(self.out(self.norm(fused))), weights


class GearAlignmentScorer(nn.Module):
    def __init__(self, dim: int, agreement_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, agreement_dim, bias=False)
        self.risk = nn.Sequential(
            nn.Linear(dim + agreement_dim + 1, max(16, dim // 2), bias=False),
            nn.SiLU(),
            nn.Linear(max(16, dim // 2), 1, bias=False),
        )

    def forward(self, h: torch.Tensor, gear_outputs: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = F.normalize(self.proj(gear_outputs), dim=-1)
        mean_projected = F.normalize(projected.mean(dim=-2), dim=-1)
        conflict = ((projected - mean_projected.unsqueeze(-2)) ** 2).sum(dim=-1).mean(dim=-1)
        risk_logit = self.risk(torch.cat([h, mean_projected, conflict.unsqueeze(-1)], dim=-1)).squeeze(-1)
        return {"conflict": conflict, "risk_logit": risk_logit}


class CrossGearCoupler(nn.Module):
    """Learned same-position message passing across gear streams."""

    def __init__(self, dim: int, num_gears: int, coupling_init: float, dropout: float) -> None:
        super().__init__()
        self.num_gears = int(num_gears)
        self.norm = RMSNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.adjacency = nn.Parameter(torch.zeros(num_gears, num_gears))
        self.gate_logit = nn.Parameter(torch.tensor(_logit(coupling_init)))

    def forward(self, gear_outputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.norm(gear_outputs)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        logits = torch.einsum("btgd,bthd->btgh", q, k) / math.sqrt(x.shape[-1])
        logits = logits + self.adjacency[None, None]
        weights = logits.softmax(dim=-1)
        message = torch.einsum("btgh,bthd->btgd", weights, v)
        gate = torch.sigmoid(self.gate_logit).to(dtype=x.dtype)
        coupled = gear_outputs + gate * self.dropout(self.out(message))
        entropy = (
            -(weights * weights.clamp_min(1e-9).log()).sum(dim=-1).mean()
            / math.log(self.num_gears)
        )
        offdiag = ~torch.eye(self.num_gears, dtype=torch.bool, device=weights.device)
        offdiag_mass = weights[..., offdiag].mean()
        return coupled, {
            "coupling_entropy": entropy,
            "coupling_gate": gate,
            "coupling_offdiag": offdiag_mass,
        }


class MultiRateGearModule(nn.Module):
    def __init__(self, config: GearTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.down = (
            nn.Identity()
            if config.gear_dim == config.dim
            else nn.Linear(config.dim, config.gear_dim, bias=False)
        )
        self.up = (
            nn.Identity()
            if config.gear_dim == config.dim
            else nn.Linear(config.gear_dim, config.dim, bias=False)
        )
        self.gears = nn.ModuleList(
            [
                GearCell(
                    config.gear_dim,
                    config.gear_slots[i],
                    config.gear_speeds[i],
                    config.gear_receptive_fields[i],
                    config.gear_update_mode,
                    config.phase_harmonics,
                    config.max_log_speed_offset,
                    config.dropout,
                    config.gear_write_summary,
                    config.gear_write_gate_init,
                    config.gear_read_gate_init,
                )
                for i in range(config.num_gears)
            ]
        )
        self.coupler = (
            CrossGearCoupler(
                config.gear_dim,
                config.num_gears,
                config.gear_coupling_init,
                config.dropout,
            )
            if config.cross_gear_coupling and config.num_gears > 1
            else None
        )
        self.fusion = GearFusion(
            config.gear_dim,
            config.num_gears,
            config.gear_residual_init,
            config.dropout,
        )
        self.alignment = GearAlignmentScorer(config.gear_dim, config.agreement_dim)

    def speed_separation_loss(self) -> torch.Tensor:
        if len(self.gears) < 2:
            return next(self.parameters()).sum() * 0.0
        log_speeds = torch.stack([gear.speed().log() for gear in self.gears])
        gaps = (log_speeds[:, None] - log_speeds[None, :]).abs()
        mask = ~torch.eye(len(self.gears), dtype=torch.bool, device=gaps.device)
        return torch.exp(-gaps[mask]).mean()

    def slot_diversity_loss(self) -> torch.Tensor:
        total = next(self.parameters()).sum() * 0.0
        count = 0
        for gear in self.gears:
            slots = F.normalize(gear.router.slots, dim=-1)
            sim = slots @ slots.T
            mask = ~torch.eye(slots.shape[0], dtype=torch.bool, device=slots.device)
            total = total + sim[mask].pow(2).mean()
            count += 1
        return total / max(count, 1)

    def forward(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        cache: list[torch.Tensor | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None] | None, dict[str, torch.Tensor]]:
        gear_input = self.down(h)
        gear_outputs = []
        next_cache = []
        balances = []
        entropies = []
        activities = []
        write_activities = []
        read_activities = []
        cache = cache or [None] * len(self.gears)
        for gear, gear_cache in zip(self.gears, cache):
            out, nc, stats = gear(gear_input, positions, gear_cache, use_cache)
            gear_outputs.append(out)
            next_cache.append(nc)
            balances.append(stats["route_balance"])
            entropies.append(stats["route_entropy"])
            activities.append(stats["update_activity"])
            write_activities.append(stats["write_activity"])
            read_activities.append(stats["read_activity"])
        stacked = torch.stack(gear_outputs, dim=-2)
        if self.coupler is not None:
            coupled, coupling_stats = self.coupler(stacked)
        else:
            coupled = stacked
            zero = stacked.sum() * 0.0
            coupling_stats = {
                "coupling_entropy": zero,
                "coupling_gate": zero,
                "coupling_offdiag": zero,
            }
        residual, fusion_weights = self.fusion(gear_input, coupled)
        residual = self.up(residual)
        alignment = self.alignment(gear_input, coupled)
        mean_outputs = F.normalize(coupled.mean(dim=(0, 1)), dim=-1)
        sim = mean_outputs @ mean_outputs.T
        offdiag = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        output_diversity = sim[offdiag].pow(2).mean() if bool(offdiag.any()) else sim.sum() * 0.0
        aux = {
            "gear_outputs": coupled,
            "fusion_weights": fusion_weights,
            "conflict": alignment["conflict"],
            "risk_logit": alignment["risk_logit"],
            "route_balance": torch.stack(balances).mean(),
            "route_entropy": torch.stack(entropies).mean(),
            "update_activity": torch.stack(activities).mean(),
            "write_activity": torch.stack(write_activities).mean(),
            "read_activity": torch.stack(read_activities).mean(),
            "coupling_entropy": coupling_stats["coupling_entropy"],
            "coupling_gate": coupling_stats["coupling_gate"],
            "coupling_offdiag": coupling_stats["coupling_offdiag"],
            "output_diversity": output_diversity,
            "speed_separation": self.speed_separation_loss(),
            "slot_diversity": self.slot_diversity_loss(),
        }
        return residual, (next_cache if use_cache else None), aux


class GearTransformerBlock(nn.Module):
    def __init__(self, config: GearTransformerConfig, use_gears: bool, own_gears: bool = True) -> None:
        super().__init__()
        if config.use_attention and config.dim % config.heads:
            raise ValueError("dim must be divisible by heads")
        self.use_attention = bool(config.use_attention)
        self.use_gears = bool(use_gears)
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.norm1 = RMSNorm(config.dim) if self.use_attention else None
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False) if self.use_attention else None
        self.proj = nn.Linear(config.dim, config.dim, bias=False) if self.use_attention else None
        self.gear_norm = RMSNorm(config.dim) if use_gears else None
        self.gears = MultiRateGearModule(config) if use_gears and own_gears else None
        self.norm2 = RMSNorm(config.dim)
        self.ff = SwiGLU(config.dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
        attn_mask: torch.Tensor | None = None,
        shared_gears: MultiRateGearModule | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any] | None, dict[str, torch.Tensor] | None]:
        b, n, d = x.shape
        attn_cache = None if cache is None else cache.get("attn")
        if attn_cache is not None:
            past = attn_cache[0].shape[2]
        elif cache is not None and "past_len" in cache:
            past = int(cache["past_len"])
        else:
            past = 0
        positions = torch.arange(past, past + n, device=x.device)
        k = v = None
        if self.use_attention:
            assert self.qkv is not None and self.norm1 is not None and self.proj is not None
            qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, self.head_dim)
            q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
            q, k = _rope(q, positions), _rope(k, positions)
            if attn_cache is not None:
                k = torch.cat([attn_cache[0], k], dim=2)
                v = torch.cat([attn_cache[1], v], dim=2)
            if attn_mask is not None:
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                attn = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=attn_cache is None and n > 1,
                )
            x = x + self.proj(attn.transpose(1, 2).reshape(b, n, d))
        gear_aux = None
        gear_cache = None if cache is None else cache.get("gears")
        next_gear_cache = None
        gear_module = self.gears if self.gears is not None else shared_gears
        if self.use_gears and gear_module is not None and self.gear_norm is not None:
            residual, next_gear_cache, gear_aux = gear_module(
                self.gear_norm(x),
                positions,
                gear_cache,
                use_cache,
            )
            x = x + residual
        x = x + self.ff(self.norm2(x))
        next_cache = None
        if use_cache:
            next_cache = {"past_len": past + n}
            if self.use_attention:
                assert k is not None and v is not None
                next_cache["attn"] = (k, v)
            if self.use_gears:
                next_cache["gears"] = next_gear_cache
        return x, next_cache, gear_aux


class FuturePredictionHead(nn.Module):
    def __init__(self, dim: int, horizons: tuple[int, ...], hidden_dim: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                str(h): nn.Sequential(
                    nn.Linear(dim, hidden_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, dim, bias=False),
                )
                for h in horizons
            }
        )

    def forward(self, hidden: torch.Tensor) -> dict[int, torch.Tensor]:
        return {int(h): head(hidden) for h, head in self.heads.items()}


class MHGTransformerLM(nn.Module):
    """Causal LM with residual multi-rate latent gear dynamics."""

    def __init__(self, config: GearTransformerConfig) -> None:
        super().__init__()
        self.config = config
        gear_layers = set(config.selected_gear_layers())
        self.gear_layer_indices = gear_layers
        self.shared_gears = (
            MultiRateGearModule(config)
            if config.share_gear_modules and gear_layers
            else None
        )
        self.token = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [
                GearTransformerBlock(
                    config,
                    layer in gear_layers,
                    own_gears=self.shared_gears is None,
                )
                for layer in range(config.layers)
            ]
        )
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.future = (
            FuturePredictionHead(config.dim, config.future_horizons, config.future_dim)
            if config.future_horizons
            else None
        )
        self.consistency = nn.Sequential(
            nn.Linear(2 * config.dim, config.dim, bias=False),
            nn.SiLU(),
            nn.Linear(config.dim, 1, bias=False),
        )
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)

    @staticmethod
    def _full_attn_mask(attention_mask, n: int, device) -> torch.Tensor | None:
        if attention_mask is None or bool(attention_mask.all()):
            return None
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
        key_valid = attention_mask.bool()[:, None, None, :]
        return causal[None, None] & key_valid

    def _forward_hidden(
        self,
        ids: torch.Tensor,
        caches: list[dict[str, Any]] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, Any]] | None, list[dict[str, torch.Tensor]]]:
        x = self.token(ids)
        attn_mask = (
            self._full_attn_mask(attention_mask, ids.shape[1], ids.device)
            if caches is None
            else None
        )
        next_caches = []
        aux_records = []
        for i, block in enumerate(self.blocks):
            x, nc, gear_aux = block(
                x,
                None if caches is None else caches[i],
                use_cache,
                attn_mask,
                self.shared_gears if i in self.gear_layer_indices else None,
            )
            if use_cache:
                next_caches.append(nc)
            if return_aux and gear_aux is not None:
                aux_records.append(gear_aux)
        return self.norm(x), (next_caches if use_cache else None), aux_records

    def forward(
        self,
        ids: torch.Tensor,
        caches: list[dict[str, Any]] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]] | None]:
        hidden, next_caches, _ = self._forward_hidden(ids, caches, use_cache, attention_mask)
        return self.head(hidden), next_caches

    @staticmethod
    def _valid_targets(tokens: torch.Tensor, loss_mask, attention_mask) -> torch.Tensor:
        valid = torch.ones_like(tokens[:, 1:], dtype=torch.bool)
        if loss_mask is not None:
            valid = valid & loss_mask[:, 1:].bool()
        if attention_mask is not None:
            valid = valid & attention_mask[:, 1:].bool()
        return valid

    @staticmethod
    def _valid_positions(tokens: torch.Tensor, attention_mask) -> torch.Tensor:
        if attention_mask is not None:
            return attention_mask.bool()
        return torch.ones_like(tokens, dtype=torch.bool)

    def _language_modeling_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.head(hidden)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        valid_float = valid.to(losses.dtype)
        return (losses * valid_float).sum() / valid_float.sum().clamp_min(1)

    def _future_losses(
        self,
        hidden: torch.Tensor,
        valid_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        zero = hidden.sum() * 0.0
        if self.future is None or not self.config.future_horizons:
            return zero, zero, None
        preds = self.future(hidden)
        latent_total = zero
        contrastive_total = zero
        terms = 0
        first_error = None
        for horizon in self.config.future_horizons:
            if hidden.shape[1] <= horizon:
                continue
            pred = F.normalize(preds[horizon][:, :-horizon], dim=-1)
            target = F.normalize(hidden[:, horizon:].detach(), dim=-1)
            mask = valid_positions[:, :-horizon] & valid_positions[:, horizon:]
            error = (1.0 - (pred * target).sum(dim=-1)).clamp_min(0.0)
            latent_total = latent_total + (
                error * mask.to(error.dtype)
            ).sum() / mask.to(error.dtype).sum().clamp_min(1)
            if first_error is None:
                first_error = error.detach()
            if self.config.future_contrastive_weight > 0.0 and bool(mask.any()):
                flat_pred = pred[mask]
                flat_target = target[mask]
                limit = min(self.config.future_contrastive_samples, flat_pred.shape[0])
                flat_pred = flat_pred[:limit]
                flat_target = flat_target[:limit]
                logits = flat_pred @ flat_target.T / self.config.future_contrastive_temperature
                labels = torch.arange(limit, device=hidden.device)
                contrastive_total = contrastive_total + F.cross_entropy(logits, labels)
            terms += 1
        if terms == 0:
            return zero, zero, None
        return latent_total / terms, contrastive_total / terms, first_error

    def _gear_diversity_loss(
        self,
        aux_records: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not aux_records:
            zero = next(self.parameters()).sum() * 0.0
            return zero, {
                "gear_route_entropy": zero.detach(),
                "gear_route_balance": zero.detach(),
                "gear_update_activity": zero.detach(),
                "gear_write_activity": zero.detach(),
                "gear_read_activity": zero.detach(),
                "gear_coupling_entropy": zero.detach(),
                "gear_coupling_gate": zero.detach(),
                "gear_coupling_offdiag": zero.detach(),
                "gear_conflict": zero.detach(),
            }
        output_diversity = torch.stack([a["output_diversity"] for a in aux_records]).mean()
        speed_separation = torch.stack([a["speed_separation"] for a in aux_records]).mean()
        slot_diversity = torch.stack([a["slot_diversity"] for a in aux_records]).mean()
        route_balance = torch.stack([a["route_balance"] for a in aux_records]).mean()
        diversity = output_diversity + 0.25 * speed_separation + 0.05 * route_balance + 0.01 * slot_diversity
        metrics = {
            "gear_route_entropy": torch.stack([a["route_entropy"] for a in aux_records]).mean().detach(),
            "gear_route_balance": route_balance.detach(),
            "gear_update_activity": torch.stack([a["update_activity"] for a in aux_records]).mean().detach(),
            "gear_write_activity": torch.stack([a["write_activity"] for a in aux_records]).mean().detach(),
            "gear_read_activity": torch.stack([a["read_activity"] for a in aux_records]).mean().detach(),
            "gear_coupling_entropy": torch.stack([a["coupling_entropy"] for a in aux_records]).mean().detach(),
            "gear_coupling_gate": torch.stack([a["coupling_gate"] for a in aux_records]).mean().detach(),
            "gear_coupling_offdiag": torch.stack([a["coupling_offdiag"] for a in aux_records]).mean().detach(),
            "gear_conflict": torch.stack([a["conflict"].mean() for a in aux_records]).mean().detach(),
        }
        return diversity, metrics

    def _alignment_loss(
        self,
        aux_records: list[dict[str, torch.Tensor]],
        future_error: torch.Tensor | None,
    ) -> torch.Tensor:
        if not aux_records or future_error is None:
            return next(self.parameters()).sum() * 0.0
        risks = torch.stack([a["risk_logit"][:, :future_error.shape[1]] for a in aux_records]).mean(dim=0)
        target = future_error.detach().clamp(0.0, 2.0) / 2.0
        return F.mse_loss(torch.sigmoid(risks), target)

    def _consistency_loss(
        self,
        hidden: torch.Tensor,
        valid_positions: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.future_horizons:
            return hidden.sum() * 0.0
        horizon = min(self.config.future_horizons)
        if hidden.shape[1] <= horizon:
            return hidden.sum() * 0.0
        prefix = hidden[:, :-horizon]
        future = hidden[:, horizon:].detach()
        mask = valid_positions[:, :-horizon] & valid_positions[:, horizon:]
        if not bool(mask.any()):
            return hidden.sum() * 0.0
        neg_future = torch.roll(future, shifts=1, dims=0)
        true_logits = self.consistency(torch.cat([prefix, future], dim=-1)).squeeze(-1)
        neg_logits = self.consistency(torch.cat([prefix, neg_future], dim=-1)).squeeze(-1)
        logits = torch.cat([true_logits[mask], neg_logits[mask]], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(true_logits[mask]),
                torch.zeros_like(neg_logits[mask]),
            ],
            dim=0,
        )
        return F.binary_cross_entropy_with_logits(logits, labels)

    def training_step(
        self,
        tokens: torch.Tensor,
        task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        meta = task_metadata or {}
        attention_mask = meta.get("attention_mask")
        loss_mask = meta.get("loss_mask")
        hidden, _, aux_records = self._forward_hidden(
            tokens,
            attention_mask=attention_mask,
            return_aux=True,
        )
        prediction_hidden = hidden[:, :-1]
        targets = tokens[:, 1:]
        valid = self._valid_targets(tokens, loss_mask, attention_mask)
        valid_positions = self._valid_positions(tokens, attention_mask)
        scales = loss_term_scales or {}

        language_modeling = self._language_modeling_loss(prediction_hidden, targets, valid)
        future_latent, future_contrastive, future_error = self._future_losses(
            hidden,
            valid_positions,
        )
        gear_diversity, gear_metrics = self._gear_diversity_loss(aux_records)
        alignment_calibration = self._alignment_loss(aux_records, future_error)
        consistency = self._consistency_loss(hidden, valid_positions)

        total = scales.get("language_modeling", 1.0) * language_modeling
        total = total + (
            self.config.future_loss_weight
            * scales.get("future_latent", 1.0)
            * future_latent
        )
        total = total + (
            self.config.future_contrastive_weight
            * scales.get("future_contrastive", 1.0)
            * future_contrastive
        )
        total = total + (
            self.config.diversity_loss_weight
            * scales.get("gear_diversity", 1.0)
            * gear_diversity
        )
        total = total + (
            self.config.alignment_loss_weight
            * scales.get("alignment_calibration", 1.0)
            * alignment_calibration
        )
        total = total + (
            self.config.consistency_loss_weight
            * scales.get("consistency", 1.0)
            * consistency
        )

        result = {
            "language_modeling": language_modeling,
            "future_latent": future_latent,
            "future_contrastive": future_contrastive,
            "gear_diversity": gear_diversity,
            "alignment_calibration": alignment_calibration,
            "consistency": consistency,
            "total": total,
        }
        result.update(gear_metrics)
        return result

    def _sample_token(self, logits: torch.Tensor, cfg) -> torch.Tensor:
        if cfg is None or cfg.deterministic:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(cfg.temperature, 1e-5)
        if cfg.top_k > 0:
            thresh = logits.topk(min(cfg.top_k, logits.shape[-1]), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        if cfg.top_p < 1.0:
            sorted_logits, idx = logits.sort(dim=-1, descending=True)
            remove = sorted_logits.softmax(-1).cumsum(-1) > cfg.top_p
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
        return torch.multinomial(logits.softmax(-1), 1)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int, sampling_config=None) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return torch.empty(prompt.shape[0], 0, dtype=torch.long, device=prompt.device)
        logits, caches = self(prompt, use_cache=True)
        token = self._sample_token(logits[:, -1], sampling_config)
        out = []
        for _ in range(max_new_tokens):
            out.append(token)
            logits, caches = self(token, caches=caches, use_cache=True)
            token = self._sample_token(logits[:, -1], sampling_config)
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def alignment_scores(self, ids: torch.Tensor) -> dict[str, float]:
        was_training = self.training
        self.eval()
        try:
            _, _, aux_records = self._forward_hidden(ids, return_aux=True)
            if not aux_records:
                return {"conflict": 0.0, "risk": 0.0, "trigger": 0.0}
            conflict = torch.stack([a["conflict"][:, -1].mean() for a in aux_records]).mean()
            risk = torch.stack([torch.sigmoid(a["risk_logit"][:, -1]).mean() for a in aux_records]).mean()
            trigger = float(conflict.item() > self.config.reasoning_trigger_threshold)
            return {"conflict": float(conflict), "risk": float(risk), "trigger": trigger}
        finally:
            self.train(was_training)

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "name": "MHGTransformerLM",
            "config": self.config.to_dict(),
            "parameters": {"total": sum(p.numel() for p in self.parameters())},
        }


@MODELS.register("gear_transformer")
def build_gear_transformer(
    model_cfg: dict,
    vocab_size: int | None = None,
) -> MHGTransformerLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return MHGTransformerLM(GearTransformerConfig(**cfg))


@MODELS.register("mlgt")
def build_multi_rate_latent_gear_transformer(
    model_cfg: dict,
    vocab_size: int | None = None,
) -> MHGTransformerLM:
    return build_gear_transformer(model_cfg, vocab_size)


@MODELS.register("gear_only")
def build_gear_only_transformer(
    model_cfg: dict,
    vocab_size: int | None = None,
) -> MHGTransformerLM:
    cfg = dict(model_cfg)
    cfg.setdefault("use_attention", False)
    cfg.setdefault("gear_layer_strategy", "all")
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return MHGTransformerLM(GearTransformerConfig(**cfg))
