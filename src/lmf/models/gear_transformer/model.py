"""Generative Multi-Scale Gear Transformer.

The gear path is a bank of causal controllers with explicitly ordered angular
speeds.  Each controller owns a phase, phase-addressed slots, and a persistent
memory.  Token/context drives and lower-to-higher phase locking update the
clocks; a coherence gate then combines the resulting fast-to-slow states.
Multi-horizon future predictions are used both as training targets and as a
small residual contribution to next-token logits.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS
from ...core.rope import apply_rope
from ..transformer.model import RMSNorm, SwiGLU
from .parallel import ParallelGearSystem


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


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return value + math.log(-math.expm1(-value))


@dataclass(frozen=True)
class GearTransformerConfig:
    vocab_size: int
    dim: int = 512
    layers: int = 8
    heads: int = 8
    max_seq_len: int = 4096
    dropout: float = 0.0
    use_attention: bool = True

    num_gears: int = 9
    gear_dim: int = 0
    # Angular phase advances in radians/token, ordered fast -> slow.
    gear_speeds: tuple[float, ...] = (
        1.4, 1.0, 0.7, 0.45, 0.3, 0.18, 0.11, 0.065, 0.035,
    )
    gear_slots: tuple[int, ...] = (32, 32, 24, 24, 20, 16, 16, 12, 12)
    gear_receptive_fields: tuple[int, ...] = (
        4, 8, 16, 32, 64, 128, 256, 512, 1024,
    )
    gear_system: str = "parallel_v4"
    gear_lane_sizes: tuple[int, ...] = ()
    gear_update_rates: tuple[float, ...] = ()
    gear_rotation_dims: int = 16
    gear_update_mode: str = "parallel"
    gear_layer_strategy: str = "upper_alternate"
    gear_layers: tuple[int, ...] = ()
    share_gear_modules: bool = False
    gear_write_summary: bool = True
    cross_gear_coupling: bool = True
    phase_harmonics: int = 2
    max_log_speed_offset: float = 0.15
    phase_drive_scale: float = 1.0
    phase_modulation_scale: float = 0.75
    phase_coupling_enabled: bool = True
    phase_coupling_init: float = 0.05
    phase_coupling_max: float = 0.25
    gear_residual_init: float = 0.12
    gear_write_gate_init: float = 0.55
    gear_read_gate_init: float = 0.50
    gear_coupling_init: float = 0.05
    gear_routing_floor: float = 0.08
    lane_routing_floor: float = 0.12
    lane_mixing_init: float = 0.05
    lane_prediction_horizons: tuple[int, ...] = ()
    lane_prediction_loss_weight: float = 0.01
    lane_token_loss_weight: float = 0.0
    prediction_loss_stride: int = 8
    phase_coupling_topology: str = "adjacent_anchor"
    phase_lock_loss_weight: float = 0.0
    temporal_context_retention: float = 0.85
    interbank_coupling_init: float = 0.15
    bank_specialization_strength: float = 0.35
    gear_bank_speed_scales: tuple[float, ...] = ()
    gear_bank_horizon_scales: tuple[float, ...] = ()
    gear_bank_temporal_strides: tuple[int, ...] = ()
    lane_dropout: float = 0.0
    routing_temperature: float = 1.0
    agreement_dim: int = 64

    future_horizons: tuple[int, ...] = (4, 16)
    future_dim: int = 0
    future_loss_weight: float = 0.02
    future_token_loss_weight: float = 0.0
    future_contrastive_weight: float = 0.0
    future_contrastive_samples: int = 256
    future_contrastive_temperature: float = 0.07
    future_logit_weight: float = 0.25
    future_residual_init: float = 0.05
    diversity_loss_weight: float = 0.001
    slot_usage_loss_weight: float = 0.003
    alignment_loss_weight: float = 0.01
    consistency_loss_weight: float = 0.01

    gear_warmup_steps: int = 100
    gear_ramp_steps: int = 400
    phase_warmup_steps: int = 300
    phase_ramp_steps: int = 700
    auxiliary_warmup_steps: int = 500
    auxiliary_ramp_steps: int = 500
    auxiliary_loss_interval: int = 1
    future_warmup_steps: int = 800
    future_ramp_steps: int = 800
    future_loss_interval: int = 1
    gear_lr_multiplier: float = 3.0

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
        object.__setattr__(
            self, "gear_lane_sizes", _as_tuple(self.gear_lane_sizes, int)
        )
        object.__setattr__(
            self, "gear_update_rates", _as_tuple(self.gear_update_rates, float)
        )
        object.__setattr__(
            self,
            "lane_prediction_horizons",
            _as_tuple(self.lane_prediction_horizons, int),
        )
        object.__setattr__(
            self,
            "gear_bank_speed_scales",
            _as_tuple(self.gear_bank_speed_scales, float),
        )
        object.__setattr__(
            self,
            "gear_bank_horizon_scales",
            _as_tuple(self.gear_bank_horizon_scales, float),
        )
        object.__setattr__(
            self,
            "gear_bank_temporal_strides",
            _as_tuple(self.gear_bank_temporal_strides, int),
        )
        object.__setattr__(self, "future_horizons", _as_tuple(self.future_horizons, int))
        if self.num_gears and not self.gear_lane_sizes:
            lane_count = min(4, self.num_gears)
            sizes = [1] * lane_count
            for index in range(self.num_gears - lane_count):
                sizes[index % lane_count] += 1
            object.__setattr__(self, "gear_lane_sizes", tuple(sizes))
        if self.num_gears and not self.gear_update_rates:
            rates = tuple(
                min(0.8, max(0.04, 2.0 / math.sqrt(float(field))))
                for field in self.gear_receptive_fields
            )
            object.__setattr__(self, "gear_update_rates", rates)
        if self.num_gears and not self.lane_prediction_horizons:
            horizons = tuple(2 ** lane for lane in range(len(self.gear_lane_sizes)))
            object.__setattr__(self, "lane_prediction_horizons", horizons)
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
        if self.use_attention and (self.dim // self.heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.layers < 1:
            raise ValueError("layers must be positive")
        if self.num_gears < 0 or self.num_gears > 20:
            raise ValueError("num_gears must be 0 (ablation) or between 5 and 20")
        if 0 < self.num_gears < 5:
            raise ValueError("a generative gear stack requires at least 5 gears")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.gear_system not in {"parallel_v5", "parallel_v4", "legacy_v3"}:
            raise ValueError(
                "gear_system must be 'parallel_v5', 'parallel_v4', or 'legacy_v3'"
            )
        if self.gear_update_mode not in {"parallel", "chunked", "dilated", "scan"}:
            raise ValueError(
                "gear_update_mode must be 'parallel', 'chunked', 'dilated', or 'scan'"
            )
        if self.gear_layer_strategy not in {
            "none",
            "explicit",
            "all",
            "alternate",
            "upper_half",
            "upper_alternate",
            "stacked_parallel",
        }:
            raise ValueError("unknown gear_layer_strategy")
        if self.num_gears:
            for name, value in (
                ("gear_speeds", self.gear_speeds),
                ("gear_slots", self.gear_slots),
                ("gear_receptive_fields", self.gear_receptive_fields),
                ("gear_update_rates", self.gear_update_rates),
            ):
                if len(value) != self.num_gears:
                    raise ValueError(f"{name} length must equal num_gears")
            if any(v <= 0.0 for v in self.gear_speeds):
                raise ValueError("gear speeds must be positive")
            if any(
                fast <= slow
                for fast, slow in zip(self.gear_speeds, self.gear_speeds[1:])
            ):
                raise ValueError("gear_speeds must be strictly decreasing (fast -> slow)")
            if any(v < 1 for v in self.gear_slots):
                raise ValueError("gear slots must be positive")
            if any(v < 1 for v in self.gear_receptive_fields):
                raise ValueError("gear receptive fields must be positive")
            if any(
                fast > slow
                for fast, slow in zip(
                    self.gear_receptive_fields,
                    self.gear_receptive_fields[1:],
                )
            ):
                raise ValueError("gear receptive fields must be non-decreasing")
            if sum(self.gear_lane_sizes) != self.num_gears:
                raise ValueError("gear_lane_sizes must sum to num_gears")
            if any(v < 1 for v in self.gear_lane_sizes):
                raise ValueError("gear lanes must contain at least one gear")
            if any(not 0.0 < v < 1.0 for v in self.gear_update_rates):
                raise ValueError("gear_update_rates must be in (0, 1)")
            if len(self.lane_prediction_horizons) != len(self.gear_lane_sizes):
                raise ValueError(
                    "lane_prediction_horizons length must equal the number of lanes"
                )
            if any(v < 1 for v in self.lane_prediction_horizons):
                raise ValueError("lane prediction horizons must be positive")
        if self.gear_rotation_dims < 0 or self.gear_rotation_dims > self.gear_dim:
            raise ValueError("gear_rotation_dims must be between 0 and gear_dim")
        if self.gear_rotation_dims % 2:
            raise ValueError("gear_rotation_dims must be even")
        if self.phase_harmonics < 1:
            raise ValueError("phase_harmonics must be positive")
        if self.max_log_speed_offset < 0.0:
            raise ValueError("max_log_speed_offset must be non-negative")
        if self.phase_drive_scale < 0.0:
            raise ValueError("phase_drive_scale must be non-negative")
        if self.phase_modulation_scale < 0.0:
            raise ValueError("phase_modulation_scale must be non-negative")
        if self.phase_coupling_init < 0.0:
            raise ValueError("phase_coupling_init must be non-negative")
        if self.phase_coupling_max <= 0.0:
            raise ValueError("phase_coupling_max must be positive")
        if self.phase_coupling_init > self.phase_coupling_max:
            raise ValueError("phase_coupling_init cannot exceed phase_coupling_max")
        if self.agreement_dim < 1:
            raise ValueError("agreement_dim must be positive")
        for name in (
            "gear_residual_init",
            "gear_write_gate_init",
            "gear_read_gate_init",
            "gear_coupling_init",
        ):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("gear_routing_floor", "lane_routing_floor"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0.0 <= self.lane_mixing_init < 1.0:
            raise ValueError("lane_mixing_init must be in [0, 1)")
        if self.phase_coupling_topology not in {
            "adjacent_anchor",
            "dense_lower",
        }:
            raise ValueError("unknown phase_coupling_topology")
        if not 0.0 < self.temporal_context_retention < 1.0:
            raise ValueError("temporal_context_retention must be in (0, 1)")
        if not 0.0 <= self.interbank_coupling_init < 1.0:
            raise ValueError("interbank_coupling_init must be in [0, 1)")
        if self.bank_specialization_strength < 0.0:
            raise ValueError("bank_specialization_strength must be non-negative")
        if not 0.0 <= self.lane_dropout < 1.0:
            raise ValueError("lane_dropout must be in [0, 1)")
        if self.routing_temperature <= 0.0:
            raise ValueError("routing_temperature must be positive")
        if self.prediction_loss_stride < 1:
            raise ValueError("prediction_loss_stride must be positive")
        for name in (
            "future_loss_weight",
            "future_token_loss_weight",
            "future_contrastive_weight",
            "future_logit_weight",
            "lane_prediction_loss_weight",
            "lane_token_loss_weight",
            "phase_lock_loss_weight",
            "diversity_loss_weight",
            "slot_usage_loss_weight",
            "alignment_loss_weight",
            "consistency_loss_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if any(h < 1 for h in self.future_horizons):
            raise ValueError("future horizons must be positive")
        if len(set(self.future_horizons)) != len(self.future_horizons):
            raise ValueError("future horizons must be unique")
        if self.future_contrastive_samples < 1:
            raise ValueError("future_contrastive_samples must be positive")
        if self.future_contrastive_temperature <= 0.0:
            raise ValueError("future_contrastive_temperature must be positive")
        if not 0.0 <= self.future_residual_init < 1.0:
            raise ValueError("future_residual_init must be in [0, 1)")
        for name in (
            "gear_warmup_steps",
            "gear_ramp_steps",
            "phase_warmup_steps",
            "phase_ramp_steps",
            "auxiliary_warmup_steps",
            "auxiliary_ramp_steps",
            "future_warmup_steps",
            "future_ramp_steps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.auxiliary_loss_interval < 1:
            raise ValueError("auxiliary_loss_interval must be positive")
        if self.future_loss_interval < 1:
            raise ValueError("future_loss_interval must be positive")
        if self.gear_lr_multiplier <= 0.0:
            raise ValueError("gear_lr_multiplier must be positive")
        bank_count = len(self.selected_gear_layers())
        for name, values in (
            ("gear_bank_speed_scales", self.gear_bank_speed_scales),
            ("gear_bank_horizon_scales", self.gear_bank_horizon_scales),
            ("gear_bank_temporal_strides", self.gear_bank_temporal_strides),
        ):
            if values and len(values) != bank_count:
                raise ValueError(f"{name} length must equal active gear bank count")
            if any(value <= 0 for value in values):
                raise ValueError(f"{name} values must be positive")
        if (
            self.gear_system == "parallel_v5"
            and bank_count > 1
            and self.share_gear_modules
        ):
            raise ValueError("parallel_v5 stacked banks cannot share gear modules")

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
        elif self.gear_layer_strategy == "stacked_parallel":
            count = min(3, self.layers)
            if count == 1:
                layers = (0,)
            else:
                layers = tuple(
                    round(index * (self.layers - 1) / (count - 1))
                    for index in range(count)
                )
        else:
            start = self.layers // 2
            layers = tuple(range(start, self.layers, 2)) or (self.layers - 1,)
        for layer in layers:
            if layer < 0 or layer >= self.layers:
                raise ValueError(f"gear layer {layer} outside model depth")
        return tuple(sorted(set(layers)))

    def gear_bank_scales(self) -> tuple[tuple[float, float], ...]:
        """Return per-bank speed and horizon specialization scales."""
        count = len(self.selected_gear_layers())
        if count == 0:
            return ()
        if self.gear_bank_speed_scales:
            speed_scales = self.gear_bank_speed_scales
        elif count == 1:
            speed_scales = (1.0,)
        else:
            speed_scales = tuple(
                1.15 - 0.30 * index / (count - 1)
                for index in range(count)
            )
        if self.gear_bank_horizon_scales:
            horizon_scales = self.gear_bank_horizon_scales
        elif count == 1:
            horizon_scales = (1.0,)
        else:
            horizon_scales = tuple(
                0.5 * (4.0 ** (index / (count - 1)))
                for index in range(count)
            )
        return tuple(zip(speed_scales, horizon_scales))

    def gear_bank_strides(self) -> tuple[int, ...]:
        count = len(self.selected_gear_layers())
        if self.gear_bank_temporal_strides:
            return self.gear_bank_temporal_strides
        if self.gear_system != "parallel_v5":
            return (1,) * count
        return tuple(2 ** index for index in range(count))

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


class HierarchicalGearClock(nn.Module):
    """Learnable monotonic clocks with token/context drive and phase locking."""

    def __init__(
        self,
        dim: int,
        base_speeds: tuple[float, ...],
        max_log_speed_offset: float,
        drive_scale: float,
        coupling_init: float,
        coupling_max: float,
    ) -> None:
        super().__init__()
        speeds = torch.tensor(base_speeds, dtype=torch.float32)
        self.num_gears = int(speeds.numel())
        self.max_log_speed_offset = float(max_log_speed_offset)
        self.drive_scale = float(drive_scale)
        self.coupling_max = float(coupling_max)
        self.register_buffer("base_first_log_speed", speeds[0].log())
        self.register_buffer("base_log_gaps", (speeds[:-1] / speeds[1:]).log())
        self.first_speed_offset = nn.Parameter(torch.zeros(()))
        self.gap_offsets = nn.Parameter(torch.zeros(self.num_gears - 1))
        self.phase_offsets = nn.Parameter(torch.zeros(self.num_gears))
        self.token_drive = nn.Linear(dim, self.num_gears, bias=False)
        self.context_drive = nn.Linear(dim, self.num_gears, bias=False)
        self.coupling_raw = nn.Parameter(
            torch.full(
                (self.num_gears, self.num_gears),
                _logit(coupling_init / coupling_max),
            )
        )
        self.coupling_phase = nn.Parameter(
            torch.zeros(self.num_gears, self.num_gears)
        )
        lower = torch.tril(
            torch.ones(self.num_gears, self.num_gears, dtype=torch.bool),
            diagonal=-1,
        )
        self.register_buffer("lower_coupling_mask", lower)

    def speeds(self) -> torch.Tensor:
        first = self.base_first_log_speed.float() + (
            self.max_log_speed_offset * torch.tanh(self.first_speed_offset.float())
        )
        gaps = self.base_log_gaps.float() * torch.exp(
            self.max_log_speed_offset * torch.tanh(self.gap_offsets.float())
        )
        log_speeds = torch.cat(
            [first[None], first - torch.cumsum(gaps, dim=0)],
            dim=0,
        )
        return log_speeds.exp()

    def forward(
        self,
        h: torch.Tensor,
        previous_phase: torch.Tensor | None = None,
        previous_context: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, dict[str, torch.Tensor]]:
        batch, length, _ = h.shape
        # Keep clocks in fp32 even when model activations use bf16/fp16; otherwise
        # long-running phase increments eventually quantize away.
        speeds = self.speeds().to(dtype=torch.float32, device=h.device)
        if previous_phase is None:
            phase = self.phase_offsets.float()[None].expand(batch, -1)
        else:
            phase = previous_phase.to(dtype=torch.float32, device=h.device)
        if previous_context is None:
            first_context = torch.zeros_like(h[:, :1])
        else:
            first_context = previous_context.to(dtype=h.dtype, device=h.device)[:, None]
        context = torch.cat([first_context, h[:, :-1]], dim=1)
        drives = self.drive_scale * torch.tanh(
            self.token_drive(h) + self.context_drive(context)
        ).float()

        strength = self.coupling_max * torch.sigmoid(self.coupling_raw.float())
        strength = strength * self.lower_coupling_mask.to(dtype=torch.float32)
        source_counts = self.lower_coupling_mask.sum(dim=-1).clamp_min(1).float()
        # r[g,j] maps source gear j's faster phase into target gear g's scale.
        ratios = speeds[:, None] / speeds[None, :]
        phases = []
        coupling_magnitudes = []
        for t in range(length):
            source = phase[:, None, :]
            target = phase[:, :, None]
            phase_error = (
                ratios[None] * source
                - target
                + self.coupling_phase.float()[None]
            )
            coupling = (
                (strength[None] * phase_error.sin()).sum(dim=-1)
                / source_counts[None]
            )
            phase = phase + speeds[None] + drives[:, t] + coupling
            phases.append(phase)
            coupling_magnitudes.append(coupling.abs().mean())
        phase_sequence = torch.stack(phases, dim=1)
        next_cache = (
            {
                "phase": phase.detach(),
                "context": h[:, -1].detach(),
            }
            if use_cache
            else None
        )
        coupling_activity = (
            torch.stack(coupling_magnitudes).mean()
            if coupling_magnitudes
            else h.sum() * 0.0
        )
        return phase_sequence, next_cache, {
            "phase_drive_activity": drives.abs().mean(),
            "phase_coupling_activity": coupling_activity,
            "fast_speed": speeds[0],
            "slow_speed": speeds[-1],
        }


class GearSlotRouter(nn.Module):
    def __init__(self, dim: int, slots: int, phase_dim: int) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.empty(slots, dim))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.phase_proj = nn.Linear(phase_dim, dim, bias=False)
        preferred = torch.arange(slots, dtype=torch.float32) * (2.0 * math.pi / slots)
        self.preferred_phase = nn.Parameter(preferred)
        self.concentration_raw = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        nn.init.normal_(self.slots, mean=0.0, std=0.02)

    def forward(
        self,
        h: torch.Tensor,
        phase: torch.Tensor,
        phase_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(h) + self.phase_proj(phase_features)
        content_scores = q @ self.slots.T / math.sqrt(h.shape[-1])
        phase_scores = (
            F.softplus(self.concentration_raw).float()
            * (
                phase.unsqueeze(-1)
                - self.preferred_phase.float()
            ).cos()
        ).to(dtype=h.dtype)
        scores = content_scores + phase_scores
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
        self.register_buffer("base_speed", torch.tensor(float(base_speed)))
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
        update_period = max(1, round(1.0 / self.base_speed_value))
        if update_mode == "dilated":
            dilation = update_period
            kernel = max(1, math.ceil(self.receptive_field / dilation))
            self.update = CausalDepthwiseUpdate(dim, kernel, dilation)
        else:
            self.update = None
        initial_decay = 1.0 - 1.0 / update_period
        self.scan_decay_logit = nn.Parameter(torch.tensor(_logit(initial_decay)))
        self.state_norm = RMSNorm(dim)
        read_in = (4 * dim if self.write_summary else 3 * dim) + self.phase_dim
        self.read_gate = nn.Linear(read_in, dim, bias=True)
        self.out_proj = nn.Linear(read_in, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.write_gate.bias, _logit(write_gate_init))
        nn.init.constant_(self.read_gate.bias, _logit(read_gate_init))

    def phase_features(
        self,
        phase: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        parts = []
        for harmonic in range(1, self.phase_harmonics + 1):
            p = phase * harmonic
            parts.extend([p.sin(), p.cos()])
        return torch.stack(parts, dim=-1).to(dtype=dtype)

    def _memory_update(
        self,
        candidate: torch.Tensor,
        update_mask: torch.Tensor,
        cache: torch.Tensor | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        decay = torch.sigmoid(self.scan_decay_logit).to(dtype=candidate.dtype)
        if cache is None:
            prev = torch.zeros(
                candidate.shape[0],
                candidate.shape[-1],
                dtype=candidate.dtype,
                device=candidate.device,
            )
        else:
            prev = cache.to(dtype=candidate.dtype, device=candidate.device)
        states = []
        for t in range(candidate.shape[1]):
            alpha = (1.0 - decay) * update_mask[:, t, :]
            prev = (1.0 - alpha) * prev + alpha * candidate[:, t, :]
            states.append(prev)
        state = torch.stack(states, dim=1)
        return state, (prev.detach() if use_cache else None)

    def _temporal_candidate(
        self,
        proposal: torch.Tensor,
        cache: torch.Tensor | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.update_mode == "scan":
            return proposal, None
        if self.update_mode == "chunked":
            window = max(1, round(1.0 / self.base_speed_value))
            return _causal_moving_average(proposal, window, cache, use_cache)
        else:
            assert self.update is not None
            return self.update(proposal, cache, use_cache)

    def forward(
        self,
        h: torch.Tensor,
        phase: torch.Tensor,
        cache: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        phase_features = self.phase_features(phase, h.dtype)
        routed, alpha = self.router(h, phase, phase_features)
        phase_b = phase_features
        if isinstance(cache, dict):
            temporal_cache = cache.get("temporal", cache.get("update"))
            summary_cache = cache.get("summary")
            memory_cache = cache.get("memory")
        else:
            temporal_cache = cache
            summary_cache = None
            memory_cache = None
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
        ).unsqueeze(-1)
        write_gate = torch.sigmoid(self.write_gate(write_context)) * periodic_mask
        candidate, next_temporal_cache = self._temporal_candidate(
            proposal,
            temporal_cache,
            use_cache,
        )
        state, next_memory_cache = self._memory_update(
            candidate,
            write_gate,
            memory_cache,
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
            {
                "temporal": next_temporal_cache,
                "summary": next_summary_cache,
                "memory": next_memory_cache,
            }
            if use_cache
            else None
        )
        route_usage = alpha.mean(dim=(0, 1))
        route_balance = ((route_usage - (1.0 / alpha.shape[-1])) ** 2).sum() * alpha.shape[-1]
        entropy_scale = math.log(alpha.shape[-1]) if alpha.shape[-1] > 1 else 1.0
        route_entropy = (
            -(alpha * alpha.clamp_min(1e-9).log()).sum(dim=-1).mean()
            / entropy_scale
        )
        usage_entropy = (
            -(route_usage * route_usage.clamp_min(1e-9).log()).sum()
            / entropy_scale
        )
        return self.dropout(out), next_cache, {
            "memory_state": state,
            "slot_message": routed,
            "route_balance": route_balance,
            "route_entropy": route_entropy,
            "usage_entropy": usage_entropy,
            "update_activity": periodic_mask.mean(),
            "write_activity": write_gate.mean(),
            "read_activity": read_gate.mean(),
        }


class GearFusion(nn.Module):
    def __init__(self, dim: int, num_gears: int, residual_init: float, dropout: float) -> None:
        super().__init__()
        self.context_score = nn.Linear(dim, dim, bias=False)
        self.state_score = nn.Linear(dim, dim, bias=False)
        self.message_score = nn.Linear(dim, dim, bias=False)
        self.score = nn.Linear(dim, 1, bias=False)
        self.state_value = nn.Linear(dim, dim, bias=False)
        self.message_value = nn.Linear(dim, dim, bias=False)
        self.agreement_gain_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0))
        )
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.residual_gate_logit = nn.Parameter(torch.tensor(_logit(residual_init)))

    def forward(
        self,
        h: torch.Tensor,
        gear_outputs: torch.Tensor,
        memory_states: torch.Tensor,
        slot_messages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = (
            gear_outputs
            + self.state_value(memory_states)
            + self.message_value(slot_messages)
        )
        normalized_values = F.normalize(values, dim=-1)
        consensus = F.normalize(normalized_values.mean(dim=-2), dim=-1)
        agreement_scores = (
            normalized_values * consensus.unsqueeze(-2)
        ).sum(dim=-1)
        scores = self.score(
            torch.tanh(
                self.context_score(h).unsqueeze(-2)
                + self.state_score(memory_states)
                + self.message_score(slot_messages)
            )
        ).squeeze(-1)
        scores = scores + (
            F.softplus(self.agreement_gain_raw).to(dtype=h.dtype)
            * agreement_scores
        )
        weights = scores.softmax(dim=-1)
        fused = (values * weights.unsqueeze(-1)).sum(dim=-2)
        agreement = (
            weights * ((agreement_scores + 1.0) * 0.5)
        ).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        gate = torch.sigmoid(self.residual_gate_logit).to(dtype=h.dtype)
        residual = gate * agreement * self.dropout(self.out(self.norm(fused)))
        return residual, weights, agreement.squeeze(-1)


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
        self.clock = HierarchicalGearClock(
            config.gear_dim,
            config.gear_speeds,
            config.max_log_speed_offset,
            config.phase_drive_scale,
            config.phase_coupling_init,
            config.phase_coupling_max,
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
        log_speeds = self.clock.speeds().log()
        adjacent_gaps = log_speeds[:-1] - log_speeds[1:]
        return torch.exp(-adjacent_gaps).mean()

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
        cache: dict[str, Any] | list[torch.Tensor | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any] | None, dict[str, torch.Tensor]]:
        gear_input = self.down(h)
        gear_outputs = []
        memory_states = []
        slot_messages = []
        next_gear_caches = []
        balances = []
        entropies = []
        usage_entropies = []
        activities = []
        write_activities = []
        read_activities = []
        if isinstance(cache, dict):
            gear_caches = cache.get("gears") or [None] * len(self.gears)
            previous_phase = cache.get("phase")
            previous_context = cache.get("context")
        else:
            gear_caches = cache or [None] * len(self.gears)
            previous_phase = None
            previous_context = None
        phases, next_clock_cache, clock_stats = self.clock(
            gear_input,
            previous_phase,
            previous_context,
            use_cache,
        )
        for index, (gear, gear_cache) in enumerate(zip(self.gears, gear_caches)):
            out, nc, stats = gear(
                gear_input,
                phases[..., index],
                gear_cache,
                use_cache,
            )
            gear_outputs.append(out)
            memory_states.append(stats["memory_state"])
            slot_messages.append(stats["slot_message"])
            next_gear_caches.append(nc)
            balances.append(stats["route_balance"])
            entropies.append(stats["route_entropy"])
            usage_entropies.append(stats["usage_entropy"])
            activities.append(stats["update_activity"])
            write_activities.append(stats["write_activity"])
            read_activities.append(stats["read_activity"])
        stacked = torch.stack(gear_outputs, dim=-2)
        stacked_states = torch.stack(memory_states, dim=-2)
        stacked_messages = torch.stack(slot_messages, dim=-2)
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
        residual, fusion_weights, gear_agreement = self.fusion(
            gear_input,
            coupled,
            stacked_states,
            stacked_messages,
        )
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
            "usage_entropy": torch.stack(usage_entropies).mean(),
            "update_activity": torch.stack(activities).mean(),
            "write_activity": torch.stack(write_activities).mean(),
            "read_activity": torch.stack(read_activities).mean(),
            "coupling_entropy": coupling_stats["coupling_entropy"],
            "coupling_gate": coupling_stats["coupling_gate"],
            "coupling_offdiag": coupling_stats["coupling_offdiag"],
            "output_diversity": output_diversity,
            "speed_separation": self.speed_separation_loss(),
            "slot_diversity": self.slot_diversity_loss(),
            "phase_drive_activity": clock_stats["phase_drive_activity"],
            "phase_coupling_activity": clock_stats["phase_coupling_activity"],
            "fast_speed": clock_stats["fast_speed"],
            "slow_speed": clock_stats["slow_speed"],
            "coherence_entropy": (
                -(fusion_weights * fusion_weights.clamp_min(1e-9).log())
                .sum(dim=-1)
                .mean()
                / math.log(len(self.gears))
            ),
            "gear_agreement": gear_agreement.mean(),
        }
        next_cache = None
        if use_cache:
            assert next_clock_cache is not None
            next_cache = {
                "phase": next_clock_cache["phase"],
                "context": next_clock_cache["context"],
                "gears": next_gear_caches,
            }
        return residual, next_cache, aux


def _build_gear_system(
    config: GearTransformerConfig,
    *,
    bank_index: int = 0,
    bank_count: int = 1,
    speed_scale: float = 1.0,
    horizon_scale: float = 1.0,
    temporal_stride: int = 1,
) -> nn.Module:
    if config.gear_system in {"parallel_v4", "parallel_v5"}:
        return ParallelGearSystem(
            config,
            bank_index=bank_index,
            bank_count=bank_count,
            speed_scale=speed_scale,
            horizon_scale=horizon_scale,
            temporal_stride=temporal_stride,
        )
    return MultiRateGearModule(config)


class GearTransformerBlock(nn.Module):
    def __init__(
        self,
        config: GearTransformerConfig,
        use_gears: bool,
        own_gears: bool = True,
        *,
        bank_index: int = 0,
        bank_count: int = 1,
        speed_scale: float = 1.0,
        horizon_scale: float = 1.0,
        temporal_stride: int = 1,
    ) -> None:
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
        self.gears = (
            _build_gear_system(
                config,
                bank_index=bank_index,
                bank_count=bank_count,
                speed_scale=speed_scale,
                horizon_scale=horizon_scale,
                temporal_stride=temporal_stride,
            )
            if use_gears and own_gears
            else None
        )
        self.norm2 = RMSNorm(config.dim)
        self.ff = SwiGLU(config.dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
        attn_mask: torch.Tensor | None = None,
        shared_gears: nn.Module | None = None,
        gear_scale: float | torch.Tensor = 1.0,
        phase_scale: float | torch.Tensor = 1.0,
        bank_carrier: torch.Tensor | None = None,
        component_scales: dict[str, float] | None = None,
        collect_aux: bool = True,
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
            q, k = apply_rope(q, positions), apply_rope(k, positions)
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
        gear_enabled = not (
            isinstance(gear_scale, (float, int)) and gear_scale <= 0.0
        )
        if (
            gear_enabled
            and self.use_gears
            and gear_module is not None
            and self.gear_norm is not None
        ):
            if isinstance(gear_module, ParallelGearSystem):
                residual, next_gear_cache, gear_aux = gear_module(
                    self.gear_norm(x),
                    positions,
                    gear_cache,
                    use_cache,
                    residual_scale=gear_scale,
                    phase_scale=phase_scale,
                    bank_carrier=bank_carrier,
                    component_scales=component_scales,
                    collect_aux=collect_aux,
                )
            else:
                residual, next_gear_cache, gear_aux = gear_module(
                    self.gear_norm(x),
                    positions,
                    gear_cache,
                    use_cache,
                )
                residual = residual * torch.as_tensor(
                    gear_scale, dtype=residual.dtype, device=residual.device
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
        ordered_gear_layers = config.selected_gear_layers()
        gear_layers = set(ordered_gear_layers)
        bank_scales = config.gear_bank_scales()
        bank_strides = config.gear_bank_strides()
        bank_by_layer = {
            layer: (
                bank_index,
                *bank_scales[bank_index],
                bank_strides[bank_index],
            )
            for bank_index, layer in enumerate(ordered_gear_layers)
        }
        self.gear_layer_indices = gear_layers
        self.shared_gears = (
            _build_gear_system(config)
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
                    bank_index=bank_by_layer.get(
                        layer, (0, 1.0, 1.0, 1)
                    )[0],
                    bank_count=max(1, len(ordered_gear_layers)),
                    speed_scale=bank_by_layer.get(
                        layer, (0, 1.0, 1.0, 1)
                    )[1],
                    horizon_scale=bank_by_layer.get(
                        layer, (0, 1.0, 1.0, 1)
                    )[2],
                    temporal_stride=bank_by_layer.get(
                        layer, (0, 1.0, 1.0, 1)
                    )[3],
                )
                for layer in range(config.layers)
            ]
        )
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        needs_future_targets = any(
            weight > 0.0
            for weight in (
                config.future_loss_weight,
                config.future_token_loss_weight,
                config.future_contrastive_weight,
                config.alignment_loss_weight,
            )
        )
        uses_future_logits = config.future_logit_weight > 0.0
        self.future = (
            FuturePredictionHead(config.dim, config.future_horizons, config.future_dim)
            if config.future_horizons and (needs_future_targets or uses_future_logits)
            else None
        )
        self.future_mix_logits = (
            nn.Parameter(torch.zeros(len(config.future_horizons)))
            if self.future is not None and uses_future_logits
            else None
        )
        self.future_to_hidden = (
            nn.Linear(config.dim, config.dim, bias=False)
            if self.future is not None and uses_future_logits
            else None
        )
        self.future_residual_gate_logit = (
            nn.Parameter(torch.tensor(_logit(config.future_residual_init)))
            if self.future is not None and uses_future_logits
            else None
        )
        self.consistency = (
            nn.Sequential(
                nn.Linear(2 * config.dim, config.dim, bias=False),
                nn.SiLU(),
                nn.Linear(config.dim, 1, bias=False),
            )
            if config.future_horizons and config.consistency_loss_weight > 0.0
            else None
        )
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def initialize_trunk_from_transformer(self, transformer: nn.Module) -> int:
        """Copy every shape-compatible non-gear parameter from a Transformer.

        This supports the fastest training path: pretrain a normal Transformer,
        attach the parallel gear system, briefly train gears with a frozen trunk,
        then jointly fine-tune.
        """
        source = transformer.state_dict()
        copied = 0
        for name, target in self.named_parameters():
            if (
                name.startswith("shared_gears.")
                or ".gears." in name
                or name.startswith("future")
                or name.startswith("consistency")
            ):
                continue
            candidate = source.get(name)
            if candidate is None or candidate.shape != target.shape:
                continue
            target.copy_(candidate.to(device=target.device, dtype=target.dtype))
            copied += target.numel()
        return copied

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
        gear_scale: float | torch.Tensor = 1.0,
        phase_scale: float | torch.Tensor = 1.0,
        component_scales: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]] | None, list[dict[str, torch.Tensor]]]:
        x = self.token(ids)
        attn_mask = (
            self._full_attn_mask(attention_mask, ids.shape[1], ids.device)
            if caches is None
            else None
        )
        next_caches = []
        aux_records = []
        bank_carrier = None
        for i, block in enumerate(self.blocks):
            if (
                component_scales is not None
                and component_scales.get(f"block_{i}", 1.0) <= 0.0
            ):
                if use_cache:
                    raise ValueError("block ablation is not supported with caches")
                continue
            x, nc, gear_aux = block(
                x,
                None if caches is None else caches[i],
                use_cache,
                attn_mask,
                self.shared_gears if i in self.gear_layer_indices else None,
                gear_scale,
                phase_scale,
                bank_carrier,
                component_scales,
                return_aux,
            )
            if use_cache:
                next_caches.append(nc)
            if return_aux and gear_aux is not None:
                aux_records.append(gear_aux)
            if gear_aux is not None and "carrier" in gear_aux:
                bank_carrier = gear_aux["carrier"]
        return self.norm(x), (next_caches if use_cache else None), aux_records

    def forward(
        self,
        ids: torch.Tensor,
        caches: list[dict[str, Any]] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]] | None]:
        hidden, next_caches, _ = self._forward_hidden(ids, caches, use_cache, attention_mask)
        return self.head(self._generation_hidden(hidden)), next_caches

    def component_logits(
        self,
        ids: torch.Tensor,
        disabled_components: tuple[str, ...] | list[str] = (),
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score tokens with selected mechanisms disabled for clean ablations."""
        disabled = set(disabled_components)
        component_scales = {name: 0.0 for name in disabled}
        gear_scale = 0.0 if "gears" in disabled else 1.0
        phase_scale = 0.0 if "phase" in disabled else 1.0
        hidden, _, _ = self._forward_hidden(
            ids,
            attention_mask=attention_mask,
            gear_scale=gear_scale,
            phase_scale=phase_scale,
            component_scales=component_scales,
        )
        future_scale = 0.0 if "future" in disabled else 1.0
        return self.head(self._generation_hidden(hidden, future_scale))

    @torch.no_grad()
    def component_ablation_metrics(
        self,
        tokens: torch.Tensor,
        components: tuple[str, ...] = (
            "gears",
            "phase",
            "phase_coupling",
            "rotation",
            "temporal_context",
            "interbank_coupling",
            "lane_mixing",
            "future",
        ),
    ) -> dict[str, dict[str, float]]:
        """Measure per-component NLL and top-1 prediction contribution."""
        was_training = self.training
        self.eval()
        try:
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            full = self.component_logits(inputs)
            full_nll = F.cross_entropy(
                full.reshape(-1, full.shape[-1]),
                targets.reshape(-1),
            )
            full_top = full.argmax(dim=-1)
            result = {
                "full": {
                    "nll": float(full_nll),
                    "delta_nll": 0.0,
                    "top1_change": 0.0,
                }
            }
            for component in components:
                ablated = self.component_logits(inputs, (component,))
                nll = F.cross_entropy(
                    ablated.reshape(-1, ablated.shape[-1]),
                    targets.reshape(-1),
                )
                result[component] = {
                    "nll": float(nll),
                    "delta_nll": float(nll - full_nll),
                    "top1_change": float(
                        (ablated.argmax(dim=-1) != full_top).float().mean()
                    ),
                }
            for bank_index, _ in enumerate(self.config.selected_gear_layers()):
                name = f"bank_{bank_index}"
                ablated = self.component_logits(inputs, (name,))
                nll = F.cross_entropy(
                    ablated.reshape(-1, ablated.shape[-1]),
                    targets.reshape(-1),
                )
                result[name] = {
                    "nll": float(nll),
                    "delta_nll": float(nll - full_nll),
                    "top1_change": float(
                        (ablated.argmax(dim=-1) != full_top).float().mean()
                    ),
                }
            for block_index in range(self.config.layers):
                name = f"block_{block_index}"
                ablated = self.component_logits(inputs, (name,))
                nll = F.cross_entropy(
                    ablated.reshape(-1, ablated.shape[-1]),
                    targets.reshape(-1),
                )
                result[name] = {
                    "nll": float(nll),
                    "delta_nll": float(nll - full_nll),
                    "top1_change": float(
                        (ablated.argmax(dim=-1) != full_top).float().mean()
                    ),
                }
            return result
        finally:
            self.train(was_training)

    @staticmethod
    def _ramp(step: int, start: int, duration: int) -> float:
        if step < start:
            return 0.0
        if duration <= 0:
            return 1.0
        return min(1.0, (step - start + 1) / duration)

    def _training_scales(self, metadata: dict[str, Any]) -> dict[str, float]:
        if "training_step" not in metadata:
            return {
                "gear": 1.0,
                "phase": 1.0,
                "auxiliary": 1.0,
                "future": 1.0,
            }
        step = int(metadata["training_step"])
        return {
            "gear": self._ramp(
                step, self.config.gear_warmup_steps, self.config.gear_ramp_steps
            ),
            "phase": self._ramp(
                step, self.config.phase_warmup_steps, self.config.phase_ramp_steps
            ),
            "auxiliary": self._ramp(
                step,
                self.config.auxiliary_warmup_steps,
                self.config.auxiliary_ramp_steps,
            ),
            "future": self._ramp(
                step,
                self.config.future_warmup_steps,
                self.config.future_ramp_steps,
            ),
        }

    def _generation_hidden(
        self,
        hidden: torch.Tensor,
        future_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Inject predicted future compatibility into the state scored by the LM head."""
        if (
            self.future is None
            or self.future_mix_logits is None
            or self.future_to_hidden is None
            or self.future_residual_gate_logit is None
            or (isinstance(future_scale, (float, int)) and future_scale <= 0.0)
        ):
            return hidden
        predictions = self.future(hidden)
        ordered = torch.stack(
            [predictions[h] for h in self.config.future_horizons],
            dim=-2,
        )
        weights = self.future_mix_logits.softmax(dim=0).to(dtype=hidden.dtype)
        future_state = (ordered * weights.view(1, 1, -1, 1)).sum(dim=-2)
        gate = torch.sigmoid(self.future_residual_gate_logit).to(dtype=hidden.dtype)
        scale = torch.as_tensor(
            future_scale, dtype=hidden.dtype, device=hidden.device
        )
        return hidden + (
            self.config.future_logit_weight
            * scale
            * gate
            * self.future_to_hidden(future_state)
        )

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
        future_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        logits = self.head(self._generation_hidden(hidden, future_scale))
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
        tokens: torch.Tensor,
        valid_positions: torch.Tensor,
        prediction_selector: int = 0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        zero = hidden.sum() * 0.0
        if self.future is None or not self.config.future_horizons:
            return zero, zero, zero, None
        preds = self.future(hidden)
        latent_total = zero
        contrastive_total = zero
        token_total = zero
        terms = 0
        token_terms = 0
        first_error = None
        for horizon_index, horizon in enumerate(self.config.future_horizons):
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
            if (
                self.config.future_token_loss_weight > 0.0
                and horizon_index
                == prediction_selector % len(self.config.future_horizons)
            ):
                stride = self.config.prediction_loss_stride
                prediction = preds[horizon][:, :-horizon:stride]
                token_targets = tokens[:, horizon::stride]
                token_mask = mask[:, ::stride]
                token_losses = F.cross_entropy(
                    self.head(prediction).reshape(-1, self.config.vocab_size),
                    token_targets.reshape(-1),
                    reduction="none",
                ).reshape_as(token_targets)
                token_mask_float = token_mask.to(token_losses.dtype)
                token_total = token_total + (
                    token_losses * token_mask_float
                ).sum() / token_mask_float.sum().clamp_min(1.0)
                token_terms += 1
            terms += 1
        if terms == 0:
            return zero, zero, zero, None
        return (
            latent_total / terms,
            contrastive_total / terms,
            token_total / max(token_terms, 1),
            first_error,
        )

    def _lane_prediction_loss(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        valid_positions: torch.Tensor,
        aux_records: list[dict[str, torch.Tensor]],
        prediction_selector: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = hidden.sum() * 0.0
        if not aux_records or not self.config.lane_prediction_horizons:
            return zero, zero
        total = zero
        token_total = zero
        terms = 0
        token_terms = 0
        selected_token_term = prediction_selector % (
            len(aux_records) * len(self.config.lane_prediction_horizons)
        )
        candidate_token_term = 0
        for record in aux_records:
            lane_hidden = record.get("lane_hidden")
            if lane_hidden is None:
                continue
            horizon_scale = float(
                record.get(
                    "bank_horizon_scale",
                    hidden.new_tensor(1.0),
                ).detach()
            )
            for lane, horizon in enumerate(
                self.config.lane_prediction_horizons
            ):
                horizon = max(1, round(horizon * horizon_scale))
                if hidden.shape[1] <= horizon:
                    continue
                prediction = F.normalize(
                    lane_hidden[:, :-horizon, lane],
                    dim=-1,
                )
                target = F.normalize(
                    hidden[:, horizon:].detach(),
                    dim=-1,
                )
                mask = (
                    valid_positions[:, :-horizon]
                    & valid_positions[:, horizon:]
                )
                error = 1.0 - (prediction * target).sum(dim=-1)
                mask_float = mask.to(error.dtype)
                total = total + (
                    error * mask_float
                ).sum() / mask_float.sum().clamp_min(1.0)
                terms += 1
                if (
                    self.config.lane_token_loss_weight > 0.0
                    and candidate_token_term == selected_token_term
                ):
                    stride = self.config.prediction_loss_stride
                    token_prediction = lane_hidden[
                        :, :-horizon:stride, lane
                    ]
                    token_targets = tokens[:, horizon::stride]
                    token_mask = mask[:, ::stride]
                    token_losses = F.cross_entropy(
                        self.head(token_prediction).reshape(
                            -1, self.config.vocab_size
                        ),
                        token_targets.reshape(-1),
                        reduction="none",
                    ).reshape_as(token_targets)
                    token_mask_float = token_mask.to(token_losses.dtype)
                    token_total = token_total + (
                        token_losses * token_mask_float
                    ).sum() / token_mask_float.sum().clamp_min(1.0)
                    token_terms += 1
                candidate_token_term += 1
            context_hidden = record.get("context_hidden")
            if context_hidden is not None and hidden.shape[1] > 1:
                context_prediction = F.normalize(
                    context_hidden[:, :-1],
                    dim=-1,
                )
                context_target = F.normalize(
                    hidden[:, 1:].detach(),
                    dim=-1,
                )
                context_mask = valid_positions[:, :-1] & valid_positions[:, 1:]
                context_error = 1.0 - (
                    context_prediction * context_target
                ).sum(dim=-1)
                context_mask_float = context_mask.to(context_error.dtype)
                total = total + (
                    context_error * context_mask_float
                ).sum() / context_mask_float.sum().clamp_min(1.0)
                terms += 1
        return (
            total / max(terms, 1),
            token_total / max(token_terms, 1),
        )

    def _gear_diversity_loss(
        self,
        aux_records: list[dict[str, torch.Tensor]],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        if not aux_records:
            zero = next(self.parameters()).sum() * 0.0
            return zero, zero, zero, {
                "gear_route_entropy": zero.detach(),
                "gear_usage_entropy": zero.detach(),
                "gear_route_balance": zero.detach(),
                "gear_update_activity": zero.detach(),
                "gear_write_activity": zero.detach(),
                "gear_read_activity": zero.detach(),
                "gear_coupling_entropy": zero.detach(),
                "gear_coupling_gate": zero.detach(),
                "gear_coupling_offdiag": zero.detach(),
                "gear_phase_drive_activity": zero.detach(),
                "gear_phase_coupling_activity": zero.detach(),
                "gear_coherence_entropy": zero.detach(),
                "gear_agreement": zero.detach(),
                "gear_rotation_activity": zero.detach(),
                "gear_minimum_phase_advance": zero.detach(),
                "gear_lane_balance": zero.detach(),
                "gear_fast_speed": zero.detach(),
                "gear_slow_speed": zero.detach(),
                "gear_conflict": zero.detach(),
                "gear_phase_lock_error": zero.detach(),
                "gear_interbank_gate": zero.detach(),
                "gear_interbank_activity": zero.detach(),
                "gear_temporal_context_gate": zero.detach(),
                "gear_active_fraction": zero.detach(),
            }
        output_diversity = torch.stack([a["output_diversity"] for a in aux_records]).mean()
        speed_separation = torch.stack([a["speed_separation"] for a in aux_records]).mean()
        slot_diversity = torch.stack([a["slot_diversity"] for a in aux_records]).mean()
        route_balance = torch.stack([a["route_balance"] for a in aux_records]).mean()
        usage_entropy = torch.stack([a["usage_entropy"] for a in aux_records]).mean()
        lane_balance = torch.stack(
            [
                a.get("lane_balance", a["gear_agreement"] * 0.0)
                for a in aux_records
            ]
        ).mean()
        diversity = output_diversity + 0.25 * speed_separation + 0.01 * slot_diversity
        slot_usage = route_balance + (1.0 - usage_entropy) + lane_balance
        phase_lock = torch.stack(
            [
                a.get("phase_lock_error", a["gear_agreement"] * 0.0)
                for a in aux_records
            ]
        ).mean()
        metrics = {
            "gear_route_entropy": torch.stack([a["route_entropy"] for a in aux_records]).mean().detach(),
            "gear_usage_entropy": usage_entropy.detach(),
            "gear_route_balance": route_balance.detach(),
            "gear_update_activity": torch.stack([a["update_activity"] for a in aux_records]).mean().detach(),
            "gear_write_activity": torch.stack([a["write_activity"] for a in aux_records]).mean().detach(),
            "gear_read_activity": torch.stack([a["read_activity"] for a in aux_records]).mean().detach(),
            "gear_coupling_entropy": torch.stack([a["coupling_entropy"] for a in aux_records]).mean().detach(),
            "gear_coupling_gate": torch.stack([a["coupling_gate"] for a in aux_records]).mean().detach(),
            "gear_coupling_offdiag": torch.stack([a["coupling_offdiag"] for a in aux_records]).mean().detach(),
            "gear_phase_drive_activity": torch.stack(
                [a["phase_drive_activity"] for a in aux_records]
            ).mean().detach(),
            "gear_phase_coupling_activity": torch.stack(
                [a["phase_coupling_activity"] for a in aux_records]
            ).mean().detach(),
            "gear_coherence_entropy": torch.stack(
                [a["coherence_entropy"] for a in aux_records]
            ).mean().detach(),
            "gear_agreement": torch.stack(
                [a["gear_agreement"] for a in aux_records]
            ).mean().detach(),
            "gear_rotation_activity": torch.stack(
                [
                    a.get("rotation_activity", a["gear_agreement"] * 0.0)
                    for a in aux_records
                ]
            ).mean().detach(),
            "gear_minimum_phase_advance": torch.stack(
                [
                    a.get("minimum_phase_advance", a["fast_speed"])
                    for a in aux_records
                ]
            ).min().detach(),
            "gear_lane_balance": lane_balance.detach(),
            "gear_fast_speed": torch.stack(
                [a["fast_speed"] for a in aux_records]
            ).mean().detach(),
            "gear_slow_speed": torch.stack(
                [a["slow_speed"] for a in aux_records]
            ).mean().detach(),
            "gear_conflict": torch.stack([a["conflict"].mean() for a in aux_records]).mean().detach(),
            "gear_phase_lock_error": phase_lock.detach(),
            "gear_interbank_gate": torch.stack(
                [a["interbank_gate"] for a in aux_records]
            ).mean().detach(),
            "gear_interbank_activity": torch.stack(
                [a["interbank_activity"] for a in aux_records]
            ).mean().detach(),
            "gear_temporal_context_gate": torch.stack(
                [a["temporal_context_gate"] for a in aux_records]
            ).mean().detach(),
            "gear_active_fraction": torch.stack(
                [a["active_fraction"] for a in aux_records]
            ).mean().detach(),
        }
        return diversity, slot_usage, phase_lock, metrics

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
        if not self.config.future_horizons or self.consistency is None:
            return hidden.sum() * 0.0
        horizon = min(self.config.future_horizons)
        if hidden.shape[1] <= horizon:
            return hidden.sum() * 0.0
        prefix = hidden[:, :-horizon]
        future = hidden[:, horizon:].detach()
        mask = valid_positions[:, :-horizon] & valid_positions[:, horizon:]
        if not bool(mask.any()):
            return hidden.sum() * 0.0
        if hidden.shape[0] > 1:
            neg_future = torch.roll(future, shifts=1, dims=0)
        else:
            neg_future = torch.roll(future, shifts=1, dims=1)
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
        schedule = self._training_scales(meta)
        prediction_selector = int(meta.get("training_step", 0))
        has_auxiliary_objective = any(
            weight > 0.0
            for weight in (
                self.config.diversity_loss_weight,
                self.config.slot_usage_loss_weight,
                self.config.lane_prediction_loss_weight,
                self.config.lane_token_loss_weight,
                self.config.phase_lock_loss_weight,
                self.config.alignment_loss_weight,
                self.config.consistency_loss_weight,
            )
        )
        collect_auxiliary = (
            has_auxiliary_objective
            and schedule["auxiliary"] > 0.0
            and prediction_selector % self.config.auxiliary_loss_interval == 0
        )
        auxiliary_objective_scale = (
            schedule["auxiliary"] * self.config.auxiliary_loss_interval
            if collect_auxiliary
            else 0.0
        )
        has_future_objective = any(
            weight > 0.0
            for weight in (
                self.config.future_loss_weight,
                self.config.future_token_loss_weight,
                self.config.future_contrastive_weight,
                self.config.alignment_loss_weight,
            )
        )
        future_objective_scale = (
            schedule["future"] * self.config.future_loss_interval
            if (
                has_future_objective
                and prediction_selector % self.config.future_loss_interval == 0
            )
            else 0.0
        )
        hidden, _, aux_records = self._forward_hidden(
            tokens,
            attention_mask=attention_mask,
            return_aux=collect_auxiliary,
            gear_scale=schedule["gear"],
            phase_scale=schedule["phase"],
        )
        prediction_hidden = hidden[:, :-1]
        targets = tokens[:, 1:]
        valid = self._valid_targets(tokens, loss_mask, attention_mask)
        valid_positions = self._valid_positions(tokens, attention_mask)
        scales = loss_term_scales or {}

        language_modeling = self._language_modeling_loss(
            prediction_hidden,
            targets,
            valid,
            schedule["future"],
        )
        if auxiliary_objective_scale > 0.0 or future_objective_scale > 0.0:
            (
                future_latent,
                future_contrastive,
                future_token,
                future_error,
            ) = self._future_losses(
                hidden,
                tokens,
                valid_positions,
                prediction_selector,
            )
        else:
            future_latent = hidden.sum() * 0.0
            future_contrastive = hidden.sum() * 0.0
            future_token = hidden.sum() * 0.0
            future_error = None
        (
            gear_diversity,
            slot_usage,
            phase_lock,
            gear_metrics,
        ) = self._gear_diversity_loss(aux_records)
        lane_prediction, lane_token = (
            self._lane_prediction_loss(
                hidden,
                tokens,
                valid_positions,
                aux_records,
                prediction_selector,
            )
            if auxiliary_objective_scale > 0.0
            else (hidden.sum() * 0.0, hidden.sum() * 0.0)
        )
        alignment_calibration = self._alignment_loss(aux_records, future_error)
        consistency = (
            self._consistency_loss(hidden, valid_positions)
            if auxiliary_objective_scale > 0.0
            else hidden.sum() * 0.0
        )

        total = scales.get("language_modeling", 1.0) * language_modeling
        total = total + (
            self.config.future_loss_weight
            * future_objective_scale
            * scales.get("future_latent", 1.0)
            * future_latent
        )
        total = total + (
            self.config.future_token_loss_weight
            * future_objective_scale
            * scales.get("future_token", 1.0)
            * future_token
        )
        total = total + (
            self.config.future_contrastive_weight
            * future_objective_scale
            * scales.get("future_contrastive", 1.0)
            * future_contrastive
        )
        total = total + (
            self.config.diversity_loss_weight
            * auxiliary_objective_scale
            * scales.get("gear_diversity", 1.0)
            * gear_diversity
        )
        total = total + (
            self.config.slot_usage_loss_weight
            * auxiliary_objective_scale
            * scales.get("slot_usage", 1.0)
            * slot_usage
        )
        total = total + (
            self.config.lane_prediction_loss_weight
            * auxiliary_objective_scale
            * scales.get("lane_prediction", 1.0)
            * lane_prediction
        )
        total = total + (
            self.config.lane_token_loss_weight
            * auxiliary_objective_scale
            * scales.get("lane_token", 1.0)
            * lane_token
        )
        total = total + (
            self.config.phase_lock_loss_weight
            * auxiliary_objective_scale
            * scales.get("phase_lock", 1.0)
            * phase_lock
        )
        total = total + (
            self.config.alignment_loss_weight
            * auxiliary_objective_scale
            * scales.get("alignment_calibration", 1.0)
            * alignment_calibration
        )
        total = total + (
            self.config.consistency_loss_weight
            * auxiliary_objective_scale
            * scales.get("consistency", 1.0)
            * consistency
        )

        result = {
            "language_modeling": language_modeling,
            "future_latent": future_latent,
            "future_token": future_token,
            "future_contrastive": future_contrastive,
            "gear_diversity": gear_diversity,
            "slot_usage": slot_usage,
            "lane_prediction": lane_prediction,
            "lane_token": lane_token,
            "phase_lock": phase_lock,
            "alignment_calibration": alignment_calibration,
            "consistency": consistency,
            "future_logit_gate": (
                torch.sigmoid(self.future_residual_gate_logit).detach()
                if self.future_residual_gate_logit is not None
                else hidden.sum().detach() * 0.0
            ),
            "gear_schedule": hidden.new_tensor(schedule["gear"]).detach(),
            "phase_schedule": hidden.new_tensor(schedule["phase"]).detach(),
            "auxiliary_schedule": hidden.new_tensor(
                schedule["auxiliary"]
            ).detach(),
            "auxiliary_objective_schedule": hidden.new_tensor(
                auxiliary_objective_scale
            ).detach(),
            "future_schedule": hidden.new_tensor(schedule["future"]).detach(),
            "future_objective_schedule": hidden.new_tensor(
                future_objective_scale
            ).detach(),
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
        for index in range(max_new_tokens):
            out.append(token)
            if index + 1 == max_new_tokens:
                break
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

    @torch.no_grad()
    def gear_diagnostics(self, ids: torch.Tensor) -> dict[str, Any]:
        """Return mechanism-level diagnostics for the active parallel gear layers."""
        was_training = self.training
        self.eval()
        try:
            _, _, records = self._forward_hidden(ids, return_aux=True)
            systems = [
                block.gears
                for block in self.blocks
                if isinstance(block.gears, ParallelGearSystem)
            ]
            if isinstance(self.shared_gears, ParallelGearSystem):
                systems = [self.shared_gears]
            if not records or not systems:
                return {"active": False, "layers": 0}
            lane_usage = torch.stack(
                [record["lane_weights"].mean(dim=(0, 1)) for record in records]
            ).mean(dim=0)
            gear_usage = torch.stack(
                [record["fusion_weights"].mean(dim=(0, 1)) for record in records]
            ).mean(dim=0)
            return {
                "active": True,
                "layers": len(records),
                "speeds": systems[0].speeds().detach().cpu().tolist(),
                "bank_speeds": [
                    system.speeds().detach().cpu().tolist()
                    for system in systems
                ],
                "gear_usage": gear_usage.detach().cpu().tolist(),
                "lane_usage": lane_usage.detach().cpu().tolist(),
                "bank_lane_usage": [
                    record["lane_weights"]
                    .mean(dim=(0, 1))
                    .detach()
                    .cpu()
                    .tolist()
                    for record in records
                ],
                "bank_active_fraction": [
                    float(record["active_fraction"])
                    for record in records
                ],
                "bank_temporal_strides": [
                    int(record["temporal_stride"])
                    for record in records
                ],
                "minimum_phase_advance": float(
                    torch.stack(
                        [record["minimum_phase_advance"] for record in records]
                    ).min()
                ),
                "rotation_activity": float(
                    torch.stack(
                        [record["rotation_activity"] for record in records]
                    ).mean()
                ),
                "phase_coupling_activity": float(
                    torch.stack(
                        [record["phase_coupling_activity"] for record in records]
                    ).mean()
                ),
                "phase_lock_error": float(
                    torch.stack(
                        [record["phase_lock_error"] for record in records]
                    ).mean()
                ),
                "interbank_activity": float(
                    torch.stack(
                        [record["interbank_activity"] for record in records]
                    ).mean()
                ),
                "interbank_gate": float(
                    torch.stack(
                        [record["interbank_gate"] for record in records]
                    ).mean()
                ),
                "slot_route_entropy": float(
                    torch.stack(
                        [record["route_entropy"] for record in records]
                    ).mean()
                ),
                "lane_entropy": float(
                    torch.stack(
                        [record["coherence_entropy"] for record in records]
                    ).mean()
                ),
            }
        finally:
            self.train(was_training)

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "name": (
                "ParallelGearTransformerV5"
                if self.config.gear_system == "parallel_v5"
                else (
                    "ParallelGearTransformerV4"
                    if self.config.gear_system == "parallel_v4"
                    else "MHGTransformerLM"
                )
            ),
            "config": self.config.to_dict(),
            "parameters": {"total": sum(p.numel() for p in self.parameters())},
        }


class SimplifiedGearTransformerLM(MHGTransformerLM):
    """Single-bank V5 retaining only mechanisms with repeatable measured value."""

    def architecture_manifest(self) -> dict[str, Any]:
        manifest = super().architecture_manifest()
        manifest["name"] = "SimplifiedGearTransformerV1"
        return manifest


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


@MODELS.register("simplified_gear_transformer")
def build_simplified_gear_transformer(
    model_cfg: dict,
    vocab_size: int | None = None,
) -> SimplifiedGearTransformerLM:
    cfg = dict(model_cfg)
    cfg["gear_system"] = "parallel_v5"
    cfg["gear_layer_strategy"] = "explicit"
    cfg.setdefault("gear_layers", (0,))
    cfg["phase_coupling_enabled"] = False
    cfg["interbank_coupling_init"] = 0.0
    cfg.setdefault("gear_bank_temporal_strides", (1,))
    cfg.setdefault("gear_dim", 16)
    cfg.setdefault("gear_rotation_dims", 16)
    cfg.setdefault("diversity_loss_weight", 0.001)
    cfg.setdefault("slot_usage_loss_weight", 0.003)
    cfg.setdefault("lane_prediction_loss_weight", 0.005)
    cfg.setdefault("lane_token_loss_weight", 0.002)
    cfg.setdefault("phase_lock_loss_weight", 0.0)
    cfg.setdefault("alignment_loss_weight", 0.005)
    cfg.setdefault("consistency_loss_weight", 0.005)
    cfg.setdefault("lane_dropout", 0.05)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    config = GearTransformerConfig(**cfg)
    if len(config.selected_gear_layers()) != 1:
        raise ValueError("simplified gear transformer requires exactly one gear bank")
    if config.gear_bank_strides() != (1,):
        raise ValueError("simplified gear transformer requires temporal stride 1")
    return SimplifiedGearTransformerLM(config)
