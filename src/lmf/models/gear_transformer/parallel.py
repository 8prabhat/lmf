"""Vectorized parallel gear trains for the V4/V5 generative gear architecture.

Gears are parallel along a dedicated tensor dimension.  Recurrence remains only
across token time, because phase and memory at token ``t`` causally depend on
their values at ``t - 1``.  Each gear owns:

* a strictly positive, monotonically ordered angular velocity;
* a phase-addressed slot bank;
* a geometrically rotated latent memory subspace;
* a state-dependent recurrent write/read update.

Gears are grouped into lanes (local, phrase, semantic, discourse by default).
V5 adds a vectorized causal context carrier, sparse mechanical phase meshing,
bank specialization, and gated carriers between parallel banks at different
Transformer depths. Floors on both routing levels prevent slow lanes from being
starved before they learn useful representations.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformer.model import RMSNorm


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _normalized_entropy(probabilities: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 1:
        return probabilities.sum() * 0.0
    return (
        -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
        / math.log(count)
    )


class PositiveParallelGearClock(nn.Module):
    """Monotonic clocks whose per-token phase advance is always positive."""

    def __init__(
        self,
        dim: int,
        base_speeds: tuple[float, ...],
        max_log_speed_offset: float,
        drive_scale: float,
        modulation_scale: float,
        coupling_init: float,
        coupling_max: float,
        coupling_mask: torch.Tensor | None = None,
        speed_scale: float = 1.0,
        predictor_corrector: bool = False,
    ) -> None:
        super().__init__()
        speeds = torch.tensor(base_speeds, dtype=torch.float32)
        self.num_gears = int(speeds.numel())
        self.max_log_speed_offset = float(max_log_speed_offset)
        self.drive_scale = float(drive_scale)
        self.modulation_scale = float(modulation_scale)
        self.coupling_max = float(coupling_max)
        self.speed_scale = float(speed_scale)
        self.predictor_corrector = bool(predictor_corrector)
        self.register_buffer("base_first_log_speed", speeds[0].log())
        self.register_buffer("base_log_gaps", (speeds[:-1] / speeds[1:]).log())
        self.first_speed_offset = nn.Parameter(torch.zeros(()))
        self.gap_offsets = nn.Parameter(torch.zeros(self.num_gears - 1))
        self.phase_offsets = nn.Parameter(torch.zeros(self.num_gears))
        self.token_drive = nn.Linear(dim, self.num_gears, bias=False)
        self.context_drive = nn.Linear(dim, self.num_gears, bias=False)
        ratio = min(max(coupling_init / coupling_max, 1e-6), 1.0 - 1e-6)
        if coupling_mask is None:
            coupling_mask = torch.tril(
                torch.ones(self.num_gears, self.num_gears, dtype=torch.bool),
                diagonal=-1,
            )
        if coupling_mask.shape != (self.num_gears, self.num_gears):
            raise ValueError("coupling_mask must have shape [num_gears, num_gears]")
        self.register_buffer("coupling_mask", coupling_mask.bool())
        coupling_edges = coupling_mask.nonzero(as_tuple=False)
        self.register_buffer("coupling_targets", coupling_edges[:, 0])
        self.register_buffer("coupling_sources", coupling_edges[:, 1])
        if len(coupling_edges):
            self.coupling_logits = nn.Parameter(
                torch.full((len(coupling_edges),), _logit(ratio))
            )
            self.coupling_phase = nn.Parameter(
                torch.zeros(len(coupling_edges))
            )
        else:
            self.register_buffer("coupling_logits", torch.empty(0))
            self.register_buffer("coupling_phase", torch.empty(0))
        self.register_buffer(
            "source_counts",
            coupling_mask.sum(dim=-1).clamp_min(1).to(torch.float32),
        )

    def speeds(self) -> torch.Tensor:
        first = self.base_first_log_speed.float() + (
            self.max_log_speed_offset * torch.tanh(self.first_speed_offset.float())
        )
        gaps = self.base_log_gaps.float() * torch.exp(
            self.max_log_speed_offset * torch.tanh(self.gap_offsets.float())
        )
        logs = torch.cat([first[None], first - gaps.cumsum(dim=0)], dim=0)
        return logs.exp() * self.speed_scale

    def _coupling(
        self,
        source_phase: torch.Tensor,
        speeds: torch.Tensor,
        coupling_scale: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.coupling_logits.numel() == 0:
            zero = source_phase.sum() * 0.0
            return torch.zeros_like(source_phase), zero
        target = self.coupling_targets
        source = self.coupling_sources
        ratios = speeds.index_select(0, target) / speeds.index_select(0, source)
        error = (
            ratios * source_phase.index_select(-1, source)
            - source_phase.index_select(-1, target)
            + self.coupling_phase.float()
        )
        scale = torch.as_tensor(
            coupling_scale,
            dtype=torch.float32,
            device=source_phase.device,
        )
        strength = (
            scale
            * self.coupling_max
            * torch.sigmoid(self.coupling_logits.float())
        )
        contributions = strength * error.sin()
        coupling = torch.zeros_like(source_phase)
        coupling = coupling.scatter_add(
            -1,
            target.view(*([1] * (source_phase.ndim - 1)), -1).expand_as(
                contributions
            ),
            contributions,
        )
        coupling = coupling / self.source_counts
        lock_error = (1.0 - error.cos()).mean()
        return coupling, lock_error.mean()

    def initial_phase(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.phase_offsets.float()[None].expand(batch, -1).to(device)

    def step(
        self,
        token: torch.Tensor,
        context: torch.Tensor,
        phase: torch.Tensor,
        phase_mechanism_scale: float | torch.Tensor = 1.0,
        coupling_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        speeds = self.speeds().to(device=token.device)
        drive = self.drive_scale * (
            self.token_drive(token) + self.context_drive(context)
        ).float()
        coupling, lock_error = self._coupling(
            phase,
            speeds,
            coupling_scale,
        )
        scale = torch.as_tensor(
            phase_mechanism_scale,
            dtype=torch.float32,
            device=token.device,
        )
        modulation = self.modulation_scale * torch.tanh(
            scale * (drive + coupling)
        )
        # Multiplicative modulation preserves a strictly positive phase advance.
        delta = speeds[None] * modulation.exp()
        next_phase = phase + delta
        return next_phase, delta, {
            "phase_drive_activity": drive.abs().mean(),
            "phase_coupling_activity": coupling.abs().mean(),
            "minimum_phase_advance": delta.min(),
            "phase_lock_error": lock_error,
            "predictor_phase": next_phase,
        }

    def sequence(
        self,
        tokens: torch.Tensor,
        contexts: torch.Tensor,
        initial_phase: torch.Tensor,
        positions: torch.Tensor,
        phase_mechanism_scale: float | torch.Tensor = 1.0,
        coupling_scale: float | torch.Tensor = 1.0,
        initial_predictor_phase: torch.Tensor | None = None,
        step_sizes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Vectorized positive phase updates with causal predictor meshing.

        In V5, a drive-only predictor phase is an associative cumulative scan.
        Sparse gear coupling reads the previous predictor phase, then corrects
        the positive phase advance used by slot routing and memory rotation.
        Keeping predictor and corrected phases separately makes full-sequence
        and cached decoding exactly equivalent without a token-wise Python loop.
        """
        batch, length, _ = tokens.shape
        speeds = self.speeds().to(device=tokens.device)
        drive = self.drive_scale * (
            self.token_drive(tokens) + self.context_drive(contexts)
        ).float()
        scale = torch.as_tensor(
            phase_mechanism_scale,
            dtype=torch.float32,
            device=tokens.device,
        )
        if step_sizes is None:
            step_sizes = torch.ones(
                length,
                dtype=torch.float32,
                device=tokens.device,
            )
        step_sizes = step_sizes.to(
            dtype=torch.float32,
            device=tokens.device,
        )
        if self.predictor_corrector:
            predictor_start = (
                initial_phase
                if initial_predictor_phase is None
                else initial_predictor_phase
            )
            predictor_modulation = self.modulation_scale * torch.tanh(
                scale * drive
            )
            predictor_delta = (
                speeds[None, None]
                * predictor_modulation.exp()
                * step_sizes[None, :, None]
            )
            predictor_phase = (
                predictor_start[:, None] + predictor_delta.cumsum(dim=1)
            )
            source_phase = torch.cat(
                [predictor_start[:, None], predictor_phase[:, :-1]],
                dim=1,
            )
        else:
            steps = positions.to(dtype=torch.float32, device=tokens.device)
            source_phase = (
                self.phase_offsets.float()[None, None]
                + steps[None, :, None] * speeds[None, None]
            ).expand(batch, -1, -1)
            predictor_phase = source_phase
        coupling, lock_error = self._coupling(
            source_phase,
            speeds,
            coupling_scale,
        )
        modulation = self.modulation_scale * torch.tanh(
            scale * (drive + coupling)
        )
        delta = (
            speeds[None, None]
            * modulation.exp()
            * step_sizes[None, :, None]
        )
        phase = initial_phase[:, None] + delta.cumsum(dim=1)
        return phase, delta, {
            "phase_drive_activity": drive.abs().mean(),
            "phase_coupling_activity": coupling.abs().mean(),
            "minimum_phase_advance": delta.min(),
            "phase_lock_error": lock_error,
            "predictor_phase": predictor_phase[:, -1],
        }


def _sparse_mechanical_mask(
    lane_sizes: tuple[int, ...],
    topology: str,
) -> torch.Tensor:
    """Build a directed fast-to-slow gear mesh without dense dilution."""
    num_gears = sum(lane_sizes)
    if topology == "dense_lower":
        return torch.tril(
            torch.ones(num_gears, num_gears, dtype=torch.bool),
            diagonal=-1,
        )
    if topology != "adjacent_anchor":
        raise ValueError(f"unknown phase coupling topology: {topology}")
    mask = torch.zeros(num_gears, num_gears, dtype=torch.bool)
    lane_starts: list[int] = []
    cursor = 0
    for count in lane_sizes:
        lane_starts.append(cursor)
        for gear in range(cursor + 1, cursor + count):
            mask[gear, gear - 1] = True
        cursor += count
    for lane in range(1, len(lane_starts)):
        current = lane_starts[lane]
        previous_anchor = lane_starts[lane - 1]
        mask[current, previous_anchor] = True
        if current > 0:
            mask[current, current - 1] = True
    return mask


class ParallelGearSystem(nn.Module):
    """Parallel multi-lane gear controller with recurrent rotated memories."""

    def __init__(
        self,
        config: Any,
        *,
        bank_index: int = 0,
        bank_count: int = 1,
        speed_scale: float = 1.0,
        horizon_scale: float = 1.0,
        temporal_stride: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.bank_index = int(bank_index)
        self.bank_count = int(bank_count)
        self.horizon_scale = float(horizon_scale)
        self.temporal_stride = int(temporal_stride)
        if self.temporal_stride < 1:
            raise ValueError("temporal_stride must be positive")
        self.is_v5 = config.gear_system == "parallel_v5"
        self.num_gears = int(config.num_gears)
        self.dim = int(config.gear_dim)
        self.rotation_dims = min(int(config.gear_rotation_dims), self.dim)
        self.rotation_dims -= self.rotation_dims % 2
        self.lane_sizes = tuple(int(v) for v in config.gear_lane_sizes)
        self.num_lanes = len(self.lane_sizes)
        self.max_slots = max(config.gear_slots)
        self.phase_harmonics = int(config.phase_harmonics)
        self.phase_dim = 2 * self.phase_harmonics

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
        coupling_mask = (
            _sparse_mechanical_mask(
                self.lane_sizes,
                config.phase_coupling_topology if self.is_v5 else "dense_lower",
            )
            if config.phase_coupling_enabled
            else torch.zeros(
                self.num_gears,
                self.num_gears,
                dtype=torch.bool,
            )
        )
        self.clock = PositiveParallelGearClock(
            self.dim,
            config.gear_speeds,
            config.max_log_speed_offset,
            config.phase_drive_scale,
            config.phase_modulation_scale,
            config.phase_coupling_init,
            config.phase_coupling_max,
            coupling_mask=coupling_mask,
            speed_scale=speed_scale,
            predictor_corrector=self.is_v5,
        )

        self.context_token = nn.Linear(self.dim, self.dim, bias=False)
        self.context_bank = (
            nn.Linear(self.dim, self.dim, bias=False)
            if self.bank_index > 0
            else None
        )
        self.context_retention_logit = nn.Parameter(
            torch.tensor(_logit(config.temporal_context_retention))
        )
        self.interbank_gate_logit = (
            nn.Parameter(torch.tensor(_logit(config.interbank_coupling_init)))
            if self.bank_index > 0
            else None
        )
        self.emits_carrier = self.bank_index < self.bank_count - 1
        self.carrier_norm = RMSNorm(self.dim) if self.emits_carrier else None
        self.carrier_out = (
            nn.Linear(self.dim, self.dim, bias=False)
            if self.emits_carrier
            else None
        )

        lane_ids: list[int] = []
        for lane, count in enumerate(self.lane_sizes):
            lane_ids.extend([lane] * count)
        self.register_buffer("gear_lane_ids", torch.tensor(lane_ids, dtype=torch.long))
        lane_mask = torch.zeros(self.num_lanes, self.num_gears, dtype=torch.bool)
        for gear, lane in enumerate(lane_ids):
            lane_mask[lane, gear] = True
        self.register_buffer("lane_mask", lane_mask)
        self.register_buffer(
            "lane_counts",
            lane_mask.sum(dim=-1).clamp_min(1).to(torch.float32),
        )

        slot_mask = torch.zeros(self.num_gears, self.max_slots, dtype=torch.bool)
        preferred_parts = []
        slot_offsets = [0]
        for gear, slots in enumerate(config.gear_slots):
            slot_mask[gear, :slots] = True
            preferred_parts.append(
                torch.arange(slots, dtype=torch.float32)
                * (2.0 * math.pi / slots)
            )
            slot_offsets.append(slot_offsets[-1] + int(slots))
        self.register_buffer("slot_mask", slot_mask)
        self.register_buffer(
            "slot_offsets",
            torch.tensor(slot_offsets, dtype=torch.long),
        )
        self.slots = nn.Parameter(
            torch.empty(sum(config.gear_slots), self.dim)
        )
        self.preferred_phase = nn.Parameter(torch.cat(preferred_parts))
        self.slot_concentration_raw = nn.Parameter(torch.zeros(self.num_gears))
        nn.init.normal_(self.slots, mean=0.0, std=0.02)

        self.input_projection = nn.Linear(
            self.dim, self.num_gears * self.dim, bias=False
        )
        self.context_projection = nn.Linear(
            self.dim, self.num_gears * self.dim, bias=False
        )
        self.slot_query_projection = nn.Linear(
            self.dim, self.num_gears * self.dim, bias=False
        )
        self.phase_query = nn.Parameter(
            torch.empty(self.num_gears, self.phase_dim, self.dim)
        )
        self.gear_roles = nn.Parameter(torch.empty(self.num_gears, self.dim))
        nn.init.normal_(self.phase_query, mean=0.0, std=0.02)
        nn.init.normal_(self.gear_roles, mean=0.0, std=0.02)

        self.memory_candidate = nn.Parameter(
            torch.empty(self.num_gears, 3 * self.dim, self.dim)
        )
        self.memory_update = nn.Parameter(
            torch.empty(self.num_gears, 3 * self.dim, self.dim)
        )
        self.memory_reset = nn.Parameter(
            torch.empty(self.num_gears, 3 * self.dim, self.dim)
        )
        self.memory_bias = nn.Parameter(torch.zeros(self.num_gears, self.dim))
        update_rates = torch.tensor(config.gear_update_rates, dtype=torch.float32)
        self.update_bias = nn.Parameter(
            torch.tensor([_logit(float(rate)) for rate in update_rates])[:, None]
            .expand(-1, self.dim)
            .clone()
        )
        self.reset_bias = nn.Parameter(torch.zeros(self.num_gears, self.dim))
        for parameter in (
            self.memory_candidate,
            self.memory_update,
            self.memory_reset,
        ):
            for gear in parameter:
                nn.init.xavier_uniform_(gear)

        self.aperture_gain = nn.Parameter(torch.full((self.num_gears,), 2.0))
        self.aperture_bias = nn.Parameter(torch.full((self.num_gears,), -0.5))
        self.state_norm = RMSNorm(self.dim)
        self.read_gate = nn.Linear(3 * self.dim, self.dim, bias=True)
        self.output_projection = nn.Parameter(
            torch.empty(self.num_gears, 2 * self.dim, self.dim)
        )
        nn.init.xavier_uniform_(self.output_projection)
        nn.init.constant_(self.read_gate.bias, _logit(config.gear_read_gate_init))

        self.gear_context_score = nn.Linear(
            self.dim, self.num_gears, bias=False
        )
        self.gear_state_score = nn.Parameter(
            torch.empty(self.num_gears, self.dim)
        )
        self.gear_message_score = nn.Parameter(
            torch.empty(self.num_gears, self.dim)
        )
        nn.init.normal_(self.gear_state_score, std=0.02)
        nn.init.normal_(self.gear_message_score, std=0.02)

        self.lane_norm = RMSNorm(self.dim)
        self.lane_q = nn.Linear(self.dim, self.dim, bias=False)
        self.lane_k = nn.Linear(self.dim, self.dim, bias=False)
        self.lane_v = nn.Linear(self.dim, self.dim, bias=False)
        self.lane_out = nn.Linear(self.dim, self.dim, bias=False)
        self.lane_mixing_gate_logit = nn.Parameter(
            torch.tensor(_logit(config.lane_mixing_init))
        )
        self.lane_context_score = nn.Linear(
            self.dim, self.num_lanes, bias=False
        )
        self.lane_value_score = nn.Parameter(
            torch.empty(self.num_lanes, self.dim)
        )
        self.lane_roles = nn.Parameter(torch.empty(self.num_lanes, self.dim))
        nn.init.normal_(self.lane_value_score, std=0.02)
        nn.init.normal_(self.lane_roles, std=0.02)
        preferred_lane = (
            0.0
            if self.bank_count <= 1
            else self.bank_index * (self.num_lanes - 1) / (self.bank_count - 1)
        )
        lane_positions = torch.arange(self.num_lanes, dtype=torch.float32)
        lane_prior = -(
            float(config.bank_specialization_strength)
            * (lane_positions - preferred_lane).abs()
        )
        self.register_buffer("bank_lane_prior", lane_prior)

        self.fused_norm = RMSNorm(self.dim)
        self.fused_out = nn.Linear(self.dim, self.dim, bias=False)
        self.residual_gate_logit = nn.Parameter(
            torch.tensor(_logit(config.gear_residual_init))
        )
        self.risk_head = (
            nn.Sequential(
                nn.Linear(self.dim + 1, max(16, self.dim // 2), bias=False),
                nn.SiLU(),
                nn.Linear(max(16, self.dim // 2), 1, bias=False),
            )
            if config.alignment_loss_weight > 0.0
            else None
        )
        self.dropout = nn.Dropout(config.dropout)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Migrate pre-audit dense/padded V5 tensors during checkpoint load."""
        slots_key = prefix + "slots"
        slots = state_dict.get(slots_key)
        if slots is not None and slots.ndim == 3:
            state_dict[slots_key] = torch.cat(
                [
                    slots[gear, : int(count)]
                    for gear, count in enumerate(self.config.gear_slots)
                ],
                dim=0,
            )
        preferred_key = prefix + "preferred_phase"
        preferred = state_dict.get(preferred_key)
        if preferred is not None and preferred.ndim == 2:
            state_dict[preferred_key] = torch.cat(
                [
                    preferred[gear, : int(count)]
                    for gear, count in enumerate(self.config.gear_slots)
                ],
                dim=0,
            )
        target = self.clock.coupling_targets.detach().cpu()
        source = self.clock.coupling_sources.detach().cpu()
        for name in ("coupling_logits", "coupling_phase"):
            key = prefix + "clock." + name
            value = state_dict.get(key)
            if value is not None and value.ndim == 2:
                state_dict[key] = value[target, source]
        if self.context_bank is None:
            state_dict.pop(prefix + "context_bank.weight", None)
            state_dict.pop(prefix + "interbank_gate_logit", None)
        if self.carrier_out is None:
            state_dict.pop(prefix + "carrier_norm.weight", None)
            state_dict.pop(prefix + "carrier_out.weight", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _component_scale(
        component_scales: dict[str, float] | None,
        name: str,
    ) -> float:
        if component_scales is None:
            return 1.0
        return float(component_scales.get(name, 1.0))

    @staticmethod
    def _affine_scan(
        retention: torch.Tensor,
        injection: torch.Tensor,
        initial: torch.Tensor,
    ) -> torch.Tensor:
        """Chunk-stable closed-form scan for ``state = a * previous + b``."""
        outputs = []
        current = initial
        chunk_size = 32
        for start in range(0, retention.shape[1], chunk_size):
            stop = min(start + chunk_size, retention.shape[1])
            chunk_retention = retention[:, start:stop].float()
            chunk_injection = injection[:, start:stop].float()
            prefix = chunk_retention.cumprod(dim=1)
            safe_prefix = prefix.clamp_min(1e-20)
            states = prefix * (
                current.float()[:, None]
                + (chunk_injection / safe_prefix).cumsum(dim=1)
            )
            outputs.append(states.to(dtype=injection.dtype))
            current = states[:, -1]
        return torch.cat(outputs, dim=1)

    def _causal_context(
        self,
        token: torch.Tensor,
        bank_carrier: torch.Tensor | None,
        initial: torch.Tensor,
        temporal_scale: float,
        interbank_scale: float,
        step_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        bank_input = torch.zeros_like(token) if bank_carrier is None else bank_carrier
        if self.context_bank is None or self.interbank_gate_logit is None:
            interbank_gate = token.new_zeros(())
            interbank_signal = torch.zeros_like(token)
        else:
            interbank_gate = (
                torch.sigmoid(self.interbank_gate_logit).to(dtype=token.dtype)
                * interbank_scale
            )
            interbank_signal = interbank_gate * self.context_bank(bank_input)
        proposal = torch.tanh(
            self.context_token(token) + interbank_signal
        )
        base_retention = torch.sigmoid(
            self.context_retention_logit
        ).to(dtype=token.dtype)
        retention_value = (
            1.0
            - temporal_scale
            * (1.0 - base_retention.pow(step_sizes.to(token.dtype)))
        )
        retention = retention_value[None, :, None].expand(
            token.shape[0], -1, 1
        )
        states = self._affine_scan(
            retention,
            (1.0 - retention) * proposal,
            initial,
        )
        previous = torch.cat([initial[:, None], states[:, :-1]], dim=1)
        return previous, states, {
            "temporal_context_gate": (1.0 - retention_value).mean(),
            "interbank_gate": interbank_gate,
            "interbank_activity": (
                interbank_gate * bank_input
            ).pow(2).mean().sqrt(),
            "interbank_signal": interbank_signal,
        }

    def speeds(self) -> torch.Tensor:
        return self.clock.speeds()

    def speed_separation_loss(self) -> torch.Tensor:
        logs = self.speeds().log()
        return torch.exp(-(logs[:-1] - logs[1:])).mean()

    def slot_diversity_loss(self) -> torch.Tensor:
        total = self.slots.sum() * 0.0
        terms = 0
        for gear in range(self.num_gears):
            start = int(self.slot_offsets[gear])
            stop = int(self.slot_offsets[gear + 1])
            slots = F.normalize(self.slots[start:stop], dim=-1)
            if slots.shape[0] <= 1:
                continue
            similarity = slots @ slots.T
            selected = ~torch.eye(
                slots.shape[0], dtype=torch.bool, device=slots.device
            )
            total = total + similarity[selected].pow(2).mean()
            terms += 1
        return total / max(terms, 1)

    def _phase_features(
        self,
        phase: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        parts = []
        for harmonic in range(1, self.phase_harmonics + 1):
            angle = harmonic * phase
            parts.extend([angle.sin(), angle.cos()])
        return torch.stack(parts, dim=-1).to(dtype=dtype)

    def _slot_messages(
        self,
        token: torch.Tensor,
        phase: torch.Tensor,
        phase_mechanism_scale: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.slot_query_projection(token).reshape(
            *token.shape[:-1], self.num_gears, self.dim
        )
        phase_features = self._phase_features(phase, token.dtype)
        scale = torch.as_tensor(
            phase_mechanism_scale,
            dtype=token.dtype,
            device=token.device,
        )
        query = query + scale * torch.einsum(
            "...gp,gpd->...gd", phase_features, self.phase_query
        )
        phase_scale = torch.as_tensor(
            phase_mechanism_scale,
            dtype=torch.float32,
            device=token.device,
        )
        messages = []
        routing_rows = []
        for gear in range(self.num_gears):
            start = int(self.slot_offsets[gear])
            stop = int(self.slot_offsets[gear + 1])
            slots = self.slots[start:stop]
            content_scores = torch.einsum(
                "...d,kd->...k", query[..., gear, :], slots
            ) / math.sqrt(self.dim)
            concentration = F.softplus(
                self.slot_concentration_raw[gear].float()
            )
            phase_scores = (
                concentration
                * (
                    phase[..., gear, None]
                    - self.preferred_phase[start:stop].float()
                ).cos()
                * phase_scale
            ).to(dtype=token.dtype)
            routing = (content_scores + phase_scores).softmax(dim=-1)
            messages.append(torch.einsum("...k,kd->...d", routing, slots))
            routing_rows.append(
                F.pad(routing, (0, self.max_slots - routing.shape[-1]))
            )
        return torch.stack(messages, dim=-2), torch.stack(routing_rows, dim=-2)

    def _rotate_memory(
        self,
        memory: torch.Tensor,
        phase_advance: torch.Tensor,
        rotation_scale: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rotation_dims == 0:
            return memory, memory.sum() * 0.0
        scale = torch.as_tensor(
            rotation_scale,
            dtype=memory.dtype,
            device=memory.device,
        )
        pairs = memory[..., : self.rotation_dims].reshape(
            *memory.shape[:-1], self.rotation_dims // 2, 2
        )
        angle = (phase_advance * scale).to(dtype=memory.dtype)[..., None]
        cosine = angle.cos()
        sine = angle.sin()
        x, y = pairs.unbind(dim=-1)
        rotated_pairs = torch.stack(
            [cosine * x - sine * y, sine * x + cosine * y],
            dim=-1,
        )
        rotated_prefix = rotated_pairs.flatten(start_dim=-2)
        rotated = torch.cat(
            [rotated_prefix, memory[..., self.rotation_dims :]],
            dim=-1,
        )
        activity = (rotated_prefix - memory[..., : self.rotation_dims]).pow(2).mean().sqrt()
        return rotated, activity

    def _affine_rotation_scan(
        self,
        retention: torch.Tensor,
        angle: torch.Tensor,
        injection: torch.Tensor,
        initial: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked closed-form scan of ``s = a R(theta) s_prev + b``.

        Rotated pairs are represented as complex values, turning each 2-D
        rotation into multiplication by ``a * exp(i theta)``. Short chunks
        prevent cumulative products from underflowing on long contexts.
        """
        outputs = []
        current = initial.float()
        chunk_size = 16
        for start in range(0, retention.shape[1], chunk_size):
            stop = min(start + chunk_size, retention.shape[1])
            r = retention[:, start:stop].float()
            theta = angle[:, start:stop].float()
            b = injection[:, start:stop].float()
            parts = []
            if self.rotation_dims:
                pair_b = b[..., : self.rotation_dims].reshape(
                    *b.shape[:-1],
                    self.rotation_dims // 2,
                    2,
                )
                pair_initial = current[..., : self.rotation_dims].reshape(
                    *current.shape[:-1],
                    self.rotation_dims // 2,
                    2,
                )
                complex_b = torch.complex(pair_b[..., 0], pair_b[..., 1])
                complex_initial = torch.complex(
                    pair_initial[..., 0],
                    pair_initial[..., 1],
                )
                coefficient = torch.polar(
                    r.squeeze(-1),
                    theta,
                )
                prefix = coefficient.cumprod(dim=1)
                magnitude = prefix.abs().clamp_min(1e-20)
                safe_prefix = torch.polar(magnitude, torch.angle(prefix))
                complex_states = prefix[..., None] * (
                    complex_initial[:, None]
                    + (complex_b / safe_prefix[..., None]).cumsum(dim=1)
                )
                rotated = torch.stack(
                    [complex_states.real, complex_states.imag],
                    dim=-1,
                ).flatten(start_dim=-2)
                parts.append(rotated)
            if self.rotation_dims < self.dim:
                suffix_b = b[..., self.rotation_dims :]
                prefix_r = r.cumprod(dim=1)
                safe_prefix_r = prefix_r.clamp_min(1e-20)
                suffix = prefix_r * (
                    current[..., self.rotation_dims :][:, None]
                    + (suffix_b / safe_prefix_r).cumsum(dim=1)
                )
                parts.append(suffix)
            states = torch.cat(parts, dim=-1)
            outputs.append(states.to(dtype=injection.dtype))
            current = states[:, -1]
        return torch.cat(outputs, dim=1)

    def _memory_sequence(
        self,
        token_gears: torch.Tensor,
        slot_message: torch.Tensor,
        initial_memory: torch.Tensor,
        initial_proxy: torch.Tensor,
        context_gears: torch.Tensor,
        phase: torch.Tensor,
        phase_advance: torch.Tensor,
        phase_mechanism_scale: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        scale = torch.as_tensor(
            phase_mechanism_scale,
            dtype=token_gears.dtype,
            device=token_gears.device,
        )
        aperture = torch.sigmoid(
            scale
            * (
                self.aperture_gain.to(dtype=token_gears.dtype)[None, None]
                * phase.cos().to(dtype=token_gears.dtype)
                + self.aperture_bias.to(dtype=token_gears.dtype)[None, None]
            )
        )[..., None]
        base_inputs = torch.cat(
            [token_gears, slot_message, context_gears],
            dim=-1,
        )
        update_bias = self.update_bias.mean(dim=-1)[None, None, :, None]

        def build_scan(
            previous_proxy: torch.Tensor,
            scan_initial: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reset = torch.sigmoid(
                torch.einsum(
                    "btgi,gid->btgd", base_inputs, self.memory_reset
                )
                + self.reset_bias[None, None]
            )
            candidate_inputs = torch.cat(
                [token_gears, slot_message, reset * previous_proxy],
                dim=-1,
            )
            candidate = torch.tanh(
                torch.einsum(
                    "btgi,gid->btgd",
                    candidate_inputs,
                    self.memory_candidate,
                )
                + self.memory_bias[None, None]
            )
            update_logits = torch.einsum(
                "btgi,gid->btgd", base_inputs, self.memory_update
            ).mean(dim=-1, keepdim=True)
            write = torch.sigmoid(update_logits + update_bias)
            write = write * (0.25 + 0.75 * aperture)
            retention = 1.0 - write
            angle = phase_advance.to(token_gears.dtype) * scale
            states = self._affine_rotation_scan(
                retention,
                angle,
                write * candidate,
                scan_initial,
            )
            return states, write

        zero_proxy = torch.zeros_like(token_gears)
        first_states, _ = build_scan(zero_proxy, initial_proxy)
        previous_proxy = torch.cat(
            [initial_proxy[:, None], first_states[:, :-1]],
            dim=1,
        )
        states, write = build_scan(previous_proxy, initial_memory)
        rotated_proxy, rotation_activity = self._rotate_memory(
            previous_proxy,
            phase_advance,
            phase_mechanism_scale,
        )
        rotation_activity = (
            rotated_proxy[..., : self.rotation_dims]
            - previous_proxy[..., : self.rotation_dims]
        ).pow(2).mean().sqrt()
        return states, aperture, {
            "write_activity": write.mean(),
            "rotation_activity": rotation_activity,
            "proxy_final": first_states[:, -1],
        }

    def _gear_outputs(
        self,
        token_gears: torch.Tensor,
        memory: torch.Tensor,
        slot_message: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        read_context = torch.cat(
            [
                token_gears,
                memory,
                slot_message,
            ],
            dim=-1,
        )
        read_gate = torch.sigmoid(self.read_gate(read_context))
        values = torch.einsum(
            "...gi,gid->...gd",
            torch.cat([memory, slot_message], dim=-1),
            self.output_projection,
        )
        return read_gate * values, read_gate

    def _floor_masked_softmax(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        floor: float,
    ) -> torch.Tensor:
        masked = scores.masked_fill(~mask, float("-inf"))
        weights = masked.softmax(dim=-1)
        uniform = mask.to(dtype=scores.dtype)
        uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (1.0 - floor) * weights + floor * uniform

    def _fuse_lanes(
        self,
        token: torch.Tensor,
        outputs: torch.Tensor,
        memory: torch.Tensor,
        slot_message: torch.Tensor,
        lane_mixing_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gear_scores = (
            self.gear_context_score(token)
            + (memory * self.gear_state_score[None]).sum(dim=-1)
            + (slot_message * self.gear_message_score[None]).sum(dim=-1)
        )
        expanded_scores = gear_scores.unsqueeze(-2).expand(
            *gear_scores.shape[:-1], self.num_lanes, self.num_gears
        )
        lane_mask = self.lane_mask.view(
            *([1] * (expanded_scores.ndim - 2)),
            self.num_lanes,
            self.num_gears,
        ).expand_as(expanded_scores)
        within_lane = self._floor_masked_softmax(
            expanded_scores,
            lane_mask,
            self.config.gear_routing_floor,
        )
        lane_values = torch.einsum(
            "...lg,...gd->...ld", within_lane, outputs
        )
        lane_values = lane_values + self.lane_roles

        normalized = self.lane_norm(lane_values)
        q = self.lane_q(normalized)
        k = self.lane_k(normalized)
        v = self.lane_v(normalized)
        attention_logits = torch.einsum(
            "...ld,...md->...lm", q, k
        ) / math.sqrt(self.dim)
        attention = attention_logits.softmax(dim=-1)
        mixed = torch.einsum("...lm,...md->...ld", attention, v)
        mixing_gate = torch.sigmoid(self.lane_mixing_gate_logit).to(
            dtype=token.dtype
        ) * lane_mixing_scale
        lane_values = lane_values + (
            mixing_gate * self.lane_out(mixed)
        )

        lane_scores = (
            self.lane_context_score(token)
            + (lane_values * self.lane_value_score).sum(dim=-1)
            + self.bank_lane_prior.to(dtype=token.dtype)
        )
        if self.training and self.config.lane_dropout > 0.0:
            keep = torch.rand_like(lane_scores).ge(self.config.lane_dropout)
            keep[..., 0] = True
            lane_scores = lane_scores.masked_fill(~keep, float("-inf"))
        lane_scores = lane_scores / self.config.routing_temperature
        lane_weights = lane_scores.softmax(dim=-1)
        lane_weights = (
            (1.0 - self.config.lane_routing_floor) * lane_weights
            + self.config.lane_routing_floor / self.num_lanes
        )
        fused = torch.einsum("...l,...ld->...d", lane_weights, lane_values)
        global_gear_weights = torch.einsum(
            "...l,...lg->...g", lane_weights, within_lane
        )

        normalized_lanes = F.normalize(lane_values, dim=-1)
        consensus = F.normalize(normalized_lanes.mean(dim=-2), dim=-1)
        agreement = (
            normalized_lanes * consensus.unsqueeze(-2)
        ).sum(dim=-1)
        agreement = (
            lane_weights * ((agreement + 1.0) * 0.5)
        ).sum(dim=-1)
        off_diagonal = ~torch.eye(
            self.num_lanes, dtype=torch.bool, device=token.device
        )
        return fused, {
            "fusion_weights": global_gear_weights,
            "lane_weights": lane_weights,
            "lane_values": lane_values,
            "lane_attention_entropy": _normalized_entropy(
                attention, self.num_lanes
            ).mean(),
            "lane_attention_offdiag": attention[..., off_diagonal].mean(),
            "lane_mixing_gate": mixing_gate,
            "gear_agreement": agreement.mean(),
        }

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
        residual_scale: float | torch.Tensor = 1.0,
        phase_scale: float | torch.Tensor = 1.0,
        bank_carrier: torch.Tensor | None = None,
        component_scales: dict[str, float] | None = None,
        collect_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, dict[str, torch.Tensor]]:
        batch, full_length, _ = hidden.shape
        if positions is None:
            positions = torch.arange(full_length, device=hidden.device)
        full_positions = positions
        active_mask = (positions % self.temporal_stride) == 0
        held_residual = (
            None if cache is None else cache.get("held_residual")
        )
        held_carrier = (
            None if cache is None else cache.get("held_carrier")
        )
        if not bool(active_mask.any()):
            residual = (
                torch.zeros_like(hidden)
                if held_residual is None
                else held_residual[:, None].expand(-1, full_length, -1)
            )
            carrier = (
                torch.zeros_like(hidden)
                if held_carrier is None
                else held_carrier[:, None].expand(-1, full_length, -1)
            )
            next_cache = dict(cache or {}) if use_cache else None
            return residual, next_cache, {
                "carrier": carrier,
                "active_fraction": hidden.new_tensor(0.0),
                "temporal_stride": hidden.new_tensor(
                    float(self.temporal_stride)
                ),
            }

        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
        hidden = hidden.index_select(1, active_indices)
        if bank_carrier is not None:
            bank_carrier = bank_carrier.index_select(1, active_indices)
        positions = positions.index_select(0, active_indices)
        previous_position = (
            int(cache.get("last_position", -1))
            if cache is not None
            else -1
        )
        step_sizes = torch.diff(
            torch.cat(
                [
                    positions.new_tensor([previous_position]),
                    positions,
                ]
            )
        ).clamp_min(1)

        gear_input = self.down(hidden)
        batch, length, _ = gear_input.shape
        if cache is None:
            initial_phase = self.clock.initial_phase(batch, hidden.device)
            initial_memory = torch.zeros(
                batch,
                self.num_gears,
                self.dim,
                dtype=gear_input.dtype,
                device=gear_input.device,
            )
            initial_proxy = torch.zeros_like(initial_memory)
            previous_context = torch.zeros(
                batch,
                self.dim,
                dtype=gear_input.dtype,
                device=gear_input.device,
            )
            initial_predictor_phase = initial_phase
        else:
            initial_phase = cache["phase"].float()
            initial_memory = cache["memory"].to(dtype=gear_input.dtype)
            initial_proxy = cache["proxy"].to(dtype=gear_input.dtype)
            previous_context = cache["context"].to(dtype=gear_input.dtype)
            initial_predictor_phase = cache.get(
                "predictor_phase",
                initial_phase,
            ).float()

        temporal_scale = self._component_scale(
            component_scales, "temporal_context"
        )
        interbank_scale = self._component_scale(
            component_scales, "interbank_coupling"
        )
        contexts, context_states, context_stats = self._causal_context(
            gear_input,
            None if bank_carrier is None else self.down(bank_carrier),
            previous_context,
            temporal_scale,
            interbank_scale,
            step_sizes,
        )
        phase, advance, clock_stats = self.clock.sequence(
            gear_input,
            contexts,
            initial_phase,
            positions,
            phase_scale,
            self._component_scale(component_scales, "phase_coupling"),
            initial_predictor_phase,
            step_sizes,
        )
        slot_messages, routing = self._slot_messages(
            gear_input,
            phase,
            phase_scale,
        )
        token_gears = self.input_projection(gear_input).reshape(
            batch, length, self.num_gears, self.dim
        ) + self.gear_roles[None, None]
        token_gears = token_gears + context_stats[
            "interbank_signal"
        ].unsqueeze(-2)
        context_gears = self.context_projection(contexts).reshape(
            batch, length, self.num_gears, self.dim
        )
        rotation_scale = self._component_scale(component_scales, "rotation")
        # During scheduled training, phase_scale stages the whole mechanical
        # path. During an explicit "phase" component ablation, retain base-speed
        # geometric rotation so phase conditioning and rotation are measured
        # independently rather than reporting the same intervention twice.
        if component_scales is None or "phase" not in component_scales:
            rotation_scale = rotation_scale * phase_scale
        raw_memory, aperture, memory_stats = self._memory_sequence(
            token_gears,
            slot_messages,
            initial_memory,
            initial_proxy,
            context_gears,
            phase,
            advance,
            rotation_scale,
        )
        memory_states = self.state_norm(raw_memory)
        gear_outputs, read_gate = self._gear_outputs(
            token_gears,
            memory_states,
            slot_messages,
        )
        fused_sequence, fusion = self._fuse_lanes(
            gear_input,
            gear_outputs,
            memory_states,
            slot_messages,
            self._component_scale(component_scales, "lane_mixing"),
        )
        fusion_weights = fusion["fusion_weights"]
        lane_weights = fusion["lane_weights"]
        residual_gate = torch.sigmoid(self.residual_gate_logit).to(hidden.dtype)
        bank_scale = self._component_scale(
            component_scales, f"bank_{self.bank_index}"
        )
        scale = torch.as_tensor(
            residual_scale, dtype=hidden.dtype, device=hidden.device
        ) * bank_scale
        residual = (
            residual_gate
            * scale
            * self.dropout(self.fused_out(self.fused_norm(fused_sequence)))
        )
        residual = self.up(residual)
        if self.carrier_out is not None and self.carrier_norm is not None:
            carrier = self.up(
                self.carrier_out(
                    self.carrier_norm(fused_sequence + context_states)
                )
            ) * bank_scale
        else:
            carrier = torch.zeros_like(hidden)

        active_residual = residual
        active_carrier = carrier
        expansion_indices = active_mask.long().cumsum(dim=0) - 1
        gather_indices = expansion_indices.clamp_min(0)

        def expand_sequence(
            value: torch.Tensor,
            held: torch.Tensor | None = None,
        ) -> torch.Tensor:
            expanded = value.index_select(1, gather_indices)
            if bool((expansion_indices < 0).any()):
                prefix = (
                    torch.zeros_like(expanded)
                    if held is None
                    else held[:, None].expand_as(expanded)
                )
                view = [1, full_length] + [1] * (expanded.ndim - 2)
                expanded = torch.where(
                    (expansion_indices < 0).view(*view),
                    prefix,
                    expanded,
                )
            return expanded

        residual = expand_sequence(active_residual, held_residual)
        carrier = expand_sequence(active_carrier, held_carrier)
        next_cache = (
            {
                "phase": phase[:, -1].detach(),
                "memory": raw_memory[:, -1].detach(),
                "proxy": memory_stats["proxy_final"].detach(),
                "context": context_states[:, -1].detach(),
                "predictor_phase": clock_stats["predictor_phase"].detach(),
                "last_position": int(positions[-1]),
                "held_residual": active_residual[:, -1].detach(),
                "held_carrier": active_carrier[:, -1].detach(),
            }
            if use_cache
            else None
        )
        if not collect_aux:
            return residual, next_cache, {"carrier": carrier}

        normalized_outputs = F.normalize(gear_outputs, dim=-1)
        mean_output = F.normalize(normalized_outputs.mean(dim=-2), dim=-1)
        conflict = (
            (normalized_outputs - mean_output.unsqueeze(-2)).pow(2)
            .sum(dim=-1)
            .mean(dim=-1)
        )
        risk_logit = (
            self.risk_head(
                torch.cat([fused_sequence, conflict.unsqueeze(-1)], dim=-1)
            ).squeeze(-1)
            if self.risk_head is not None
            else torch.zeros_like(conflict)
        )

        slot_usage = routing.mean(dim=(0, 1))
        valid_usage = slot_usage.masked_fill(~self.slot_mask, 0.0)
        uniform = self.slot_mask.to(slot_usage.dtype)
        uniform = uniform / uniform.sum(dim=-1, keepdim=True)
        route_balance = (
            (valid_usage - uniform).pow(2).sum(dim=-1)
        ).mean()
        per_gear_route_entropy = []
        for gear, slots in enumerate(self.config.gear_slots):
            per_gear_route_entropy.append(
                _normalized_entropy(routing[:, :, gear, :slots], slots).mean()
            )
        route_entropy = torch.stack(per_gear_route_entropy).mean()
        usage_entropy = torch.stack(
            [
                _normalized_entropy(slot_usage[gear, :slots], slots)
                for gear, slots in enumerate(self.config.gear_slots)
            ]
        ).mean()
        mean_outputs = F.normalize(gear_outputs.mean(dim=(0, 1)), dim=-1)
        similarity = mean_outputs @ mean_outputs.T
        off_diagonal = ~torch.eye(
            self.num_gears, dtype=torch.bool, device=hidden.device
        )
        output_diversity = similarity[off_diagonal].pow(2).mean()
        coherence_entropy = _normalized_entropy(
            lane_weights, self.num_lanes
        ).mean()

        aux = {
            "gear_outputs": gear_outputs,
            "memory_states": memory_states,
            "slot_messages": slot_messages,
            "fusion_weights": fusion_weights,
            "lane_weights": lane_weights,
            "lane_hidden": self.up(fusion["lane_values"]),
            "context_hidden": self.up(context_states),
            "conflict": conflict,
            "risk_logit": risk_logit,
            "route_balance": route_balance,
            "route_entropy": route_entropy,
            "usage_entropy": usage_entropy,
            "update_activity": aperture.mean(),
            "write_activity": memory_stats["write_activity"],
            "read_activity": read_gate.mean(),
            "coupling_entropy": fusion["lane_attention_entropy"],
            "coupling_gate": torch.sigmoid(self.lane_mixing_gate_logit),
            "coupling_offdiag": fusion["lane_attention_offdiag"],
            "output_diversity": output_diversity,
            "speed_separation": self.speed_separation_loss(),
            "slot_diversity": self.slot_diversity_loss(),
            "phase_drive_activity": clock_stats["phase_drive_activity"],
            "phase_coupling_activity": clock_stats[
                "phase_coupling_activity"
            ],
            "phase_lock_error": clock_stats["phase_lock_error"],
            "minimum_phase_advance": clock_stats["minimum_phase_advance"],
            "rotation_activity": memory_stats["rotation_activity"],
            "fast_speed": self.speeds()[0],
            "slow_speed": self.speeds()[-1],
            "coherence_entropy": coherence_entropy,
            "gear_agreement": fusion["gear_agreement"],
            "lane_balance": (
                lane_weights.mean(dim=(0, 1)) - (1.0 / self.num_lanes)
            ).pow(2).mean(),
            "carrier": carrier,
            "temporal_context_gate": context_stats["temporal_context_gate"],
            "interbank_gate": context_stats["interbank_gate"],
            "interbank_activity": context_stats["interbank_activity"],
            "bank_index": hidden.new_tensor(float(self.bank_index)),
            "bank_horizon_scale": hidden.new_tensor(self.horizon_scale),
            "active_fraction": hidden.new_tensor(length / full_length),
            "temporal_stride": hidden.new_tensor(float(self.temporal_stride)),
        }
        for name, value in tuple(aux.items()):
            if (
                torch.is_tensor(value)
                and value.ndim >= 2
                and value.shape[1] == length
            ):
                aux[name] = expand_sequence(value)
        aux["carrier"] = carrier
        return residual, next_cache, aux
