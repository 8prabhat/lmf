"""Canonical Pure Parallel Gear language model.

Sequence information is carried only by fixed-size rotor states.  There is no
token-history tensor, token-to-token similarity, attention, retrieval, or KV
cache in this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS


class GearRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        source = value.float()
        normalized = source * torch.rsqrt(
            source.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight


@dataclass(frozen=True)
class PureParallelGearConfig:
    vocab_size: int
    dim: int = 192
    layers: int = 4
    ffn_dim: int | None = None
    num_banks: int = 4
    bank_roles: tuple[str, ...] | None = None
    gears_per_bank: int = 8
    rotor_channels: int = 4
    predictor_gears: int = 4
    settling_rounds: int = 2
    max_sentence_tokens: int = 128
    intra_sentence_clutch_tokens: int = 32
    max_seq_len: int = 4096
    theta_limit: float = math.pi / 3.0
    torque_limit: float = 0.15
    retention_floor: float = 0.94
    retention_ceiling: float = 0.9999
    rotor_radius_limit: float = 4.0
    coupling_limit: float = math.pi / 4.0
    omega_limit: float = math.pi / 2.0
    dropout: float = 0.0
    rotor_energy_weight: float = 1e-4
    omega_saturation_weight: float = 1e-4
    clutch_collapse_weight: float = 1e-5
    clutch_target_mean: float = 0.35
    clutch_balance_weight: float = 0.05
    gear_residual_floor: float = 0.05
    predictor_residual_floor: float = 0.20
    regularizer_decay_fraction: float = 0.5
    minimum_regularizer_scale: float = 0.0
    diagnostic_max_tokens: int = 512
    boundary_policy: str = "deterministic"
    ablations: tuple[str, ...] = ()
    boundary_settling: bool = True
    cross_bank_coupling: bool = True
    overlapping_coupling: bool = True
    learned_angular_velocity: bool = True
    use_load_state: bool = True
    use_predictor_gear: bool = True
    use_local_swiglu: bool = True
    use_fast_weight_memory: bool = False
    fast_weight_banks: int = 4
    fast_weight_key_dim: int = 16
    fast_weight_value_dim: int = 16
    fast_weight_decay: float = 0.99
    fast_weight_chunk_tokens: int = 128
    copy_gate_target_mean: float = 0.10
    copy_gate_balance_weight: float = 0.02
    fast_weight_energy_weight: float = 1e-4
    fast_weight_energy_limit: float = 50.0

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.dim < 8 or self.layers < 1:
            raise ValueError("dim must be >= 8 and layers must be positive")
        if self.num_banks < 1 or self.gears_per_bank < 2:
            raise ValueError("gear topology requires at least one bank and two gears")
        if self.bank_roles is None:
            defaults = (
                "surface_syntax",
                "relations_entities",
                "discourse_continuity",
                "planning_constraints",
            )
            roles = tuple(
                defaults[index] if index < len(defaults) else f"bank_{index}"
                for index in range(self.num_banks)
            )
            object.__setattr__(self, "bank_roles", roles)
        else:
            object.__setattr__(self, "bank_roles", tuple(self.bank_roles))
        if len(self.bank_roles) != self.num_banks:
            raise ValueError("bank_roles must contain one label per bank")
        if self.rotor_channels < 1 or self.predictor_gears < 2:
            raise ValueError("rotor_channels must be positive and predictor_gears >= 2")
        if self.settling_rounds < 0:
            raise ValueError("settling_rounds cannot be negative")
        if self.max_sentence_tokens < 2 or self.max_seq_len < 2:
            raise ValueError("sequence and sentence limits must be at least 2")
        if (
            self.intra_sentence_clutch_tokens != 0
            and self.intra_sentence_clutch_tokens < 2
        ):
            raise ValueError(
                "intra_sentence_clutch_tokens must be zero or at least 2"
            )
        if any(
            value <= 0.0
            for value in (
                self.theta_limit,
                self.torque_limit,
                self.rotor_radius_limit,
                self.coupling_limit,
                self.omega_limit,
            )
        ):
            raise ValueError("mechanical limits must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.retention_floor < self.retention_ceiling <= 1.0:
            raise ValueError(
                "retention bounds must satisfy 0 < floor < ceiling <= 1"
            )
        if not 0.0 <= self.gear_residual_floor < 1.0:
            raise ValueError("gear_residual_floor must be in [0, 1)")
        if not 0.0 <= self.predictor_residual_floor < 1.0:
            raise ValueError("predictor_residual_floor must be in [0, 1)")
        if self.ffn_dim is None:
            object.__setattr__(self, "ffn_dim", 4 * self.dim)
        object.__setattr__(self, "ablations", tuple(self.ablations))
        if int(self.ffn_dim) < self.dim:
            raise ValueError("ffn_dim must be at least dim")
        if not 0.0 < self.regularizer_decay_fraction <= 1.0:
            raise ValueError("regularizer_decay_fraction must be in (0, 1]")
        if not 0.0 <= self.minimum_regularizer_scale <= 1.0:
            raise ValueError("minimum_regularizer_scale must be in [0, 1]")
        if self.boundary_policy not in {"deterministic", "fixed"}:
            raise ValueError("boundary_policy must be deterministic or fixed")
        if self.use_fast_weight_memory:
            if self.fast_weight_banks < 1:
                raise ValueError("fast_weight_banks must be positive")
            if self.fast_weight_key_dim < 1 or self.fast_weight_value_dim < 1:
                raise ValueError("fast_weight key/value dims must be positive")
            if not 0.0 < self.fast_weight_decay < 1.0:
                raise ValueError("fast_weight_decay must be in (0, 1)")
            if self.fast_weight_chunk_tokens < 2:
                raise ValueError("fast_weight_chunk_tokens must be at least 2")
            if not 0.0 < self.copy_gate_target_mean < 1.0:
                raise ValueError("copy_gate_target_mean must be in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GearState:
    """Constant-size state for one gear layer."""

    rotor: torch.Tensor
    omega: torch.Tensor
    load: torch.Tensor
    sentence_length: torch.Tensor
    segment_id: torch.Tensor

    def detach(self) -> "GearState":
        return GearState(
            self.rotor.detach(),
            self.omega.detach(),
            self.load.detach(),
            self.sentence_length.detach(),
            self.segment_id.detach(),
        )

    def to(self, *args, **kwargs) -> "GearState":
        rotor = self.rotor.to(*args, **kwargs)
        return GearState(
            rotor,
            self.omega.to(*args, **kwargs),
            self.load.to(*args, **kwargs),
            self.sentence_length.cpu(),
            self.segment_id.cpu(),
        )


@dataclass
class FastWeightMemoryState:
    """Decayed key/value accumulator for the associative memory.

    matrix[b] approximates sum_i decay^(t-i) * outer(key_i, value_i) up to
    the current token, carried across chunks exactly (see
    FastWeightMemory._chunked_scan) and reset to zero at segment changes.
    """

    matrix: torch.Tensor
    segment_id: torch.Tensor

    def detach(self) -> "FastWeightMemoryState":
        return FastWeightMemoryState(self.matrix.detach(), self.segment_id.detach())

    def to(self, *args, **kwargs) -> "FastWeightMemoryState":
        return FastWeightMemoryState(
            self.matrix.to(*args, **kwargs), self.segment_id.cpu()
        )


@dataclass
class GearCache:
    """All generation state; its size is independent of context length."""

    layers: tuple[GearState, ...]
    predictor: GearState | None
    tokens_processed: torch.Tensor
    memory: FastWeightMemoryState | None = None

    def detach(self) -> "GearCache":
        return GearCache(
            tuple(state.detach() for state in self.layers),
            None if self.predictor is None else self.predictor.detach(),
            self.tokens_processed.detach(),
            None if self.memory is None else self.memory.detach(),
        )


def _rotate(vector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate [..., 2] vectors by angles with shape [...]."""
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    x, y = vector.unbind(dim=-1)
    return torch.stack(
        (cosine * x - sine * y, sine * x + cosine * y),
        dim=-1,
    )


@dataclass
class _ChunkSpec:
    """One sentence/clutch span within a single row, found by _chunk_plan."""

    start: int
    stop: int
    boundary: bool
    micro_clutch: bool
    needs_reset: bool
    segment: int
    sentence_length_before: int


@dataclass
class _MemoryChunkSpec:
    """One safety-capped span within a single row, found by
    FastWeightMemory._memory_chunk_plan. Unlike _ChunkSpec, this never
    breaks at sentence ends or the intra-sentence clutch interval -- the
    memory is meant to persist across sentences within a document, only
    resetting at segment (document) changes."""

    start: int
    stop: int
    needs_reset: bool
    segment: int


class PureGearLayer(nn.Module):
    """Independent rotor banks with explicit sentence-boundary clutching."""

    def __init__(
        self,
        config: PureParallelGearConfig,
        *,
        banks: int | None = None,
        gears: int | None = None,
        use_ffn: bool = True,
        residual_floor: float | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.banks = int(config.num_banks if banks is None else banks)
        self.gears = int(config.gears_per_bank if gears is None else gears)
        self.channels = int(config.rotor_channels)
        self.use_ffn = bool(use_ffn)
        self.residual_floor = float(
            config.gear_residual_floor
            if residual_floor is None
            else residual_floor
        )
        state_width = self.banks * self.gears * self.channels

        # Disjoint-pair indices for the two "brick wall" intra-bank mixing
        # passes (settle() alternates them so every adjacent gear pair gets
        # mixed once per round). Pairs within a single pass never share a
        # gear index, so all pairs in a pass can be mixed in one batched op
        # instead of looping _mix_gears once per pair.
        even_lefts = list(range(0, self.gears - 1, 2))
        odd_lefts = list(range(1, self.gears - 1, 2))
        self.register_buffer(
            "_even_gear_lefts", torch.tensor(even_lefts, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_even_gear_rights",
            torch.tensor([index + 1 for index in even_lefts], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_odd_gear_lefts", torch.tensor(odd_lefts, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_odd_gear_rights",
            torch.tensor([index + 1 for index in odd_lefts], dtype=torch.long),
            persistent=False,
        )

        self.input_norm = GearRMSNorm(config.dim)
        self.angle_projection = nn.Linear(config.dim, state_width, bias=False)
        self.clutch_projection = nn.Linear(config.dim, state_width)
        self.torque_projection = nn.Linear(config.dim, 2 * state_width, bias=False)
        self.retention_projection = nn.Linear(config.dim, state_width)
        if self.banks == 1:
            # The predictor is a short-horizon differential integrator, not a
            # fifth slow/planner bank, so it uses the full retention range.
            retention_low = torch.tensor([config.retention_floor])
            retention_high = torch.tensor([config.retention_ceiling])
        else:
            fraction = torch.linspace(0.0, 1.0, self.banks + 1)
            edges = 1.0 - (1.0 - fraction).square()
            span = config.retention_ceiling - config.retention_floor
            values = config.retention_floor + span * edges
            retention_low = values[:-1]
            retention_high = values[1:]
        self.register_buffer(
            "retention_low",
            retention_low[:, None, None],
            persistent=True,
        )
        self.register_buffer(
            "retention_high",
            retention_high[:, None, None],
            persistent=True,
        )

        phase = torch.zeros(self.banks, self.gears, self.channels)
        for bank in range(self.banks):
            for gear in range(self.gears):
                phase[bank, gear] = (
                    2.0 * math.pi * gear / self.gears
                    + math.pi * bank / max(self.banks, 1)
                )
        self.initial_phase = nn.Parameter(phase)
        speed_scales = torch.logspace(
            math.log10(0.01),
            math.log10(0.35),
            steps=self.banks,
        )
        speed = torch.empty(self.banks, self.gears, self.channels)
        for bank, scale in enumerate(speed_scales):
            speed[bank] = scale / torch.arange(
                1,
                self.gears + 1,
                dtype=torch.float32,
            )[:, None]
        if config.learned_angular_velocity:
            self.base_omega = nn.Parameter(speed)
        else:
            self.register_buffer("base_omega", speed)

        if config.boundary_settling:
            self.pair_kernel = nn.Parameter(
                torch.empty(
                    self.banks,
                    self.gears - 1,
                    self.channels,
                    8,
                )
            )
        else:
            self.register_buffer(
                "pair_kernel",
                torch.zeros(
                    self.banks,
                    self.gears - 1,
                    self.channels,
                    8,
                ),
            )
        if config.boundary_settling and config.cross_bank_coupling and self.banks > 1:
            self.cross_kernel = nn.Parameter(
                torch.empty(
                    self.banks,
                    self.gears,
                    self.channels,
                    8,
                )
            )
        else:
            self.register_buffer(
                "cross_kernel",
                torch.zeros(
                    self.banks,
                    self.gears,
                    self.channels,
                    8,
                ),
            )
        if config.boundary_settling:
            nn.init.normal_(self.pair_kernel, std=0.08)
        if config.boundary_settling and config.cross_bank_coupling and self.banks > 1:
            nn.init.normal_(self.cross_kernel, std=0.08)
        intra_gate = torch.full(
            (
                max(config.settling_rounds, 1),
                self.banks,
                self.gears - 1,
                self.channels,
            ),
            -1.5,
        )
        if config.boundary_settling:
            self.intra_gate = nn.Parameter(intra_gate)
        else:
            self.register_buffer("intra_gate", intra_gate)
        cross_gate = torch.full(
            (
                max(config.settling_rounds, 1),
                self.banks,
                self.gears,
                self.channels,
            ),
            -2.0,
        )
        if config.boundary_settling and config.cross_bank_coupling and self.banks > 1:
            self.cross_gate = nn.Parameter(cross_gate)
        else:
            self.register_buffer("cross_gate", cross_gate)
        if config.boundary_settling and config.use_load_state:
            response = torch.zeros(
                self.banks, self.gears, self.channels, 2
            )
            response[..., 0] = 0.08
            response[..., 1] = 0.03
            self.load_response = nn.Parameter(response)
        else:
            self.register_buffer(
                "load_response",
                torch.zeros(
                    self.banks, self.gears, self.channels, 2
                ),
            )
        if config.boundary_settling and config.learned_angular_velocity:
            response = torch.zeros(
                self.banks, self.gears, self.channels, 2
            )
            response[..., 0] = 0.04
            response[..., 1] = 0.02
            self.omega_response = nn.Parameter(response)
        else:
            self.register_buffer(
                "omega_response",
                torch.zeros(
                    self.banks, self.gears, self.channels, 2
                ),
            )

        feature_dim = self._feature_dim()
        self.readout_norm = GearRMSNorm(feature_dim)
        self.readout = nn.Linear(feature_dim, config.dim, bias=False)
        initial_scale = max(self.residual_floor + 0.05, 0.10)
        initial_fraction = (
            initial_scale - self.residual_floor
        ) / max(1.0 - self.residual_floor, 1e-6)
        initial_fraction = min(max(initial_fraction, 1e-4), 1.0 - 1e-4)
        self.gear_residual = nn.Parameter(
            torch.tensor(
                math.log(initial_fraction / (1.0 - initial_fraction))
            )
        )
        self.dropout = nn.Dropout(config.dropout)
        if self.use_ffn:
            self.ffn_norm = GearRMSNorm(config.dim)
            self.ffn_in = nn.Linear(config.dim, 2 * int(config.ffn_dim))
            self.ffn_out = nn.Linear(int(config.ffn_dim), config.dim, bias=False)
            self.ffn_residual = nn.Parameter(torch.tensor(0.05))

        nn.init.normal_(self.angle_projection.weight, std=0.01)
        nn.init.normal_(self.torque_projection.weight, std=0.01)
        nn.init.zeros_(self.retention_projection.weight)
        nn.init.zeros_(self.retention_projection.bias)
        nn.init.zeros_(self.clutch_projection.weight)
        target_mean = config.clutch_target_mean
        neutral_logit = math.log(target_mean / (1.0 - target_mean))
        nn.init.constant_(self.clutch_projection.bias, neutral_logit)
        nn.init.xavier_uniform_(self.readout.weight)
        if self.use_ffn:
            nn.init.xavier_uniform_(self.ffn_in.weight)
            nn.init.normal_(self.ffn_out.weight, std=0.005)

    def _feature_dim(self) -> int:
        state = self.banks * self.gears * self.channels
        rotor = state * 2
        radial = state
        motion = state * 3
        adjacent = self.banks * (self.gears - 1) * self.channels * 2
        dynamics = state * (
            3 if self.config.use_load_state else 2
        )
        cross = max(self.banks - 1, 0) * self.gears * self.channels * 2
        return rotor + radial + motion + adjacent + dynamics + cross

    def initial_state(
        self,
        batch: int,
        device: torch.device,
    ) -> GearState:
        phase = self.initial_phase.float()
        rotor = torch.stack((phase.cos(), phase.sin()), dim=-1)
        rotor = rotor.unsqueeze(0).expand(batch, -1, -1, -1, -1).clone()
        omega = self.config.omega_limit * torch.tanh(
            self.base_omega.float() / self.config.omega_limit
        )
        omega = omega.unsqueeze(0).expand(batch, -1, -1, -1).clone()
        load = torch.zeros_like(omega)
        return GearState(
            rotor.to(device),
            omega.to(device),
            load.to(device),
            torch.zeros(batch, dtype=torch.long),
            torch.full((batch,), -1, dtype=torch.long),
        )

    def _project_token_controls(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source = self.input_norm(hidden).float()
        shape = (
            *hidden.shape[:-1],
            self.banks,
            self.gears,
            self.channels,
        )
        delta = self.config.theta_limit * torch.tanh(
            self.angle_projection(source).float().reshape(shape)
        )
        clutch = torch.sigmoid(
            self.clutch_projection(source).float().reshape(shape)
        )
        torque = self.config.torque_limit * clutch[..., None] * torch.tanh(
            self.torque_projection(source).float().reshape(*shape, 2)
        )
        retention_fraction = torch.sigmoid(
            self.retention_projection(source).float().reshape(shape)
        )
        retention = self.retention_low + (
            self.retention_high - self.retention_low
        ) * retention_fraction
        return delta, clutch, torque, retention

    def _scan_token_dynamics(
        self,
        delta: torch.Tensor,
        torque: torch.Tensor,
        retention: torch.Tensor,
        state: GearState,
        *,
        fixed_omega: bool,
    ) -> torch.Tensor:
        omega = (
            self.config.omega_limit
            * torch.tanh(
                self.base_omega.float()[None] / self.config.omega_limit
            )
            if fixed_omega
            else state.omega
        )
        phase = torch.cumsum(delta + omega, dim=0)
        log_scale = torch.cumsum(retention.clamp_min(1e-6).log(), dim=0)
        scale = log_scale.exp()
        local_torque = (
            _rotate(torque, -phase)
            * (-log_scale).exp()[..., None]
        )
        transported = state.rotor + torch.cumsum(local_torque, dim=0)
        rotor = scale[..., None] * _rotate(transported, phase)
        return rotor

    def _token_dynamics(
        self,
        hidden: torch.Tensor,
        state: GearState,
        *,
        fixed_omega: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta, clutch, torque, retention = self._project_token_controls(
            hidden
        )
        rotor = self._scan_token_dynamics(
            delta,
            torque,
            retention,
            state,
            fixed_omega=fixed_omega,
        )
        return rotor, clutch, retention

    @staticmethod
    def _pair_features(
        rotor: torch.Tensor,
        omega: torch.Tensor,
        load: torch.Tensor,
        left: int,
        right: int,
    ) -> torch.Tensor:
        a = rotor[:, :, left]
        b = rotor[:, :, right]
        return torch.cat(
            (
                a,
                b,
                omega[:, :, left, :, None],
                omega[:, :, right, :, None],
                load[:, :, left, :, None],
                load[:, :, right, :, None],
            ),
            dim=-1,
        )

    def _mix_gears(
        self,
        rotor: torch.Tensor,
        omega: torch.Tensor,
        load: torch.Tensor,
        left: int,
        right: int,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        features = self._pair_features(rotor, omega, load, left, right)
        kernel = self.pair_kernel[:, left].float()
        raw = (features * kernel[None]).sum(dim=-1)
        angle = (
            torch.sigmoid(gate.float())[None]
            * self.config.coupling_limit
            * torch.tanh(raw)
        )
        a = rotor[:, :, left]
        b = rotor[:, :, right]
        cosine, sine = angle.cos()[..., None], angle.sin()[..., None]
        values = [rotor[:, :, index] for index in range(self.gears)]
        values[left] = cosine * a - sine * b
        values[right] = sine * a + cosine * b
        return torch.stack(values, dim=2)

    def _mix_gear_pairs(
        self,
        rotor: torch.Tensor,
        omega: torch.Tensor,
        load: torch.Tensor,
        lefts: torch.Tensor,
        rights: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized _mix_gears over a batch of mutually-disjoint pairs.

        lefts/rights index distinct, non-overlapping gear positions, so all
        pairs can be mixed in one shot instead of one _mix_gears call per
        pair -- this is the loop that dominated settle()'s cost.
        """
        a = rotor.index_select(2, lefts)
        b = rotor.index_select(2, rights)
        features = torch.cat(
            (
                a,
                b,
                omega.index_select(2, lefts)[..., None],
                omega.index_select(2, rights)[..., None],
                load.index_select(2, lefts)[..., None],
                load.index_select(2, rights)[..., None],
            ),
            dim=-1,
        )
        kernel = self.pair_kernel.index_select(1, lefts).float()
        raw = (features * kernel[None]).sum(dim=-1)
        angle = (
            torch.sigmoid(gate.float())[None]
            * self.config.coupling_limit
            * torch.tanh(raw)
        )
        cosine, sine = angle.cos()[..., None], angle.sin()[..., None]
        new_a = cosine * a - sine * b
        new_b = sine * a + cosine * b
        rotor = rotor.index_copy(2, lefts, new_a).index_copy(2, rights, new_b)
        # sigmoid(gate).mean() averages over (banks, pairs, channels); the
        # original loop summed one mean() per pair, so rescale by pair count
        # to match (each pair contributes equally many banks*channels terms).
        activity = lefts.numel() * torch.sigmoid(gate.float()).mean()
        return rotor, activity

    def _mix_banks(
        self,
        rotor: torch.Tensor,
        omega: torch.Tensor,
        load: torch.Tensor,
        left: int,
        right: int,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        a, b = rotor[:, left], rotor[:, right]
        features = torch.cat(
            (
                a,
                b,
                omega[:, left, :, :, None],
                omega[:, right, :, :, None],
                load[:, left, :, :, None],
                load[:, right, :, :, None],
            ),
            dim=-1,
        )
        kernel = self.cross_kernel[left].float()
        raw = (features * kernel[None]).sum(dim=-1)
        angle = (
            torch.sigmoid(gate.float())[None]
            * self.config.coupling_limit
            * torch.tanh(raw)
        )
        cosine, sine = angle.cos()[..., None], angle.sin()[..., None]
        values = [rotor[:, index] for index in range(self.banks)]
        values[left] = cosine * a - sine * b
        values[right] = sine * a + cosine * b
        return torch.stack(values, dim=1)

    def settle(
        self,
        state: GearState,
        *,
        cross_bank: bool = True,
        commuting_only: bool = False,
        use_load: bool = True,
    ) -> tuple[GearState, torch.Tensor]:
        rotor = state.rotor
        activity = rotor.new_zeros(())
        for round_index in range(self.config.settling_rounds):
            gate_round = min(round_index, self.intra_gate.shape[0] - 1)
            if self._even_gear_lefts.numel() > 0:
                rotor, pair_activity = self._mix_gear_pairs(
                    rotor,
                    state.omega,
                    state.load,
                    self._even_gear_lefts,
                    self._even_gear_rights,
                    self.intra_gate[gate_round, :, self._even_gear_lefts],
                )
                activity = activity + pair_activity
            if not commuting_only and self._odd_gear_lefts.numel() > 0:
                rotor, pair_activity = self._mix_gear_pairs(
                    rotor,
                    state.omega,
                    state.load,
                    self._odd_gear_lefts,
                    self._odd_gear_rights,
                    self.intra_gate[gate_round, :, self._odd_gear_lefts],
                )
                activity = activity + pair_activity
            if cross_bank and self.banks > 1:
                for left in range(self.banks):
                    right = (left + 1) % self.banks
                    rotor = self._mix_banks(
                        rotor,
                        state.omega,
                        state.load,
                        left,
                        right,
                        self.cross_gate[gate_round, left],
                    )
                    activity = activity + torch.sigmoid(
                        self.cross_gate[gate_round, left]
                    ).mean()

        magnitude = rotor.square().sum(dim=-1).clamp_min(1e-8).sqrt()
        log_energy = magnitude.log().clamp(-4.0, 4.0)
        normalized = rotor / magnitude[..., None]
        bounded_magnitude = magnitude.clamp_max(
            self.config.rotor_radius_limit
        )
        bounded_rotor = normalized * bounded_magnitude[..., None]
        if use_load and self.config.use_load_state:
            orientation = normalized[..., 0] - normalized[..., 1]
            load = torch.tanh(
                state.load
                + self.load_response[..., 0].float() * log_energy
                + self.load_response[..., 1].float() * orientation
            )
        else:
            load = torch.zeros_like(state.load)
        omega = (
            self.config.omega_limit
            * torch.tanh(
                state.omega / self.config.omega_limit
                + self.omega_response[..., 0].float() * load
                + self.omega_response[..., 1].float() * log_energy
            )
            if self.config.learned_angular_velocity
            else state.omega
        )
        count = max(1, self.config.settling_rounds)
        return (
            GearState(
                bounded_rotor,
                omega,
                load,
                torch.zeros_like(state.sentence_length),
                state.segment_id,
            ),
            activity / count,
        )

    def _readout(
        self,
        rotor: torch.Tensor,
        omega: torch.Tensor,
        load: torch.Tensor,
        clutch: torch.Tensor,
        previous_rotor: torch.Tensor,
    ) -> torch.Tensor:
        normalized = F.normalize(rotor, dim=-1)
        previous_normalized = F.normalize(previous_rotor, dim=-1)
        radius = rotor.square().sum(dim=-1).clamp_min(1e-8).sqrt()
        previous_radius = (
            previous_rotor.square().sum(dim=-1).clamp_min(1e-8).sqrt()
        )
        log_radius = radius.log().clamp(-4.0, 4.0)
        radial_change = (log_radius - previous_radius.log()).clamp(
            -4.0, 4.0
        )
        motion_dot = (previous_normalized * normalized).sum(dim=-1)
        motion_cross = (
            previous_normalized[..., 0] * normalized[..., 1]
            - previous_normalized[..., 1] * normalized[..., 0]
        )
        motion = torch.stack(
            (motion_dot, motion_cross, radial_change),
            dim=-1,
        )
        left, right = normalized[:, :, :-1], normalized[:, :, 1:]
        dot = (left * right).sum(dim=-1)
        cross = left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]
        adjacent = torch.stack((dot, cross), dim=-1)
        cross_bank = (
            normalized[:, 1:] - normalized[:, :-1]
            if self.banks > 1
            else normalized[:, :0]
        )
        parts = [
            normalized.flatten(1),
            log_radius.flatten(1),
            motion.flatten(1),
            adjacent.flatten(1),
            omega.flatten(1),
        ]
        if self.config.use_load_state:
            parts.append(load.flatten(1))
        parts.extend((clutch.flatten(1), cross_bank.flatten(1)))
        features = torch.cat(parts, dim=-1)
        return self.readout(self.readout_norm(features))

    @staticmethod
    def _state_row(state: GearState, row: int) -> GearState:
        return GearState(
            state.rotor[row : row + 1],
            state.omega[row : row + 1],
            state.load[row : row + 1],
            state.sentence_length[row : row + 1],
            state.segment_id[row : row + 1],
        )

    def _chunk_plan(
        self,
        token_mask_row: torch.Tensor,
        segment_ids_row: torch.Tensor,
        sentence_end_mask_row: torch.Tensor,
        initial_segment: int,
        initial_sentence_length: int,
    ) -> tuple[list[_ChunkSpec], list[int]]:
        """Find one row's sentence/clutch spans without touching any tensor
        math -- pure Python/CPU bookkeeping so forward() can batch every
        row's chunk processing together instead of looping row by row.

        Mirrors the boundary conditions that used to be interleaved with
        tensor processing in the per-row loop, unchanged.
        """
        length = int(token_mask_row.shape[0])
        chunks: list[_ChunkSpec] = []
        zero_positions: list[int] = []
        position = 0
        current_segment = initial_segment
        sentence_length = initial_sentence_length
        while position < length:
            if not bool(token_mask_row[position]):
                zero_positions.append(position)
                position += 1
                continue

            segment = int(segment_ids_row[position])
            needs_reset = segment != current_segment
            if needs_reset:
                current_segment = segment
                sentence_length = 0

            remaining = self.config.max_sentence_tokens - sentence_length
            clutch_interval = self.config.intra_sentence_clutch_tokens
            clutch_remaining = (
                remaining
                if clutch_interval == 0
                else clutch_interval - sentence_length % clutch_interval
            )
            stop = min(
                length,
                position + max(1, min(remaining, clutch_remaining)),
            )
            boundary = False
            for candidate in range(position, stop):
                if (
                    not bool(token_mask_row[candidate])
                    or int(segment_ids_row[candidate]) != segment
                ):
                    stop = candidate
                    break
                if bool(sentence_end_mask_row[candidate]):
                    stop = candidate + 1
                    boundary = True
                    break
            if stop == position:
                continue
            if stop - position >= remaining:
                boundary = True
            micro_clutch = (
                not boundary
                and clutch_interval > 0
                and stop - position >= clutch_remaining
            )
            chunks.append(
                _ChunkSpec(
                    position,
                    stop,
                    boundary,
                    micro_clutch,
                    needs_reset,
                    segment,
                    sentence_length,
                )
            )
            sentence_length = 0 if boundary else sentence_length + (stop - position)
            position = stop
        return chunks, zero_positions

    @staticmethod
    def _stack_states(states: list[GearState]) -> GearState:
        return GearState(
            torch.cat([state.rotor for state in states], dim=0),
            torch.cat([state.omega for state in states], dim=0),
            torch.cat([state.load for state in states], dim=0),
            torch.cat([state.sentence_length for state in states], dim=0),
            torch.cat([state.segment_id for state in states], dim=0),
        )

    def _forward_batched(
        self,
        hidden: torch.Tensor,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
        state: GearState,
        *,
        fixed_omega: bool,
        settling_enabled: bool,
        cross_bank: bool,
        commuting_only: bool,
        use_load: bool,
    ) -> tuple[torch.Tensor, GearState, dict[str, torch.Tensor]]:
        """Batch every row's chunk processing per chunk-step instead of
        looping row by row and chunk by chunk.

        Chunk boundaries are content-dependent, so finding them stays a
        (tensor-free, near-free) Python pass per row via _chunk_plan. The
        expensive rotor math runs once per chunk-step across every row that
        still has a chunk at that step, instead of once per row per chunk --
        this is what makes both CPU loop overhead and MPS kernel-launch
        overhead drop roughly batch_size-fold.

        _scan_token_dynamics needs no change at all: its cumsum is along
        dim=0 only, so adding an extra "which row" dimension after it (delta
        shaped [T, rows, banks, gears, channels] instead of [T, banks, gears,
        channels]) keeps every row's cumulative sum independent via ordinary
        broadcasting. settle() and _mix_gear_pairs/_mix_banks already accept
        an arbitrary leading dim (no time axis there at all). Only _readout
        hardcodes dim=1/dim=2 as banks/gears, so its calls are flattened to
        [T*rows, banks, gears, channels, 2] and reshaped back afterward.
        """
        device = hidden.device
        batch, length = hidden.shape[0], hidden.shape[1]
        delta, clutch_controls, torque, retention_controls = (
            self._project_token_controls(hidden)
        )

        plans: list[list[_ChunkSpec]] = []
        zero_lists: list[list[int]] = []
        for row in range(batch):
            chunks, zeros = self._chunk_plan(
                token_mask[row],
                segment_ids[row],
                sentence_end_mask[row],
                int(state.segment_id[row].item()),
                int(state.sentence_length[row].item()),
            )
            plans.append(chunks)
            zero_lists.append(zeros)

        output = hidden.new_zeros(batch, length, self.dim)
        row_states = [self._state_row(state, row) for row in range(batch)]
        row_rotor_energy: list[list[torch.Tensor]] = [[] for _ in range(batch)]
        row_clutch: list[list[torch.Tensor]] = [[] for _ in range(batch)]
        row_retention: list[list[torch.Tensor]] = [[] for _ in range(batch)]
        # The legacy per-row loop takes a per-row mean of that row's own
        # settle() couplings (defaulting to zero for a row that never
        # settles, e.g. fully masked), then means *those* across rows --
        # not a flat mean over every settle() call. Mirror that exactly so
        # a row with zero settles dilutes the aggregate the same way.
        row_coupling: list[list[torch.Tensor]] = [[] for _ in range(batch)]

        max_chunks = max((len(plan) for plan in plans), default=0)
        for step in range(max_chunks):
            active_rows = [row for row in range(batch) if step < len(plans[row])]
            specs = [plans[row][step] for row in active_rows]
            num_active = len(active_rows)
            step_max_len = max(spec.stop - spec.start for spec in specs)

            delta_step = delta.new_zeros(
                step_max_len, num_active, self.banks, self.gears, self.channels
            )
            torque_step = torque.new_zeros(
                step_max_len, num_active, self.banks, self.gears, self.channels, 2
            )
            retention_step = retention_controls.new_ones(
                step_max_len, num_active, self.banks, self.gears, self.channels
            )
            clutch_step = clutch_controls.new_zeros(
                step_max_len, num_active, self.banks, self.gears, self.channels
            )
            rotor_in = hidden.new_zeros(
                num_active, self.banks, self.gears, self.channels, 2
            )
            omega_in = hidden.new_zeros(num_active, self.banks, self.gears, self.channels)
            load_in = hidden.new_zeros(num_active, self.banks, self.gears, self.channels)

            for index, row in enumerate(active_rows):
                spec = specs[index]
                chunk_len = spec.stop - spec.start
                delta_step[:chunk_len, index] = delta[row, spec.start : spec.stop]
                torque_step[:chunk_len, index] = torque[row, spec.start : spec.stop]
                retention_step[:chunk_len, index] = retention_controls[
                    row, spec.start : spec.stop
                ]
                clutch_step[:chunk_len, index] = clutch_controls[
                    row, spec.start : spec.stop
                ]
                if spec.needs_reset:
                    fresh = self.initial_state(1, device)
                    fresh.segment_id.fill_(spec.segment)
                    row_states[row] = fresh
                row_state = row_states[row]
                rotor_in[index] = row_state.rotor[0]
                omega_in[index] = row_state.omega[0]
                load_in[index] = row_state.load[0]

            scan_state = GearState(
                rotor_in,
                omega_in,
                load_in,
                hidden.new_zeros(num_active, dtype=torch.long),
                hidden.new_zeros(num_active, dtype=torch.long),
            )
            rotor_step = self._scan_token_dynamics(
                delta_step,
                torque_step,
                retention_step,
                scan_state,
                fixed_omega=fixed_omega,
            )
            previous_rotor_step = torch.cat(
                (rotor_in[None], rotor_step[:-1]), dim=0
            )
            omega_step = omega_in[None].expand(step_max_len, -1, -1, -1, -1).clone()
            load_step = load_in[None].expand(step_max_len, -1, -1, -1, -1).clone()

            settle_indices = [
                index for index, spec in enumerate(specs) if spec.boundary or spec.micro_clutch
            ]
            if settle_indices:
                settle_positions = torch.tensor(
                    [specs[index].stop - specs[index].start - 1 for index in settle_indices],
                    device=device,
                )
                gather_index = torch.tensor(settle_indices, device=device)
                boundary_rotor = rotor_step[settle_positions, gather_index]
                if settling_enabled:
                    boundary_state = GearState(
                        boundary_rotor,
                        omega_in.index_select(0, gather_index),
                        load_in.index_select(0, gather_index),
                        hidden.new_zeros(len(settle_indices), dtype=torch.long),
                        hidden.new_zeros(len(settle_indices), dtype=torch.long),
                    )
                    settled_state, coupling = self.settle(
                        boundary_state,
                        cross_bank=cross_bank,
                        commuting_only=commuting_only,
                        use_load=use_load,
                    )
                    for settled_index in settle_indices:
                        row_coupling[active_rows[settled_index]].append(coupling)
                    put_index = (settle_positions, gather_index)
                    rotor_step = rotor_step.index_put(put_index, settled_state.rotor)
                    omega_step = omega_step.index_put(put_index, settled_state.omega)
                    load_step = load_step.index_put(put_index, settled_state.load)

            output_step = self._readout(
                rotor_step.reshape(
                    step_max_len * num_active, self.banks, self.gears, self.channels, 2
                ),
                omega_step.reshape(
                    step_max_len * num_active, self.banks, self.gears, self.channels
                ),
                load_step.reshape(
                    step_max_len * num_active, self.banks, self.gears, self.channels
                ),
                clutch_step.reshape(
                    step_max_len * num_active, self.banks, self.gears, self.channels
                ),
                previous_rotor_step.reshape(
                    step_max_len * num_active, self.banks, self.gears, self.channels, 2
                ),
            ).reshape(step_max_len, num_active, self.dim)

            for index, row in enumerate(active_rows):
                spec = specs[index]
                chunk_len = spec.stop - spec.start
                last = chunk_len - 1
                output[row, spec.start : spec.stop] = output_step[:chunk_len, index]
                row_rotor_energy[row].append(
                    rotor_step[:chunk_len, index].square().sum(dim=-1)
                )
                row_clutch[row].append(clutch_step[:chunk_len, index])
                row_retention[row].append(retention_step[:chunk_len, index])

                next_sentence_length = (
                    0 if spec.boundary else spec.sentence_length_before + chunk_len
                )
                row_states[row] = GearState(
                    rotor_step[last : last + 1, index],
                    omega_step[last : last + 1, index],
                    load_step[last : last + 1, index],
                    torch.full_like(row_states[row].sentence_length, next_sentence_length),
                    torch.full_like(row_states[row].segment_id, spec.segment),
                )

        empty_state = hidden.new_zeros(0, self.banks, self.gears, self.channels)
        rotor_energy_parts = [
            torch.cat(parts, dim=0) if parts else empty_state
            for parts in row_rotor_energy
        ]
        clutch_parts = [
            torch.cat(parts, dim=0) if parts else empty_state
            for parts in row_clutch
        ]
        retention_parts = [
            torch.cat(parts, dim=0) if parts else empty_state
            for parts in row_retention
        ]
        next_state = self._stack_states(row_states)
        per_row_coupling = [
            torch.stack(parts).mean() if parts else hidden.new_zeros(())
            for parts in row_coupling
        ]
        return output, next_state, {
            "rotor_energy": torch.cat(rotor_energy_parts, dim=0)
            if any(part.numel() for part in rotor_energy_parts)
            else empty_state,
            "clutch": torch.cat(clutch_parts, dim=0)
            if any(part.numel() for part in clutch_parts)
            else empty_state,
            "retention": torch.cat(retention_parts, dim=0)
            if any(part.numel() for part in retention_parts)
            else empty_state,
            "coupling_activity": (
                torch.stack(per_row_coupling).mean()
                if per_row_coupling
                else hidden.new_zeros(())
            ),
            "omega": next_state.omega,
            "load": next_state.load,
            "rotor": next_state.rotor,
        }

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
        state: GearState | None = None,
        ablations: Iterable[str] = (),
    ) -> tuple[torch.Tensor, GearState, dict[str, torch.Tensor]]:
        hidden_dtype = hidden.dtype
        hidden = hidden.float()
        batch, length = hidden.shape[0], hidden.shape[1]
        if state is None:
            state = self.initial_state(batch, hidden.device)
        disabled = frozenset(ablations)
        fixed_omega = (
            not self.config.learned_angular_velocity
            or "fixed_angular_velocities" in disabled
        )
        settling_enabled = (
            self.config.boundary_settling and "no_boundary_settling" not in disabled
        )
        cross_bank = (
            self.config.cross_bank_coupling
            and "no_cross_bank_coupling" not in disabled
        )
        commuting_only = (
            not self.config.overlapping_coupling
            or "commuting_coupling_only" in disabled
        )
        use_load = self.config.use_load_state and "no_load_state" not in disabled

        control_token_mask = token_mask.detach().to(device="cpu", dtype=torch.bool)
        control_segment_ids = segment_ids.detach().to(device="cpu", dtype=torch.long)
        control_sentence_end_mask = sentence_end_mask.detach().to(
            device="cpu", dtype=torch.bool
        )
        gear_output, next_state, record = self._forward_batched(
            hidden,
            control_token_mask,
            control_segment_ids,
            control_sentence_end_mask,
            state,
            fixed_omega=fixed_omega,
            settling_enabled=settling_enabled,
            cross_bank=cross_bank,
            commuting_only=commuting_only,
            use_load=use_load,
        )
        residual_scale = self.residual_floor + (
            1.0 - self.residual_floor
        ) * torch.sigmoid(self.gear_residual)
        record["gear_residual_scale"] = residual_scale
        hidden = hidden + residual_scale * self.dropout(gear_output)
        if self.use_ffn and "no_local_swiglu" not in disabled:
            value, gate = self.ffn_in(self.ffn_norm(hidden)).chunk(2, dim=-1)
            feedforward = self.ffn_out(F.silu(gate) * value)
            hidden = hidden + torch.tanh(self.ffn_residual) * self.dropout(feedforward)
        return hidden.to(hidden_dtype), next_state, record


class FastWeightMemory(nn.Module):
    """Decayed key/value associative memory, embedding-grounded.

    Owned by PureParallelGearLM (not PureGearLayer) because it needs the
    raw token ids and the shared, tied embedding/head table: values are
    drawn from the actual embedding of the token that occurred
    (value_down_proj(token_embedding(token_id_t))), and the direct logit
    path reads them back out through the same head
    (head(value_up_proj(read_t))) -- both require the LM-level embedding
    table, not anything a single gear layer has access to.

    Numerical safety: S_t = decay*S_{t-1} + outer(key_t, value_t) is a
    linear recurrence with a closed-form parallel scan via
    rescale-then-cumsum, but rescaling by decay^-i over a whole multi-
    thousand-token segment overflows in fp32 (0.99^-4096 ~ 6e17) -- the
    same numerical fragility found and fixed this session for the rotor's
    omega recurrence. The fix is the same one used there: only run the
    rescale-cumsum within bounded chunks (<= fast_weight_chunk_tokens,
    where 0.99^-128 ~ 3.6 is safe), carrying the accumulated matrix across
    chunks via a plain decay multiply -- which the same formula already
    produces at the chunk's last position, so no separate carry step is
    needed.
    """

    def __init__(self, config: PureParallelGearConfig) -> None:
        super().__init__()
        self.config = config
        self.banks = int(config.fast_weight_banks)
        self.key_dim = int(config.fast_weight_key_dim)
        self.value_dim = int(config.fast_weight_value_dim)
        self.chunk_tokens = int(config.fast_weight_chunk_tokens)
        self.decay = float(config.fast_weight_decay)

        self.key_proj = nn.Linear(config.dim, self.banks * self.key_dim, bias=False)
        self.query_proj = nn.Linear(config.dim, self.banks * self.key_dim, bias=False)
        self.value_down_proj = nn.Linear(config.dim, self.value_dim, bias=False)
        self.value_up_proj = nn.Linear(
            self.banks * self.value_dim, config.dim, bias=False
        )
        self.gate_proj = nn.Linear(config.dim + self.banks * self.value_dim, 1)
        self.memory_out_proj = nn.Linear(
            self.banks * self.value_dim, config.dim, bias=False
        )
        self.memory_residual = nn.Parameter(torch.tensor(0.0))

        nn.init.normal_(self.key_proj.weight, std=0.02)
        nn.init.normal_(self.query_proj.weight, std=0.02)
        nn.init.normal_(self.value_down_proj.weight, std=0.02)
        nn.init.normal_(self.value_up_proj.weight, std=0.02)
        nn.init.zeros_(self.memory_out_proj.weight)
        nn.init.zeros_(self.gate_proj.weight)
        target = config.copy_gate_target_mean
        nn.init.constant_(self.gate_proj.bias, math.log(target / (1.0 - target)))

    def initial_state(self, batch: int, device: torch.device) -> FastWeightMemoryState:
        matrix = torch.zeros(
            batch, self.banks, self.key_dim, self.value_dim, device=device
        )
        segment_id = torch.full((batch,), -1, dtype=torch.long)
        return FastWeightMemoryState(matrix, segment_id)

    def _memory_chunk_plan(
        self,
        token_mask_row: torch.Tensor,
        segment_ids_row: torch.Tensor,
        initial_segment: int,
    ) -> tuple[list[_MemoryChunkSpec], list[int]]:
        length = int(token_mask_row.shape[0])
        chunks: list[_MemoryChunkSpec] = []
        zero_positions: list[int] = []
        position = 0
        current_segment = initial_segment
        while position < length:
            if not bool(token_mask_row[position]):
                zero_positions.append(position)
                position += 1
                continue
            segment = int(segment_ids_row[position])
            needs_reset = segment != current_segment
            if needs_reset:
                current_segment = segment
            stop = min(length, position + self.chunk_tokens)
            for candidate in range(position, stop):
                if (
                    not bool(token_mask_row[candidate])
                    or int(segment_ids_row[candidate]) != segment
                ):
                    stop = candidate
                    break
            if stop == position:
                continue
            chunks.append(_MemoryChunkSpec(position, stop, needs_reset, segment))
            position = stop
        return chunks, zero_positions

    def forward(
        self,
        hidden: torch.Tensor,
        token_embeddings: torch.Tensor,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        head: nn.Linear,
        state: FastWeightMemoryState | None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, FastWeightMemoryState, dict
    ]:
        device = hidden.device
        batch, length = hidden.shape[0], hidden.shape[1]
        if state is None:
            state = self.initial_state(batch, device)

        key = F.normalize(
            self.key_proj(hidden).reshape(batch, length, self.banks, self.key_dim),
            dim=-1,
        )
        query = F.normalize(
            self.query_proj(hidden).reshape(batch, length, self.banks, self.key_dim),
            dim=-1,
        )
        value = self.value_down_proj(token_embeddings)

        control_token_mask = token_mask.detach().to(device="cpu", dtype=torch.bool)
        control_segment_ids = segment_ids.detach().to(device="cpu", dtype=torch.long)

        plans: list[list[_MemoryChunkSpec]] = []
        for row in range(batch):
            chunks, _ = self._memory_chunk_plan(
                control_token_mask[row],
                control_segment_ids[row],
                int(state.segment_id[row].item()),
            )
            plans.append(chunks)

        read_all = hidden.new_zeros(batch, length, self.banks, self.value_dim)
        row_states = [
            FastWeightMemoryState(
                state.matrix[row : row + 1], state.segment_id[row : row + 1]
            )
            for row in range(batch)
        ]
        row_memory_energy: list[list[torch.Tensor]] = [[] for _ in range(batch)]

        max_chunks = max((len(plan) for plan in plans), default=0)
        log_decay = math.log(self.decay)
        for step in range(max_chunks):
            active_rows = [row for row in range(batch) if step < len(plans[row])]
            specs = [plans[row][step] for row in active_rows]
            num_active = len(active_rows)
            step_max_len = max(spec.stop - spec.start for spec in specs)

            key_step = key.new_zeros(step_max_len, num_active, self.banks, self.key_dim)
            value_step = value.new_zeros(step_max_len, num_active, self.value_dim)
            matrix_in = hidden.new_zeros(
                num_active, self.banks, self.key_dim, self.value_dim
            )

            for index, row in enumerate(active_rows):
                spec = specs[index]
                chunk_len = spec.stop - spec.start
                key_step[:chunk_len, index] = key[row, spec.start : spec.stop]
                value_step[:chunk_len, index] = value[row, spec.start : spec.stop]
                if spec.needs_reset:
                    row_states[row] = FastWeightMemoryState(
                        hidden.new_zeros(1, self.banks, self.key_dim, self.value_dim),
                        torch.full((1,), spec.segment, dtype=torch.long),
                    )
                matrix_in[index] = row_states[row].matrix[0]

            t_index = torch.arange(
                1, step_max_len + 1, device=device, dtype=hidden.dtype
            )
            rescale = torch.exp(-t_index * log_decay)
            decay_pow = torch.exp(t_index * log_decay)

            outer_kv = key_step[..., None] * value_step[:, :, None, None, :]
            rescaled = outer_kv * rescale[:, None, None, None, None]
            cumsum_rescaled = torch.cumsum(rescaled, dim=0)
            local = decay_pow[:, None, None, None, None] * cumsum_rescaled
            matrix_t = decay_pow[:, None, None, None, None] * matrix_in[None] + local

            query_step = key.new_zeros(
                step_max_len, num_active, self.banks, self.key_dim
            )
            for index, row in enumerate(active_rows):
                spec = specs[index]
                chunk_len = spec.stop - spec.start
                query_step[:chunk_len, index] = query[row, spec.start : spec.stop]
            read_step = (query_step[..., None] * matrix_t).sum(dim=-2)

            for index, row in enumerate(active_rows):
                spec = specs[index]
                chunk_len = spec.stop - spec.start
                last = chunk_len - 1
                read_all[row, spec.start : spec.stop] = read_step[:chunk_len, index]
                row_memory_energy[row].append(
                    matrix_t[:chunk_len, index].square().sum(dim=(-1, -2))
                )
                row_states[row] = FastWeightMemoryState(
                    matrix_t[last : last + 1, index],
                    torch.full((1,), spec.segment, dtype=torch.long),
                )

        next_state = FastWeightMemoryState(
            torch.cat([row_state.matrix for row_state in row_states], dim=0),
            torch.cat([row_state.segment_id for row_state in row_states], dim=0),
        )
        empty = hidden.new_zeros(0, self.banks)
        energy_parts = [
            torch.cat(parts, dim=0) if parts else empty for parts in row_memory_energy
        ]
        memory_energy = (
            torch.cat(energy_parts, dim=0)
            if any(part.numel() for part in energy_parts)
            else empty
        )

        read_flat = read_all.reshape(batch, length, self.banks * self.value_dim)
        gate_input = torch.cat([hidden, read_flat], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        memory_out = self.memory_out_proj(read_flat)
        logit_bias = head(self.value_up_proj(read_flat))

        return (
            memory_out,
            logit_bias,
            gate,
            next_state,
            {"memory_energy": memory_energy},
        )


class PureParallelGearLM(nn.Module):
    """Decoder LM with gear-only sequence mixing."""

    force_fp32_parameters = True

    def __init__(self, config: PureParallelGearConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [
                PureGearLayer(
                    config,
                    use_ffn=config.use_local_swiglu,
                    residual_floor=config.gear_residual_floor,
                )
                for _ in range(config.layers)
            ]
        )
        self.predictor = (
            PureGearLayer(
                config,
                banks=1,
                gears=config.predictor_gears,
                use_ffn=False,
                residual_floor=config.predictor_residual_floor,
            )
            if config.use_predictor_gear
            else None
        )
        self.final_norm = GearRMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.memory = (
            FastWeightMemory(config) if config.use_fast_weight_memory else None
        )
        self._boundary_detector = None
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)

    def configure_boundary_detector(self, tokenizer: Any) -> None:
        from ...data.sentence_boundaries import SentenceBoundaryDetector

        self._boundary_detector = SentenceBoundaryDetector(
            tokenizer,
            max_sentence_tokens=self.config.max_sentence_tokens,
        )

    def _masks(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor | None,
        segment_ids: torch.Tensor | None,
        sentence_end_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_mask is None:
            token_mask = torch.ones(token_ids.shape, dtype=torch.bool)
        if segment_ids is None:
            segment_ids = torch.zeros(token_ids.shape, dtype=torch.long)
        if sentence_end_mask is None:
            sentence_end_mask = torch.zeros(token_ids.shape, dtype=torch.bool)
        for name, value in (
            ("token_mask", token_mask),
            ("segment_ids", segment_ids),
            ("sentence_end_mask", sentence_end_mask),
        ):
            if value.shape != token_ids.shape:
                raise ValueError(f"{name} must match token_ids shape")
        token_mask = token_mask.detach().to(device="cpu", dtype=torch.bool)
        segment_ids = segment_ids.detach().to(device="cpu", dtype=torch.long)
        sentence_end_mask = sentence_end_mask.detach().to(
            device="cpu", dtype=torch.bool
        )
        if bool((token_mask & (segment_ids < 0)).any()):
            raise ValueError("attended tokens must have non-negative segment_ids")
        if bool((sentence_end_mask & ~token_mask).any()):
            raise ValueError("sentence boundaries cannot be placed on padding")
        if self.config.boundary_policy == "fixed":
            sentence_end_mask = torch.zeros_like(sentence_end_mask)
            for row in range(token_ids.shape[0]):
                span = 0
                previous = None
                for position in range(token_ids.shape[1]):
                    if not bool(token_mask[row, position]):
                        continue
                    segment = int(segment_ids[row, position])
                    if previous is None or segment != previous:
                        span = 0
                    span += 1
                    if span >= self.config.max_sentence_tokens:
                        sentence_end_mask[row, position] = True
                        span = 0
                    previous = segment
        return token_mask, segment_ids, sentence_end_mask

    def _forward_hidden(
        self,
        token_ids: torch.Tensor,
        *,
        cache: GearCache | None = None,
        token_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        sentence_end_mask: torch.Tensor | None = None,
        ablations: Iterable[str] = (),
    ) -> tuple[
        torch.Tensor,
        GearCache,
        list[dict[str, torch.Tensor]],
        dict[str, torch.Tensor],
    ]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        token_mask, segment_ids, sentence_end_mask = self._masks(
            token_ids,
            token_mask,
            segment_ids,
            sentence_end_mask,
        )
        disabled = frozenset(self.config.ablations) | frozenset(ablations)
        control_token_mask = token_mask
        control_segment_ids = segment_ids
        control_sentence_end_mask = sentence_end_mask
        token_embeddings = self.token(token_ids)
        hidden = token_embeddings
        next_states, records = [], []
        for index, layer in enumerate(self.layers):
            hidden, state, record = layer(
                hidden,
                token_mask=control_token_mask,
                segment_ids=control_segment_ids,
                sentence_end_mask=control_sentence_end_mask,
                state=None if cache is None else cache.layers[index],
                ablations=disabled,
            )
            next_states.append(state)
            records.append(record)
        if self.predictor is None or "no_predictor_gear" in disabled:
            predictor_state = None
        else:
            hidden, predictor_state, predictor_record = self.predictor(
                hidden,
                token_mask=control_token_mask,
                segment_ids=control_segment_ids,
                sentence_end_mask=control_sentence_end_mask,
                state=None if cache is None else cache.predictor,
                ablations=disabled,
            )
            records.append(predictor_record)
        memory_extras: dict[str, torch.Tensor] = {}
        memory_state = None if cache is None else cache.memory
        if self.memory is not None:
            memory_out, logit_bias, gate, memory_state, memory_diag = self.memory(
                hidden,
                token_embeddings,
                control_token_mask,
                control_segment_ids,
                self.head,
                memory_state,
            )
            hidden = (
                hidden
                + gate * torch.tanh(self.memory.memory_residual) * memory_out
            )
            memory_extras = {"gate": gate, "logit_bias": logit_bias, **memory_diag}
        processed = (
            token_mask.sum(dim=1)
            if cache is None
            else cache.tokens_processed + token_mask.sum(dim=1)
        )
        return (
            self.final_norm(hidden),
            GearCache(tuple(next_states), predictor_state, processed, memory_state),
            records,
            memory_extras,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        cache: GearCache | None = None,
        use_cache: bool = False,
        token_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        sentence_end_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, GearCache | None]:
        if token_mask is None:
            token_mask = attention_mask
        hidden, next_cache, _, memory_extras = self._forward_hidden(
            token_ids,
            cache=cache,
            token_mask=token_mask,
            segment_ids=segment_ids,
            sentence_end_mask=sentence_end_mask,
        )
        logits = self.head(hidden)
        if memory_extras:
            logits = logits + memory_extras["gate"] * memory_extras["logit_bias"]
        return logits, (next_cache if use_cache else None)

    @staticmethod
    def _valid_targets(
        tokens: torch.Tensor,
        loss_mask: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        segment_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        valid = torch.ones_like(tokens[:, 1:], dtype=torch.bool)
        if loss_mask is not None:
            valid &= loss_mask[:, 1:].bool()
        if token_mask is not None:
            valid &= token_mask[:, 1:].bool()
        if segment_ids is not None:
            valid &= segment_ids[:, 1:] == segment_ids[:, :-1]
        return valid

    def training_step(
        self,
        tokens: torch.Tensor,
        task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        metadata = task_metadata or {}
        token_mask = metadata.get("token_mask", metadata.get("attention_mask"))
        segment_ids = metadata.get("segment_ids")
        sentence_end_mask = metadata.get("sentence_end_mask")
        hidden, _, records, memory_extras = self._forward_hidden(
            tokens,
            token_mask=token_mask,
            segment_ids=segment_ids,
            sentence_end_mask=sentence_end_mask,
        )
        logits = self.head(hidden[:, :-1])
        if memory_extras:
            logits = logits + (
                memory_extras["gate"][:, :-1] * memory_extras["logit_bias"][:, :-1]
            )
        targets = tokens[:, 1:]
        per_token = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        valid = self._valid_targets(
            tokens,
            metadata.get("loss_mask"),
            token_mask,
            segment_ids,
        )
        language_modeling = per_token[valid].mean() if bool(valid.any()) else per_token.mean() * 0.0
        energies = torch.cat(
            [record["rotor_energy"].reshape(-1) for record in records]
        )
        radius = energies.clamp_min(1e-8).sqrt()
        rotor_energy = (
            (0.25 - radius).clamp_min(0.0).square()
            + (
                radius - self.config.rotor_radius_limit
            ).clamp_min(0.0).square()
        ).mean()
        omega = torch.cat([record["omega"].reshape(-1) for record in records])
        omega_saturation = (
            (omega.abs() / self.config.omega_limit - 0.9).clamp_min(0.0).square()
        ).mean()
        clutch = torch.cat([record["clutch"].reshape(-1) for record in records])
        retention = torch.cat(
            [record["retention"].reshape(-1) for record in records]
        )
        residual_scales = torch.stack(
            [record["gear_residual_scale"] for record in records]
        )
        dead_gear_fraction = torch.stack(
            [
                (
                    (
                        record["clutch"]
                        .float()
                        .mean(dim=(0, 3))
                        < 0.01
                    )
                    | (
                        record["clutch"]
                        .float()
                        .mean(dim=(0, 3))
                        > 0.99
                    )
                )
                .float()
                .mean()
                for record in records
            ]
        ).mean()
        # Stability regularization must prevent saturation without prescribing
        # a semantic utilization pattern or forcing every gear to have the
        # same mean/variance.  The previous target-mean/target-variance loss
        # kept clutches close to their initialization and made the write path
        # nearly token-invariant.
        clutch_collapse = (
            (0.05 - clutch).clamp_min(0.0).square()
            + (clutch - 0.95).clamp_min(0.0).square()
        ).mean()
        # Measure and protect each individual gear, not only a bank average.
        clutch_balance = torch.stack(
            [
                (
                    (
                        0.05
                        - record["clutch"].float().mean(dim=(0, 3))
                    )
                    .clamp_min(0.0)
                    .square()
                    + (
                        record["clutch"].float().mean(dim=(0, 3))
                        - 0.95
                    )
                    .clamp_min(0.0)
                    .square()
                )
                .mean()
                for record in records
            ]
        ).mean()
        scales = loss_term_scales or {}
        regularizer_scale = float(metadata.get("regularizer_scale", 1.0))
        total = scales.get("language_modeling", 1.0) * language_modeling
        total = total + regularizer_scale * (
            self.config.rotor_energy_weight
            * scales.get("rotor_energy", 1.0)
            * rotor_energy
            + self.config.omega_saturation_weight
            * scales.get("omega_saturation", 1.0)
            * omega_saturation
            + self.config.clutch_collapse_weight
            * scales.get("clutch_collapse", 1.0)
            * clutch_collapse
            + self.config.clutch_balance_weight
            * scales.get("clutch_balance", 1.0)
            * clutch_balance
        )
        metrics = {
            "language_modeling": language_modeling,
            "rotor_energy": rotor_energy,
            "omega_saturation": omega_saturation,
            "clutch_collapse": clutch_collapse,
            "clutch_balance": clutch_balance,
            "retention_mean": retention.mean(),
            "retention_min": retention.min(),
            "retention_max": retention.max(),
            "retention_std": retention.std(unbiased=False),
            "gear_residual_mean": residual_scales.mean(),
            "gear_residual_min": residual_scales.min(),
            "dead_gear_fraction": dead_gear_fraction,
            # Backward-compatible diagnostic name for existing result readers.
            "dead_bank_fraction": dead_gear_fraction,
            "regularizer_scale": language_modeling.new_tensor(regularizer_scale),
        }
        if memory_extras:
            gate_mean = memory_extras["gate"].mean()
            # Unannealed (no regularizer_scale factor): like clutch_balance,
            # a gate saturated fully open/closed has a vanishing gradient
            # and can't self-correct once annealing has weakened the pull
            # back toward the target mean.
            copy_gate_balance = (gate_mean - self.config.copy_gate_target_mean).square()
            memory_energy = memory_extras["memory_energy"]
            memory_radius = memory_energy.clamp_min(1e-8).sqrt()
            memory_energy_penalty = (
                (memory_radius - self.config.fast_weight_energy_limit)
                .clamp_min(0.0)
                .square()
                .mean()
            )
            total = total + self.config.copy_gate_balance_weight * copy_gate_balance
            total = total + regularizer_scale * (
                self.config.fast_weight_energy_weight * memory_energy_penalty
            )
            metrics["copy_gate_mean"] = gate_mean
            metrics["copy_gate_balance"] = copy_gate_balance
            metrics["memory_energy"] = memory_energy_penalty
        metrics["total"] = total
        return metrics

    @torch.no_grad()
    def diagnostics(
        self,
        token_ids: torch.Tensor,
        **kwargs,
    ) -> list[dict[str, torch.Tensor]]:
        if token_ids.shape[1] > self.config.diagnostic_max_tokens:
            token_ids = token_ids[:, -self.config.diagnostic_max_tokens :]
            kwargs = {
                key: (
                    value[:, -self.config.diagnostic_max_tokens :]
                    if torch.is_tensor(value) and value.ndim == 2
                    else value
                )
                for key, value in kwargs.items()
            }
        _, _, records, _ = self._forward_hidden(token_ids, **kwargs)
        return records

    @torch.no_grad()
    def component_logits(
        self,
        token_ids: torch.Tensor,
        disabled_components: Iterable[str] = (),
        **kwargs,
    ) -> torch.Tensor:
        hidden, _, _, memory_extras = self._forward_hidden(
            token_ids,
            ablations=tuple(disabled_components),
            **kwargs,
        )
        logits = self.head(hidden)
        if memory_extras:
            logits = logits + memory_extras["gate"] * memory_extras["logit_bias"]
        return logits

    @staticmethod
    def _sample_token(logits: torch.Tensor, config) -> torch.Tensor:
        if config is None or config.deterministic:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(float(config.temperature), 1e-5)
        if config.top_k > 0:
            threshold = logits.topk(
                min(config.top_k, logits.shape[-1]), dim=-1
            ).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if config.top_p < 1.0:
            sorted_logits, indices = logits.sort(dim=-1, descending=True)
            remove = sorted_logits.softmax(-1).cumsum(-1) > config.top_p
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                -1, indices, sorted_logits
            )
        return torch.multinomial(logits.softmax(dim=-1), 1)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        sampling_config=None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return torch.empty(
                prompt.shape[0], 0, dtype=torch.long, device=prompt.device
            )
        prompt_boundaries = None
        sentence_tokens: list[list[int]] | None = None
        if self._boundary_detector is not None:
            prompt_boundaries = torch.stack(
                [
                    self._boundary_detector.scan_tokens(
                        row.tolist(), close_final=False
                    )[1]
                    for row in prompt.detach().cpu()
                ]
            )
            sentence_tokens = [
                self._boundary_detector.trailing_sentence(row.tolist())
                for row in prompt.detach().cpu()
            ]
        logits, cache = self(
            prompt,
            use_cache=True,
            sentence_end_mask=prompt_boundaries,
        )
        token = self._sample_token(logits[:, -1], sampling_config)
        output = []
        for index in range(max_new_tokens):
            output.append(token)
            if index + 1 == max_new_tokens:
                break
            boundary = torch.zeros(token.shape, dtype=torch.bool)
            if self._boundary_detector is not None and sentence_tokens is not None:
                for row, value in enumerate(token[:, 0].tolist()):
                    sentence_tokens[row].append(int(value))
                    if self._boundary_detector.is_boundary_incremental(
                        sentence_tokens[row]
                    ):
                        boundary[row, 0] = True
                        sentence_tokens[row] = []
            logits, cache = self(
                token,
                cache=cache,
                use_cache=True,
                sentence_end_mask=boundary,
            )
            token = self._sample_token(logits[:, -1], sampling_config)
        return torch.cat(output, dim=1)

    def architecture_manifest(self) -> dict[str, Any]:
        state_values = (
            self.config.layers
            * self.config.num_banks
            * self.config.gears_per_bank
            * self.config.rotor_channels
            * 4
            + (
                self.config.predictor_gears * self.config.rotor_channels * 4
                if self.config.use_predictor_gear
                else 0
            )
            + (
                self.config.fast_weight_banks
                * self.config.fast_weight_key_dim
                * self.config.fast_weight_value_dim
                if self.config.use_fast_weight_memory
                else 0
            )
        )
        return {
            "name": "PureParallelGear",
            "version": 2,
            "config": self.config.to_dict(),
            "parameters": {
                "total": sum(parameter.numel() for parameter in self.parameters())
            },
            "state": {
                "floating_values_per_example": state_values,
                "cache_complexity": (
                    "O(layers*banks*gears*channels), independent of context"
                    if self.memory is None
                    else (
                        "O(layers*banks*gears*channels + "
                        "memory_banks*key_dim*value_dim), independent of context"
                    )
                ),
                "max_sentence_tokens": self.config.max_sentence_tokens,
                "settling_rounds": self.config.settling_rounds,
                "intra_sentence_clutch_tokens": (
                    self.config.intra_sentence_clutch_tokens
                ),
                "bank_roles": list(self.config.bank_roles),
                "retention_ranges": [
                    [
                        float(self.layers[0].retention_low[bank, 0, 0]),
                        float(self.layers[0].retention_high[bank, 0, 0]),
                    ]
                    for bank in range(self.layers[0].banks)
                ]
            },
            "invariants": {
                "sequence_mixing": (
                    "persistent_affine_rotors_and_explicit_clutches_only"
                    if self.memory is None
                    else "persistent_affine_rotors_plus_fast_weight_memory"
                ),
                "sentence_execution": "parallel_affine_scan",
                "self_attention": False,
                # The memory has cosine-similarity key/query projections,
                # but no fast-weight memory means no qkv-style mechanism at
                # all -- only flip this when it's actually present.
                "qkv_projections": self.memory is not None,
                "token_similarity": self.memory is not None,
                "history_retrieval": self.memory is not None,
                "history_tensor": False,
                "kv_cache": False,
                "token_routing": False,
                "transformer_blocks": False,
                "fast_weight_memory": self.memory is not None,
            },
        }


@MODELS.register("pure_parallel_gear")
def build_pure_parallel_gear(
    model_cfg: dict,
    vocab_size: int | None = None,
) -> PureParallelGearLM:
    config = dict(model_cfg)
    if vocab_size is not None:
        config["vocab_size"] = vocab_size
    return PureParallelGearLM(PureParallelGearConfig(**config))
