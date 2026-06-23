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
from ..bounded_hybrid_gear.scan import complex_mul
from .segment_scan import broadcast_affine, gather_chunk_summary, local_token_scan


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
    use_segment_scan: bool = True
    future_horizons: tuple[int, ...] | None = None
    future_aux_weight: float = 0.10
    bank_settle_cadence: tuple[int, ...] | None = None
    adaptive_settling_depth: bool = False
    content_triggered_settling: bool = False
    content_settle_threshold: float | None = None
    content_settle_min_gap: int = 4
    unify_memory_consolidation: bool = False
    intra_sequence_gradient_clip: float = 50.0

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
        if self.future_horizons is None:
            # Slower banks (higher index, per the retention_low/high
            # geometry above) predict further ahead -- one fixed token
            # horizon per bank, geometrically spaced the same way the
            # bounded_hybrid_gear sibling architecture's proven future-aux
            # loss does (its default for 4 banks is exactly (4, 16, 64, 256)).
            object.__setattr__(
                self,
                "future_horizons",
                tuple(4 * 4**bank for bank in range(self.num_banks)),
            )
        else:
            object.__setattr__(
                self, "future_horizons", tuple(int(value) for value in self.future_horizons)
            )
        if len(self.future_horizons) != self.num_banks:
            raise ValueError("future_horizons must contain one horizon per bank")
        if any(horizon < 1 for horizon in self.future_horizons):
            raise ValueError("future_horizons must all be positive")
        if self.future_aux_weight < 0.0:
            raise ValueError("future_aux_weight must be non-negative")
        if self.bank_settle_cadence is None:
            # Defaults to every bank settling on every real sentence
            # boundary -- identical to pre-Phase-3.2 behavior -- since,
            # unlike future_horizons above, this changes settle()'s actual
            # numeric output (which banks mix this event), not just an
            # additive training-only loss term. Per this project's own
            # standard (no quality claim ships without a paired ablation),
            # a non-trivial cadence is opt-in via explicit config, not a
            # silent default change.
            object.__setattr__(
                self, "bank_settle_cadence", tuple(1 for _ in range(self.num_banks))
            )
        else:
            object.__setattr__(
                self,
                "bank_settle_cadence",
                tuple(int(value) for value in self.bank_settle_cadence),
            )
        if len(self.bank_settle_cadence) != self.num_banks:
            raise ValueError("bank_settle_cadence must contain one cadence per bank")
        if any(cadence < 1 for cadence in self.bank_settle_cadence):
            raise ValueError("bank_settle_cadence must all be positive")
        if self.content_settle_threshold is None:
            # A fixed absolute default here (the original implementation
            # used 0.5) is disconnected from clutch_target_mean, the knob
            # that actually sets clutch's calibrated center -- found via a
            # real ablation run: with the project's default
            # clutch_target_mean=0.35 and clutch_collapse only guarding
            # against the [0.05, 0.95] saturation extremes (nothing pulls
            # clutch up specifically), a real 320K-token training run
            # never produced a single token above 0.43, so a 0.5 threshold
            # was structurally unreachable -- not merely undertrained. The
            # opposite failure mode is just as real: a config with
            # clutch_target_mean=0.6 would have made a fixed 0.5 threshold
            # fire on the *majority* of tokens by default. Deriving the
            # default from clutch_target_mean fixes both directions at
            # once; the margin (capped at half the remaining headroom to
            # 1.0) is a deliberately modest, clearly-named constant, not a
            # value validated by its own paired ablation yet.
            object.__setattr__(
                self,
                "content_settle_threshold",
                self.clutch_target_mean
                + min(0.10, 0.5 * (1.0 - self.clutch_target_mean)),
            )
        if not 0.0 < self.content_settle_threshold < 1.0:
            raise ValueError("content_settle_threshold must be in (0, 1)")
        if self.content_settle_threshold <= self.clutch_target_mean:
            raise ValueError(
                "content_settle_threshold must exceed clutch_target_mean"
            )
        if self.content_settle_min_gap < 1:
            raise ValueError("content_settle_min_gap must be at least 1")
        if self.intra_sequence_gradient_clip <= 0.0:
            raise ValueError("intra_sequence_gradient_clip must be positive")

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
    # Count of real sentence boundaries (not micro_clutch firings) seen so
    # far, per row -- read by settle() to gate gear-ratio multi-rate bank
    # settling (Phase 3.2: bank_settle_cadence). Optional, defaulting to
    # None (= no cadence gating, every bank always active, identical to
    # pre-Phase-3.2 behavior) so every pre-existing call site that builds a
    # GearState without it -- test legacy references, _state_row,
    # settle()'s own boundary_state placeholders elsewhere -- keeps working
    # unchanged. Device-resident (unlike sentence_length/segment_id, which
    # are deliberately CPU for _chunk_plan's Python bookkeeping) since it
    # only ever feeds a device-side per-bank mask inside settle(), called
    # once per chunk-step/round -- forcing a CPU<->device sync there would
    # reintroduce exactly the dispatch overhead Phase 1/2 removed.
    boundary_count: torch.Tensor | None = None

    def detach(self) -> "GearState":
        return GearState(
            self.rotor.detach(),
            self.omega.detach(),
            self.load.detach(),
            self.sentence_length.detach(),
            self.segment_id.detach(),
            self.boundary_count.detach() if self.boundary_count is not None else None,
        )

    def to(self, *args, **kwargs) -> "GearState":
        rotor = self.rotor.to(*args, **kwargs)
        return GearState(
            rotor,
            self.omega.to(*args, **kwargs),
            self.load.to(*args, **kwargs),
            self.sentence_length.cpu(),
            self.segment_id.cpu(),
            (
                self.boundary_count.to(*args, **kwargs)
                if self.boundary_count is not None
                else None
            ),
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
    breaks at the intra-sentence clutch interval, and only breaks at
    sentence ends (`boundary`) when config.unify_memory_consolidation is
    on (Phase 3.5) -- the memory is meant to persist across sentences
    within a document by default, only resetting at segment (document)
    changes; `boundary` additionally marks a forced split (not a reset)
    at a sentence end, the point where the consolidation gate applies."""

    start: int
    stop: int
    needs_reset: bool
    segment: int
    boundary: bool = False


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
        # config.bank_settle_cadence is defined relative to the main gear
        # stack's banks (config.num_banks); the predictor layer (banks=1,
        # see __init__'s `banks` override) is a separate short-horizon
        # integrator, not one of those timescale banks, so it always gets
        # an unconditional (cadence=1) buffer regardless of config.
        cadence = (
            config.bank_settle_cadence
            if self.banks == config.num_banks
            else tuple(1 for _ in range(self.banks))
        )
        self.register_buffer(
            "_bank_settle_cadence",
            torch.tensor(cadence, dtype=torch.long),
            persistent=False,
        )
        # Static fact about config, not data -- lets settle() skip
        # computing/applying the cadence mask entirely in the (default)
        # all-1s case, so opting out of Phase 3.2 costs nothing, not even
        # an extra elementwise compare-and-where per round.
        self._has_bank_cadence = any(value != 1 for value in cadence)

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
        # Phase 3.3: soft per-bank gate deciding how many of settle()'s
        # rounds k>=1 actually take effect for a given settle event, based
        # on that bank's rotor energy at entry -- registered as a Parameter
        # only when actually used (config.adaptive_settling_depth), mirroring
        # intra_gate/cross_gate/load_response/omega_response's own
        # conditional-Parameter-vs-buffer pattern above, so a disabled
        # (default) layer carries zero extra trainable parameters for this.
        if config.boundary_settling and config.adaptive_settling_depth:
            self.depth_response = nn.Parameter(torch.ones(self.banks))
            self.depth_threshold = nn.Parameter(torch.ones(self.banks))
        else:
            self.register_buffer("depth_response", torch.ones(self.banks))
            self.register_buffer("depth_threshold", torch.ones(self.banks))

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
            torch.zeros(batch, dtype=torch.long, device=device),
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
        active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized _mix_gears over a batch of mutually-disjoint pairs.

        lefts/rights index distinct, non-overlapping gear positions, so all
        pairs can be mixed in one shot instead of one _mix_gears call per
        pair -- this is the loop that dominated settle()'s cost.

        `active` ([rows, banks], bool), when given, is this round's
        gear-ratio cadence mask (Phase 3.2): a False entry means that
        bank does not settle this event at all, for this row, so it must
        come out exactly as it went in -- not just skip this specific
        pair-mixing op, but be fully unaffected. `activity`'s computation
        deliberately ignores this mask: it measures the learned gate's
        raw propensity to mix (a quantity training_step regularizes
        against collapsing to zero), not this event's actual realized
        effect, since bank_settle_cadence is a fixed structural schedule,
        not something the gate itself ever decides.
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
        mixed = rotor.index_copy(2, lefts, new_a).index_copy(2, rights, new_b)
        rotor = (
            torch.where(active[:, :, None, None, None], mixed, rotor)
            if active is not None
            else mixed
        )
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
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`active` ([rows, banks], bool), when given, is this round's
        gear-ratio cadence mask (Phase 3.2): this (left, right) pair only
        mixes for a row where *both* banks are active this event -- a bank
        not engaging this cycle can't be meshed with by its neighbor
        either, exactly like a disengaged gear in a real gear train."""
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
        new_a = cosine * a - sine * b
        new_b = sine * a + cosine * b
        if active is not None:
            pair_active = (active[:, left] & active[:, right])[:, None, None, None]
            new_a = torch.where(pair_active, new_a, a)
            new_b = torch.where(pair_active, new_b, b)
        # This ring step only ever touches 2 of `banks` bank slots; the prior
        # rebuild-the-whole-list-then-restack pattern paid an O(banks)
        # tensor-copy cost on every one of the ring's `banks` sequential
        # steps (O(banks^2) total) just to change 2 entries. A targeted
        # index_copy touches only those 2 slots, bit-exact, same sequential
        # dependency (still settle()'s only non-vectorized loop -- the ring
        # is a true data dependency: consecutive pairs share an endpoint, so
        # it cannot be split into disjoint passes the way the intra-bank
        # gear pairs were without changing settle()'s numeric output).
        index = torch.tensor([left, right], device=rotor.device)
        update = torch.stack((new_a, new_b), dim=1)
        return rotor.index_copy(1, index, update)

    def _bank_active(self, boundary_count: torch.Tensor | None) -> torch.Tensor | None:
        """Gear-ratio cadence mask (Phase 3.2), [rows, banks] bool: True
        where a bank settles this event for that row. None whenever
        there's nothing to gate -- either bank_settle_cadence is the
        default all-1s (every bank, every event) or the caller didn't
        supply a real boundary_count -- so callers can treat None as
        "every bank always active" and skip masking entirely."""
        if not self._has_bank_cadence or boundary_count is None:
            return None
        return (boundary_count[:, None] % self._bank_settle_cadence[None, :]) == 0

    def _round_depth_gate(
        self, log_energy_entry: torch.Tensor | None, round_index: int
    ) -> torch.Tensor | None:
        """Soft per-(row, bank) continuation weight for settle() round
        `round_index` (Phase 3.3), in [0, 1]: how much of *this* round's
        mixing actually counts, versus reverting to the rotor as it stood
        before this round. Round 0 always fully counts (every settle event
        gets at least one full round, unconditionally) -- only k>=1 are
        gated, and the threshold `depth_threshold[bank] * round_index`
        rises with k, so each additional round needs progressively more
        rotor energy at entry to justify itself. None when there's nothing
        to gate (feature disabled or round 0), matching _bank_active's
        convention so callers can skip the blend entirely.
        """
        if not self.config.adaptive_settling_depth or round_index == 0:
            return None
        return torch.sigmoid(
            self.depth_response[None] * log_energy_entry
            - self.depth_threshold[None] * round_index
        )

    def settle(
        self,
        state: GearState,
        *,
        cross_bank: bool = True,
        commuting_only: bool = False,
        use_load: bool = True,
    ) -> tuple[GearState, torch.Tensor]:
        rotor = state.rotor
        active = self._bank_active(state.boundary_count)
        log_energy_entry = (
            # [rows, banks] -- averaged over gears/channels, since the
            # depth gate is a per-bank decision (one "is this boundary
            # information-dense" signal per bank), not a per-gear one.
            rotor.square().sum(dim=-1).clamp_min(1e-8).sqrt().log().clamp(-4.0, 4.0)
            .mean(dim=(2, 3))
            if self.config.adaptive_settling_depth
            else None
        )
        activity = rotor.new_zeros(())
        for round_index in range(self.config.settling_rounds):
            rotor_before_round = rotor
            gate_round = min(round_index, self.intra_gate.shape[0] - 1)
            if self._even_gear_lefts.numel() > 0:
                rotor, pair_activity = self._mix_gear_pairs(
                    rotor,
                    state.omega,
                    state.load,
                    self._even_gear_lefts,
                    self._even_gear_rights,
                    self.intra_gate[gate_round, :, self._even_gear_lefts],
                    active,
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
                    active,
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
                        active,
                    )
                    activity = activity + torch.sigmoid(
                        self.cross_gate[gate_round, left]
                    ).mean()
            depth_gate = self._round_depth_gate(log_energy_entry, round_index)
            if depth_gate is not None:
                weight = depth_gate[:, :, None, None, None]
                rotor = weight * rotor + (1.0 - weight) * rotor_before_round

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
        if active is not None:
            # A bank inactive this event is fully unaffected by it -- not
            # just unmixed (already guaranteed by the per-round masking
            # above) but also no load/omega update, exactly like a
            # disengaged gear that simply isn't turning this cycle.
            bank_active = active[:, :, None, None]
            bounded_rotor = torch.where(
                bank_active[..., None], bounded_rotor, state.rotor
            )
            load = torch.where(bank_active, load, state.load)
            omega = torch.where(bank_active, omega, state.omega)
        count = max(1, self.config.settling_rounds)
        return (
            GearState(
                bounded_rotor,
                omega,
                load,
                torch.zeros_like(state.sentence_length),
                state.segment_id,
                state.boundary_count,
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

    def _content_trigger(
        self, clutch_controls: torch.Tensor
    ) -> torch.Tensor | None:
        """Phase 3.4: per-token boolean ([batch, length], CPU), True where
        the already-learned clutch signal alone -- averaged over banks/
        gears/channels, then thresholded -- is high enough to force a
        settle here, independent of sentence_end_mask. None when the
        feature is off, matching the rest of this file's "None means no
        gating" convention.

        This is a deterministic function of an already-learned signal
        (clutch_controls, trained via its existing uses: torque gating
        plus the clutch_collapse/clutch_balance regularizers in
        training_step), not a directly gradient-trained trigger -- a hard
        threshold comparison has no useful gradient w.r.t. either operand,
        and chunk-plan's downstream consumer is Python-level integer
        indexing, not a differentiable tensor op, so there is nowhere for
        a straight-through-style gradient on the trigger itself to flow
        to. Making the trigger *decision* genuinely learned via gradient
        descent would need every token to be a soft, blended settle point
        (the way Phase 3.3 blends settle() rounds), which would undo the
        chunked/discrete structure Phases 1-2 spent their effort
        vectorizing -- out of scope here.
        """
        if not self.config.content_triggered_settling:
            return None
        mean_clutch = clutch_controls.mean(dim=(2, 3, 4))
        return (
            (mean_clutch > self.config.content_settle_threshold)
            .detach()
            .to(device="cpu", dtype=torch.bool)
        )

    def _chunk_plan(
        self,
        token_mask_row: torch.Tensor,
        segment_ids_row: torch.Tensor,
        sentence_end_mask_row: torch.Tensor,
        initial_segment: int,
        initial_sentence_length: int,
        content_trigger_row: torch.Tensor | None = None,
    ) -> tuple[list[_ChunkSpec], list[int]]:
        """Find one row's sentence/clutch spans without touching any tensor
        math -- pure Python/CPU bookkeeping so forward() can batch every
        row's chunk processing together instead of looping row by row.

        Mirrors the boundary conditions that used to be interleaved with
        tensor processing in the per-row loop, unchanged.

        `content_trigger_row` (Phase 3.4, optional, [length] bool), when
        given, marks positions where the already-learned clutch signal
        alone (mean over banks/gears/channels, thresholded by
        config.content_settle_threshold) is enough to force a boundary
        here -- independent of `sentence_end_mask_row`. None (the default,
        also what's passed when content_triggered_settling is off) means
        no extra triggering, identical to pre-Phase-3.4 behavior.
        """
        length = int(token_mask_row.shape[0])
        chunks: list[_ChunkSpec] = []
        zero_positions: list[int] = []
        position = 0
        current_segment = initial_segment
        sentence_length = initial_sentence_length
        min_gap = self.config.content_settle_min_gap
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
                # content-triggered settling (Phase 3.4): the min_gap floor
                # bounds worst-case chunk count regardless of how the
                # learned clutch signal happens to behave (e.g. before it's
                # well-trained), the same role intra_sentence_clutch_tokens
                # already plays against the fixed micro_clutch cadence.
                if (
                    content_trigger_row is not None
                    and candidate - position + 1 >= min_gap
                    and bool(content_trigger_row[candidate])
                ):
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

    def _clip_recurrent_gradient(self, tensor: torch.Tensor) -> torch.Tensor:
        """Bound the gradient crossing a chunk-step boundary in the carried
        rotor/omega/load state, without touching the forward value at all.

        Found via direct empirical debugging: a real (lr=3e-3, long-context)
        training run produced a non-finite *combined* gradient norm even
        though every individual chunk's forward rotor value stayed small and
        finite throughout (confirmed by instrumenting every
        `_scan_token_dynamics` call in the crashing step: 696 chunk-steps
        chained together, max chunk length 32, forward rotor magnitudes
        ~1-1.6 the whole way through). That rules out a single-chunk forward
        overflow -- the failure is purely a backward-pass phenomenon: a per-
        chunk-step gradient gain only slightly above 1 (from a settle()
        Jacobian pushed there by an aggressive learning rate) compounds
        multiplicatively across hundreds of sequential chunk-steps within
        one training step's backward pass, the textbook exploding-gradient-
        through-depth failure mode for any long recurrence. The existing
        `gradient_explosion_threshold` clipping in the trainer can't prevent
        this: it only inspects the *combined* norm after backward finishes,
        by which point the sum-of-squares in that computation has already
        overflowed float32. This hook clips the gradient at the one point
        that actually matters -- where it crosses from one chunk-step's
        state into the next -- the same place a textbook truncated-BPTT
        gradient clip would sit, so a runaway gain can never compound past
        this bound regardless of how many chunk-steps end up chained.

        A no-op by construction whenever the gradient is already finite and
        below the threshold (the case for every existing test and every
        healthy training step observed in this project), so this changes
        nothing about well-behaved training -- it only engages in exactly
        the pathological regime that would otherwise hard-crash.

        Clips per row, on RMS (per-element magnitude within that row) --
        not the raw combined L2 norm over the whole tensor, and not jointly
        across rows -- for two reasons found via real test failures:
        1. A small test model (2 banks x 4 gears x 2 channels, batch 3) hit
           a perfectly legitimate combined norm of ~58 under a gradient-
           amplifying `.square().sum()` test loss, well past a threshold
           picked by eyeballing only the full-scale real-corpus model's
           norm. The combined L2 norm grows with element count (batch x
           banks x gears x channels), which varies a lot across configs/
           tests and has nothing to do with whether any individual value
           is actually dangerous; RMS does not have that artifact.
        2. `_forward_batched` calls this once per chunk-step on a tensor
           covering every row active that step, batched together, while
           the test suite's legacy reference implementations process one
           row at a time -- so clipping jointly across whichever rows
           happen to be batched together would make the result depend on
           batching, not just on each row's own gradient. Per-row keeps it
           identical to clipping each row in total isolation, regardless
           of which other rows are processed alongside it.
        """
        if not tensor.requires_grad:
            return tensor
        threshold = self.config.intra_sequence_gradient_clip

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            flat = grad.reshape(grad.shape[0], -1)
            finite_row = torch.isfinite(flat).all(dim=1)
            # Already corrupted by the time it reached this boundary for
            # that row -- rescaling a non-finite row can't recover a
            # meaningful direction, so drop just that row's contribution
            # rather than propagate the corruption further upstream.
            safe_flat = torch.where(
                finite_row[:, None], flat, torch.zeros_like(flat)
            )
            # Found via direct debugging: naive pow(2).mean().sqrt() on a
            # row whose elements are individually finite but large (this
            # is exactly the regime this hook exists to catch) can square
            # past float32's ~3.4e38 ceiling and overflow *its own*
            # intermediate computation to inf -- which then makes
            # `threshold / rms` evaluate to exactly 0.0, silently zeroing
            # the entire row's gradient instead of rescaling it (confirmed
            # as the real mechanism behind an "exactly 0.0 grad_norm"
            # training stall: the elements never tripped the finite_row
            # check above, since they were finite before squaring).
            # Scaling by each row's own max-abs value before squaring -- a
            # standard numerically-stable norm technique -- keeps the
            # squared term in [0, 1] regardless of the input's raw
            # magnitude, so this can no longer overflow no matter how
            # large (short of already being non-finite) the gradient is.
            row_max = safe_flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
            rms = (safe_flat / row_max).pow(2).mean(dim=1).sqrt() * row_max.squeeze(1)
            factor = torch.where(
                rms > threshold,
                threshold / rms.clamp_min(1e-30),
                torch.ones_like(rms),
            )
            return (safe_flat * factor[:, None]).reshape(grad.shape)

        tensor.register_hook(_hook)
        return tensor

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
        """Batch every row's chunk processing per chunk-step, and -- within
        a chunk-step -- gather/scatter every active row's slice with one
        vectorized indexing op instead of a Python loop over active rows.

        Chunk *boundaries* are still found by the per-row Python
        `_chunk_plan` pass (content-dependent; not yet vectorized -- see its
        docstring). What this method avoids is launching one small device
        kernel per active row per chunk-step to slice rows in and out of the
        step's working tensors: on MPS/CUDA, O(max_chunks * batch) tiny
        sequential launches per layer per training step is where wall-clock
        time actually went, far more than the Python bookkeeping itself.

        rotor_energy/clutch/retention in the returned record are pure
        per-token regularizer inputs consumed only via `.mean()` over all
        valid tokens (see training_step) -- never zipped against a specific
        token's position -- so collecting them as one flat, order-agnostic
        list per layer (rather than the original's per-row-then-concatenate
        order) is exactly equivalent for every consumer. `output` and
        `next_state` are not order-agnostic and are written to their exact
        (row, position) / (row,) destinations via masked scatter.
        """
        device = hidden.device
        batch, length = hidden.shape[0], hidden.shape[1]
        delta, clutch_controls, torque, retention_controls = (
            self._project_token_controls(hidden)
        )
        content_trigger = self._content_trigger(clutch_controls)

        plans: list[list[_ChunkSpec]] = []
        for row in range(batch):
            chunks, _zeros = self._chunk_plan(
                token_mask[row],
                segment_ids[row],
                sentence_end_mask[row],
                int(state.segment_id[row].item()),
                int(state.sentence_length[row].item()),
                content_trigger[row] if content_trigger is not None else None,
            )
            plans.append(chunks)

        output = hidden.new_zeros(batch, length, self.dim)
        fresh = self.initial_state(1, device)
        fresh_rotor, fresh_omega, fresh_load = (
            fresh.rotor[0], fresh.omega[0], fresh.load[0]
        )

        current_rotor = state.rotor.clone()
        current_omega = state.omega.clone()
        current_load = state.load.clone()
        current_sentence_length = state.sentence_length.clone()
        current_segment_id = state.segment_id.clone()
        current_boundary_count = (
            state.boundary_count.clone() if state.boundary_count is not None else None
        )

        energy_parts: list[torch.Tensor] = []
        clutch_parts: list[torch.Tensor] = []
        retention_parts: list[torch.Tensor] = []
        row_coupling: list[list[torch.Tensor]] = [[] for _ in range(batch)]
        settle_row_parts: list[torch.Tensor] = []
        settle_position_parts: list[torch.Tensor] = []
        settle_segment_parts: list[torch.Tensor] = []
        settle_rotor_parts: list[torch.Tensor] = []

        max_chunks = max((len(plan) for plan in plans), default=0)
        for step in range(max_chunks):
            active_rows = [row for row in range(batch) if step < len(plans[row])]
            specs = [plans[row][step] for row in active_rows]
            num_active = len(active_rows)
            step_max_len = max(spec.stop - spec.start for spec in specs)

            chunk_len_list = [spec.stop - spec.start for spec in specs]
            # One combined host->device transfer instead of four separate
            # torch.tensor(..., device=device) calls -- each pays its own
            # CPU-staging + dispatch cost, a fixed per-step overhead that
            # doesn't shrink with batch size and so dominates disproportion-
            # ately at small batch (measured: this loop is net slower than
            # the pre-vectorization per-row Python loop at batch=4, even
            # though it wins decisively at batch>=32).
            combined = torch.tensor(
                [
                    (
                        spec.stop - spec.start,
                        spec.start,
                        int(spec.needs_reset),
                        row,
                        int(spec.boundary),
                    )
                    for row, spec in zip(active_rows, specs)
                ],
                device=device,
                dtype=torch.long,
            )
            chunk_lens = combined[:, 0]
            starts = combined[:, 1]
            needs_reset_t = combined[:, 2].bool()
            rows_t = combined[:, 3]
            boundary_t_device = combined[:, 4].bool()
            # CPU-only siblings, built straight from the Python lists rather
            # than via chunk_lens.cpu()/rows_t.cpu() -- the latter would
            # force a device->host sync waiting on the transfer above.
            chunk_lens_cpu = torch.tensor(chunk_len_list, dtype=torch.long)
            rows_t_cpu = torch.tensor(active_rows, dtype=torch.long)
            boundary_t = torch.tensor(
                [spec.boundary for spec in specs], dtype=torch.bool
            )
            segment_t = torch.tensor(
                [spec.segment for spec in specs], dtype=torch.long
            )
            sentence_length_before_t = torch.tensor(
                [spec.sentence_length_before for spec in specs], dtype=torch.long
            )

            time_offsets = torch.arange(step_max_len, device=device)
            valid = time_offsets[:, None] < chunk_lens[None, :]  # [T, A]
            positions = (starts[None, :] + time_offsets[:, None]).clamp(
                0, length - 1
            )
            gather_rows = rows_t[None, :].expand(step_max_len, -1)

            def _gather(source: torch.Tensor, fill: float) -> torch.Tensor:
                gathered = source[gather_rows, positions]
                extra = source.dim() - 2
                mask = valid.reshape(*valid.shape, *([1] * extra))
                return torch.where(mask, gathered, torch.full_like(gathered, fill))

            delta_step = _gather(delta, 0.0)
            torque_step = _gather(torque, 0.0)
            retention_step = _gather(retention_controls, 1.0)
            clutch_step = _gather(clutch_controls, 0.0)

            reset_mask = needs_reset_t[:, None, None, None]
            rotor_in = torch.where(
                reset_mask[..., None].expand(
                    -1, self.banks, self.gears, self.channels, 2
                ),
                fresh_rotor[None].expand(num_active, -1, -1, -1, -1),
                current_rotor.index_select(0, rows_t),
            )
            omega_in = torch.where(
                reset_mask.expand(-1, self.banks, self.gears, self.channels),
                fresh_omega[None].expand(num_active, -1, -1, -1),
                current_omega.index_select(0, rows_t),
            )
            load_in = torch.where(
                reset_mask.expand(-1, self.banks, self.gears, self.channels),
                fresh_load[None].expand(num_active, -1, -1, -1),
                current_load.index_select(0, rows_t),
            )

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
                        (
                            current_boundary_count.index_select(0, rows_t).index_select(
                                0, gather_index
                            )
                            if current_boundary_count is not None
                            else None
                        ),
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
                    settle_row_parts.append(
                        torch.tensor(
                            [active_rows[index] for index in settle_indices],
                            dtype=torch.long,
                            device=device,
                        )
                    )
                    settle_position_parts.append(
                        torch.tensor(
                            [specs[index].stop - 1 for index in settle_indices],
                            dtype=torch.long,
                            device=device,
                        )
                    )
                    settle_segment_parts.append(
                        torch.tensor(
                            [specs[index].segment for index in settle_indices],
                            dtype=torch.long,
                            device=device,
                        )
                    )
                    settle_rotor_parts.append(settled_state.rotor)

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

            flat_valid = valid.reshape(-1)
            flat_rows = gather_rows.reshape(-1)[flat_valid]
            flat_positions = positions.reshape(-1)[flat_valid]
            flat_output = output_step.reshape(step_max_len * num_active, self.dim)[
                flat_valid
            ]
            output[flat_rows, flat_positions] = flat_output.to(output.dtype)

            flat_energy = (
                rotor_step.square().sum(dim=-1)
                .reshape(step_max_len * num_active, self.banks, self.gears, self.channels)
            )[flat_valid]
            flat_clutch = clutch_step.reshape(
                step_max_len * num_active, self.banks, self.gears, self.channels
            )[flat_valid]
            flat_retention = retention_step.reshape(
                step_max_len * num_active, self.banks, self.gears, self.channels
            )[flat_valid]
            energy_parts.append(flat_energy)
            clutch_parts.append(flat_clutch)
            retention_parts.append(flat_retention)

            last_index = chunk_lens - 1
            arange_active = torch.arange(num_active, device=device)
            last_rotor = rotor_step[last_index, arange_active]
            last_omega = omega_step[last_index, arange_active]
            last_load = load_step[last_index, arange_active]
            next_sentence_length = torch.where(
                boundary_t,
                torch.zeros_like(sentence_length_before_t),
                sentence_length_before_t + chunk_lens_cpu,
            )
            current_rotor = self._clip_recurrent_gradient(
                current_rotor.index_copy(0, rows_t, last_rotor)
            )
            current_omega = self._clip_recurrent_gradient(
                current_omega.index_copy(0, rows_t, last_omega)
            )
            current_load = self._clip_recurrent_gradient(
                current_load.index_copy(0, rows_t, last_load)
            )
            current_sentence_length = current_sentence_length.index_copy(
                0, rows_t_cpu, next_sentence_length
            )
            current_segment_id = current_segment_id.index_copy(
                0, rows_t_cpu, segment_t
            )
            if current_boundary_count is not None:
                next_boundary_count = (
                    current_boundary_count.index_select(0, rows_t)
                    + boundary_t_device.long()
                )
                current_boundary_count = current_boundary_count.index_copy(
                    0, rows_t, next_boundary_count
                )

        empty_state = hidden.new_zeros(0, self.banks, self.gears, self.channels)
        next_state = GearState(
            current_rotor,
            current_omega,
            current_load,
            current_sentence_length,
            current_segment_id,
            current_boundary_count,
        )
        per_row_coupling = [
            torch.stack(parts).mean() if parts else hidden.new_zeros(())
            for parts in row_coupling
        ]
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        empty_rotor = hidden.new_zeros(0, self.banks, self.gears, self.channels, 2)
        return output, next_state, {
            "rotor_energy": torch.cat(energy_parts, dim=0) if energy_parts else empty_state,
            "clutch": torch.cat(clutch_parts, dim=0) if clutch_parts else empty_state,
            "retention": torch.cat(retention_parts, dim=0) if retention_parts else empty_state,
            "coupling_activity": (
                torch.stack(per_row_coupling).mean()
                if per_row_coupling
                else hidden.new_zeros(())
            ),
            "omega": next_state.omega,
            "load": next_state.load,
            "rotor": next_state.rotor,
            # Per-settle-event (boundary or micro_clutch) snapshots, used
            # only by the LM-level future-rotor auxiliary loss (Phase 3.1):
            # which row/position/segment each settle fired at, and that
            # event's settled rotor state.
            "settle_row": (
                torch.cat(settle_row_parts) if settle_row_parts else empty_long
            ),
            "settle_position": (
                torch.cat(settle_position_parts) if settle_position_parts else empty_long
            ),
            "settle_segment": (
                torch.cat(settle_segment_parts) if settle_segment_parts else empty_long
            ),
            "settle_rotor": (
                torch.cat(settle_rotor_parts, dim=0) if settle_rotor_parts else empty_rotor
            ),
        }

    def _forward_segment_scan(
        self,
        hidden: torch.Tensor,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
        state: GearState,
        *,
        settling_enabled: bool,
        cross_bank: bool,
        commuting_only: bool,
        use_load: bool,
    ) -> tuple[torch.Tensor, GearState, dict[str, torch.Tensor]]:
        """Exact alternative to `_forward_batched`, valid only when angular
        velocity is fixed (the caller must only use this when `fixed_omega`
        is True -- see segment_scan.py's module docstring for why). Moves
        the per-token affine recurrence out of the chunk-step loop entirely
        (Level 0: one whole-sequence scan), leaving only a per-chunk-
        summary propagation across the genuinely irreducible settle()/reset
        dependency (Level 1: same round count as `_forward_batched`, but
        each round now touches only [batch, banks, gears, channels, ...]
        summaries, no per-token tensors), then one vectorized broadcast
        back to every token (Level 2).

        Unlike `_forward_batched`, this method needs `token_mask` both as a
        CPU-only input to the Python `_chunk_plan` bookkeeping *and* as a
        device-resident mask for the final vectorized output write -- so,
        unlike `_forward_batched`'s contract, callers must pass the
        original device-resident `token_mask`/`segment_ids`/
        `sentence_end_mask` here, not pre-cast CPU copies; the CPU copies
        needed for `_chunk_plan` are built internally below.
        """
        device = hidden.device
        batch, length = hidden.shape[0], hidden.shape[1]
        delta, clutch_controls, torque, retention_controls = (
            self._project_token_controls(hidden)
        )

        control_token_mask = token_mask.detach().to(device="cpu", dtype=torch.bool)
        control_segment_ids = segment_ids.detach().to(device="cpu", dtype=torch.long)
        control_sentence_end_mask = sentence_end_mask.detach().to(
            device="cpu", dtype=torch.bool
        )
        content_trigger = self._content_trigger(clutch_controls)
        plans: list[list[_ChunkSpec]] = []
        for row in range(batch):
            chunks, _zeros = self._chunk_plan(
                control_token_mask[row],
                control_segment_ids[row],
                control_sentence_end_mask[row],
                int(state.segment_id[row].item()),
                int(state.sentence_length[row].item()),
                content_trigger[row] if content_trigger is not None else None,
            )
            plans.append(chunks)
        max_chunks = max((len(plan) for plan in plans), default=0)

        fresh = self.initial_state(1, device)
        fresh_rotor, fresh_omega, fresh_load = (
            fresh.rotor[0], fresh.omega[0], fresh.load[0]
        )
        empty_state = hidden.new_zeros(0, self.banks, self.gears, self.channels)

        if max_chunks == 0:
            empty_long = torch.zeros(0, dtype=torch.long, device=device)
            empty_rotor = hidden.new_zeros(0, self.banks, self.gears, self.channels, 2)
            return (
                hidden.new_zeros(batch, length, self.dim),
                state,
                {
                    "rotor_energy": empty_state,
                    "clutch": empty_state,
                    "retention": empty_state,
                    "coupling_activity": hidden.new_zeros(()),
                    "omega": state.omega,
                    "load": state.load,
                    "rotor": state.rotor,
                    "settle_row": empty_long,
                    "settle_position": empty_long,
                    "settle_segment": empty_long,
                    "settle_rotor": empty_rotor,
                },
            )

        # ---- Level 0: one whole-sequence affine scan, no loop. ---------
        # omega is fixed (not carried state) by this path's precondition,
        # so every chunk's phase recurrence can use the same constant --
        # this is exactly the coupling that makes the general (learned
        # omega) case unable to take this shortcut.
        omega_const = self.config.omega_limit * torch.tanh(
            self.base_omega.float() / self.config.omega_limit
        )
        angle = delta + omega_const[None, None]
        scale = retention_controls
        multiplier = torch.stack(
            (scale * angle.cos(), scale * angle.sin()), dim=-1
        )
        bias = torque

        row_index_list: list[int] = []
        start_position_list: list[int] = []
        chunk_end_list: list[list[int]] = []
        needs_reset_list: list[list[int]] = []
        settle_list: list[list[int]] = []
        valid_round_list: list[list[int]] = []
        boundary_list: list[list[int]] = []
        sentence_length_before_list: list[list[int]] = []
        chunk_len_list: list[list[int]] = []
        segment_list: list[list[int]] = []
        for row, plan in enumerate(plans):
            pad = max_chunks - len(plan)
            for spec in plan:
                row_index_list.append(row)
                start_position_list.append(spec.start)
            chunk_end_list.append(
                [spec.stop - 1 for spec in plan] + [0] * pad
            )
            needs_reset_list.append(
                [int(spec.needs_reset) for spec in plan] + [0] * pad
            )
            settle_list.append(
                [int(spec.boundary or spec.micro_clutch) for spec in plan]
                + [0] * pad
            )
            valid_round_list.append([1] * len(plan) + [0] * pad)
            boundary_list.append(
                [int(spec.boundary) for spec in plan] + [0] * pad
            )
            sentence_length_before_list.append(
                [spec.sentence_length_before for spec in plan] + [0] * pad
            )
            chunk_len_list.append(
                [spec.stop - spec.start for spec in plan] + [0] * pad
            )
            segment_list.append(
                [spec.segment for spec in plan]
                + [int(state.segment_id[row])] * pad
            )

        reset_rows = torch.tensor(row_index_list, device=device, dtype=torch.long)
        reset_positions = torch.tensor(
            start_position_list, device=device, dtype=torch.long
        )
        reset_mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
        reset_mask[reset_rows, reset_positions] = True

        local_value, local_multiplier = local_token_scan(multiplier, bias, reset_mask)

        chunk_end_index = torch.tensor(chunk_end_list, device=device, dtype=torch.long)
        chunk_local_bias, chunk_local_multiplier = gather_chunk_summary(
            local_value, local_multiplier, chunk_end_index
        )

        needs_reset_all = torch.tensor(needs_reset_list, dtype=torch.bool, device=device)
        settle_all = torch.tensor(settle_list, dtype=torch.bool, device=device)
        valid_round_all = torch.tensor(valid_round_list, dtype=torch.bool, device=device)
        valid_round_all_cpu = torch.tensor(valid_round_list, dtype=torch.bool)
        boundary_all = torch.tensor(boundary_list, dtype=torch.bool)
        sentence_length_before_all = torch.tensor(
            sentence_length_before_list, dtype=torch.long
        )
        chunk_len_all = torch.tensor(chunk_len_list, dtype=torch.long)
        segment_all = torch.tensor(segment_list, dtype=torch.long)
        # Device copies built once here, not per-round inside the Level-1
        # loop below, to avoid a repeated device sync when recording each
        # settle event's segment id for the future-rotor auxiliary loss,
        # or (boundary_all_device) when incrementing the gear-ratio
        # cadence counter (Phase 3.2).
        segment_all_device = segment_all.to(device)
        boundary_all_device = boundary_all.to(device)

        # ---- Level 1: per-chunk-summary propagation. Irreducible in
        # round count (settle()'s nonlinearity, by design, cannot be
        # parallel-scanned over), but every round here only touches
        # [batch, banks, gears, channels, ...] summaries -- no per-token
        # tensor survives in this loop, unlike `_forward_batched`'s
        # per-round token-window gather/scatter.
        current_rotor = state.rotor.clone()
        current_load = state.load.clone()
        current_sentence_length = state.sentence_length.clone()
        current_segment_id = state.segment_id.clone()
        current_boundary_count = (
            state.boundary_count.clone() if state.boundary_count is not None else None
        )

        entry_rotor_parts: list[torch.Tensor] = []
        entry_load_parts: list[torch.Tensor] = []
        exit_rotor_parts: list[torch.Tensor] = []
        exit_load_parts: list[torch.Tensor] = []
        row_coupling: list[list[torch.Tensor]] = [[] for _ in range(batch)]
        settle_row_parts: list[torch.Tensor] = []
        settle_position_parts: list[torch.Tensor] = []
        settle_segment_parts: list[torch.Tensor] = []
        settle_rotor_parts: list[torch.Tensor] = []

        for k in range(max_chunks):
            valid_k = valid_round_all[:, k]
            needs_reset_k = needs_reset_all[:, k]

            reset_mask_k = needs_reset_k[:, None, None, None]
            rotor_in = torch.where(
                reset_mask_k[..., None].expand(
                    -1, self.banks, self.gears, self.channels, 2
                ),
                fresh_rotor[None].expand(batch, -1, -1, -1, -1),
                current_rotor,
            )
            load_in = torch.where(
                reset_mask_k.expand(-1, self.banks, self.gears, self.channels),
                fresh_load[None].expand(batch, -1, -1, -1),
                current_load,
            )
            entry_rotor_parts.append(rotor_in)
            entry_load_parts.append(load_in)

            natural_exit_rotor = (
                complex_mul(chunk_local_multiplier[:, k], rotor_in)
                + chunk_local_bias[:, k]
            )

            settle_mask_k = settle_all[:, k] & valid_k
            if settling_enabled and bool(settle_mask_k.any()):
                settle_rows = settle_mask_k.nonzero(as_tuple=True)[0]
                boundary_state = GearState(
                    natural_exit_rotor.index_select(0, settle_rows),
                    omega_const[None].expand(settle_rows.numel(), -1, -1, -1),
                    load_in.index_select(0, settle_rows),
                    hidden.new_zeros(settle_rows.numel(), dtype=torch.long),
                    hidden.new_zeros(settle_rows.numel(), dtype=torch.long),
                    (
                        current_boundary_count.index_select(0, settle_rows)
                        if current_boundary_count is not None
                        else None
                    ),
                )
                settled_state, coupling = self.settle(
                    boundary_state,
                    cross_bank=cross_bank,
                    commuting_only=commuting_only,
                    use_load=use_load,
                )
                exit_rotor_k = natural_exit_rotor.index_copy(
                    0, settle_rows, settled_state.rotor
                )
                exit_load_k = load_in.index_copy(0, settle_rows, settled_state.load)
                for row in settle_rows.tolist():
                    row_coupling[row].append(coupling)
                settle_row_parts.append(settle_rows)
                settle_position_parts.append(chunk_end_index[settle_rows, k])
                settle_segment_parts.append(segment_all_device[settle_rows, k])
                settle_rotor_parts.append(settled_state.rotor)
            else:
                exit_rotor_k = natural_exit_rotor
                exit_load_k = load_in

            exit_rotor_parts.append(exit_rotor_k)
            exit_load_parts.append(exit_load_k)

            current_rotor = self._clip_recurrent_gradient(
                torch.where(
                    valid_k[:, None, None, None, None], exit_rotor_k, current_rotor
                )
            )
            current_load = self._clip_recurrent_gradient(
                torch.where(
                    valid_k[:, None, None, None], exit_load_k, current_load
                )
            )

            boundary_k = boundary_all[:, k]
            next_sentence_length = torch.where(
                boundary_k,
                torch.zeros_like(sentence_length_before_all[:, k]),
                sentence_length_before_all[:, k] + chunk_len_all[:, k],
            )
            valid_k_cpu = valid_round_all_cpu[:, k]
            current_sentence_length = torch.where(
                valid_k_cpu, next_sentence_length, current_sentence_length
            )
            current_segment_id = torch.where(
                valid_k_cpu, segment_all[:, k], current_segment_id
            )
            if current_boundary_count is not None:
                current_boundary_count = torch.where(
                    valid_k & boundary_all_device[:, k],
                    current_boundary_count + 1,
                    current_boundary_count,
                )

        next_state = GearState(
            current_rotor,
            omega_const[None].expand(batch, -1, -1, -1).clone(),
            current_load,
            current_sentence_length,
            current_segment_id,
            current_boundary_count,
        )

        # ---- Level 2: one vectorized broadcast back to every token. ----
        entry_rotor_history = torch.stack(entry_rotor_parts, dim=1)
        entry_load_history = torch.stack(entry_load_parts, dim=1)
        exit_rotor_history = torch.stack(exit_rotor_parts, dim=1)
        exit_load_history = torch.stack(exit_load_parts, dim=1)

        chunk_index_of = (reset_mask.cumsum(dim=1) - 1).clamp_min(0)
        final_rotor = broadcast_affine(
            local_multiplier, local_value, entry_rotor_history, chunk_index_of
        )
        load_gather_index = chunk_index_of.reshape(batch, length, 1, 1, 1).expand(
            -1, -1, self.banks, self.gears, self.channels
        )
        load_broadcast = entry_load_history.gather(1, load_gather_index)
        rotor_gather_index = chunk_index_of.reshape(
            batch, length, 1, 1, 1, 1
        ).expand(-1, -1, self.banks, self.gears, self.channels, 2)
        entry_rotor_broadcast = entry_rotor_history.gather(1, rotor_gather_index)

        # settle()/reset overwrites exactly each chunk's own last token
        # (matching `_forward_batched`'s index_put at that position) --
        # everywhere else the broadcast formula above is already exact.
        flat_rows = (
            torch.arange(batch, device=device)[:, None]
            .expand(-1, max_chunks)
            .reshape(-1)
        )
        flat_positions = chunk_end_index.reshape(-1)
        flat_valid = valid_round_all.reshape(-1)
        scatter_rows = flat_rows[flat_valid]
        scatter_positions = flat_positions[flat_valid]
        final_rotor[scatter_rows, scatter_positions] = (
            exit_rotor_history.reshape(-1, self.banks, self.gears, self.channels, 2)[
                flat_valid
            ].to(final_rotor.dtype)
        )
        load_broadcast[scatter_rows, scatter_positions] = (
            exit_load_history.reshape(-1, self.banks, self.gears, self.channels)[
                flat_valid
            ].to(load_broadcast.dtype)
        )

        is_chunk_start = reset_mask[..., None, None, None, None]
        shifted_rotor = torch.cat(
            (final_rotor[:, :1], final_rotor[:, :-1]), dim=1
        )
        previous_rotor = torch.where(
            is_chunk_start, entry_rotor_broadcast, shifted_rotor
        )

        omega_for_readout = omega_const[None, None].expand(batch, length, -1, -1, -1)
        output = self._readout(
            final_rotor.reshape(batch * length, self.banks, self.gears, self.channels, 2),
            omega_for_readout.reshape(batch * length, self.banks, self.gears, self.channels),
            load_broadcast.reshape(batch * length, self.banks, self.gears, self.channels),
            clutch_controls.reshape(batch * length, self.banks, self.gears, self.channels),
            previous_rotor.reshape(
                batch * length, self.banks, self.gears, self.channels, 2
            ),
        ).reshape(batch, length, self.dim)
        valid_token = token_mask.to(device=device, dtype=torch.bool)
        output = torch.where(valid_token[..., None], output, torch.zeros_like(output))
        per_row_coupling = [
            torch.stack(parts).mean() if parts else hidden.new_zeros(())
            for parts in row_coupling
        ]
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        empty_rotor = hidden.new_zeros(0, self.banks, self.gears, self.channels, 2)
        return output, next_state, {
            "rotor_energy": final_rotor.square().sum(dim=-1)[valid_token],
            "clutch": clutch_controls[valid_token],
            "retention": retention_controls[valid_token],
            "coupling_activity": (
                torch.stack(per_row_coupling).mean()
                if per_row_coupling
                else hidden.new_zeros(())
            ),
            "omega": next_state.omega,
            "load": next_state.load,
            "rotor": next_state.rotor,
            "settle_row": (
                torch.cat(settle_row_parts) if settle_row_parts else empty_long
            ),
            "settle_position": (
                torch.cat(settle_position_parts) if settle_position_parts else empty_long
            ),
            "settle_segment": (
                torch.cat(settle_segment_parts) if settle_segment_parts else empty_long
            ),
            "settle_rotor": (
                torch.cat(settle_rotor_parts, dim=0) if settle_rotor_parts else empty_rotor
            ),
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

        if self.config.use_segment_scan and fixed_omega:
            # Exact only when omega is fixed -- see segment_scan.py's
            # module docstring. Falls back to `_forward_batched` otherwise
            # rather than raising, since `fixed_omega` is itself derived
            # from ablations that can vary call-to-call.
            gear_output, next_state, record = self._forward_segment_scan(
                hidden,
                token_mask,
                segment_ids,
                sentence_end_mask,
                state,
                settling_enabled=settling_enabled,
                cross_bank=cross_bank,
                commuting_only=commuting_only,
                use_load=use_load,
            )
        else:
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

        # Phase 3.5: "settle = consolidate" -- a learned per-bank gate
        # applied to the accumulated matrix specifically at sentence-
        # boundary chunk splits (not at segment-change resets, which
        # already zero it outright). Registered as a real Parameter only
        # when actually used, mirroring PureGearLayer's own conditional-
        # Parameter-vs-buffer pattern for gated mechanisms (intra_gate,
        # depth_response, etc.) so a disabled (default) memory carries no
        # extra trainable parameters. Init to sigmoid(3.0)~=0.95 -- a mild
        # consolidation pulse, not a near-total wipe, so enabling this
        # doesn't suddenly destabilize an otherwise-unrelated training run.
        if config.unify_memory_consolidation:
            self.consolidation_gate = nn.Parameter(torch.full((self.banks,), 3.0))
        else:
            self.register_buffer("consolidation_gate", torch.full((self.banks,), 3.0))

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
        sentence_end_mask_row: torch.Tensor | None = None,
    ) -> tuple[list[_MemoryChunkSpec], list[int]]:
        """`sentence_end_mask_row` (Phase 3.5, optional): when given (only
        when config.unify_memory_consolidation is on), a sentence end also
        forces a chunk split here -- a *split*, not a reset, so it leaves
        the exact accumulated matrix value unchanged on its own (the
        rescale-cumsum recurrence this chunks is exact regardless of where
        it's split); it only matters because it gives forward() a place to
        apply the consolidation gate. None (the default) means no extra
        splitting, identical to pre-Phase-3.5 behavior.
        """
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
            boundary = False
            for candidate in range(position, stop):
                if (
                    not bool(token_mask_row[candidate])
                    or int(segment_ids_row[candidate]) != segment
                ):
                    stop = candidate
                    break
                if (
                    sentence_end_mask_row is not None
                    and bool(sentence_end_mask_row[candidate])
                ):
                    stop = candidate + 1
                    boundary = True
                    break
            if stop == position:
                continue
            chunks.append(_MemoryChunkSpec(position, stop, needs_reset, segment, boundary))
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
        sentence_end_mask: torch.Tensor | None = None,
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
        control_sentence_end_mask = (
            sentence_end_mask.detach().to(device="cpu", dtype=torch.bool)
            if self.config.unify_memory_consolidation and sentence_end_mask is not None
            else None
        )

        plans: list[list[_MemoryChunkSpec]] = []
        for row in range(batch):
            chunks, _ = self._memory_chunk_plan(
                control_token_mask[row],
                control_segment_ids[row],
                int(state.segment_id[row].item()),
                (
                    control_sentence_end_mask[row]
                    if control_sentence_end_mask is not None
                    else None
                ),
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
                carried_matrix = matrix_t[last : last + 1, index]
                if spec.boundary:
                    # Phase 3.5 "settle = consolidate": a learned per-bank
                    # pulse applied only at a real sentence-boundary chunk
                    # split, not at the ordinary chunk_tokens-truncation
                    # splits or segment-change resets above.
                    carried_matrix = carried_matrix * torch.sigmoid(
                        self.consolidation_gate
                    )[None, :, None, None]
                row_states[row] = FastWeightMemoryState(
                    carried_matrix,
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
        # Future-rotor auxiliary loss (Phase 3.1): training-only, zero
        # inference cost, same pattern bounded_hybrid_gear's
        # BlockHybridGearV4LM already uses (future_heads + future_horizons)
        # -- one linear head per bank projecting that bank's settled rotor
        # state into embedding space, trained to predict the embedding of
        # the token future_horizons[bank] positions ahead of each settle
        # point. Ported, not re-derived: same per-bank-horizon structure,
        # same cosine-distance training signal, same detached target.
        self.future_heads = nn.ModuleList(
            [
                nn.Linear(
                    config.gears_per_bank * config.rotor_channels * 2,
                    config.dim,
                    bias=False,
                )
                for _ in range(config.num_banks)
            ]
        )
        self.register_buffer(
            "_future_horizon_offsets",
            torch.tensor(config.future_horizons, dtype=torch.long),
            persistent=False,
        )
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
                control_sentence_end_mask,
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

    def _future_loss(
        self,
        record: dict[str, torch.Tensor],
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Training-only auxiliary loss: predict, from each settle event's
        settled rotor state, the embedding of the token `future_horizons
        [bank]` positions ahead -- ported from bounded_hybrid_gear's
        BlockHybridGearV4LM._future_loss (same per-bank-horizon structure,
        same cosine-distance signal against a detached target embedding).
        Zero inference cost: only ever called from training_step, never
        from forward()/generation.
        """
        settle_row = record["settle_row"]
        if settle_row.numel() == 0:
            return self.token.weight.sum() * 0.0
        settle_position = record["settle_position"]
        settle_segment = record["settle_segment"]
        settle_rotor = record["settle_rotor"]

        length = tokens.shape[1]
        horizons = self._future_horizon_offsets
        target_position = settle_position[:, None] + horizons[None]
        within = target_position < length
        clamped = target_position.clamp_max(length - 1)
        rows = settle_row[:, None].expand_as(clamped)
        target_tokens = tokens[rows, clamped]
        target_segments = segment_ids[rows, clamped]
        target_valid = token_mask[rows, clamped]

        weights = torch.stack(
            [head.weight for head in self.future_heads], dim=0
        )
        flat_rotor = settle_rotor.flatten(2)
        prediction = F.normalize(
            torch.einsum("nkf,kdf->nkd", flat_rotor, weights).float(), dim=-1
        )
        target = F.normalize(
            self.token(target_tokens).detach().float(), dim=-1
        )

        valid = within & target_valid & (target_segments == settle_segment[:, None])
        distance = 1.0 - (prediction * target).sum(dim=-1)
        valid_float = valid.to(distance.dtype)
        per_bank_count = valid_float.sum(dim=0).clamp_min(1)
        per_bank_loss = (distance * valid_float).sum(dim=0) / per_bank_count
        return per_bank_loss.mean()

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
        # Ported from bounded_hybrid_gear's BlockHybridGearV4LM: an
        # auxiliary predictive objective, not a stability regularizer, so
        # it is weighted directly (not nested inside regularizer_scale's
        # annealing) and skipped entirely when its weight is zero -- it's
        # the one term here with a non-trivial compute cost (an embedding
        # lookup plus a per-bank einsum), unlike the cheap elementwise
        # regularizers below.
        future_scale = float(metadata.get("future_aux_scale", 1.0))
        future_weight = (
            self.config.future_aux_weight
            * future_scale
            * scales.get("future_rotor", 1.0)
        )
        if future_weight != 0.0:
            # _future_loss indexes tokens/masks with device-resident
            # settle_row/clamped index tensors (see _future_loss), so
            # these must stay on tokens' own device -- unlike
            # self._masks(), which deliberately forces CPU for
            # _chunk_plan's Python bookkeeping (not needed here).
            resolved_token_mask = (
                token_mask.to(device=tokens.device, dtype=torch.bool)
                if token_mask is not None
                else torch.ones_like(tokens, dtype=torch.bool)
            )
            resolved_segment_ids = (
                segment_ids.to(device=tokens.device, dtype=torch.long)
                if segment_ids is not None
                else torch.zeros_like(tokens)
            )
            future_rotor_loss = self._future_loss(
                records[len(self.layers) - 1],
                tokens,
                resolved_token_mask,
                resolved_segment_ids,
            )
        else:
            future_rotor_loss = hidden.sum() * 0.0
        total = scales.get("language_modeling", 1.0) * language_modeling
        total = total + future_weight * future_rotor_loss
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
            "future_rotor": future_rotor_loss,
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
                ],
                "control_path": (
                    "cpu_planned_sentence_chunks_with_sequential_settling"
                    if self.config.boundary_settling
                    else "cpu_planned_sentence_chunks"
                ),
            },
            "invariants": {
                "sequence_mixing": (
                    "persistent_affine_rotors_and_explicit_clutches_only"
                    if self.memory is None
                    else "persistent_affine_rotors_plus_fast_weight_memory"
                ),
                "sentence_execution": (
                    "parallel_affine_scan_with_sequential_boundary_settling"
                    if self.config.boundary_settling
                    else "parallel_affine_scan"
                ),
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
                "host_scalar_control_flow": True,
                "sequence_square_tensor": False,
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
