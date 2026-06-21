"""Pure Parallel Gear V3, bounded-memory hybrid, and bounded Transformer.

V3 replaces boundary-driven recurrent settling with a reset-aware associative
complex affine scan. The strict model has constant state and no token-addressed
history. The hybrid adds explicitly bounded local grouped-query attention.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS
from ..transformer.model import RMSNorm
from .attention import BoundedLocalAttention, LocalKVCache
from .mps_scan import mps_affine_scan
from .scan import chunked_affine_scan, complex_mul


DEFAULT_BANK_ROLES = (
    "surface_syntax",
    "relations_entities",
    "discourse_continuity",
    "planning_constraints",
)
DEFAULT_HALF_LIFE_BANDS = (
    (4.0, 16.0),
    (16.0, 64.0),
    (64.0, 256.0),
    (256.0, 2048.0),
)
DEFAULT_PERIOD_BANDS = DEFAULT_HALF_LIFE_BANDS


@dataclass(frozen=True)
class PureParallelGearV3Config:
    vocab_size: int
    dim: int = 192
    layers: int = 4
    ffn_dim: int | None = None
    num_banks: int = 4
    bank_roles: tuple[str, ...] = DEFAULT_BANK_ROLES
    gears_per_bank: int = 8
    rotor_channels: int = 1
    cell_dim: int = 16
    bank_rank: int = 16
    scan_chunk_tokens: int = 128
    half_life_bands: tuple[tuple[float, float], ...] = DEFAULT_HALF_LIFE_BANDS
    period_bands: tuple[tuple[float, float], ...] = DEFAULT_PERIOD_BANDS
    future_horizons: tuple[int, ...] = (4, 16, 64, 256)
    future_aux_weight: float = 0.10
    future_aux_decay_fraction: float = 0.80
    enforce_timescale_hierarchy: bool = True
    dropout: float = 0.0
    max_seq_len: int = 16384

    def __post_init__(self) -> None:
        object.__setattr__(self, "bank_roles", tuple(self.bank_roles))
        object.__setattr__(
            self,
            "half_life_bands",
            tuple(tuple(float(item) for item in band) for band in self.half_life_bands),
        )
        object.__setattr__(
            self,
            "period_bands",
            tuple(tuple(float(item) for item in band) for band in self.period_bands),
        )
        object.__setattr__(
            self,
            "future_horizons",
            tuple(int(value) for value in self.future_horizons),
        )
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least two")
        if self.dim < 8 or self.layers < 1:
            raise ValueError("dim must be >= 8 and layers must be positive")
        if self.num_banks < 1 or self.gears_per_bank < 2:
            raise ValueError("at least one bank and two gears are required")
        if self.rotor_channels < 1 or self.cell_dim < 2 or self.bank_rank < 1:
            raise ValueError("rotor, cell, and bank dimensions must be positive")
        if self.scan_chunk_tokens < 2:
            raise ValueError("scan_chunk_tokens must be at least two")
        if len(self.bank_roles) != self.num_banks:
            raise ValueError("bank_roles must contain one role per bank")
        if len(self.half_life_bands) != self.num_banks:
            raise ValueError("half_life_bands must contain one range per bank")
        if len(self.period_bands) != self.num_banks:
            raise ValueError("period_bands must contain one range per bank")
        if len(self.future_horizons) != self.num_banks:
            raise ValueError("future_horizons must contain one horizon per bank")
        previous_half_high = 0.0
        previous_period_high = 0.0
        for half_life, period in zip(self.half_life_bands, self.period_bands):
            if half_life[0] <= 0.0 or half_life[0] > half_life[1]:
                raise ValueError("invalid half-life range")
            if period[0] <= 0.0 or period[0] > period[1]:
                raise ValueError("invalid period range")
            if (
                self.enforce_timescale_hierarchy
                and half_life[0] < previous_half_high
            ):
                raise ValueError("half-life bands must not overlap or cross")
            if (
                self.enforce_timescale_hierarchy
                and period[0] < previous_period_high
            ):
                raise ValueError("period bands must not overlap or cross")
            previous_half_high = half_life[1]
            previous_period_high = period[1]
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.future_aux_weight < 0.0:
            raise ValueError("future_aux_weight must be non-negative")
        if not 0.0 < self.future_aux_decay_fraction <= 1.0:
            raise ValueError("future_aux_decay_fraction must be in (0, 1]")
        if self.ffn_dim is None:
            hidden = int(2 * (4 * self.dim) / 3)
            object.__setattr__(self, "ffn_dim", 32 * ((hidden + 31) // 32))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HybridParallelGearConfig(PureParallelGearV3Config):
    attention_window: int = 128
    attention_heads: int = 8
    attention_kv_heads: int = 2
    attention_every: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dim % self.attention_heads:
            raise ValueError("dim must be divisible by attention_heads")
        if self.attention_heads % self.attention_kv_heads:
            raise ValueError("attention_heads must be divisible by attention_kv_heads")
        if self.attention_window < 2 or self.attention_every < 1:
            raise ValueError("attention window/every must be positive")


@dataclass(frozen=True)
class BoundedTransformerConfig:
    vocab_size: int
    dim: int = 192
    layers: int = 4
    ffn_dim: int | None = None
    heads: int = 8
    kv_heads: int = 2
    attention_window: int = 128
    dropout: float = 0.0
    max_seq_len: int = 16384

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.dim < 8 or self.layers < 1:
            raise ValueError("invalid bounded Transformer dimensions")
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError("invalid grouped-query head dimensions")
        if self.attention_window < 2:
            raise ValueError("attention_window must be at least two")
        if self.ffn_dim is None:
            hidden = int(2 * (4 * self.dim) / 3)
            object.__setattr__(self, "ffn_dim", 32 * ((hidden + 31) // 32))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GearScanState:
    rotor: torch.Tensor
    segment_id: torch.Tensor

    def detach(self) -> "GearScanState":
        return GearScanState(self.rotor.detach(), self.segment_id.detach())

    def to(self, *args, **kwargs) -> "GearScanState":
        rotor = self.rotor.to(*args, **kwargs)
        return GearScanState(
            rotor,
            self.segment_id.to(device=rotor.device),
        )


@dataclass
class PureGearV3Cache:
    gear_states: tuple[GearScanState, ...]
    tokens_processed: torch.Tensor

    def detach(self) -> "PureGearV3Cache":
        return PureGearV3Cache(
            tuple(state.detach() for state in self.gear_states),
            self.tokens_processed.detach(),
        )

    def to(self, *args, **kwargs) -> "PureGearV3Cache":
        states = tuple(state.to(*args, **kwargs) for state in self.gear_states)
        device = states[0].rotor.device if states else self.tokens_processed.device
        return PureGearV3Cache(
            states,
            self.tokens_processed.to(device=device),
        )


@dataclass
class HybridGearCache:
    gear_states: tuple[GearScanState, ...]
    local_kv: tuple[LocalKVCache, ...]
    tokens_processed: torch.Tensor

    def detach(self) -> "HybridGearCache":
        return HybridGearCache(
            tuple(state.detach() for state in self.gear_states),
            tuple(cache.detach() for cache in self.local_kv),
            self.tokens_processed.detach(),
        )

    def to(self, *args, **kwargs) -> "HybridGearCache":
        states = tuple(state.to(*args, **kwargs) for state in self.gear_states)
        kv = tuple(cache.to(*args, **kwargs) for cache in self.local_kv)
        device = states[0].rotor.device if states else self.tokens_processed.device
        return HybridGearCache(
            states,
            kv,
            self.tokens_processed.to(device=device),
        )


@dataclass
class BoundedTransformerCache:
    local_kv: tuple[LocalKVCache, ...]
    tokens_processed: torch.Tensor

    def detach(self) -> "BoundedTransformerCache":
        return BoundedTransformerCache(
            tuple(cache.detach() for cache in self.local_kv),
            self.tokens_processed.detach(),
        )

    def to(self, *args, **kwargs) -> "BoundedTransformerCache":
        kv = tuple(cache.to(*args, **kwargs) for cache in self.local_kv)
        device = kv[0].key.device if kv else self.tokens_processed.device
        return BoundedTransformerCache(kv, self.tokens_processed.to(device=device))


class GearSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        up, gate = self.in_proj(value).chunk(2, dim=-1)
        return self.out_proj(up * F.silu(gate))


class PureGearV3Layer(nn.Module):
    """One contractive multi-timescale rotor layer."""

    def __init__(self, config: PureParallelGearV3Config) -> None:
        super().__init__()
        self.config = config
        self.banks = config.num_banks
        self.gears = config.gears_per_bank
        self.channels = config.rotor_channels
        self.cells = self.banks * self.gears * self.channels
        self.timescale_controls = self.banks * self.channels * 2
        self.write_controls = self.cells * 3
        self.residual_scale = 1.0 / math.sqrt(2.0 * config.layers)

        self.control_norm = RMSNorm(config.dim)
        self.control_projection = nn.Linear(
            config.dim,
            self.timescale_controls + self.write_controls,
            bias=True,
        )
        self.boundary_drive = nn.Parameter(torch.zeros(config.dim))

        phase = torch.empty(self.banks, self.gears, self.channels)
        for bank in range(self.banks):
            for gear in range(self.gears):
                phase[bank, gear] = (
                    2.0 * math.pi * gear / self.gears
                    + math.pi * bank / self.banks
                )
        self.initial_phase = nn.Parameter(phase)

        half_life = torch.tensor(config.half_life_bands, dtype=torch.float32)
        period = torch.tensor(config.period_bands, dtype=torch.float32)
        self.register_buffer(
            "log_half_life_low",
            half_life[:, 0].log()[:, None, None],
        )
        self.register_buffer(
            "log_half_life_span",
            (half_life[:, 1].log() - half_life[:, 0].log())[:, None, None],
        )
        self.register_buffer(
            "log_period_low",
            period[:, 0].log()[:, None, None],
        )
        self.register_buffer(
            "log_period_span",
            (period[:, 1].log() - period[:, 0].log())[:, None, None],
        )
        direction = torch.where(
            torch.arange(self.gears) % 2 == 0,
            torch.ones(self.gears),
            -torch.ones(self.gears),
        )
        self.register_buffer(
            "rotation_direction",
            direction[None, :, None],
        )
        self.register_buffer(
            "gear_log_fraction",
            torch.linspace(0.0, 1.0, self.gears)[None, :, None],
        )

        self.cell_encoder = nn.Sequential(
            nn.Linear(7, config.cell_dim, bias=False),
            nn.SiLU(),
            nn.Linear(config.cell_dim, config.cell_dim, bias=False),
            nn.SiLU(),
        )
        self.pool_logits = nn.Parameter(
            torch.zeros(self.banks, self.gears, self.channels)
        )
        self.bank_down = nn.Linear(config.cell_dim, config.bank_rank, bias=False)
        self.bank_up = nn.Linear(
            self.banks * config.bank_rank, config.dim, bias=False
        )
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = GearSwiGLU(config.dim, int(config.ffn_dim))
        self.dropout = nn.Dropout(config.dropout)

        nn.init.normal_(self.control_projection.weight, std=0.02)
        nn.init.zeros_(self.control_projection.bias)
        nn.init.normal_(self.bank_up.weight, std=0.01)
        nn.init.normal_(self.ffn.out_proj.weight, std=0.01)

    def initial_rotor(self, batch: int, device: torch.device) -> torch.Tensor:
        phase = self.initial_phase.float()
        rotor = torch.stack((phase.cos(), phase.sin()), dim=-1)
        return rotor[None].expand(batch, -1, -1, -1, -1).to(device)

    def initial_state(self, batch: int, device: torch.device) -> GearScanState:
        return GearScanState(
            self.initial_rotor(batch, device),
            torch.full((batch,), -1, device=device, dtype=torch.long),
        )

    def _controls(
        self,
        hidden: torch.Tensor,
        sentence_end_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        source = self.control_norm(hidden)
        source = source + sentence_end_mask[:, :, None].to(source.dtype) * (
            self.boundary_drive.to(source.dtype)
        )
        controls = self.control_projection(source).float()
        timescale = controls[..., : self.timescale_controls].reshape(
            hidden.shape[0],
            hidden.shape[1],
            self.banks,
            self.channels,
            2,
        )
        write_controls = controls[..., self.timescale_controls :].reshape(
            hidden.shape[0],
            hidden.shape[1],
            self.banks,
            self.gears,
            self.channels,
            3,
        )
        half_fraction = torch.sigmoid(
            timescale[..., 0]
        )[:, :, :, None, :]
        token_period_fraction = torch.sigmoid(
            timescale[..., 1]
        )[:, :, :, None, :]
        period_fraction = 0.5 * (
            token_period_fraction + self.gear_log_fraction
        )
        write_gate = torch.sigmoid(write_controls[..., 0])
        write = torch.tanh(write_controls[..., 1:3])

        half_life = torch.exp(
            self.log_half_life_low
            + half_fraction * self.log_half_life_span
        ).expand(-1, -1, -1, self.gears, -1)
        retention = torch.exp(
            math.log(0.5) / half_life
        )
        period = torch.exp(
            self.log_period_low + period_fraction * self.log_period_span
        )
        angle = self.rotation_direction * (2.0 * math.pi / period)
        multiplier = retention[..., None] * torch.stack(
            (angle.cos(), angle.sin()), dim=-1
        )
        bias = (
            (1.0 - retention.square()).clamp_min(1e-8).sqrt()[..., None]
            * write_gate[..., None]
            * write
        )
        return multiplier, bias, retention, write_gate, half_life, period

    def _transitions(
        self,
        multiplier: torch.Tensor,
        bias: torch.Tensor,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        state: GearScanState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_segment = torch.cat(
            (state.segment_id[:, None], segment_ids[:, :-1]),
            dim=1,
        )
        reset = token_mask & (segment_ids != previous_segment)
        identity = torch.zeros_like(multiplier)
        identity[..., 0] = 1.0
        multiplier = torch.where(
            token_mask[..., None, None, None, None],
            multiplier,
            identity,
        )
        bias = torch.where(
            token_mask[..., None, None, None, None],
            bias,
            torch.zeros_like(bias),
        )
        return multiplier, bias, reset

    def _readout(
        self,
        rotor: torch.Tensor,
        previous: torch.Tensor,
        retention: torch.Tensor,
        write_gate: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        radius = rotor.square().sum(dim=-1).clamp_min(1e-8).sqrt()
        delta = rotor - previous
        features = torch.cat(
            (
                rotor,
                radius[..., None],
                delta,
                retention[..., None],
                write_gate[..., None],
            ),
            dim=-1,
        ).to(hidden_dtype)
        encoded = self.cell_encoder(features)
        weights = self.pool_logits.flatten(1).softmax(dim=-1).reshape(
            self.banks, self.gears, self.channels
        )
        bank_state = (
            encoded * weights[None, None, ..., None].to(encoded.dtype)
        ).sum(dim=(3, 4))
        mixed = self.bank_down(bank_state).flatten(2)
        return self.bank_up(F.silu(mixed)), bank_state

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
        state: GearScanState | None = None,
    ) -> tuple[torch.Tensor, GearScanState, dict[str, torch.Tensor]]:
        batch, length, _ = hidden.shape
        state = state or self.initial_state(batch, hidden.device)
        multiplier, bias, retention, write_gate, half_life, period = self._controls(
            hidden, sentence_end_mask
        )
        multiplier, bias, reset = self._transitions(
            multiplier,
            bias,
            token_mask,
            segment_ids,
            state,
        )
        initial = self.initial_rotor(batch, hidden.device)
        reset_initial = self.initial_rotor(batch, hidden.device)
        reset_bias = (
            complex_mul(multiplier, reset_initial[:, None]) + bias
        )
        scan_multiplier = torch.where(
            reset[..., None, None, None, None],
            torch.zeros_like(multiplier),
            multiplier,
        )
        scan_bias = torch.where(
            reset[..., None, None, None, None],
            reset_bias,
            bias,
        )
        if hidden.device.type == "mps":
            rotor = mps_affine_scan(
                scan_multiplier,
                scan_bias,
                state.rotor.float(),
            )
        else:
            rotor, _, _ = chunked_affine_scan(
                scan_multiplier,
                scan_bias,
                state.rotor.float(),
                chunk_size=self.config.scan_chunk_tokens,
            )
        initial = reset_initial
        previous = torch.cat((state.rotor[:, None], rotor[:, :-1]), dim=1)
        previous = torch.where(
            reset[..., None, None, None, None],
            initial[:, None],
            previous,
        )
        gear_output, bank_state = self._readout(
            rotor,
            previous,
            retention,
            write_gate,
            hidden.dtype,
        )
        hidden = hidden + self.residual_scale * self.dropout(gear_output)
        hidden = hidden + self.residual_scale * self.dropout(
            self.ffn(self.ffn_norm(hidden))
        )

        valid_count = token_mask.long().sum(dim=1)
        last_index = (valid_count - 1).clamp_min(0)
        gather_rotor = last_index.reshape(batch, 1, 1, 1, 1, 1).expand(
            -1, 1, self.banks, self.gears, self.channels, 2
        )
        final_rotor = rotor.gather(1, gather_rotor).squeeze(1)
        final_segment = segment_ids.gather(1, last_index[:, None]).squeeze(1)
        has_token = valid_count > 0
        next_state = GearScanState(
            torch.where(
                has_token[:, None, None, None, None],
                final_rotor,
                state.rotor,
            ),
            torch.where(has_token, final_segment, state.segment_id),
        )
        return hidden, next_state, {
            "rotor": rotor,
            "retention": retention,
            "write_gate": write_gate,
            "half_life": half_life,
            "period": period,
            "bank_state": bank_state,
            "reset": reset,
        }


class _LanguageModelLossMixin:
    config: Any
    token: nn.Embedding
    head: nn.Linear

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
            values, indices = logits.sort(dim=-1, descending=True)
            remove = values.softmax(-1).cumsum(-1) > config.top_p
            remove[..., 0] = False
            values = values.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                -1, indices, values
            )
        return torch.multinomial(logits.softmax(dim=-1), 1)

    @staticmethod
    def _masks(
        tokens: torch.Tensor,
        metadata: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_mask = metadata.get("token_mask", metadata.get("attention_mask"))
        segment_ids = metadata.get("segment_ids")
        sentence_end = metadata.get("sentence_end_mask")
        if token_mask is None:
            token_mask = torch.ones_like(tokens, dtype=torch.bool)
        if segment_ids is None:
            segment_ids = torch.zeros_like(tokens, dtype=torch.long)
        if sentence_end is None:
            sentence_end = torch.zeros_like(tokens, dtype=torch.bool)
        return (
            token_mask.to(device=tokens.device, dtype=torch.bool),
            segment_ids.to(device=tokens.device, dtype=torch.long),
            sentence_end.to(device=tokens.device, dtype=torch.bool),
        )

    @staticmethod
    def _valid_targets(
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        loss_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Clone before in-place masking: token_mask also participates in the
        # differentiable transition graph through torch.where.
        valid = token_mask[:, 1:].bool().clone()
        valid &= segment_ids[:, 1:] == segment_ids[:, :-1]
        if loss_mask is not None:
            valid &= loss_mask[:, 1:].to(device=tokens.device).bool()
        return valid

    def _language_modeling_loss(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.head(hidden[:, :-1])
        targets = tokens[:, 1:]
        losses = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        valid_float = valid.to(losses.dtype)
        return (losses * valid_float).sum() / valid_float.sum().clamp_min(1)


class PureParallelGearV3LM(nn.Module, _LanguageModelLossMixin):
    """Strict constant-state Gear V3 language model."""

    def __init__(self, config: PureParallelGearV3Config) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [PureGearV3Layer(config) for _ in range(config.layers)]
        )
        self.final_norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.future_heads = nn.ModuleList(
            [
                nn.Linear(config.cell_dim, config.dim, bias=False)
                for _ in range(config.num_banks)
            ]
        )
        nn.init.normal_(self.token.weight, std=0.02)

    def _forward_hidden(
        self,
        tokens: torch.Tensor,
        *,
        cache: PureGearV3Cache | None = None,
        use_cache: bool = False,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        PureGearV3Cache | None,
        list[dict[str, torch.Tensor]],
    ]:
        hidden = self.token(tokens)
        states = []
        records = []
        for index, layer in enumerate(self.layers):
            hidden, state, record = layer(
                hidden,
                token_mask=token_mask,
                segment_ids=segment_ids,
                sentence_end_mask=sentence_end_mask,
                state=None if cache is None else cache.gear_states[index],
            )
            states.append(state)
            records.append(record)
        processed = token_mask.sum(dim=1)
        if cache is not None:
            processed = processed + cache.tokens_processed
        next_cache = (
            PureGearV3Cache(tuple(states), processed) if use_cache else None
        )
        return self.final_norm(hidden), next_cache, records

    def forward(
        self,
        tokens: torch.Tensor,
        cache: PureGearV3Cache | None = None,
        use_cache: bool = False,
        token_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        sentence_end_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PureGearV3Cache | None]:
        metadata = {
            "token_mask": token_mask if token_mask is not None else attention_mask,
            "segment_ids": segment_ids,
            "sentence_end_mask": sentence_end_mask,
        }
        masks = self._masks(tokens, metadata)
        hidden, next_cache, _ = self._forward_hidden(
            tokens,
            cache=cache,
            use_cache=use_cache,
            token_mask=masks[0],
            segment_ids=masks[1],
            sentence_end_mask=masks[2],
        )
        return self.head(hidden), next_cache

    def _future_loss(
        self,
        records: list[dict[str, torch.Tensor]],
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not records:
            return self.token.weight.sum() * 0.0
        bank_state = records[-1]["bank_state"]
        losses = []
        for bank, horizon in enumerate(self.config.future_horizons):
            if horizon >= tokens.shape[1]:
                continue
            prediction = F.normalize(
                self.future_heads[bank](bank_state[:, :-horizon, bank]).float(),
                dim=-1,
            )
            target = F.normalize(
                self.token(tokens[:, horizon:]).detach().float(),
                dim=-1,
            )
            valid = (
                token_mask[:, :-horizon]
                & token_mask[:, horizon:]
                & (segment_ids[:, :-horizon] == segment_ids[:, horizon:])
            )
            distance = 1.0 - (prediction * target).sum(dim=-1)
            valid_float = valid.to(distance.dtype)
            losses.append(
                (distance * valid_float).sum()
                / valid_float.sum().clamp_min(1)
            )
        return (
            torch.stack(losses).mean()
            if losses
            else self.token.weight.sum() * 0.0
        )

    def stream_training_step(
        self,
        tokens: torch.Tensor,
        *,
        cache: PureGearV3Cache | None = None,
        detach_cache: bool = False,
        task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ) -> tuple[dict[str, torch.Tensor], PureGearV3Cache]:
        metadata = task_metadata or {}
        token_mask, segment_ids, sentence_end = self._masks(tokens, metadata)
        hidden, next_cache, records = self._forward_hidden(
            tokens,
            cache=cache,
            use_cache=True,
            token_mask=token_mask,
            segment_ids=segment_ids,
            sentence_end_mask=sentence_end,
        )
        assert next_cache is not None
        valid = self._valid_targets(
            tokens,
            token_mask,
            segment_ids,
            metadata.get("loss_mask"),
        )
        language_modeling = self._language_modeling_loss(hidden, tokens, valid)
        future = self._future_loss(records, tokens, token_mask, segment_ids)
        scales = loss_term_scales or {}
        future_scale = float(metadata.get("future_aux_scale", 1.0))
        total = scales.get("language_modeling", 1.0) * language_modeling
        total = total + (
            self.config.future_aux_weight
            * future_scale
            * scales.get("future_state", 1.0)
            * future
        )
        metrics = {
            "language_modeling": language_modeling,
            "future_state": future,
            "future_aux_scale": language_modeling.new_tensor(future_scale),
            "retention_mean": torch.cat(
                [record["retention"].reshape(-1) for record in records]
            ).mean(),
            "write_gate_mean": torch.cat(
                [record["write_gate"].reshape(-1) for record in records]
            ).mean(),
            "total": total,
        }
        return metrics, (next_cache.detach() if detach_cache else next_cache)

    def training_step(
        self,
        tokens: torch.Tensor,
        task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        metrics, _ = self.stream_training_step(
            tokens,
            task_metadata=task_metadata,
            loss_term_scales=loss_term_scales,
        )
        return metrics

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, sampling_config=None):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return torch.empty(
                prompt.shape[0], 0, dtype=torch.long, device=prompt.device
            )
        logits, cache = self(prompt, use_cache=True)
        token = self._sample_token(logits[:, -1], sampling_config)
        output = []
        for index in range(max_new_tokens):
            output.append(token)
            if index + 1 == max_new_tokens:
                break
            logits, cache = self(token, cache=cache, use_cache=True)
            token = self._sample_token(logits[:, -1], sampling_config)
        return torch.cat(output, dim=1)

    def architecture_manifest(self) -> dict[str, Any]:
        state_values = (
            self.config.layers
            * self.config.num_banks
            * self.config.gears_per_bank
            * self.config.rotor_channels
            * 2
        )
        return {
            "name": "PureParallelGearV3",
            "version": 3,
            "config": self.config.to_dict(),
            "parameters": {
                "total": sum(parameter.numel() for parameter in self.parameters())
            },
            "state": {
                "floating_values_per_example": state_values,
                "cache_complexity": "O(layers*banks*gears*channels)",
                "scan": "two_level_associative_complex_affine",
                "half_life_bands": [list(value) for value in self.config.half_life_bands],
                "period_bands": [list(value) for value in self.config.period_bands],
                "timescale_hierarchy_enforced": (
                    self.config.enforce_timescale_hierarchy
                ),
            },
            "invariants": {
                "self_attention": False,
                "qkv_projections": False,
                "token_similarity": False,
                "history_retrieval": False,
                "history_tensor": False,
                "kv_cache": False,
                "token_routing": False,
                "transformer_blocks": False,
                "host_scalar_control_flow": False,
                "sequence_square_tensor": False,
            },
        }


class HybridParallelGearLM(PureParallelGearV3LM):
    """V3 with explicitly bounded local grouped-query attention."""

    config: HybridParallelGearConfig

    def __init__(self, config: HybridParallelGearConfig) -> None:
        super().__init__(config)
        attention_layers = [
            index
            for index in range(config.layers)
            if (index + 1) % config.attention_every == 0
        ]
        self.attention_layer_indices = tuple(attention_layers)
        self.attention_norms = nn.ModuleList(
            [RMSNorm(config.dim) for _ in attention_layers]
        )
        self.attentions = nn.ModuleList(
            [
                BoundedLocalAttention(
                    config.dim,
                    config.attention_heads,
                    config.attention_kv_heads,
                    config.attention_window,
                )
                for _ in attention_layers
            ]
        )
        self.attention_residual_scale = 1.0 / math.sqrt(
            max(1.0, 2.0 * len(attention_layers))
        )

    def _forward_hidden(
        self,
        tokens: torch.Tensor,
        *,
        cache: HybridGearCache | None = None,
        use_cache: bool = False,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        HybridGearCache | None,
        list[dict[str, torch.Tensor]],
    ]:
        hidden = self.token(tokens)
        states = []
        records = []
        kv_states = []
        attention_index = 0
        attention_set = set(self.attention_layer_indices)
        for index, layer in enumerate(self.layers):
            hidden, state, record = layer(
                hidden,
                token_mask=token_mask,
                segment_ids=segment_ids,
                sentence_end_mask=sentence_end_mask,
                state=None if cache is None else cache.gear_states[index],
            )
            states.append(state)
            records.append(record)
            if index in attention_set:
                attention_output, kv = self.attentions[attention_index](
                    self.attention_norms[attention_index](hidden),
                    token_mask=token_mask,
                    segment_ids=segment_ids,
                    cache=(
                        None
                        if cache is None
                        else cache.local_kv[attention_index]
                    ),
                    use_cache=use_cache,
                )
                hidden = hidden + self.attention_residual_scale * attention_output
                if use_cache:
                    assert kv is not None
                    kv_states.append(kv)
                attention_index += 1
        processed = token_mask.sum(dim=1)
        if cache is not None:
            processed = processed + cache.tokens_processed
        next_cache = (
            HybridGearCache(tuple(states), tuple(kv_states), processed)
            if use_cache
            else None
        )
        return self.final_norm(hidden), next_cache, records

    def architecture_manifest(self) -> dict[str, Any]:
        manifest = super().architecture_manifest()
        manifest["name"] = "HybridParallelGear"
        manifest["state"]["local_attention_window"] = self.config.attention_window
        manifest["state"]["cache_complexity"] = (
            "O(gear_state + attention_layers*window*kv_heads*head_dim)"
        )
        manifest["invariants"].update(
            {
                "self_attention": True,
                "qkv_projections": True,
                "token_similarity": True,
                "history_retrieval": True,
                "history_tensor": True,
                "kv_cache": True,
                "bounded_history": True,
            }
        )
        return manifest


class BoundedTransformerBlock(nn.Module):
    def __init__(self, config: BoundedTransformerConfig) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(2.0 * config.layers)
        self.attention_norm = RMSNorm(config.dim)
        self.attention = BoundedLocalAttention(
            config.dim,
            config.heads,
            config.kv_heads,
            config.attention_window,
        )
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = GearSwiGLU(config.dim, int(config.ffn_dim))
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        cache: LocalKVCache | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, LocalKVCache | None]:
        attended, next_cache = self.attention(
            self.attention_norm(hidden),
            token_mask=token_mask,
            segment_ids=segment_ids,
            cache=cache,
            use_cache=use_cache,
        )
        hidden = hidden + self.scale * self.dropout(attended)
        hidden = hidden + self.scale * self.dropout(
            self.ffn(self.ffn_norm(hidden))
        )
        return hidden, next_cache


class BoundedTransformerLM(nn.Module, _LanguageModelLossMixin):
    """Parameter-matchable bounded-memory Transformer control."""

    def __init__(self, config: BoundedTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [BoundedTransformerBlock(config) for _ in range(config.layers)]
        )
        self.final_norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        nn.init.normal_(self.token.weight, std=0.02)

    def _forward_hidden(
        self,
        tokens: torch.Tensor,
        *,
        cache: BoundedTransformerCache | None = None,
        use_cache: bool = False,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, BoundedTransformerCache | None, list]:
        del sentence_end_mask
        hidden = self.token(tokens)
        kv = []
        for index, block in enumerate(self.blocks):
            hidden, next_kv = block(
                hidden,
                token_mask=token_mask,
                segment_ids=segment_ids,
                cache=None if cache is None else cache.local_kv[index],
                use_cache=use_cache,
            )
            if use_cache:
                assert next_kv is not None
                kv.append(next_kv)
        processed = token_mask.sum(dim=1)
        if cache is not None:
            processed = processed + cache.tokens_processed
        next_cache = (
            BoundedTransformerCache(tuple(kv), processed)
            if use_cache
            else None
        )
        return self.final_norm(hidden), next_cache, []

    def forward(
        self,
        tokens: torch.Tensor,
        cache: BoundedTransformerCache | None = None,
        use_cache: bool = False,
        token_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        sentence_end_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        masks = self._masks(
            tokens,
            {
                "token_mask": token_mask if token_mask is not None else attention_mask,
                "segment_ids": segment_ids,
                "sentence_end_mask": sentence_end_mask,
            },
        )
        hidden, next_cache, _ = self._forward_hidden(
            tokens,
            cache=cache,
            use_cache=use_cache,
            token_mask=masks[0],
            segment_ids=masks[1],
            sentence_end_mask=masks[2],
        )
        return self.head(hidden), next_cache

    def stream_training_step(
        self,
        tokens: torch.Tensor,
        *,
        cache: BoundedTransformerCache | None = None,
        detach_cache: bool = False,
        task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ):
        metadata = task_metadata or {}
        token_mask, segment_ids, sentence_end = self._masks(tokens, metadata)
        hidden, next_cache, _ = self._forward_hidden(
            tokens,
            cache=cache,
            use_cache=True,
            token_mask=token_mask,
            segment_ids=segment_ids,
            sentence_end_mask=sentence_end,
        )
        assert next_cache is not None
        valid = self._valid_targets(
            tokens, token_mask, segment_ids, metadata.get("loss_mask")
        )
        language_modeling = self._language_modeling_loss(hidden, tokens, valid)
        scale = (loss_term_scales or {}).get("language_modeling", 1.0)
        metrics = {
            "language_modeling": language_modeling,
            "total": scale * language_modeling,
        }
        return metrics, (next_cache.detach() if detach_cache else next_cache)

    def training_step(self, tokens, task_metadata=None, loss_term_scales=None):
        metrics, _ = self.stream_training_step(
            tokens,
            task_metadata=task_metadata,
            loss_term_scales=loss_term_scales,
        )
        return metrics

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, sampling_config=None):
        if max_new_tokens == 0:
            return torch.empty(
                prompt.shape[0], 0, dtype=torch.long, device=prompt.device
            )
        logits, cache = self(prompt, use_cache=True)
        token = self._sample_token(logits[:, -1], sampling_config)
        output = []
        for index in range(max_new_tokens):
            output.append(token)
            if index + 1 == max_new_tokens:
                break
            logits, cache = self(token, cache=cache, use_cache=True)
            token = self._sample_token(logits[:, -1], sampling_config)
        return torch.cat(output, dim=1)

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "name": "BoundedTransformer",
            "version": 3,
            "config": self.config.to_dict(),
            "parameters": {
                "total": sum(parameter.numel() for parameter in self.parameters())
            },
            "state": {
                "local_attention_window": self.config.attention_window,
                "cache_complexity": "O(layers*window*kv_heads*head_dim)",
            },
            "invariants": {
                "self_attention": True,
                "qkv_projections": True,
                "history_tensor": True,
                "kv_cache": True,
                "bounded_history": True,
                "host_scalar_control_flow": False,
                "sequence_square_tensor": False,
                "gear_state": False,
            },
        }


@MODELS.register("pure_parallel_gear_v3")
def build_pure_parallel_gear_v3(model_cfg: dict, vocab_size: int | None = None):
    values = dict(model_cfg)
    if vocab_size is not None:
        values["vocab_size"] = vocab_size
    return PureParallelGearV3LM(PureParallelGearV3Config(**values))


@MODELS.register("hybrid_parallel_gear")
def build_hybrid_parallel_gear(model_cfg: dict, vocab_size: int | None = None):
    values = dict(model_cfg)
    if vocab_size is not None:
        values["vocab_size"] = vocab_size
    return HybridParallelGearLM(HybridParallelGearConfig(**values))


@MODELS.register("bounded_transformer")
def build_bounded_transformer(model_cfg: dict, vocab_size: int | None = None):
    values = dict(model_cfg)
    if vocab_size is not None:
        values["vocab_size"] = vocab_size
    return BoundedTransformerLM(BoundedTransformerConfig(**values))
