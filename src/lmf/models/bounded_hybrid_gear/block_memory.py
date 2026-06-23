"""Block-rate Bounded Hybrid Gear memory.

Uses the proven bounded-attention trunk at token rate and updates persistent
Gear memory once per fixed token block.  Tokens in a block can only consume
memory produced by completed prior blocks, preserving causality while reducing
scan length by ``block_tokens``. Three switchable fusion strategies (additive,
selective-FiLM, bank-router) are registered as separate model names.
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
from .attention import LocalKVCache
from .model import (
    DEFAULT_BANK_ROLES,
    DEFAULT_HALF_LIFE_BANDS,
    DEFAULT_PERIOD_BANDS,
    BoundedTransformerBlock,
    GearScanState,
    PureParallelGearV3Config,
    _LanguageModelLossMixin,
)
from .mps_scan import mps_affine_scan
from .scan import chunked_affine_scan, complex_mul


@dataclass(frozen=True)
class BlockHybridGearV4Config:
    vocab_size: int
    dim: int = 192
    layers: int = 4
    ffn_dim: int | None = None
    heads: int = 8
    kv_heads: int = 2
    attention_window: int = 128
    block_tokens: int = 128
    gear_every: int = 2
    gear_placement: str = "pre_next"
    fusion_mode: str = "additive"
    fusion_rank: int = 32
    num_banks: int = 4
    bank_roles: tuple[str, ...] = DEFAULT_BANK_ROLES
    gears_per_bank: int = 8
    rotor_channels: int = 1
    cell_dim: int = 16
    bank_rank: int = 16
    half_life_bands: tuple[tuple[float, float], ...] = DEFAULT_HALF_LIFE_BANDS
    period_bands: tuple[tuple[float, float], ...] = DEFAULT_PERIOD_BANDS
    future_horizons: tuple[int, ...] = (4, 16, 64, 256)
    future_aux_weight: float = 0.10
    future_aux_decay_fraction: float = 0.80
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
        if self.vocab_size < 2 or self.dim < 8 or self.layers < 1:
            raise ValueError("invalid V4 dimensions")
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError("invalid V4 grouped-query head dimensions")
        if self.attention_window < 2 or self.block_tokens < 2:
            raise ValueError("attention_window and block_tokens must be >= 2")
        if self.gear_every < 1:
            raise ValueError("gear_every must be positive")
        if self.gear_placement not in {"pre_next", "post_group"}:
            raise ValueError(
                "gear_placement must be 'pre_next' or 'post_group'"
            )
        if self.fusion_mode not in {
            "additive",
            "selective_film",
            "bank_router",
        }:
            raise ValueError(
                "fusion_mode must be additive, selective_film, or bank_router"
            )
        if self.fusion_rank < 1:
            raise ValueError("fusion_rank must be positive")
        if self.layers < self.gear_every:
            raise ValueError("V4 requires at least one block-rate Gear memory")
        if self.ffn_dim is None:
            hidden = int(2 * (4 * self.dim) / 3)
            object.__setattr__(self, "ffn_dim", 32 * ((hidden + 31) // 32))
        # Reuse the strict V3 validator for all mechanical invariants.
        PureParallelGearV3Config(
            vocab_size=self.vocab_size,
            dim=self.dim,
            layers=max(1, self.layers // self.gear_every),
            ffn_dim=self.dim,
            num_banks=self.num_banks,
            bank_roles=self.bank_roles,
            gears_per_bank=self.gears_per_bank,
            rotor_channels=self.rotor_channels,
            cell_dim=self.cell_dim,
            bank_rank=self.bank_rank,
            half_life_bands=self.half_life_bands,
            period_bands=self.period_bands,
            future_horizons=self.future_horizons,
            future_aux_weight=self.future_aux_weight,
            future_aux_decay_fraction=self.future_aux_decay_fraction,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlockGearMemoryCache:
    state: GearScanState
    context: torch.Tensor
    accumulator: torch.Tensor
    accumulator_count: torch.Tensor
    accumulator_segment: torch.Tensor
    block_position: torch.Tensor
    bank_context: torch.Tensor

    def detach(self) -> "BlockGearMemoryCache":
        return BlockGearMemoryCache(
            self.state.detach(),
            self.context.detach(),
            self.accumulator.detach(),
            self.accumulator_count.detach(),
            self.accumulator_segment.detach(),
            self.block_position.detach(),
            self.bank_context.detach(),
        )

    def to(self, *args, **kwargs) -> "BlockGearMemoryCache":
        state = self.state.to(*args, **kwargs)
        context = self.context.to(*args, **kwargs)
        device = context.device
        return BlockGearMemoryCache(
            state,
            context,
            self.accumulator.to(*args, **kwargs),
            self.accumulator_count.to(device=device),
            self.accumulator_segment.to(device=device),
            self.block_position.to(device=device),
            self.bank_context.to(*args, **kwargs),
        )


@dataclass
class BlockHybridGearV4Cache:
    local_kv: tuple[LocalKVCache, ...]
    gear_memory: tuple[BlockGearMemoryCache, ...]
    tokens_processed: torch.Tensor
    block_offset: int = 0

    def detach(self) -> "BlockHybridGearV4Cache":
        return BlockHybridGearV4Cache(
            tuple(cache.detach() for cache in self.local_kv),
            tuple(cache.detach() for cache in self.gear_memory),
            self.tokens_processed.detach(),
            self.block_offset,
        )

    def to(self, *args, **kwargs) -> "BlockHybridGearV4Cache":
        local_kv = tuple(cache.to(*args, **kwargs) for cache in self.local_kv)
        gear_memory = tuple(
            cache.to(*args, **kwargs) for cache in self.gear_memory
        )
        if gear_memory:
            device = gear_memory[0].context.device
        elif local_kv:
            device = local_kv[0].key.device
        else:
            device = self.tokens_processed.device
        return BlockHybridGearV4Cache(
            local_kv,
            gear_memory,
            self.tokens_processed.to(device=device),
            self.block_offset,
        )


class StaticBlockRotorMemory(nn.Module):
    """Hardware-efficient fixed-timescale rotor updated at block rate."""

    def __init__(self, config: BlockHybridGearV4Config) -> None:
        super().__init__()
        self.banks = config.num_banks
        self.gears = config.gears_per_bank
        self.channels = config.rotor_channels
        self.cells = self.banks * self.gears * self.channels
        self.write_projection = nn.Linear(
            config.dim,
            self.cells * 3,
            bias=True,
        )
        phase = torch.empty(self.banks, self.gears, self.channels)
        for bank in range(self.banks):
            for gear in range(self.gears):
                phase[bank, gear] = (
                    2.0 * math.pi * gear / self.gears
                    + math.pi * bank / self.banks
                )
        self.initial_phase = nn.Parameter(phase)

        bands = torch.tensor(config.half_life_bands, dtype=torch.float32)
        periods = torch.tensor(config.period_bands, dtype=torch.float32)
        gear_fraction = torch.linspace(0.0, 1.0, self.gears)
        half_life = torch.exp(
            bands[:, 0].log()[:, None]
            + 0.5
            * (bands[:, 1].log() - bands[:, 0].log())[:, None]
        ).expand(-1, self.gears)
        period = torch.exp(
            periods[:, 0].log()[:, None]
            + gear_fraction[None]
            * (periods[:, 1].log() - periods[:, 0].log())[:, None]
        )
        retention = torch.exp(math.log(0.5) / half_life)
        direction = torch.where(
            torch.arange(self.gears) % 2 == 0,
            torch.ones(self.gears),
            -torch.ones(self.gears),
        )
        angle = direction[None] * (2.0 * math.pi / period)
        multiplier = retention[..., None, None] * torch.stack(
            (angle.cos(), angle.sin()),
            dim=-1,
        )[:, :, None]
        self.register_buffer("fixed_multiplier", multiplier)
        self.register_buffer(
            "fixed_retention",
            retention[..., None].expand(-1, -1, self.channels),
        )
        self.register_buffer("fixed_period", period)

        self.cell_encoder = nn.Linear(7, config.cell_dim, bias=False)
        self.pool_logits = nn.Parameter(
            torch.zeros(self.banks, self.gears, self.channels)
        )
        self.bank_up = nn.Linear(
            self.banks * config.cell_dim,
            config.dim,
            bias=False,
        )
        nn.init.normal_(self.write_projection.weight, std=0.02)
        nn.init.zeros_(self.write_projection.bias)
        nn.init.normal_(self.bank_up.weight, std=0.01)

    def initial_rotor(
        self,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        phase = self.initial_phase.float()
        rotor = torch.stack((phase.cos(), phase.sin()), dim=-1)
        return rotor[None].expand(batch, -1, -1, -1, -1).to(device)

    def initial_state(
        self,
        batch: int,
        device: torch.device,
    ) -> GearScanState:
        return GearScanState(
            self.initial_rotor(batch, device),
            torch.full((batch,), -1, device=device, dtype=torch.long),
        )

    def forward(
        self,
        source: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
        state: GearScanState,
    ):
        del sentence_end_mask
        batch, length, _ = source.shape
        controls = self.write_projection(source).reshape(
            batch,
            length,
            self.banks,
            self.gears,
            self.channels,
            3,
        )
        write_gate = torch.sigmoid(controls[..., 0]).float()
        write = torch.tanh(controls[..., 1:3]).float()
        retention = self.fixed_retention[None, None].expand(
            batch, length, -1, -1, -1
        )
        multiplier = self.fixed_multiplier[None, None].expand(
            batch, length, -1, -1, -1, -1
        )
        bias = (
            (1.0 - retention.square()).clamp_min(1e-8).sqrt()[..., None]
            * write_gate[..., None]
            * write
        )
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
        reset_initial = self.initial_rotor(batch, source.device)
        reset_bias = complex_mul(
            multiplier, reset_initial[:, None]
        ) + bias
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
        if source.device.type == "mps":
            rotor = mps_affine_scan(
                scan_multiplier.float(),
                scan_bias.float(),
                state.rotor.float(),
            )
        else:
            rotor, _, _ = chunked_affine_scan(
                scan_multiplier.float(),
                scan_bias.float(),
                state.rotor.float(),
                chunk_size=max(2, length),
            )
        previous = torch.cat((state.rotor[:, None], rotor[:, :-1]), dim=1)
        previous = torch.where(
            reset[..., None, None, None, None],
            reset_initial[:, None],
            previous,
        )
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
        ).to(source.dtype)
        weights = self.pool_logits.flatten(1).softmax(dim=-1).reshape(
            self.banks, self.gears, self.channels
        )
        pooled_features = (
            features * weights[None, None, ..., None].to(features.dtype)
        ).sum(dim=(3, 4))
        bank_state = F.silu(self.cell_encoder(pooled_features))
        output = self.bank_up(bank_state.flatten(2))

        valid_count = token_mask.long().sum(dim=1)
        last_index = (valid_count - 1).clamp_min(0)
        gather_rotor = last_index.reshape(
            batch, 1, 1, 1, 1, 1
        ).expand(
            -1, 1, self.banks, self.gears, self.channels, 2
        )
        final_rotor = rotor.gather(1, gather_rotor).squeeze(1)
        final_segment = segment_ids.gather(
            1, last_index[:, None]
        ).squeeze(1)
        has_token = valid_count > 0
        next_state = GearScanState(
            torch.where(
                has_token[:, None, None, None, None],
                final_rotor,
                state.rotor,
            ),
            torch.where(has_token, final_segment, state.segment_id),
        )
        return output, next_state, {
            "rotor": rotor,
            "retention": retention,
            "write_gate": write_gate,
            "period": self.fixed_period,
            "bank_state": bank_state,
            "reset": reset,
        }


class SelectiveGearFiLM(nn.Module):
    """Token- and channel-selective modulation driven by prior-block Gear state."""

    def __init__(self, dim: int, rank: int, residual_scale: float) -> None:
        super().__init__()
        del rank
        self.residual_scale = float(residual_scale)
        self.context_norm = RMSNorm(dim)
        self.projection = nn.Linear(dim, 2 * dim + 1, bias=True)
        nn.init.normal_(self.projection.weight, std=0.01)
        nn.init.zeros_(self.projection.bias)

    def controls(self, context: torch.Tensor) -> torch.Tensor:
        return self.projection(self.context_norm(context))

    def apply_controls(
        self,
        hidden: torch.Tensor,
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift, gate_logits = torch.split(
            controls,
            (hidden.shape[-1], hidden.shape[-1], 1),
            dim=-1,
        )
        gate = torch.sigmoid(gate_logits)
        modulation = gate * (
            torch.tanh(scale) * hidden + shift
        )
        return hidden + self.residual_scale * modulation, gate

    def apply_block_controls(
        self,
        hidden: torch.Tensor,
        controls: torch.Tensor,
        block_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, dim = hidden.shape
        blocks = length // block_tokens
        hidden_blocks = hidden.reshape(batch, blocks, block_tokens, dim)
        scale, shift, gate_logits = torch.split(
            controls,
            (dim, dim, 1),
            dim=-1,
        )
        scale = torch.tanh(scale)[:, :, None]
        shift = shift[:, :, None]
        gate = torch.sigmoid(gate_logits)[:, :, None]
        modulation = gate * (scale * hidden_blocks + shift)
        return (
            (
                hidden_blocks + self.residual_scale * modulation
            ).reshape_as(hidden),
            gate.expand(-1, -1, block_tokens, -1).reshape(
                batch, length, 1
            ),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply_controls(hidden, self.controls(context))


class GearBankRouter(nn.Module):
    """Token queries retrieve from four fixed-size persistent Gear bank slots."""

    def __init__(
        self,
        dim: int,
        bank_dim: int,
        rank: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.residual_scale = float(residual_scale)
        self.query = nn.Linear(dim, rank, bias=False)
        self.key = nn.Linear(bank_dim, rank, bias=False)
        self.value = nn.Linear(bank_dim, dim, bias=False)
        self.gate = nn.Linear(dim, 1, bias=True)
        nn.init.normal_(self.query.weight, std=0.02)
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.value.weight, std=0.02)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        banks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(hidden).float()
        key = self.key(banks).float()
        value = self.value(banks)
        if banks.ndim == hidden.ndim:
            scores = torch.matmul(
                query,
                key.transpose(-1, -2),
            )
            probability = (scores * (self.rank ** -0.5)).softmax(dim=-1)
            retrieved = torch.matmul(
                probability.to(value.dtype),
                value,
            )
        else:
            scores = (query[..., None, :] * key).sum(dim=-1)
            probability = (scores * (self.rank ** -0.5)).softmax(dim=-1)
            retrieved = (
                probability.to(value.dtype)[..., None] * value
            ).sum(dim=-2)
        gate = torch.sigmoid(self.gate(hidden))
        return (
            hidden + self.residual_scale * gate * retrieved,
            gate,
        )


class BlockGearMemory(nn.Module):
    def __init__(self, config: BlockHybridGearV4Config) -> None:
        super().__init__()
        gear_layers = max(1, config.layers // config.gear_every)
        self.block_tokens = config.block_tokens
        self.dim = config.dim
        self.residual_scale = 1.0 / math.sqrt(2.0 * gear_layers)
        self.memory = StaticBlockRotorMemory(config)
        self.fusion_mode = config.fusion_mode
        self.fusion = (
            SelectiveGearFiLM(
                config.dim,
                config.fusion_rank,
                self.residual_scale,
            )
            if config.fusion_mode == "selective_film"
            else None
        )
        self.bank_router = (
            GearBankRouter(
                config.dim,
                config.cell_dim,
                config.fusion_rank,
                self.residual_scale,
            )
            if config.fusion_mode == "bank_router"
            else None
        )

    def _fuse(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        block_context: torch.Tensor | None = None,
        bank_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bank_router is not None:
            if bank_context is None:
                raise ValueError("bank_router requires bank_context")
            if (
                bank_context.ndim == 4
                and bank_context.shape[1] * self.block_tokens
                == hidden.shape[1]
            ):
                batch, blocks, banks, bank_dim = bank_context.shape
                hidden_blocks = hidden.reshape(
                    batch, blocks, self.block_tokens, self.dim
                )
                output, gate = self.bank_router(
                    hidden_blocks,
                    bank_context,
                )
                return (
                    output.reshape_as(hidden),
                    gate.reshape(batch, -1, 1),
                )
            return self.bank_router(hidden, bank_context)
        if self.fusion is not None:
            if block_context is not None:
                controls = self.fusion.controls(block_context)
                return self.fusion.apply_block_controls(
                    hidden,
                    controls,
                    self.block_tokens,
                )
            return self.fusion(hidden, context)
        return (
            hidden + self.residual_scale * context,
            torch.ones(
                *hidden.shape[:-1],
                1,
                device=hidden.device,
                dtype=hidden.dtype,
            ),
        )

    def initial_cache(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockGearMemoryCache:
        return BlockGearMemoryCache(
            self.memory.initial_state(batch, device),
            torch.zeros(batch, self.dim, device=device, dtype=dtype),
            torch.zeros(batch, self.dim, device=device, dtype=dtype),
            torch.zeros(batch, device=device, dtype=torch.long),
            torch.full((batch,), -1, device=device, dtype=torch.long),
            torch.zeros(batch, device=device, dtype=torch.long),
            torch.zeros(
                batch,
                self.memory.banks,
                self.memory.cell_encoder.out_features,
                device=device,
                dtype=dtype,
            ),
        )

    def _block_summary(
        self,
        source: torch.Tensor,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
    ):
        batch, length, dim = source.shape
        blocks = (length + self.block_tokens - 1) // self.block_tokens
        padded_length = blocks * self.block_tokens
        padding = padded_length - length
        if padding:
            source = F.pad(source, (0, 0, 0, padding))
            token_mask = F.pad(token_mask, (0, padding), value=False)
            segment_ids = F.pad(segment_ids, (0, padding), value=-1)
            sentence_end_mask = F.pad(
                sentence_end_mask, (0, padding), value=False
            )
        source_blocks = source.reshape(
            batch, blocks, self.block_tokens, dim
        )
        valid_blocks = token_mask.reshape(
            batch, blocks, self.block_tokens
        )
        segment_blocks = segment_ids.reshape(
            batch, blocks, self.block_tokens
        )
        sentence_blocks = sentence_end_mask.reshape(
            batch, blocks, self.block_tokens
        )
        valid_count = valid_blocks.long().sum(dim=-1)
        last_index = (valid_count - 1).clamp_min(0)
        block_segment = segment_blocks.gather(
            2, last_index[..., None]
        ).squeeze(-1)
        has_tokens = valid_count > 0
        block_valid = valid_count == self.block_tokens
        block_segment = torch.where(
            has_tokens,
            block_segment,
            torch.full_like(block_segment, -1),
        )
        selected = valid_blocks & (
            segment_blocks == block_segment[..., None]
        )
        denominator = selected.sum(dim=-1, keepdim=True).clamp_min(1)
        summary = (
            source_blocks
            * selected[..., None].to(source_blocks.dtype)
        ).sum(dim=2) / denominator.to(source_blocks.dtype)
        block_sentence_end = (sentence_blocks & selected).any(dim=-1)
        first_segment = segment_blocks[:, :, 0]
        prefix_same = (
            (segment_blocks == first_segment[..., None]) & valid_blocks
        ).long().cumprod(dim=2).bool()
        return (
            summary,
            block_valid,
            block_segment,
            block_sentence_end,
            prefix_same,
            padded_length,
            selected.sum(dim=-1),
            (
                source_blocks
                * selected[..., None].to(source_blocks.dtype)
            ).sum(dim=2),
        )

    def forward_blocks(
        self,
        hidden: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
        cache: BlockGearMemoryCache | None,
        simple_sequence: bool = False,
    ):
        batch, length, _ = hidden.shape
        cache = cache or self.initial_cache(
            batch, hidden.device, hidden.dtype
        )
        if simple_sequence and length % self.block_tokens == 0:
            blocks = length // self.block_tokens
            hidden_blocks = hidden.reshape(
                batch, blocks, self.block_tokens, self.dim
            )
            summary = hidden_blocks.mean(dim=2)
            block_valid = torch.ones(
                batch, blocks, device=hidden.device, dtype=torch.bool
            )
            block_segment = segment_ids[:, :: self.block_tokens]
            block_sentence_end = sentence_end_mask.reshape(
                batch, blocks, self.block_tokens
            ).any(dim=-1)
            prefix_same = torch.ones(
                batch,
                blocks,
                self.block_tokens,
                device=hidden.device,
                dtype=torch.bool,
            )
            padded_length = length
            selected_count = torch.full(
                (batch, blocks),
                self.block_tokens,
                device=hidden.device,
                dtype=torch.long,
            )
            selected_sum = summary * self.block_tokens
        else:
            (
                summary,
                block_valid,
                block_segment,
                block_sentence_end,
                prefix_same,
                padded_length,
                selected_count,
                selected_sum,
            ) = self._block_summary(
                hidden,
                token_mask,
                segment_ids,
                sentence_end_mask,
            )
        memory_output, state, record = self.memory(
            summary,
            token_mask=block_valid,
            segment_ids=block_segment,
            sentence_end_mask=block_sentence_end,
            state=cache.state,
        )
        prior_context = torch.cat(
            (cache.context[:, None], memory_output[:, :-1]),
            dim=1,
        )
        prior_segment = torch.cat(
            (cache.state.segment_id[:, None], block_segment[:, :-1]),
            dim=1,
        )
        prior_bank_context = torch.cat(
            (cache.bank_context[:, None], record["bank_state"][:, :-1]),
            dim=1,
        )
        if simple_sequence and length % self.block_tokens == 0:
            # ``simple_sequence`` is the fast path used by whole-document
            # training lanes. A lane can still switch documents between two
            # windows, leaving the previous document in ``cache.context``.
            # Mask that carried context before broadcasting it into the first
            # block of the new document. The scan itself resets by segment,
            # but fusion happens from the *prior* state and therefore needs
            # its own isolation check.
            prior_is_same_segment = (
                (prior_segment == block_segment) & block_valid
            )
            block_context = (
                prior_context
                * prior_is_same_segment[..., None].to(prior_context.dtype)
            )
            context = block_context.repeat_interleave(
                self.block_tokens, dim=1
            )
            bank_context = (
                prior_bank_context
                * prior_is_same_segment[..., None, None].to(
                    prior_bank_context.dtype
                )
            )
        else:
            block_context = None
            segment_blocks = F.pad(
                segment_ids,
                (0, padded_length - length),
                value=-1,
            ).reshape(batch, -1, self.block_tokens)
            first_segment = segment_blocks[:, :, 0]
            apply_context = (
                prefix_same
                & (prior_segment[..., None] == first_segment[..., None])
            )
            context = (
                prior_context[:, :, None]
                * apply_context[..., None].to(prior_context.dtype)
            ).reshape(batch, padded_length, self.dim)[:, :length]
            bank_context = (
                prior_bank_context[:, :, None]
                * apply_context[..., None, None].to(
                    prior_bank_context.dtype
                )
            ).reshape(
                batch,
                padded_length,
                self.memory.banks,
                -1,
            )[:, :length]
        output, modulation_gate = self._fuse(
            hidden,
            context,
            block_context,
            bank_context,
        )

        block_count = block_valid.long().sum(dim=1)
        last_block = (block_count - 1).clamp_min(0)
        gather_context = last_block[:, None, None].expand(
            -1, 1, self.dim
        )
        final_context = memory_output.gather(
            1, gather_context
        ).squeeze(1)
        gather_bank = last_block[:, None, None, None].expand(
            -1,
            1,
            self.memory.banks,
            record["bank_state"].shape[-1],
        )
        final_bank_context = record["bank_state"].gather(
            1, gather_bank
        ).squeeze(1)
        has_block = block_count > 0
        remainder = length % self.block_tokens
        if remainder:
            partial_count = selected_count[:, -1]
            partial_sum = selected_sum[:, -1]
            partial_segment = block_segment[:, -1]
        else:
            partial_count = torch.zeros_like(cache.accumulator_count)
            partial_sum = torch.zeros_like(cache.accumulator)
            partial_segment = state.segment_id
        next_cache = BlockGearMemoryCache(
            state,
            torch.where(has_block[:, None], final_context, cache.context),
            partial_sum,
            partial_count,
            partial_segment,
            torch.full_like(cache.block_position, remainder),
            torch.where(
                has_block[:, None, None],
                final_bank_context,
                cache.bank_context,
            ),
        )
        record.update(
            {
                "block_segment": block_segment,
                "block_valid": block_valid,
                "block_end_position": (
                    torch.arange(
                        block_segment.shape[1],
                        device=hidden.device,
                    )
                    * self.block_tokens
                    + self.block_tokens
                    - 1
                ),
                "modulation_gate": modulation_gate,
            }
        )
        return output, next_cache, record

    def forward_token(
        self,
        hidden: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
        cache: BlockGearMemoryCache | None,
    ):
        batch = hidden.shape[0]
        cache = cache or self.initial_cache(
            batch, hidden.device, hidden.dtype
        )
        segment = segment_ids[:, 0]
        valid = token_mask[:, 0]
        context_valid = valid & (cache.state.segment_id == segment)
        context = (
            cache.context[:, None]
            * context_valid[:, None, None].to(hidden.dtype)
        )
        bank_context = (
            cache.bank_context[:, None]
            * context_valid[:, None, None, None].to(
                cache.bank_context.dtype
            )
        )
        output, modulation_gate = self._fuse(
            hidden,
            context,
            bank_context=bank_context,
        )

        same_accumulator = (
            cache.accumulator_segment == segment
        ) & (cache.accumulator_count > 0)
        accumulator = torch.where(
            same_accumulator[:, None],
            cache.accumulator + hidden[:, 0],
            hidden[:, 0],
        )
        count = torch.where(
            valid,
            torch.where(
                same_accumulator,
                cache.accumulator_count + 1,
                torch.ones_like(cache.accumulator_count),
            ),
            cache.accumulator_count,
        )
        block_position = torch.where(
            valid,
            cache.block_position + 1,
            cache.block_position,
        )
        complete = valid & (block_position >= self.block_tokens)
        summary = accumulator / count.clamp_min(1)[:, None].to(hidden.dtype)
        memory_output, state, record = self.memory(
            summary[:, None],
            token_mask=complete[:, None],
            segment_ids=segment[:, None],
            sentence_end_mask=sentence_end_mask,
            state=cache.state,
        )
        next_context = torch.where(
            complete[:, None],
            memory_output[:, 0],
            cache.context,
        )
        next_bank_context = torch.where(
            complete[:, None, None],
            record["bank_state"][:, 0],
            cache.bank_context,
        )
        next_cache = BlockGearMemoryCache(
            state,
            next_context,
            torch.where(
                complete[:, None],
                torch.zeros_like(accumulator),
                accumulator,
            ),
            torch.where(
                complete,
                torch.zeros_like(count),
                count,
            ),
            torch.where(
                valid,
                segment,
                cache.accumulator_segment,
            ),
            torch.where(
                complete,
                torch.zeros_like(block_position),
                block_position,
            ),
            next_bank_context,
        )
        record.update(
            {
                "block_segment": segment[:, None],
                "block_valid": complete[:, None],
                "block_end_position": torch.zeros(
                    1, device=hidden.device, dtype=torch.long
                ),
                "modulation_gate": modulation_gate,
            }
        )
        return output, next_cache, record


class BlockHybridGearV4LM(nn.Module, _LanguageModelLossMixin):
    def __init__(self, config: BlockHybridGearV4Config) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.dim)
        bounded_config = type(
            "_BoundedConfig",
            (),
            {
                "dim": config.dim,
                "layers": config.layers,
                "ffn_dim": config.ffn_dim,
                "heads": config.heads,
                "kv_heads": config.kv_heads,
                "attention_window": config.attention_window,
                "dropout": config.dropout,
            },
        )()
        self.blocks = nn.ModuleList(
            [BoundedTransformerBlock(bounded_config) for _ in range(config.layers)]
        )
        if config.gear_placement == "pre_next":
            self.gear_layer_indices = tuple(
                index
                for index in range(config.layers - 1)
                if index % config.gear_every == 0
            )
        else:
            self.gear_layer_indices = tuple(
                index
                for index in range(config.layers)
                if (index + 1) % config.gear_every == 0
            )
        self.gear_memories = nn.ModuleList(
            [BlockGearMemory(config) for _ in self.gear_layer_indices]
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
        self.register_buffer(
            "_future_horizon_offsets",
            torch.tensor(config.future_horizons, dtype=torch.long),
            persistent=False,
        )
        nn.init.normal_(self.token.weight, std=0.02)

    def _forward_hidden(
        self,
        tokens: torch.Tensor,
        *,
        cache: BlockHybridGearV4Cache | None,
        use_cache: bool,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        sentence_end_mask: torch.Tensor,
        simple_sequence: bool = False,
    ):
        hidden = self.token(tokens)
        next_kv = []
        next_memory = []
        records = []
        gear_index = 0
        gear_set = set(self.gear_layer_indices)
        for index, block in enumerate(self.blocks):
            hidden, local_cache = block(
                hidden,
                token_mask=token_mask,
                segment_ids=segment_ids,
                cache=None if cache is None else cache.local_kv[index],
                use_cache=use_cache,
            )
            if use_cache:
                assert local_cache is not None
                next_kv.append(local_cache)
            if index in gear_set:
                memory = self.gear_memories[gear_index]
                memory_cache = (
                    None if cache is None else cache.gear_memory[gear_index]
                )
                if tokens.shape[1] == 1:
                    hidden, memory_cache, record = memory.forward_token(
                        hidden,
                        token_mask=token_mask,
                        segment_ids=segment_ids,
                        sentence_end_mask=sentence_end_mask,
                        cache=memory_cache,
                    )
                else:
                    hidden, memory_cache, record = memory.forward_blocks(
                        hidden,
                        token_mask=token_mask,
                        segment_ids=segment_ids,
                        sentence_end_mask=sentence_end_mask,
                        cache=memory_cache,
                        simple_sequence=simple_sequence,
                    )
                next_memory.append(memory_cache)
                records.append(record)
                gear_index += 1
        processed = token_mask.sum(dim=1)
        if cache is not None:
            processed = processed + cache.tokens_processed
        next_cache = (
            BlockHybridGearV4Cache(
                tuple(next_kv),
                tuple(next_memory),
                processed,
                (
                    (0 if cache is None else cache.block_offset)
                    + tokens.shape[1]
                )
                % self.config.block_tokens,
            )
            if use_cache
            else None
        )
        return self.final_norm(hidden), next_cache, records

    def forward(
        self,
        tokens: torch.Tensor,
        cache: BlockHybridGearV4Cache | None = None,
        use_cache: bool = False,
        token_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        sentence_end_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        if (
            cache is not None
            and cache.block_offset != 0
            and tokens.shape[1] > 1
        ):
            pieces = []
            next_cache = cache
            for position in range(tokens.shape[1]):
                logits, next_cache = self(
                    tokens[:, position : position + 1],
                    cache=next_cache,
                    use_cache=True,
                    token_mask=(
                        None
                        if token_mask is None
                        else token_mask[:, position : position + 1]
                    ),
                    segment_ids=(
                        None
                        if segment_ids is None
                        else segment_ids[:, position : position + 1]
                    ),
                    sentence_end_mask=(
                        None
                        if sentence_end_mask is None
                        else sentence_end_mask[:, position : position + 1]
                    ),
                    attention_mask=(
                        None
                        if attention_mask is None
                        else attention_mask[:, position : position + 1]
                    ),
                )
                pieces.append(logits)
            return torch.cat(pieces, dim=1), next_cache
        simple_sequence = (
            token_mask is None
            and attention_mask is None
            and segment_ids is None
            and tokens.shape[1] % self.config.block_tokens == 0
        )
        masks = self._masks(
            tokens,
            {
                "token_mask": (
                    token_mask if token_mask is not None else attention_mask
                ),
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
            simple_sequence=simple_sequence,
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
        record = records[-1]
        bank_state = record["bank_state"]
        block_segment = record["block_segment"]
        block_valid = record["block_valid"]
        block_end = record["block_end_position"]
        horizons = self._future_horizon_offsets
        target_position = block_end[:, None] + horizons[None]
        within = target_position < tokens.shape[1]
        clamped = target_position.clamp_max(tokens.shape[1] - 1)
        target_tokens = tokens[:, clamped]
        target_segments = segment_ids[:, clamped]
        target_valid = token_mask[:, clamped]

        weights = torch.stack(
            [head.weight for head in self.future_heads],
            dim=0,
        )
        prediction = F.normalize(
            torch.einsum(
                "bckr,kdr->bckd",
                bank_state,
                weights,
            ).float(),
            dim=-1,
        )
        target = F.normalize(
            self.token(target_tokens).detach().float(),
            dim=-1,
        )
        valid = (
            block_valid[:, :, None]
            & within[None]
            & target_valid
            & (block_segment[:, :, None] == target_segments)
        )
        distance = 1.0 - (prediction * target).sum(dim=-1)
        valid_float = valid.to(distance.dtype)
        per_bank_count = valid_float.sum(dim=(0, 1)).clamp_min(1)
        per_bank_loss = (
            distance * valid_float
        ).sum(dim=(0, 1)) / per_bank_count
        return per_bank_loss.mean()

    def stream_training_step(
        self,
        tokens: torch.Tensor,
        *,
        cache: BlockHybridGearV4Cache | None = None,
        detach_cache: bool = False,
        task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ):
        metadata = task_metadata or {}
        simple_sequence = (
            (
                (
                    "segment_ids" not in metadata
                    and "attention_mask" not in metadata
                    and "token_mask" not in metadata
                )
                or bool(metadata.get("single_segment_rows", False))
            )
            and tokens.shape[1] % self.config.block_tokens == 0
        )
        token_mask, segment_ids, sentence_end = self._masks(tokens, metadata)
        hidden, next_cache, records = self._forward_hidden(
            tokens,
            cache=cache,
            use_cache=True,
            token_mask=token_mask,
            segment_ids=segment_ids,
            sentence_end_mask=sentence_end,
            simple_sequence=simple_sequence,
        )
        assert next_cache is not None
        valid = self._valid_targets(
            tokens,
            token_mask,
            segment_ids,
            metadata.get("loss_mask"),
        )
        language_modeling = self._language_modeling_loss(hidden, tokens, valid)
        scales = loss_term_scales or {}
        future_scale = float(metadata.get("future_aux_scale", 1.0))
        future_weight = (
            self.config.future_aux_weight
            * future_scale
            * scales.get("future_state", 1.0)
        )
        future = (
            self._future_loss(records, tokens, token_mask, segment_ids)
            if future_weight != 0.0
            else hidden.sum() * 0.0
        )
        total = scales.get("language_modeling", 1.0) * language_modeling
        total = total + future_weight * future
        metrics = {
            "language_modeling": language_modeling,
            "future_state": future,
            "future_aux_scale": language_modeling.new_tensor(future_scale),
            "total": total,
        }
        return metrics, (next_cache.detach() if detach_cache else next_cache)

    def training_step(
        self,
        tokens,
        task_metadata=None,
        loss_term_scales=None,
    ):
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
        gear_layers = len(self.gear_memories)
        state_values = (
            gear_layers
            * self.config.num_banks
            * self.config.gears_per_bank
            * self.config.rotor_channels
            * 2
        )
        selective = self.config.fusion_mode == "selective_film"
        bank_router = self.config.fusion_mode == "bank_router"
        return {
            "name": (
                "BoundedHybridGearBlockBankRouter"
                if bank_router
                else (
                    "BoundedHybridGearBlockSelectiveFiLM"
                    if selective
                    else "BoundedHybridGearBlockAdditive"
                )
            ),
            "version": 4.3 if bank_router else (4.2 if selective else 4.1),
            "config": self.config.to_dict(),
            "parameters": {
                "total": sum(parameter.numel() for parameter in self.parameters())
            },
            "state": {
                "gear_floating_values_per_example": state_values,
                "local_attention_window": self.config.attention_window,
                "gear_update_tokens": self.config.block_tokens,
                "gear_placement": self.config.gear_placement,
                "gear_fusion": self.config.fusion_mode,
                "fusion_rank": self.config.fusion_rank,
                "cache_complexity": (
                    "O(gear_state + gear_layers*dim + "
                    "layers*window*kv_heads*head_dim)"
                ),
                "half_life_bands": [
                    list(value) for value in self.config.half_life_bands
                ],
                "period_bands": [
                    list(value) for value in self.config.period_bands
                ],
            },
            "invariants": {
                "self_attention": True,
                "bounded_history": True,
                "kv_cache": True,
                "block_rate_gear_memory": True,
                "current_block_memory_read": False,
                "token_channel_selective_modulation": selective,
                "fixed_gear_bank_retrieval": bank_router,
                "sequence_square_tensor": False,
                "host_scalar_control_flow": False,
            },
        }


@MODELS.register("bounded_hybrid_gear_block_additive")
def build_bounded_hybrid_gear_block_additive(
    model_cfg: dict,
    vocab_size: int | None = None,
):
    values = dict(model_cfg)
    if vocab_size is not None:
        values["vocab_size"] = vocab_size
    return BlockHybridGearV4LM(BlockHybridGearV4Config(**values))


@MODELS.register("bounded_hybrid_gear_block_selective_film")
def build_bounded_hybrid_gear_block_selective_film(
    model_cfg: dict,
    vocab_size: int | None = None,
):
    values = dict(model_cfg)
    values["fusion_mode"] = "selective_film"
    if vocab_size is not None:
        values["vocab_size"] = vocab_size
    return BlockHybridGearV4LM(BlockHybridGearV4Config(**values))


@MODELS.register("bounded_hybrid_gear_block_bank_router")
def build_bounded_hybrid_gear_block_bank_router(
    model_cfg: dict,
    vocab_size: int | None = None,
):
    values = dict(model_cfg)
    values["fusion_mode"] = "bank_router"
    if vocab_size is not None:
        values["vocab_size"] = vocab_size
    return BlockHybridGearV4LM(BlockHybridGearV4Config(**values))
