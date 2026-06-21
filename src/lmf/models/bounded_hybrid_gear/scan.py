"""Associative complex affine scans used by Pure Parallel Gear V3.

Each transition represents ``z -> multiplier * z + bias`` with complex values
stored as real tensors whose final dimension is ``(real, imaginary)``.
Composition is associative, including document resets: a reset transition uses
a zero multiplier and a bias that already contains the reset-state update.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def complex_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply complex tensors represented by a final dimension of size two."""
    left_real, left_imag = left.unbind(dim=-1)
    right_real, right_imag = right.unbind(dim=-1)
    return torch.stack(
        (
            left_real * right_real - left_imag * right_imag,
            left_real * right_imag + left_imag * right_real,
        ),
        dim=-1,
    )


def compose_affine(
    left_multiplier: torch.Tensor,
    left_bias: torch.Tensor,
    right_multiplier: torch.Tensor,
    right_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``right(left(z))`` for two complex affine transforms."""
    multiplier = complex_mul(right_multiplier, left_multiplier)
    bias = complex_mul(right_multiplier, left_bias) + right_bias
    return multiplier, bias


def _identity_like(value: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (value.shape[0], length, *value.shape[2:-1], 2)
    multiplier = value.new_zeros(shape)
    multiplier[..., 0] = 1.0
    return multiplier, value.new_zeros(shape)


def hillis_steele_affine_scan(
    multiplier: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive associative prefix scan along dimension one."""
    if multiplier.shape != bias.shape or multiplier.shape[-1] != 2:
        raise ValueError("multiplier and bias must share shape [..., 2]")
    length = multiplier.shape[1]
    offset = 1
    while offset < length:
        tail_multiplier, tail_bias = compose_affine(
            multiplier[:, :-offset],
            bias[:, :-offset],
            multiplier[:, offset:],
            bias[:, offset:],
        )
        multiplier = torch.cat((multiplier[:, :offset], tail_multiplier), dim=1)
        bias = torch.cat((bias[:, :offset], tail_bias), dim=1)
        offset *= 2
    return multiplier, bias


def chunked_affine_scan(
    multiplier: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    *,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-level inclusive scan with bounded local numerical depth.

    Local chunks are scanned independently, chunk summaries are scanned with
    the same associative operator, and the exclusive chunk carry is composed
    into every local prefix. No operation allocates a sequence-square tensor.
    """
    if chunk_size < 2:
        raise ValueError("chunk_size must be at least two")
    if multiplier.shape != bias.shape or multiplier.shape[-1] != 2:
        raise ValueError("multiplier and bias must share shape [B, T, ..., 2]")
    if initial.shape != multiplier.shape[:1] + multiplier.shape[2:]:
        raise ValueError("initial state shape must match scan state without time")

    batch, length = multiplier.shape[:2]
    if length == 0:
        return (
            multiplier,
            multiplier,
            bias,
        )
    chunks = (length + chunk_size - 1) // chunk_size
    padded_length = chunks * chunk_size
    padding = padded_length - length
    if padding:
        identity_multiplier, identity_bias = _identity_like(multiplier, padding)
        multiplier = torch.cat((multiplier, identity_multiplier), dim=1)
        bias = torch.cat((bias, identity_bias), dim=1)

    state_shape = multiplier.shape[2:]
    local_multiplier = multiplier.reshape(
        batch * chunks, chunk_size, *state_shape
    )
    local_bias = bias.reshape(batch * chunks, chunk_size, *state_shape)
    local_multiplier, local_bias = hillis_steele_affine_scan(
        local_multiplier,
        local_bias,
    )
    local_multiplier = local_multiplier.reshape(
        batch, chunks, chunk_size, *state_shape
    )
    local_bias = local_bias.reshape(batch, chunks, chunk_size, *state_shape)

    chunk_multiplier = local_multiplier[:, :, -1]
    chunk_bias = local_bias[:, :, -1]
    scanned_chunk_multiplier, scanned_chunk_bias = hillis_steele_affine_scan(
        chunk_multiplier,
        chunk_bias,
    )
    carry_multiplier, carry_bias = _identity_like(chunk_multiplier, 1)
    if chunks > 1:
        carry_multiplier = torch.cat(
            (carry_multiplier, scanned_chunk_multiplier[:, :-1]),
            dim=1,
        )
        carry_bias = torch.cat((carry_bias, scanned_chunk_bias[:, :-1]), dim=1)

    full_multiplier, full_bias = compose_affine(
        carry_multiplier[:, :, None],
        carry_bias[:, :, None],
        local_multiplier,
        local_bias,
    )
    full_multiplier = full_multiplier.reshape(batch, padded_length, *state_shape)
    full_bias = full_bias.reshape(batch, padded_length, *state_shape)
    states = complex_mul(full_multiplier, initial[:, None]) + full_bias
    return (
        states[:, :length],
        full_multiplier[:, :length],
        full_bias[:, :length],
    )


def _gather_time(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    gather = index.reshape(index.shape[0], index.shape[1], *([1] * (value.ndim - 2)))
    gather = gather.expand(-1, -1, *value.shape[2:])
    return value.gather(1, gather)


def chunked_rotor_scan(
    multiplier: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
    reset: torch.Tensor,
    reset_initial: torch.Tensor,
    *,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast two-level scan specialized for contractive rotation multipliers.

    Within each bounded chunk, commuting ``a*R(phi)`` multipliers admit a
    closed-form cumsum. Reset-aware segment baselines are obtained with a
    tensor-only cumulative maximum. Only chunk summaries use Hillis--Steele,
    reducing accelerator launch depth from ``log2(chunk_size)`` operations over
    the full rotor tensor to ``log2(num_chunks)`` operations over summaries.
    """
    if multiplier.shape != bias.shape or multiplier.shape[-1] != 2:
        raise ValueError("multiplier and bias must share shape [B, T, ..., 2]")
    if reset.shape != multiplier.shape[:2]:
        raise ValueError("reset must have shape [batch, sequence]")
    if initial.shape != multiplier.shape[:1] + multiplier.shape[2:]:
        raise ValueError("initial shape does not match scan state")
    if reset_initial.shape != initial.shape:
        raise ValueError("reset_initial must match initial")

    batch, length = multiplier.shape[:2]
    chunks = (length + chunk_size - 1) // chunk_size
    padded_length = chunks * chunk_size
    padding = padded_length - length
    if padding:
        identity_multiplier, identity_bias = _identity_like(multiplier, padding)
        multiplier = torch.cat((multiplier, identity_multiplier), dim=1)
        bias = torch.cat((bias, identity_bias), dim=1)
        reset = F.pad(reset, (0, padding), value=False)

    state_shape = multiplier.shape[2:]
    local_multiplier = multiplier.reshape(
        batch * chunks, chunk_size, *state_shape
    )
    local_bias_input = bias.reshape(batch * chunks, chunk_size, *state_shape)
    local_reset = reset.reshape(batch * chunks, chunk_size)

    magnitude = local_multiplier.square().sum(dim=-1).clamp_min(1e-20).sqrt()
    angle = torch.atan2(
        local_multiplier[..., 1],
        local_multiplier[..., 0],
    )
    raw_log_scale = torch.cumsum(magnitude.log(), dim=1)
    raw_phase = torch.cumsum(angle, dim=1)
    raw_scale = raw_log_scale.exp()
    raw_rotation = torch.stack((raw_phase.cos(), raw_phase.sin()), dim=-1)
    raw_prefix_multiplier = raw_scale[..., None] * raw_rotation
    raw_transport = (
        complex_mul(
            local_bias_input,
            torch.stack((raw_phase.cos(), -raw_phase.sin()), dim=-1),
        )
        / raw_scale[..., None].clamp_min(1e-20)
    )
    raw_transport_sum = torch.cumsum(raw_transport, dim=1)
    raw_prefix_bias = complex_mul(raw_prefix_multiplier, raw_transport_sum)

    position = torch.arange(chunk_size, device=reset.device)[None]
    reset_marker = torch.where(local_reset, position, -torch.ones_like(position))
    last_reset = torch.cummax(reset_marker, dim=1).values
    has_reset = last_reset >= 0
    baseline_index = (last_reset - 1).clamp_min(0)
    baseline_exists = last_reset > 0

    phase_baseline = _gather_time(raw_phase, baseline_index)
    log_baseline = _gather_time(raw_log_scale, baseline_index)
    phase_baseline = torch.where(
        baseline_exists.reshape(*baseline_exists.shape, *([1] * (raw_phase.ndim - 2))),
        phase_baseline,
        torch.zeros_like(phase_baseline),
    )
    log_baseline = torch.where(
        baseline_exists.reshape(*baseline_exists.shape, *([1] * (raw_log_scale.ndim - 2))),
        log_baseline,
        torch.zeros_like(log_baseline),
    )
    segment_phase = raw_phase - phase_baseline
    segment_log_scale = raw_log_scale - log_baseline
    segment_scale = segment_log_scale.exp()
    inverse_segment_rotation = torch.stack(
        (segment_phase.cos(), -segment_phase.sin()),
        dim=-1,
    )
    segment_transport = (
        complex_mul(local_bias_input, inverse_segment_rotation)
        / segment_scale[..., None].clamp_min(1e-20)
    )
    segment_transport_cumsum = torch.cumsum(segment_transport, dim=1)
    transport_baseline = _gather_time(segment_transport_cumsum, baseline_index)
    transport_baseline = torch.where(
        baseline_exists.reshape(
            *baseline_exists.shape,
            *([1] * (segment_transport_cumsum.ndim - 2)),
        ),
        transport_baseline,
        torch.zeros_like(transport_baseline),
    )
    segment_transport_sum = segment_transport_cumsum - transport_baseline
    segment_multiplier = segment_scale[..., None] * torch.stack(
        (segment_phase.cos(), segment_phase.sin()),
        dim=-1,
    )
    reset_base = reset_initial[:, None].expand(
        batch, chunks, *reset_initial.shape[1:]
    ).reshape(batch * chunks, *reset_initial.shape[1:])
    reset_state = complex_mul(
        segment_multiplier,
        reset_base[:, None] + segment_transport_sum,
    )

    reset_mask = has_reset.reshape(
        *has_reset.shape, *([1] * (local_multiplier.ndim - 2))
    )
    local_prefix_multiplier = torch.where(
        reset_mask,
        torch.zeros_like(raw_prefix_multiplier),
        raw_prefix_multiplier,
    )
    local_prefix_bias = torch.where(
        reset_mask,
        reset_state,
        raw_prefix_bias,
    )
    local_prefix_multiplier = local_prefix_multiplier.reshape(
        batch, chunks, chunk_size, *state_shape
    )
    local_prefix_bias = local_prefix_bias.reshape(
        batch, chunks, chunk_size, *state_shape
    )

    chunk_multiplier = local_prefix_multiplier[:, :, -1]
    chunk_bias = local_prefix_bias[:, :, -1]
    scanned_chunk_multiplier, scanned_chunk_bias = hillis_steele_affine_scan(
        chunk_multiplier,
        chunk_bias,
    )
    carry_multiplier, carry_bias = _identity_like(chunk_multiplier, 1)
    if chunks > 1:
        carry_multiplier = torch.cat(
            (carry_multiplier, scanned_chunk_multiplier[:, :-1]), dim=1
        )
        carry_bias = torch.cat(
            (carry_bias, scanned_chunk_bias[:, :-1]), dim=1
        )
    full_multiplier, full_bias = compose_affine(
        carry_multiplier[:, :, None],
        carry_bias[:, :, None],
        local_prefix_multiplier,
        local_prefix_bias,
    )
    full_multiplier = full_multiplier.reshape(batch, padded_length, *state_shape)
    full_bias = full_bias.reshape(batch, padded_length, *state_shape)
    states = complex_mul(full_multiplier, initial[:, None]) + full_bias
    return states[:, :length], full_multiplier[:, :length], full_bias[:, :length]
