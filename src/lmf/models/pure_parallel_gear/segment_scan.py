"""Whole-sequence parallel-scan forward path for Pure Parallel Gear.

Only exact when angular velocity is held fixed (``fixed_omega=True`` at the
call site, i.e. ``config.learned_angular_velocity=False`` or the
``fixed_angular_velocities`` ablation) -- when omega is instead carried,
learned state (the default), each chunk's per-token phase recurrence
depends on that chunk's own incoming omega value, which is itself only
resolved by the same sequential settle()/reset propagation this module
computes. That coupling means the per-token math cannot be precomputed
independently of the propagation in the general case; with omega fixed,
the coupling vanishes and the per-token math becomes a single whole-
sequence computation, with everything sequential confined to a much
cheaper per-segment-summary propagation. See
`docs/.../pure_parallel_gear` Phase 2 notes for the full derivation.

Reuses the complex-affine scan primitives already proven for the bounded
hybrid gear architecture (`bounded_hybrid_gear/scan.py`) rather than
re-deriving them: Pure Parallel Gear's `_scan_token_dynamics` is the same
commuting scale*rotation multiplier with the same rotate-then-cumsum raw
bias transport `chunked_rotor_scan` already implements.
"""

from __future__ import annotations

import torch

from ..bounded_hybrid_gear.scan import _gather_time, complex_mul


def local_token_scan(
    multiplier: torch.Tensor,
    bias: torch.Tensor,
    reset: torch.Tensor,
    *,
    eps: float = 1e-20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token rotor value and composed multiplier, both relative to each
    token's own segment start (the most recent True in `reset`, looking
    back along dim 1) -- i.e. as if that segment's carry-in were zero.

    Deliberately *not* `chunked_rotor_scan` with `reset_initial=0`: that
    function's reset handling is built to *snap* the state to a known,
    fixed value at a reset (zeroing the composed multiplier there, since
    the point of a reset is "the past doesn't matter, here is the new
    value") -- exactly wrong for this use, where the real carry-in at
    each segment start is not yet known (it depends on settle()'s output,
    resolved later, sequentially, in Level 1). This computes the same
    commuting scale*rotation cumsum/cummax-baseline math `chunked_rotor_
    scan` uses internally, but returns the *unzeroed* segment-relative
    multiplier so a real carry-in can be composed in afterward once known
    (see `broadcast_affine`). Single direct cumsum over the whole sequence
    -- no two-level chunking -- since Pure Parallel Gear's sequence lengths
    don't need `chunked_rotor_scan`'s long-sequence memory/precision
    bound, only its reset-aware baselining idea.
    """
    magnitude = multiplier.square().sum(dim=-1).clamp_min(eps).sqrt()
    angle = torch.atan2(multiplier[..., 1], multiplier[..., 0])
    raw_log_scale = torch.cumsum(magnitude.clamp_min(eps).log(), dim=1)
    raw_phase = torch.cumsum(angle, dim=1)

    position = torch.arange(reset.shape[1], device=reset.device)[None]
    reset_marker = torch.where(reset, position, -torch.ones_like(position))
    last_reset = torch.cummax(reset_marker, dim=1).values
    baseline_index = (last_reset - 1).clamp_min(0)
    baseline_exists = last_reset > 0

    extra_dims = (1,) * (raw_phase.ndim - 2)
    exists_mask = baseline_exists.reshape(*baseline_exists.shape, *extra_dims)
    phase_baseline = torch.where(
        exists_mask, _gather_time(raw_phase, baseline_index), torch.zeros_like(raw_phase)
    )
    log_baseline = torch.where(
        exists_mask,
        _gather_time(raw_log_scale, baseline_index),
        torch.zeros_like(raw_log_scale),
    )

    segment_phase = raw_phase - phase_baseline
    segment_log_scale = raw_log_scale - log_baseline
    segment_scale = segment_log_scale.exp()
    inverse_rotation = torch.stack(
        (segment_phase.cos(), -segment_phase.sin()), dim=-1
    )
    transport = (
        complex_mul(bias, inverse_rotation) / segment_scale[..., None].clamp_min(eps)
    )
    transport_cumsum = torch.cumsum(transport, dim=1)
    transport_exists_mask = baseline_exists.reshape(
        *baseline_exists.shape, *([1] * (transport_cumsum.ndim - 2))
    )
    transport_baseline = torch.where(
        transport_exists_mask,
        _gather_time(transport_cumsum, baseline_index),
        torch.zeros_like(transport_cumsum),
    )
    transport_sum = transport_cumsum - transport_baseline

    segment_multiplier = segment_scale[..., None] * torch.stack(
        (segment_phase.cos(), segment_phase.sin()), dim=-1
    )
    local_value = complex_mul(segment_multiplier, transport_sum)
    return local_value, segment_multiplier


def gather_chunk_summary(
    local_value: torch.Tensor,
    local_multiplier: torch.Tensor,
    chunk_end_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice each chunk's own last-token local value/multiplier out of the
    whole-sequence tensors -- the per-chunk summary Level 1 propagates."""
    index = chunk_end_index.reshape(
        *chunk_end_index.shape, *([1] * (local_value.ndim - 2))
    ).expand(-1, -1, *local_value.shape[2:])
    summary_value = local_value.gather(1, index)
    summary_multiplier = local_multiplier.gather(1, index)
    return summary_value, summary_multiplier


def broadcast_affine(
    local_multiplier: torch.Tensor,
    local_value: torch.Tensor,
    carry_in_per_chunk: torch.Tensor,
    chunk_index_of: torch.Tensor,
) -> torch.Tensor:
    """final[b, t] = local_multiplier[b, t] (x) carry_in_per_chunk[b, chunk_index_of[b, t]] + local_value[b, t].

    `chunk_index_of[b, t]` is the index, along dim 1 of `carry_in_per_chunk`,
    of the chunk token t belongs to -- every token in one chunk shares the
    same gathered carry-in, broadcast out of the (much smaller) per-chunk
    summary tensor in one vectorized gather, no loop.
    """
    index = chunk_index_of.reshape(
        *chunk_index_of.shape, *([1] * (carry_in_per_chunk.ndim - 2))
    ).expand(-1, -1, *carry_in_per_chunk.shape[2:])
    carry_in_per_token = carry_in_per_chunk.gather(1, index)
    return complex_mul(local_multiplier, carry_in_per_token) + local_value
