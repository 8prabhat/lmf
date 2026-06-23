"""Gated delta-rule associative memory — the matmul-first core of Spectral Memory.

The recurrence each memory bank runs is the *gated delta rule* (an
error-correcting fast-weight update), with a per-token, per-bank decay ``a_t``::

    S_t = a_t * S_{t-1} + b_t * (v_t - a_t * S_{t-1} @ k_t) @ k_t^T
    o_t = S_t @ q_t

``S`` is a ``[d_v, d_k]`` matrix memory mapping keys -> values. The bracketed
term writes the *residual* between the desired value and what the memory would
currently recall for that key — this is what gives associative recall with a
bounded state, the ingredient plain convolution / linear-attention lacks.

Two equivalent implementations live here:

* :func:`delta_rule_chunked` — the training/eval path. Splits the sequence into
  chunks; within a chunk everything is dense matmuls (the WY/UT-transform form),
  across chunks a small state is carried. ~all heavy compute is GEMM, which is
  the fast path on Apple Silicon (MPS) — no custom sequential scan, no depthwise
  conv, no FFT. This mirrors the chunking philosophy of
  ``bounded_hybrid_gear.scan.chunked_affine_scan`` but generalized from scalar
  complex-affine state to a matrix delta state.
* :func:`delta_rule_recurrent` — the O(1)/token reference used for decoding and,
  crucially, as the ground truth the chunked path is tested against
  (``tests/test_spectral_memory.py`` asserts the two agree).

Decays are handled in log space so every cross-token factor is in ``(0, 1]`` —
numerically stable even for fast banks over long chunks. Segment ids (packed
documents) are honoured by masking cross-segment connectivity; because packed
segments are contiguous runs, an equality mask is exact.
"""

from __future__ import annotations

import math

import torch


def _inv_unit_lower(strict_lower: torch.Tensor) -> torch.Tensor:
    """Invert ``I + N`` for a strictly-lower-triangular ``N`` (batched).

    ``N`` is nilpotent (``N**L == 0`` for an ``L x L`` block), so
    ``(I + N)^{-1} = sum_i (-N)^i`` is a finite sum. Using the geometric-series
    doubling identity ``prod_k (I + R^{2^k})`` with ``R = -N`` computes it in
    ``ceil(log2 L)`` batched matmuls — MPS-safe (pure matmul, no
    ``solve_triangular`` which MPS lacks).
    """
    length = strict_lower.shape[-1]
    eye = torch.eye(length, device=strict_lower.device, dtype=strict_lower.dtype)
    if length == 1:
        return eye.expand_as(strict_lower).clone()
    r = -strict_lower
    inv = eye + r
    power = r
    for _ in range(1, math.ceil(math.log2(length))):
        power = power @ power
        inv = inv @ (eye + power)
    return inv


def delta_rule_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_a: torch.Tensor,
    beta: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
    chunk: int = 64,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-parallel gated delta rule. All heavy ops are dense matmuls.

    Shapes (banks are folded into the batch dimension by the caller):
        q, k, v   : [N, T, d]   (k expected L2-normalized over d)
        log_a     : [N, T]      per-token log decay (<= 0)
        beta      : [N, T]      per-token write strength in (0, 1)
        segment_ids: [N, T] or None
        state     : [N, d_v, d_k] incoming memory, or None (zeros)

    Returns (output [N, T, d_v], final_state [N, d_v, d_k]).
    """
    n, t_len, d_k = q.shape
    d_v = v.shape[-1]
    in_dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    log_a, beta = log_a.float(), beta.float()

    s = (
        torch.zeros(n, d_v, d_k, device=q.device, dtype=torch.float32)
        if state is None
        else state.float()
    )
    # An incoming ``state`` continues the segment of the first token (the
    # recurrent reference never resets at step 0), so seed s_prev with seg[:, 0].
    s_prev = None if segment_ids is None else segment_ids[:, 0]
    outputs: list[torch.Tensor] = []

    pos = 0
    while pos < t_len:
        length = min(chunk, t_len - pos)
        qc = q[:, pos : pos + length]
        kc = k[:, pos : pos + length]
        vc = v[:, pos : pos + length]
        la = log_a[:, pos : pos + length]
        bc = beta[:, pos : pos + length]

        ell = torch.cumsum(la, dim=1)                       # [N, L]
        diff = ell[:, :, None] - ell[:, None, :]            # [N, L, L]
        causal = torch.tril(torch.ones(length, length, device=q.device, dtype=torch.bool))
        decay = torch.where(causal, torch.exp(diff), torch.zeros_like(diff))  # p_t / p_j

        if segment_ids is not None:
            sc = segment_ids[:, pos : pos + length]
            same = (sc[:, :, None] == sc[:, None, :]).float()
            decay = decay * same
            enter = (sc == s_prev[:, None]).float()
        else:
            enter = torch.ones(n, length, device=q.device)

        kk = kc @ kc.transpose(1, 2)                        # [N, L, L]
        qk = qc @ kc.transpose(1, 2)
        a_mat = torch.tril(decay * kk, -1)                  # strictly lower, seg-masked
        m_mat = decay * qk                                  # causal+seg incl. diagonal
        n_mat = bc[:, :, None] * a_mat                      # beta * A
        inv = _inv_unit_lower(n_mat)

        p = torch.exp(ell)                                  # [N, L]
        ks = kc @ s.transpose(1, 2)                         # [N, L, d_v]
        qs = qc @ s.transpose(1, 2)
        v_eff = vc - (enter * p)[:, :, None] * ks
        w = inv @ (bc[:, :, None] * v_eff)                  # [N, L, d_v]
        out = (enter * p)[:, :, None] * qs + m_mat @ w
        outputs.append(out)

        # carry state to next chunk
        if segment_ids is not None:
            last_seg = sc[:, -1]
            last_mask = (sc == last_seg[:, None]).float()
            enter_last = enter[:, -1]
        else:
            last_mask = torch.ones(n, length, device=q.device)
            enter_last = enter[:, -1]
        ratio_end = torch.exp(ell[:, -1:] - ell)            # p_L / p_t  (<= 1)
        coef = (last_mask * ratio_end)[:, :, None] * w
        s = enter_last[:, None, None] * p[:, -1][:, None, None] * s + coef.transpose(1, 2) @ kc
        if segment_ids is not None:
            s_prev = sc[:, -1]
        pos += length

    return torch.cat(outputs, dim=1).to(in_dtype), s.to(in_dtype)


def delta_rule_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_a: torch.Tensor,
    beta: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token reference (O(1)/token). Ground truth for the chunked path.

    Same shapes/semantics as :func:`delta_rule_chunked`.
    """
    n, t_len, d_k = q.shape
    d_v = v.shape[-1]
    in_dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    a = torch.exp(log_a.float())
    beta = beta.float()

    s = (
        torch.zeros(n, d_v, d_k, device=q.device, dtype=torch.float32)
        if state is None
        else state.float()
    )
    prev_seg = None
    outputs: list[torch.Tensor] = []
    for step in range(t_len):
        if segment_ids is not None:
            seg_t = segment_ids[:, step]
            if prev_seg is not None:
                reset = (seg_t != prev_seg).float()[:, None, None]
                s = s * (1.0 - reset)
            prev_seg = seg_t
        at = a[:, step][:, None, None]
        bt = beta[:, step][:, None, None]
        kt, vt, qt = k[:, step], v[:, step], q[:, step]
        sk = (s * kt[:, None, :]).sum(-1)                   # S @ k_t  -> [N, d_v]
        corr = vt - at[:, :, 0] * sk                        # v_t - a_t * S k_t
        s = at * s + bt * (corr[:, :, None] * kt[:, None, :])
        outputs.append((s * qt[:, None, :]).sum(-1))        # S @ q_t
    return torch.stack(outputs, dim=1).to(in_dtype), s.to(in_dtype)
