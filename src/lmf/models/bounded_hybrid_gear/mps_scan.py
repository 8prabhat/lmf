"""Fused FP32 complex affine scan for Apple Metal.

The MPS eager implementation of an associative scan dispatches many small
dependent kernels.  This module keeps the mathematically identical recurrence
but executes one Metal thread per independent rotor cell.  The backward is
fused as a reverse recurrence, avoiding a Python or eager-op graph over time.
"""

from __future__ import annotations

import threading

import torch


_SHADER_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void affine_scan_forward(
    device const float* multiplier [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device const float* initial [[buffer(2)]],
    device float* states [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    constant uint& cells [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    uint batch_index = index / cells;
    uint cell = index - batch_index * cells;
    uint initial_offset = (batch_index * cells + cell) * 2;
    float state_real = initial[initial_offset];
    float state_imag = initial[initial_offset + 1];
    for (uint position = 0; position < length; ++position) {
        uint offset = ((batch_index * length + position) * cells + cell) * 2;
        float multiplier_real = multiplier[offset];
        float multiplier_imag = multiplier[offset + 1];
        float next_real = (
            multiplier_real * state_real
            - multiplier_imag * state_imag
            + bias[offset]
        );
        float next_imag = (
            multiplier_real * state_imag
            + multiplier_imag * state_real
            + bias[offset + 1]
        );
        states[offset] = next_real;
        states[offset + 1] = next_imag;
        state_real = next_real;
        state_imag = next_imag;
    }
}

kernel void affine_scan_backward(
    device const float* multiplier [[buffer(0)]],
    device const float* initial [[buffer(1)]],
    device const float* states [[buffer(2)]],
    device const float* grad_states [[buffer(3)]],
    device float* grad_multiplier [[buffer(4)]],
    device float* grad_bias [[buffer(5)]],
    device float* grad_initial [[buffer(6)]],
    constant uint& length [[buffer(7)]],
    constant uint& cells [[buffer(8)]],
    uint index [[thread_position_in_grid]]
) {
    uint batch_index = index / cells;
    uint cell = index - batch_index * cells;
    float carry_real = 0.0f;
    float carry_imag = 0.0f;
    for (uint reverse_position = 0; reverse_position < length; ++reverse_position) {
        uint position = length - reverse_position - 1;
        uint offset = ((batch_index * length + position) * cells + cell) * 2;
        float adjoint_real = grad_states[offset] + carry_real;
        float adjoint_imag = grad_states[offset + 1] + carry_imag;
        grad_bias[offset] = adjoint_real;
        grad_bias[offset + 1] = adjoint_imag;

        float previous_real;
        float previous_imag;
        if (position == 0) {
            uint initial_offset = (batch_index * cells + cell) * 2;
            previous_real = initial[initial_offset];
            previous_imag = initial[initial_offset + 1];
        } else {
            uint previous_offset = (
                (batch_index * length + position - 1) * cells + cell
            ) * 2;
            previous_real = states[previous_offset];
            previous_imag = states[previous_offset + 1];
        }
        grad_multiplier[offset] = (
            adjoint_real * previous_real
            + adjoint_imag * previous_imag
        );
        grad_multiplier[offset + 1] = (
            -adjoint_real * previous_imag
            + adjoint_imag * previous_real
        );

        float multiplier_real = multiplier[offset];
        float multiplier_imag = multiplier[offset + 1];
        carry_real = (
            multiplier_real * adjoint_real
            + multiplier_imag * adjoint_imag
        );
        carry_imag = (
            -multiplier_imag * adjoint_real
            + multiplier_real * adjoint_imag
        );
    }
    uint initial_offset = (batch_index * cells + cell) * 2;
    grad_initial[initial_offset] = carry_real;
    grad_initial[initial_offset + 1] = carry_imag;
}
"""

_shader_library = None
_shader_lock = threading.Lock()


def _library():
    global _shader_library
    if _shader_library is None:
        with _shader_lock:
            if _shader_library is None:
                _shader_library = torch.mps.compile_shader(_SHADER_SOURCE)
    return _shader_library


class _MPSAffineScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        multiplier: torch.Tensor,
        bias: torch.Tensor,
        initial: torch.Tensor,
    ) -> torch.Tensor:
        multiplier = multiplier.contiguous()
        bias = bias.contiguous()
        initial = initial.contiguous()
        batch, length = multiplier.shape[:2]
        cells = multiplier.numel() // (batch * length * 2)
        states = torch.empty_like(multiplier)
        _library().affine_scan_forward(
            multiplier,
            bias,
            initial,
            states,
            length,
            cells,
            threads=batch * cells,
        )
        ctx.save_for_backward(multiplier, initial, states)
        ctx.length = length
        ctx.cells = cells
        return states

    @staticmethod
    def backward(ctx, grad_states: torch.Tensor):
        multiplier, initial, states = ctx.saved_tensors
        grad_states = grad_states.contiguous()
        grad_multiplier = torch.empty_like(multiplier)
        grad_bias = torch.empty_like(multiplier)
        grad_initial = torch.empty_like(initial)
        batch = multiplier.shape[0]
        _library().affine_scan_backward(
            multiplier,
            initial,
            states,
            grad_states,
            grad_multiplier,
            grad_bias,
            grad_initial,
            ctx.length,
            ctx.cells,
            threads=batch * ctx.cells,
        )
        return grad_multiplier, grad_bias, grad_initial


def mps_affine_scan(
    multiplier: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Run the fused scan, requiring contiguous FP32 MPS inputs."""
    if multiplier.device.type != "mps":
        raise ValueError("mps_affine_scan requires MPS tensors")
    if multiplier.dtype != torch.float32:
        raise ValueError("mps_affine_scan requires FP32 accumulation")
    if multiplier.shape != bias.shape or multiplier.shape[-1] != 2:
        raise ValueError("multiplier and bias must share shape [B, T, ..., 2]")
    if initial.shape != multiplier.shape[:1] + multiplier.shape[2:]:
        raise ValueError("initial state shape must match scan state without time")
    return _MPSAffineScan.apply(multiplier, bias, initial)
