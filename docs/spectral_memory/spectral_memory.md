# Spectral Memory (SM-LM)

A clean-slate, recall-capable, Mac-fast language-model family. Full rationale and
the design's evidence base live in the technical spec
(`~/Downloads/sm_lm_mac_gpu_technical_specification.md`). This doc is the in-repo
pointer for the implementation.

## Concept

A **filter bank of content-addressable, error-correcting associative memories
spread across the temporal-timescale spectrum**, fused by a learned read, with a
thin slice of local attention for exact copy.

- **MTDM — Multi-Timescale Delta Memory** (`model.py:MultiTimescaleDeltaMemory`):
  `banks` matrix memories `S^h ∈ R[d_v, d_k]`, each updated by the **gated delta
  rule** (error-correcting write), each hard-banded to a distinct log-spaced
  decay half-life so the bank set tiles the timescale axis and cannot collapse.
  This banding is the point of departure from Gated DeltaNet's free per-head
  decay.
- **CBIR — Cross-Band Interference Router**: input-dependent (sigmoid/softmax)
  gate fusing the per-bank reads.
- **SWA — Sliding-Window Attention** (`model.py:SlidingWindowAttention`): 1-2
  designated layers (`attention_layers`) for exact local recall / induction.
- **token-shift** instead of depthwise conv (Mac-friendly local mixing).

## Why it meets the goals

- **Predictive power**: delta rule (recall) + multi-timescale banks
  (short+long state-tracking) + thin attention (exact copy).
- **Fast inference**: each bank carries a fixed `d_v × d_k` state → O(1)/token.
- **Fast Mac training**: the gated delta rule runs through the chunk-parallel,
  matmul-only path in `delta_scan.py` (`delta_rule_chunked`) — no custom scan, no
  depthwise conv, no FFT. Mirrors the chunking philosophy of
  `bounded_hybrid_gear.scan.chunked_affine_scan`, generalized to matrix state.

## Files

| File | Contents |
|------|----------|
| `delta_scan.py` | `delta_rule_chunked` (training/eval), `delta_rule_recurrent` (O(1) decode + test ground truth), `_inv_unit_lower` (MPS-safe unit-lower-triangular inverse via geometric-series doubling) |
| `model.py` | `SpectralMemoryConfig`, `MultiTimescaleDeltaMemory`, `SlidingWindowAttention`, `SpectralMemoryBlock`, `SpectralMemoryLM`, `build_spectral_memory` (`@MODELS.register("spectral_memory")`) |
| `trainer.py` | `build_spectral_memory_trainer` (`@TRAINERS.register("spectral_memory")`, reuses `NativeLMTrainer`) |
| `configs/spectral_memory.yaml` | smoke / prototype / larger blocks |
| `tests/test_spectral_memory.py` | spec §15 gates: chunk≡recurrent, causality, grads, shapes, tiny-overfit, parallel≡decode, decay bands |

## Correctness invariants (tested)

- `delta_rule_chunked` **==** `delta_rule_recurrent` (output and final state), with
  and without segments/incoming state — the equivalence that lets the chunk path
  train and the recurrent path decode.
- Strict causality: perturbing future tokens never changes past logits.
- Parallel forward **==** incremental KV/state-cached decode.

## Known limitations / TODO

- **MLX port**: the `_inv_unit_lower` + chunk matmuls are deliberately MPS-safe;
  an optional fused `mps_delta_scan.py` (mirroring `mps_affine_scan`) is future
  work if MPS throughput is the bottleneck (spec §8, Risk R1).
- **Segment isolation** is handled in both scan paths via contiguous-segment
  equality masks; verify against your packing if documents are interleaved.
- Decode reuses a full KV buffer for SWA layers (window enforced by mask);
  truncating to `window` with correct RoPE offsets is a later optimization.
