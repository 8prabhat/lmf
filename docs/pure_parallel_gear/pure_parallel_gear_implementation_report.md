# Pure Parallel Gear implementation report

Date: June 20, 2026

## Status

The previous V1/V2 source families, configs, scripts, tests, and active
documentation have been replaced by one registered family:
`pure_parallel_gear`. SentencePiece BPE is unchanged.

The canonical family now uses architecture-manifest version 2. Version-1
checkpoints are intentionally incompatible.

This report records implementation verification only. It does **not** claim
that the model beats a Transformer. Proxy, 3M, 15M, blinded generation, and
conditional 50M training artifacts must be produced before any quality or
efficiency claim is accepted.

## Implemented mechanism

- Four independently rotating gear banks with configurable role labels.
- Eight gears per bank and four two-dimensional rotor channels per gear.
- Token-conditioned angle, clutch, torque, and affine-retention projections.
- Distinct bounded retention ranges for fast through slow/planner banks.
- Verified cumulative affine closed-form recurrence.
- Sentence/segment-span production scans with packed reset and boundary control.
- Even, overlapping odd, and cross-bank noncommutative Givens clutching.
- Pair/channel-specific intra-bank and cross-bank coupling kernels.
- Boundary-carried velocity and torsional load.
- Bounded radial state preserved through settling and exposed in every readout.
- Deterministic punctuation/EOS/document boundaries plus a 128-token cap.
- Stacked gear layers and an optional structural predictor gear bank.
- Nonzero architectural predictor residual floor.
- Constant-size `GearCache`; no historical token tensors.
- Token-local RMSNorm, SwiGLU, residuals, embeddings, and tied output head.

The architecture manifest explicitly declares attention, Q/K/V, token
similarity, historical retrieval, routing, Transformer blocks, and KV caching
absent.

## Verified engineering results

The MPS qualification run completed successfully on:

- Python 3.13.5
- PyTorch 2.12.0
- Apple MPS

Measured qualification results:

| Check | Result |
| --- | ---: |
| Closed-form vs sequential maximum rotor error | `3.5762787e-07` |
| Closed-form vs sequential maximum input-gradient error | `1.7881393e-07` |
| Full-sequence vs incremental maximum logit error | `4.6193600e-07` |
| Cache bytes at 16, 64, and 256 tokens | `664`, `664`, `664` |
| Model parameter dtype | FP32 |
| Adam moment dtype | FP32 |
| Forbidden source mechanism scan | Passed |

The repository-wide pytest suite passed. Focused Pure Gear tests cover:

- causal invariance;
- packed-document reset;
- full/incremental equivalence;
- constant cache;
- order sensitivity;
- noncommutative clutch order;
- independent pre-clutch gears;
- finite nonzero gradients;
- frozen sentence metadata;
- structural ablation parameter removal;
- parameter matching;
- explicit legacy-checkpoint rejection.

## Architecture-version-2 smoke evidence

A fresh parameter-matched MPS smoke run used:

- 20,362 supervised tokens per model;
- seed `20262000`;
- identical immutable train/validation/test manifests;
- independently selected learning rates with the same search budget;
- 208,031 Gear parameters;
- 208,080 Transformer parameters;
- 207,972 GRU parameters.

| Metric | Transformer | GRU | Gear v2 |
| --- | ---: | ---: | ---: |
| Test macro NLL | 7.8823 | 7.9903 | 7.9467 |
| Test top-1 | 2.34% | 7.08% | 2.09% |
| Training tokens/s | 61,798 | 1,659 | 644 |
| 512-token prefill tokens/s | 354,336 | — | 4,593 |
| 2K incremental tokens/s | 1,227 | — | 144 |
| 2K cache bytes | 1,310,720 | — | 4,408 |

At this tiny budget Gear v2 is 0.82% worse than Transformer macro NLL and
better than the matched GRU. This is encouraging but not a statistical quality
claim.

The mechanism probe is more important than the headline NLL:

| Post-hoc diagnostic | NLL change versus full |
| --- | ---: |
| No predictor gear | +0.01597 |
| No boundary settling | +0.00089 |
| Fixed angular velocity | +0.00271 |
| No local SwiGLU | +0.00001 |
| No cross-bank coupling | -0.00018 |

The predictor, boundary settling, and learned velocity now carry measurable
predictive signal. The local SwiGLU is nearly neutral at this budget, unlike
architecture version 1 where it carried most measured quality. Cross-bank
coupling remains unproven and requires a retrained multi-seed ablation.

Mechanism-state diagnostics:

- main-layer residual scales remained near `0.10`;
- predictor residual remained near `0.25`;
- predictor removal is no longer neutral;
- bank retention means were approximately `0.953`, `0.976`, `0.991`, and
  `0.998`;
- slow/planner rotor radii remained materially larger than fast-bank radii;
- all coupling paths were active.

Efficiency is not solved. Sentence-span scans improved 512-token Gear prefill
by roughly eight times versus the previous implementation, but Transformer
prefill and incremental decoding remain much faster on MPS. Constant cache is
confirmed.

## Training and evidence workflow

The implementation supplies:

- `prepare_pure_parallel_gear_data.py`
- `qualify_pure_parallel_gear.py`
- `benchmark_pure_parallel_gear.py`
- `run_pure_parallel_gear_ablations.py`
- `evaluate_pure_parallel_gear_360.py`
- `evaluate_pure_parallel_gear_generation.py`
- `finalize_pure_parallel_gear_gate.py`

The benchmark uses identical immutable manifests, parameter matching within
0.5%, equal LR search budgets, equal-token and equal-wall-clock runs, and an
explicit parameter-token equal-compute proxy. Transformer is the primary
baseline; a parameter-matched GRU is the recurrent control.

The finalizer requires frozen protocol thresholds, paired bootstrap intervals,
Holm-adjusted primary tests, complete seed runs, profiler evidence, true peak
memory evidence, and three independent blind raters. The current MPS allocator
snapshot is explicitly not accepted as peak memory. The finalizer hashes all
source artifacts and emits JSON, Markdown, and SVG comparison plots.

## Remaining compute work

The following are intentionally not marked complete by implementation tests:

1. Five-seed proxy and 3M training.
2. Retrained proxy/3M structural ablations.
3. Three-seed 15M quality gate.
4. At least 210 fixed generation prompts with three independent raters.
5. Conditional two-seed 50M sealed confirmation.

Missing or failed evidence causes a failed/inconclusive gate. Thresholds are not
changed after results are observed.

## Limitations

A finite state cannot guarantee exact arbitrary-length recall. The equations do
not prove coherence, factual consistency, useful novelty, training speed, or a
quality advantage. Those remain empirical questions governed by the frozen
evaluation protocol.
