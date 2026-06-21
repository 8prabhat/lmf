# Pure Parallel Gear

This is the single canonical Pure Parallel Gear architecture. The previous V1
and V2 implementations were removed because both used historical token
selection or retrieval, which violated the intended mechanism.

## Mechanical state and update

Every layer owns four independent banks of eight gears. Each gear contains four
two-dimensional rotors, bounded angular velocity, torsional load, and a
token-local clutch. No tensor in the generation cache grows with context.
The banks are labelled surface/syntax, relations/entities,
discourse/continuity, and planning/constraints to define timescale priors and
diagnostics; these labels are not claims that semantic roles are guaranteed.

For one sentence, the version-2 affine rotor recurrence is:

\[
q_t=a_tR(\Delta\theta_t)q_{t-1}+u_t
\]

where each bank has a distinct bounded retention range. Fast banks can forget
more aggressively; slow/planner banks retain state longer. The recurrence has
an exact cumulative affine form:

\[
A_t=\prod_{j=1}^{t}a_j,\qquad
\Phi_t=\sum_{j=1}^{t}\Delta\theta_j
\]

\[
q_t=A_tR(\Phi_t)\left(
q_0+\sum_{i=1}^{t}A_i^{-1}R(-\Phi_i)u_i
\right).
\]

Production executes this scan over complete sentence/segment spans. Sequential
Python depth is proportional to span count rather than token count.

At deterministic sentence boundaries, even adjacent pairs, then overlapping
odd pairs, then cross-bank ring pairs receive state-dependent Givens rotations.
The overlapping order is intentionally noncommutative. Angular velocity and load
are updated only after clutch settling. A forced boundary at 128 tokens prevents
unbounded sentence depth. Lightweight explicit clutches also occur every 32
tokens inside long sentences, allowing banks to bind information before the
terminal sentence boundary without token-history retrieval.

Gear state is read through normalized rotor coordinates, log radius, radial and
angular state changes, relative phases, velocity, load, clutch activation, and
cross-bank differences. Boundary settling preserves bounded radius rather than
normalizing every rotor to unit length. RMSNorm, SwiGLU, residuals, embeddings,
and a tied vocabulary projection are token-local and do not perform sequence
mixing.

Pair and cross-bank clutch kernels are pair/channel-specific. The predictor
bank has a nonzero architectural residual floor, so optimization cannot disable
the entire predictor merely by shrinking one scalar.

## Architecture contract

`PureParallelGearLM.architecture_manifest()` records the following as false:

- self-attention and Q/K/V projections;
- token-to-token similarity;
- historical retrieval or token-history tensors;
- KV caching;
- routing over previous tokens;
- Transformer blocks.

Old V1/V2 checkpoints are explicitly rejected. Packed documents reset every
gear state. Full-sequence and incremental logits are required to match.

## Data and training

The existing SentencePiece BPE tokenizer is unchanged. Immutable paired
manifests include `sentence_ids`, `sentence_end_mask`,
`forced_boundary_mask`, and the frozen detector hash. The detector handles
terminal punctuation, closing quotes/brackets, common abbreviations, decimal
numbers, EOS, document changes, and the 128-token cap.

Gear parameters and AdamW moments remain FP32 on MPS. Slow speed/coupling
dynamics use a lower learning-rate multiplier; token-conditioned write
projections use the normal model LR. Dynamics receive tighter gradient
clipping. Weight decay is disabled for dynamics, norms, biases, and residual
scales. The semantic objective is only next-token cross-entropy; rotor energy,
velocity saturation, and anti-saturation clutch penalties are annealed
stability terms.

## Reproducible workflow

```bash
PYTHONPATH=src .venv/bin/python scripts/qualify_pure_parallel_gear.py \
  --device mps \
  --output outputs/pure_parallel_gear/qualification.json

PYTHONPATH=src .venv/bin/python scripts/prepare_pure_parallel_gear_data.py \
  index \
  --corpus-root /path/to/sentencepiece_bpe_corpus \
  --tokenizer-name sentencepiece_bpe_edu_subset_v1 \
  --bos-id 32768 --eos-id 32769 \
  --output-root outputs/pure_parallel_gear/index

PYTHONPATH=src .venv/bin/python scripts/benchmark_pure_parallel_gear.py \
  --stage 3m \
  --train-manifest-template 'outputs/pure_parallel_gear/train_seed_{seed}' \
  --validation-manifest outputs/pure_parallel_gear/validation \
  --confirmation-manifest outputs/pure_parallel_gear/development_test \
  --qualification outputs/pure_parallel_gear/qualification.json \
  --output-dir outputs/pure_parallel_gear/3m

PYTHONPATH=src .venv/bin/python scripts/run_pure_parallel_gear_ablations.py \
  --stage 3m \
  --train-manifest-template 'outputs/pure_parallel_gear/train_seed_{seed}' \
  --validation-manifest outputs/pure_parallel_gear/validation \
  --qualification outputs/pure_parallel_gear/qualification.json \
  --output-dir outputs/pure_parallel_gear/3m_ablations
```

Post-training evaluation is split deliberately:

- `evaluate_pure_parallel_gear_360.py` produces natural, compositional,
  robustness, efficiency, memory, profiler, gradient, state, and ablation
  evidence.
- `evaluate_pure_parallel_gear_generation.py` produces at least 210 fixed
  prompts, three continuation lengths, multiple decoding settings, and a
  separately keyed blind package.
- `finalize_pure_parallel_gear_gate.py` requires three independent blind-rating
  files, computes bootstrap intervals, applies frozen thresholds, hashes every
  input artifact, and writes `final_evidence.json` plus `report.md`.

The 50M run is blocked until the 15M quality gate passes.

## Claims that are not permitted

A finite gear state cannot guarantee exact arbitrary-length recall. Coherence,
factual consistency, and useful novelty cannot be proven from the equations.
They count only when supported by held-out, multi-seed, blinded empirical
evidence. Missing evidence produces an inconclusive or failed gate, never a
retroactive threshold change.
