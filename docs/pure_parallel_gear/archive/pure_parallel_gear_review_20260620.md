# Pure Parallel Gear architecture and implementation review

Date: 2026-06-20

## Architecture version 2 implementation status

The recommended core changes have now been implemented:

- affine token-conditioned retention;
- non-overlapping fast/medium/slow/planner retention ranges;
- preserved bounded rotor radius and radial-change readout;
- pair/channel-specific coupling kernels and dynamic responses;
- explicit 32-token intra-sentence clutches;
- nonzero predictor residual floor;
- sentence/segment-span cumulative execution instead of a token-step loop;
- architecture manifest version 2 with version-1 checkpoint rejection.

The numerical findings below describe the pre-revision checkpoint and explain
why the changes were made. They are not claimed as results for version 2;
version 2 requires fresh training.

## Executive verdict

The current model is causal, uses constant-size recurrent state, has no hidden
attention path, and its streamed logits match full-sequence logits. Those are
real strengths.

The current evidence does not yet show that gear composition is responsible
for language-model quality. At the latest fair 100K-token proxy, Gear reaches
test macro NLL 7.0543 versus 6.7999 for the matched Transformer. More
importantly, the learned residual scales and ablations show that the network is
using the token-local SwiGLU while suppressing the gear and predictor paths.

The architecture should not be scaled unchanged. The next revision should
preserve the no-attention contract while changing the state transition,
readout, predictor, and training execution path.

## Evidence from the current checkpoint

The 100K-token matched checkpoint contains 158,281 Gear parameters versus
158,272 Transformer parameters.

- Gear test macro NLL: 7.0543
- Transformer test macro NLL: 6.7999
- Gear top-1 accuracy: 1.05%
- Transformer top-1 accuracy: 9.52%

Learned residual scales:

- layer 0 gear: 0.0379
- layer 1 gear: 0.0293
- predictor gear: 0.0012
- layer 0 local FFN: 0.1264
- layer 1 local FFN: 0.1469

The predictor is effectively switched off. Both main gear paths shrink while
the token-local paths grow.

Post-hoc diagnostic NLL on the same validation rows:

| Variant | NLL | Change from full |
| --- | ---: | ---: |
| Full | 7.26686 | 0 |
| No predictor gear | 7.26687 | +0.00001 |
| No boundary settling | 7.26632 | -0.00054 |
| No cross-bank coupling | 7.26658 | -0.00028 |
| Fixed angular velocities | 7.26613 | -0.00073 |
| No local SwiGLU | 7.29145 | +0.02459 |

These are diagnostic post-hoc ablations, not causal retrained ablations, but
the direction is consistent: the current gear-specific composition is nearly
neutral while the token-local FFN matters.

## Architectural findings

### 1. The transition has no selective forgetting

Within a sentence the implemented recurrence is

\[
q_t = R(\alpha_t)q_{t-1} + u_t.
\]

The Jacobian with respect to the previous rotor is an orthogonal rotation, so
its norm is one. This helps gradient preservation, but the state cannot
selectively forget or overwrite old information. Writes accumulate until a
boundary normalization.

Recommended replacement:

\[
q_t = a_t R(\alpha_t)q_{t-1} + u_t,
\qquad
a_t = a_{\min} + (1-a_{\min})\sigma(W_a h_t).
\]

This adds token-conditioned retention without attention. It also preserves a
parallelizable affine composition:

\[
(A_2,b_2)\circ(A_1,b_1)
=
(A_2A_1,\;A_2b_1+b_2).
\]

Therefore a segmented associative scan remains possible.

### 2. The readout discards radial information

The readout normalizes every rotor before exposing it to the model. Therefore

\[
f(q)=f(cq),\quad c>0.
\]

Two different histories that produce collinear states with different
magnitudes are indistinguishable inside a sentence. Magnitude is converted
into load only during boundary settling.

Add per-rotor `log_radius = log(||q|| + eps)` and radial change to every token
readout. Use bounded energy control instead of discarding radius.

Suggested state features:

- normalized rotor coordinates;
- log radius;
- token-to-token radial and phase change;
- adjacent relative phase;
- omega and load;
- cross-bank differential features.

### 3. Token writes are open-loop

At a given layer, angle, clutch, and torque are projected from the current
token representation, not from that layer's previous gear state. The layer
cannot decide to overwrite a memory because of what it already stores.

Fully state-dependent nonlinear control would destroy the simple closed-form
scan. The first revision should therefore use token-conditioned affine decay
and structured erase/write channels. If that is insufficient, test a
state-conditioned controller as a separate sequential ablation rather than
silently sacrificing parallel training.

### 4. Timescale separation is only an initialization

Fast, medium, slow, and planner banks receive different initial angular
velocities, but training can collapse them into the same regime.

Parameterize each bank inside a distinct learnable retention/frequency range.
The ranges are inductive biases, not semantic claims. Report the learned
timescale distributions and fail an experiment if all banks collapse.

### 5. Boundary composition is too low-rank

Each layer shares one eight-value pair kernel across every intra-bank pair and
one across every cross-bank pair. Gates differ, but the interaction function is
almost identical everywhere.

Use pair-specific low-rank kernels or bank-pair embeddings. This increases
compositional capacity without creating token history or attention.

### 6. Settling destroys information after mixing

Givens mixing is energy-preserving, but the implementation then normalizes
each rotor independently. Relative energy moved between gears is lost and
compressed through a shared two-parameter load response.

Keep the orthogonal coupling, preserve bounded radius, and update load with
gear-specific or low-rank responses. Do not normalize every rotor to unit
length.

### 7. Cross-bank interaction is too late

Persistent banks interact only at sentence boundaries. Many bindings needed
for next-token prediction form inside a sentence.

Add explicit lightweight intra-sentence clutch events at fixed small intervals
or at deterministic punctuation/sub-clause boundaries. These remain temporary
mechanical clutches, not attention. Their benefit must be verified against the
loss of parallel depth.

### 8. The predictor bank is not functioning as a predictor

It is another generic gear layer attached through a freely shrinkable residual.
The learned residual is approximately 0.0012, and disabling it has no material
effect.

Either remove it or redesign prediction so state differentials are directly
in the lexical path:

\[
z_t = \operatorname{RMSNorm}
\left(W_hh_t + W_ss_t + W_\Delta(s_t-s_{t-1})\right),
\quad
\text{logits}=E z_t.
\]

The predictor should have a measurable, retrained ablation effect before it is
retained.

### 9. The implementation is not token-parallel

The source currently loops over every sequence position. It vectorizes across
batch rows but not across time. At 2K context, five-repeat synchronized
measurements show:

- Transformer prefill p50: 0.00443 seconds;
- Gear prefill p50: 3.805 seconds.

Implement the promised segmented sentence scan. Process all tokens in each
sentence with cumulative affine transforms, and loop only over sentence index.
With a 128-token cap, a 2K sequence should have roughly 16 sequential sentence
steps rather than 2,048 Python steps.

### 10. Training resets state at every manifest row

The model supports persistent generation state, but training does not carry
GearCache across contiguous chunks. This limits learning of document-scale
state behavior.

Add a stateful document-stream mode:

- contiguous chunks from the same document;
- carry GearCache between chunks;
- detach state at a declared truncated-BPTT interval;
- reset only at document boundaries;
- keep a stateless mode as an ablation.

## Training-code findings and fixes

The review fixed these defects:

1. MPS control-state device mismatch in the vectorized Gear loop.
2. Per-token MPS synchronization caused by accelerator boolean branches.
3. Angle/clutch/torque write projections incorrectly received the reduced
   dynamics learning rate.
4. Clutch regularization forced a target mean and variance instead of merely
   preventing saturation.
5. Dead-state diagnostics averaged whole banks and could miss individual dead
   gears.
6. Equal-time LR schedules advanced during validation and logging.
7. Optimization throughput included validation time.
8. GRU packed-segment scanning read MPS scalars token by token.
9. Prompt generation incorrectly forced a sentence boundary at the final
   prompt token.
10. Gear generation benchmarks were not guaranteed to configure the frozen
    boundary detector.

Additional training changes still needed:

- log gear residual scales, radial statistics, write/erase entropy, per-bank
  effective rank, and timescale overlap;
- introduce stateful contiguous-document training;
- implement the segmented affine scan before throughput claims;
- use retrained structural ablations, not only post-hoc masks;
- test a controlled token-local-path dropout or contextual-path curriculum if
  the model continues to bypass state.

## Comparison-code findings and fixes

The old efficiency result was not valid:

- MPS work was timed without synchronization;
- “long-context generation” always generated from at most 32 or 64 prompt
  tokens;
- the measured generation time included a fresh prefill;
- the capacity probe reused one mutable corpus cursor across models;
- limited validation used the first rows rather than deterministic spread
  sampling;
- irrelevant-prefix NLL scored the random prefix itself;
- generation reference NLL omitted Gear sentence boundaries;
- allocator memory after backward was labeled peak memory.

The stage benchmark also constructs and parameter-matches models in Python; it
does not load `configs/pure_parallel_gear.yaml`. Changing that YAML alone does
not change the tested architecture. Result artifacts now record every actual
instantiated config and architecture manifest.

At 2K, corrected five-repeat incremental throughput is 1,295.3 tok/s for the
Transformer and 233.2 tok/s for Gear. Gear cache is 4,408 bytes versus
1,048,576 bytes for Transformer. This confirms the constant-cache advantage,
but not a decoding-speed advantage. Gear prefill is approximately 859x slower.

True peak training memory is not currently available from the MPS allocator
API used by the harness. The final gate now treats that result as unavailable
instead of accepting a post-backward snapshot as peak memory.

## Required next experiment

Do not run 3M yet. First implement and compare four small retrained variants:

1. current corrected model;
2. affine retention plus radial readout;
3. variant 2 plus redesigned predictor fusion;
4. variant 3 plus segmented sentence scan.

Use at least three seeds and 500K-1M supervised tokens. Include:

- matched total parameters;
- matched non-embedding parameters;
- same manifest row IDs in the same order;
- synchronized prefill and cached decoding;
- token-local-only control;
- state-matched affine recurrent control;
- retrained component ablations.

Advance only if the gear-specific ablations become materially positive and the
model stops shrinking its gear/predictor residuals.

## Verification after fixes

- MPS engineering qualification: passed.
- Closed-form output error: `5.36e-7`.
- Closed-form input-gradient error: `2.38e-7`.
- Full versus streamed logit error: `2.38e-7`.
- Cache remained constant at 16, 64, and 256 prompt tokens.
- Parameters and optimizer moments remained FP32.
- Repository test suite: 230 tests passed, with 2 deliberately deselected.
- Focused post-change suite: 48 tests passed.
