# Simplified Gear Transformer V1

> **RETRACTED AS A PURE-GEAR ARCHITECTURE.**
>
> This model is a Transformer/gear hybrid and does not implement the requested
> independent gear mechanism. Its results must not be cited as evidence for a
> pure parallel-gear language model. The corrected architecture is
> The independent canonical architecture is documented in
> `docs/pure_parallel_gear.md`.

Status: retained only as a historical hybrid experiment.

Registry names:

- Model: `simplified_gear_transformer`
- Trainer: `simplified_gear_transformer`

## Design objective

Simplified Gear V1 reallocates the full V5 parameter budget toward mechanisms
that repeatedly improved held-out NLL while removing mechanisms whose measured
effect was approximately zero.

## Topology

```text
tokens
  -> Transformer block 0
       -> one stride-1 parallel gear bank
            -> phase-conditioned slot routing
            -> 16-dimensional rotated recurrent memories
            -> temporal context
            -> within-lane routing and lane mixing
       -> residual return to the trunk
  -> Transformer blocks 1 and 2
  -> multi-horizon future-logit residual
  -> tied LM head
```

The benchmark-scale architecture uses:

- Transformer trunk: `dim=72`, `layers=3`, `heads=4`;
- one gear bank at block 0;
- five gears in four lanes `[2, 1, 1, 1]`;
- `gear_dim=16`, increased from V5's benchmark width of 8;
- geometric rotation over all 16 gear dimensions;
- temporal stride 1;
- future horizons 2 and 4.

## Retained mechanisms

- Positive monotonic gear clocks.
- Token-driven phase conditioning.
- Phase-addressed slots.
- Geometric memory rotation.
- Persistent recurrent gear memories.
- Temporal context carrier within the bank.
- Gear-to-lane routing and lane mixing.
- Future latent/token supervision and future-logit residual.
- Diversity, slot-usage, lane-prediction, alignment, and consistency training
  objectives.

The auxiliary heads are training-time mechanisms. The primary inference graph
remains one bank plus the future-logit residual.

## Removed mechanisms

- The second slow stacked bank.
- Inter-bank carrier and coupling.
- Mechanical phase coupling.
- Phase-lock loss.
- Bank-specialization prior.
- Slow-bank temporal holding.

Disabled coupling tensors are not allocated as trainable zero-length
parameters. The simplified model has no unused first-bank inter-bank projection
and no unused final carrier projection.

## Parameter matching

Benchmark model:

- Simplified Gear V1: 312,509 parameters.
- Matched Transformer: 311,688 parameters.
- Relative difference: +0.263% for Simplified Gear.

The matched Transformer uses the same depth (`layers=3`) and is widened to
`dim=78`, with three attention heads. Holding depth constant avoids selecting a
pathologically narrow/deep baseline merely because its parameter count is a
fraction of a percent closer.

## Gradient and execution invariants

- Every trainable tensor receives a finite, non-zero gradient in a
  representative full-objective backward pass.
- Full-sequence and cached decoding equivalence remains covered by the parent
  V5 tests.
- The simplified bank has no phase-coupling parameters, inter-bank projection,
  or outgoing carrier.
- All three Transformer blocks have positive acute ablation impact in every
  benchmark seed.

## Current result

Across five seeds and 300 matched updates:

- Simplified Gear NLL: `3.7930 ± 0.0538`.
- Matched Transformer NLL: `3.6099 ± 0.0659`.
- Simplified Gear lost all five seeds by `0.1830 ± 0.0458` NLL.
- Simplified Gear improved over the earlier full Gear V5 by `0.0611` mean NLL.
- Isolated training throughput improved from roughly 23.0K tokens/s for full
  V5 to 27.7K tokens/s.

The architecture is a meaningful improvement over full V5 but does not beat the
modern Transformer baseline. Temporal context is effectively neutral in the new
five-seed result and is the next structural removal candidate.
