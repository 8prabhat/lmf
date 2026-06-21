# Pure Parallel Gear V3

V3 is a separate architecture family. V2 remains available as an immutable
historical control.

## Families

- `pure_parallel_gear_v3`: strict constant-state recurrent rotor model with no
  attention or token-addressed history.
- `hybrid_parallel_gear`: V3 plus fixed-window grouped-query local attention.
- `bounded_transformer`: the same bounded attention and feed-forward path
  without gears.

## Core transition

Each rotor cell applies a contractive complex affine transition:

\[
q_t=a_tR(\phi_t)q_{t-1}+\sqrt{1-a_t^2}u_t.
\]

Transitions compose associatively. Document starts are represented as affine
transforms with zero multiplier, allowing resets to participate in the same
two-level scan without CPU planning or token loops.

Banks have non-overlapping half-life and rotation-period ranges:

| Bank | Half-life | Period |
| --- | ---: | ---: |
| Surface | 4–16 | 4–16 |
| Relations | 16–64 | 16–64 |
| Discourse | 64–256 | 64–256 |
| Planning | 256–2048 | 256–2048 |

The readout encodes each rotor cell locally, pools cells within banks, and uses
a low-rank bank mixer. There is no flattened global rotor feature vector,
boundary-settling loop, recurrent load/velocity, or shrinkable global gear
residual.

Within each bank, periods are structurally log-spaced across gears and rotation
direction alternates. Token controllers select a bank/channel offset inside
the bank's fixed range; they cannot reorder gears or move a bank outside its
declared range.

## Training

`PureParallelGearV3Trainer` supports contiguous document lanes and two-chunk
truncated BPTT. The final-layer bank states predict stopped-gradient future
token embeddings at horizons 4, 16, 64, and 256. This auxiliary objective
anneals to zero; next-token cross-entropy remains the final objective.

## Honesty contract

Strict V3 reports attention and history as absent. Hybrid Gear reports bounded
history and KV caching as present. Checkpoints are architecture-specific and
cannot be loaded across V2, V3, hybrid, or bounded Transformer families.

The retrained ablation matrix is recorded in
`configs/pure_parallel_gear_v3_ablations.yaml`. Legacy flattened-readout and
boundary-settling controls intentionally use V2 rather than reintroducing
removed mechanisms into the V3 production path.
