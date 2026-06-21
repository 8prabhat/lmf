# Gear Transformer V2 Mechanism

This revision makes the gear path more than a phase-conditioned residual
adapter. The token stream still keeps the full Transformer attention/MLP path,
but the gear stream now has explicit write, update, cross-gear coupling, and
read stages.

## Block-Level Flow

```text
token hidden h
  -> causal attention
  -> compact gear input u = W_down LN(h)
  -> per-gear write:
       phase features
       soft slot routing
       causal token summary over gear receptive field
       learned write gate
       causal gear state update
  -> cross-gear coupling:
       learned same-position gear graph
       off-diagonal gear message passing
  -> gear readback:
       learned read gate
       fusion over coupled gears
       W_up projection to token width
  -> MLP
```

## What Changed From V1

- **Causal token summaries**: each gear receives a causal moving summary over
  its own receptive field, instead of only seeing the current token hidden
  state.
- **Write gates**: each gear learns how strongly to absorb the proposed update,
  conditioned on token state, routed slot, causal summary, and phase.
- **Read gates**: each gear learns how strongly to expose its latent state back
  to the token stream.
- **Cross-gear coupling**: gear outputs exchange messages through a learned
  same-token gear graph before fusion.
- **Compact side channel**: the gear mechanism runs in `gear_dim`, then projects
  back to the token stream; the main token path remains full width.

## New Diagnostics

Training now exposes:

- `gear_write_activity`
- `gear_read_activity`
- `gear_coupling_entropy`
- `gear_coupling_gate`
- `gear_coupling_offdiag`
- `gear_conflict`

Useful failure signals:

- write/read activity near zero: gears are inactive.
- coupling off-diagonal near zero: gears are not communicating.
- coupling entropy near one forever: gear graph may be too uniform.
- conflict always high: alignment signal is uncalibrated.
- conflict always low: gears may have collapsed into the same behavior.

## Ablation Switches

The core ablation config now includes:

- `model.gear_write_summary`: causal summary on/off.
- `model.cross_gear_coupling`: cross-gear graph on/off.
- `model.gear_coupling_init`: initial coupling strength.
- `model.gear_dim`: compact gear width.

These are required to show that the richer mechanism is helping, not merely
adding complexity.
