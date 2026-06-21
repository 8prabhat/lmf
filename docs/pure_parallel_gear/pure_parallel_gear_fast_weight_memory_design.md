# Fast-Weight Associative Memory — Stage 1 design (revised for predictive power)

Implemented as an experimental, off-by-default V2 hybrid in
`src/lmf/models/pure_parallel_gear/model.py`. It is not part of strict V3 or
the bounded-attention Hybrid Gear architecture. This document preserves the
design rationale and known self-copy/overshadowing risk. Three
changes below exist purely to strengthen the path from "memory retrieved
something" to "the right token's logit went up" — the engineering choices
from the original sketch (reuse `GearState`/`GearCache`, fold into the
existing per-timestep loop, decay-based stability) are unchanged and still
good; they were never the bottleneck for predictive power.

## What was weak in the original sketch, and the fix for each

### 1. Values were arbitrary features, not grounded in token identity

Originally: `value_t = value_proj(hidden_t)` — an arbitrary learned
projection. Nothing tied the stored content to an actual token's identity,
so even a perfect read had no direct route to "boost vocab id `v`'s logit" —
it had to hope the read vector accidentally pointed in a useful direction
after passing through `final_norm` and the tied head.

**Fix**: ground values in the token embedding space. Write
`value_t = value_down_proj(token_embedding(token_id_t))` — a small
`Linear(dim, value_dim)` applied to the *actual embedding of the token that
occurred*, not a generic hidden-state feature. A read is then literally a
decayed blend of previously-seen tokens' own embeddings. Add a matching
`value_up_proj: Linear(value_dim, dim)` so a read can be lifted back into
`dim` and passed through the existing (tied) head — `self.head(value_up_proj(read_t))`
is now a literal "which vocab id does this resemble" bias, because the head's
weight is the same embedding table the value was drawn from.

### 2. Constant blend strength, not content-dependent

Originally: a single learned scalar (`tanh(memory_residual)`), identical for
every token regardless of whether the memory actually found a strong match.
This is the same pattern used for `gear_residual`/`ffn_residual` — fine for
those (they're always "on"), wrong here: a copy-style mechanism is useful
exactly when it's confident and should otherwise stay quiet, or it just
injects noise into positions where copying isn't relevant. This was also a
capability I had in the *windowed-attention* sketch (a content-dependent
sigmoid gate) and dropped when simplifying to stage 1 — that drop cost real
predictive power, not just elegance.

**Fix**: bring the gate back, computed from both the current context and
what the memory found: `gate_t = sigmoid(Linear(dim + value_dim, 1)([source_t, read_t]))`.
This mirrors the precedent in Pointer-Generator networks (See et al. 2017),
where the copy gate is computed from context + retrieved content, not a
static parameter. With both this and #1, the mechanism can learn "when
`read_t` strongly resembles a specific recent token, lean on it; otherwise
don't."

### 3. Signal only reached logits indirectly, via the hidden-state residual

Originally the only injection point was `hidden = hidden + scale * memory_out`,
several layers/norms away from the final logits. Combined with #1's
embedding grounding, there's now a much shorter, more direct path available
and there's no reason not to use it.

**Fix**: in addition to the residual injection (kept, for representational
benefit to later layers/FFN), add a direct additive term to the *final
logits*: `logits = head(hidden) + gate_t * head(value_up_proj(read_t))`. The
gate and the embedding-grounded value mean this term is, by construction, an
interpretable vocabulary bias toward recently-seen tokens — not a hope that
a residual nudge survives the rest of the forward pass intact.

## Honest tradeoff this revision introduces

All three fixes make the mechanism strictly more powerful *and* more
literally a copy mechanism — which raises, more acutely than before, the
overshadowing concern raised earlier in this conversation: a strong,
content-gated, embedding-grounded copy path is a generically strong
language-modeling trick on its own, independent of anything specific to
gear. The content-dependent gate (#2) is the main safeguard — it lets the
model suppress the path when it isn't useful rather than applying it
uniformly — but it does not eliminate the confound. **The matched-baseline
ablation recommended earlier (the same mechanism wired onto GRU and
transformer, parameter-matched) is now more important, not less**: with
this revision, "gear + memory beats transformer" is even less informative on
its own than it would have been with the weaker original sketch, precisely
because the mechanism is now strong enough to plausibly carry the result by
itself.

## Config additions (revised)

```python
use_fast_weight_memory: bool = False
fast_weight_key_dim: int = 16          # d_k per bank
fast_weight_value_dim: int = 16        # d_v per bank (compressed embedding width)
fast_weight_decay: float = 0.99        # per-token decay; prevents unbounded growth
copy_gate_target_mean: float = 0.10    # expected fraction of tokens where copying helps
copy_gate_balance_weight: float = 0.02 # discourages gate collapsing to always-on/always-off
```

`copy_gate_target_mean` defaults low (0.10): most tokens should *not* be
copy-driven (most next-token decisions are genuinely generative), so the
gate is regularized toward mostly-closed, same spirit as
`clutch_target_mean` already does for the clutch mechanism. This also heads
off a new version of the same failure mode fixed earlier this session: a
gate with no balance pressure can saturate fully open (the mechanism
dominates every prediction — the overshadowing risk above, realized) or
fully closed (the mechanism never fires, wasted parameters), the same
dead-bank-style collapse shape, just on a new state variable. The fix is the
same one already validated in this codebase: a `copy_gate_balance`
diagnostic identical in form to `clutch_balance`, penalizing per-bank
deviation from the target mean, kept at full regularizer strength for the
whole run rather than annealed (because, like clutch, a saturated gate's
gradient vanishes and can't self-correct).

## Per-token computation (revised)

`key_proj`, `query_proj` (`Linear(dim, banks·key_dim)`) and `value_down_proj`
(`Linear(dim, value_dim)`, applied to the token embedding, not hidden state)
are precomputed for the whole sequence before the per-timestep loop, same as
before. Keys and queries are L2-normalized per bank before the dot product —
a standard, near-free fix (turns the match into cosine similarity) that
sharpens retrieval without changing asymptotic cost; raw unnormalized linear
functionals are a known weak point of plain linear-attention-style memory
and normalization is the standard mitigation (see also `cosFormer`-style
efficient attention variants).

Inside the existing per-timestep loop, per bank:

```
key_t   = normalize(key_proj(source_t)[b])
query_t = normalize(query_proj(source_t)[b])
value_t = value_down_proj(token_embedding(token_id_t))   # grounded, fix #1

S[b]    = decay * S[b] + outer(key_t, value_t)            # write
read_t  = query_t @ S[b]                                  # read, post-write

gate_t  = sigmoid(gate_proj(concat(source_t, read_t)))     # fix #2

memory_out  = memory_out_proj(read_t_all_banks)            # for hidden residual
logit_bias  = head(value_up_proj(read_t_all_banks))        # fix #3, direct path
```

`hidden = hidden + gate_t * tanh(memory_residual) * memory_out` keeps the
existing safe residual pattern for the representational path. The logit-bias
path is added once, after the head, in `forward()`/`training_step()`:
`logits = head(hidden) + gate_t * logit_bias`.

## Stability (revised)

In addition to the `memory_energy` diagnostic and guard from the original
sketch (unchanged — plain decayed summation can still saturate, this is
still the expected stage-1 weak point), add `copy_gate_balance` to
`training_step()`'s metrics and to the trainer's regularizer set, following
the `clutch_balance` precedent exactly: penalize `(gate.mean(per bank) −
copy_gate_target_mean)²`, unannealed, for the reason given above.

## Tests (additions to the original list)

7. **Embedding grounding sanity check**: with the gate forced to 1 and decay
   near-instant memory of a single recent token, confirm
   `head(value_up_proj(read_t))` puts its largest mass on the actual vocab id
   of that recent token — a direct test that fix #1's grounding does what
   it's supposed to, independent of training.
8. **Gate is content-dependent, not constant**: feed two inputs differing
   only in whether a target token was recently seen, confirm the gate's
   output differs measurably between them (catches a regression to the
   constant-scalar behavior the original sketch had).
9. **Gate-balance regularizer prevents collapse**: train briefly with
   `copy_gate_balance_weight` at zero vs. at the configured default on an
   adversarial setup that would otherwise push the gate to saturate, confirm
   the regularized run's gate stays closer to `copy_gate_target_mean` —
   mirrors how `clutch_balance` was validated against the dead-bank crash.
10. **Direct logit path actually moves predictions**: ablation test — same
    input, same trained weights, compare logits with the direct `logit_bias`
    term zeroed out vs. included; confirm a measurable, non-trivial
    difference (proves fix #3 isn't a no-op that the residual path alone
    would have already achieved).

## Cost estimate (revised)

`value_down_proj`/`value_up_proj` add two more small linears
(`dim ↔ value_dim`) plus the gate's `Linear(dim + value_dim, 1)` — still
tiny relative to the existing `angle_projection`/`clutch_projection` family.
The normalize ops are O(key_dim) elementwise, free. Net effect on the
per-step cost estimate from the original sketch is negligible — still well
under `settle()`'s existing cost, still no unfold/softmax/vocab-sized
scatter, still folds into the per-timestep loop with no new Python-level
iteration.

## Validation plan (unchanged, with one addition)

Same 500K-token capacity-probe re-run as the original plan, comparing
against `outputs/pure_parallel_gear_360_proxy/capacity_probe_20260620/results.json`.
Additionally track `copy_gate_balance` and the realized mean gate value
alongside `memory_energy` and val_top1 — if val_top1 breaks the plateau but
the mean gate value is pinned near 1 (not near `copy_gate_target_mean`),
that's the overshadowing risk materializing and a signal to run the
matched-baseline ablation (same mechanism on GRU/transformer) before
attributing the win to gear specifically.
