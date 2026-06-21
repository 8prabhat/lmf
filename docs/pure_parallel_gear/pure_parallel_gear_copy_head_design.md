# Local Copy Head — design sketch (not yet implemented)

## Problem this addresses

The capacity probe (`outputs/pure_parallel_gear_360_proxy/capacity_probe_20260620/results.json`)
showed gear's validation top1 accuracy moves once (50K→100K tokens) then freezes
bit-for-bit for the remaining 400K tokens of a 500K-token run, while training loss
keeps falling. Transformer and GRU both keep improving over the same budget. This
points to a representational ceiling, not undertraining: gear has no mechanism to
address a specific prior token by identity, only a fixed-size compressed rotor
state, so it cannot reliably *copy* a token it has already seen (repeated names,
numbers, entities) even when that's the locally correct answer.

## Why this is an architecture-scope decision, not just a code change

`PureParallelGearLM`'s docstring (`model.py:3-5`) and `architecture_manifest()`
(`model.py:1201-1238`) declare hard invariants: `self_attention: False`,
`qkv_projections: False`, `token_similarity: False`, `history_tensor: False`,
`kv_cache: False`. A copy mechanism is, by definition, query/key similarity over
a window of recent tokens — it directly contradicts those invariants. This sketch
proposes adding it as an explicit, off-by-default, honestly-labeled hybrid mode,
not silently changing what "Pure Parallel Gear" means.

## Config additions

```python
use_local_copy_head: bool = False   # off by default; existing behavior unchanged
copy_window: int = 32               # fixed causal window, independent of seq_len
copy_head_dim: int = 32             # small Q/K projection width
```

## New module: `LocalCopyHead`

Lives alongside `PureGearLayer` in `model.py`, attached once at the top level
(`PureParallelGearLM`), operating on final hidden states — not inside each gear
layer. Three small linear projections: `query_proj`, `key_proj` (dim →
`copy_head_dim`, no bias) and `gate_proj` (dim → 1).

**Gate initialization is the critical safety detail.** `gate_proj.bias` initializes
to a strongly negative value (e.g. `-4.0`), so `sigmoid(gate) ≈ 0.02` at step 0 —
the model starts in almost-pure-gear mode and must *learn* to lean on the copy
path. This mirrors the lesson from the dead-bank-collapse fix earlier in this
conversation: an untrained, randomly-initialized new pathway must not be allowed
to dominate the output from step 1, or it risks destabilizing training the same
way the unbiased clutch initialization did.

## Forward computation (training — full sequence in parallel)

1. `query = query_proj(hidden)`, `key = key_proj(hidden)` — shape `[batch, seq, copy_head_dim]`.
2. Left-pad `key` and `token_ids` by `copy_window - 1`, then `unfold(dim=1, size=copy_window, step=1)`
   to get, for every position `t`, a window of the `copy_window` positions ending at `t`
   (causal by construction — no extra masking needed for causality itself).
3. Mask window slots where `segment_ids[window] != segment_ids[t]` (no copying across
   packed documents — same boundary respected by `test_packed_segments_reset_all_gear_state`)
   or where the original `token_mask` was False, set to `-inf` pre-softmax.
4. `scores = einsum('bsd,bswd->bsw', query, key_window) / sqrt(copy_head_dim)`,
   `attn = softmax(scores, dim=-1)` over the window.
5. Scatter attention weight onto actual vocab ids:
   `copy_distribution = zeros[batch, seq, vocab].scatter_add_(2, window_token_ids, attn)`.
   This is what makes it a *copy* mechanism — probability mass lands on specific
   token identities seen in the window, summed if a token repeats in-window.
6. `gate = sigmoid(gate_proj(hidden))`.
7. Final distribution: `p = (1 - gate) * softmax(head(hidden)) + gate * copy_distribution`.

For the **training loss only**, avoid materializing the full `[batch, seq, vocab]`
mixed tensor — gather the vocab log-softmax at the target id and separately gather
the copy mass landing on the target id (sum of `attn` over window slots whose
token id equals the target), then combine via `logsumexp` for numerical stability:

```
log p_target = logsumexp([log(1-gate) + log_softmax_vocab[target],
                           log(gate)   + log(copy_mass_at_target + 1e-8)])
```

This avoids ever forming the `[batch, seq, vocab]` tensor in the loss path (it's
only needed where sampling requires a full distribution — see below).

## Generation / cache integration

`generate()` produces one token at a time via `GearCache`. Add one new cache field,
`copy_window: CopyWindowCache`, holding fixed-size ring buffers (`recent_token_ids`,
`recent_keys`, `recent_segment_ids`, each `[batch, copy_window]`) — shift-and-append
each step since `copy_window` is small (e.g. 32), no real circular indexing needed.
At each incremental step: append the new (token id, key, segment id), compute
`query` from the new hidden state, attend over the buffer exactly as in training,
build `copy_distribution`, and mix with `softmax(head(hidden))` — here the full
mixed distribution *is* materialized since `_sample_token` needs it.

**Cache-size honesty:** this changes the cache from genuinely zero token-history
to `O(copy_window)` token-history, independent of total context length `T`. Still
far smaller than a transformer's `O(T)` KV cache, but `architecture_manifest()`
must report `kv_cache: True` / `history_tensor: True` when `use_local_copy_head`
is on — no silently keeping the old `False` flags once this exists.

## Required tests (mirroring existing repo conventions)

1. **Off-by-default no-op**: `use_local_copy_head=False` must produce bit-identical
   `forward()`/`training_step()` output to current code — protects all 19 existing tests.
2. **Gate starts near zero**: confirms the safety-critical init.
3. **Causal non-leakage**: future tokens don't change past logits (mirrors
   `test_future_tokens_cannot_change_past_logits`).
4. **Segment-boundary respect**: copy window doesn't span packed documents (mirrors
   `test_packed_segments_reset_all_gear_state`).
5. **Incremental == full-forward parity**: token-by-token generation via cache
   matches one parallel forward pass (mirrors `test_full_and_streaming_logits_match_with_constant_cache`).
6. **It actually copies**: construct a short sequence with a repeated rare token,
   force the gate toward 1, and assert the model assigns materially higher
   probability to the repeat than the vocab-only path would — the one test that
   proves the mechanism functions, not just that it's wired up.
7. **Manifest honesty**: `architecture_manifest()` flips `self_attention`/`history_tensor`/
   `kv_cache` to `True` when the flag is on.

## Cost estimate

Two `dim → copy_head_dim` linear layers plus a `dim → 1` gate — negligible parameter
count. Compute is `O(seq × copy_window × copy_head_dim)`, fixed-window so it doesn't
scale with total context length the way full attention would; expect it to be
cheap relative to `settle()`'s existing cost, to be confirmed by profiling once built.

## Validation plan after implementation

Re-run the same `probe_pure_parallel_gear_capacity.py` 500K-token comparison with
`use_local_copy_head=True` and compare against the saved baseline curve at
`outputs/pure_parallel_gear_360_proxy/capacity_probe_20260620/results.json`. Success
criterion: validation top1 should keep climbing past the ~100K-token point where
the un-augmented model flatlined, rather than reproducing the same frozen plateau.
