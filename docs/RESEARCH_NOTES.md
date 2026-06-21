# Research Notes

Condensed record of every architecture decision, tokenizer experiment, and
pilot result produced in this repo. Replaces the previous five `docs/*.md`
files and the fourteen `results/**/*.md` summaries, which carried a lot of
repeated caveats and have been folded in here. Raw JSON artifacts referenced
below still live under `results/` and `outputs/tokenizer/spt_bench/`.

Shared caveat that applies to every pilot table in this document unless
stated otherwise: single seed, 200-1,000 training steps, a handful of
validation batches, CPU/MPS smoke hardware. These are directional pilots for
choosing the next iteration, not final benchmarks.

## 0. Canonical Directory Map

Every model family's source, config, docs, tests, and (gitignored) results
live under the same name. This table is the index; see Section 2 for the full
registry-name-to-description mapping.

| Family folder | Registry key(s) | `src/lmf/models/` | Configs | Docs | Tests |
| --- | --- | --- | --- | --- | --- |
| transformer | `transformer`, `mght` | `transformer/` | `transformer_baseline.yaml` | `docs/transformer/mght_variant.md` | `test_transformer.py` |
| gru | `gru_lm` | `gru/` | -- | -- | (benchmark-only) |
| opet | `opet` | `opet/` | `opet_baseline.yaml`, `ablations/opet_smoke.yaml` | -- | `test_opet.py` |
| rhca | `rhca` | `rhca/` | `rhca.yaml` | -- | `test_rhca_model.py`, `test_rhca_recall.py`, `test_carried_state.py`, `test_codebook.py`, `test_dynamics_rope.py` |
| gear_transformer | `gear_transformer`/`mlgt`, `gear_only`, `simplified_gear_transformer` | `gear_transformer/` | `gear_transformer.yaml`, `ablations/gear_transformer_*.yaml` | `docs/gear_transformer/` (+ `archive/` for superseded V2/retracted designs) | `test_gear_transformer.py` |
| pure_parallel_gear | `pure_parallel_gear` | `pure_parallel_gear/` | `pure_parallel_gear.yaml` | `docs/pure_parallel_gear/` (+ `archive/` for a dated review) | `test_pure_parallel_gear.py` |
| bounded_hybrid_gear | `pure_parallel_gear_v3`, `hybrid_parallel_gear`, `bounded_transformer`, `bounded_hybrid_gear_block_additive`, `bounded_hybrid_gear_block_selective_film`, `bounded_hybrid_gear_block_bank_router` | `bounded_hybrid_gear/` | `bounded_hybrid_gear*.yaml` | `docs/bounded_hybrid_gear/` | `test_bounded_hybrid_gear.py` |
| mecm | `mecm` | `mecm/` (+ shared scaffolding in `_shared/`) | `multigear_baseline_models.yaml`, `ablations/mecm_*.yaml` | `docs/multigear_baseline_models/three_architectures_comparison.md` | `test_mecm.py` |
| mcpm | `mcpm` | `mcpm/` (+ shared scaffolding in `_shared/`) | `multigear_baseline_models.yaml`, `ablations/mcpm_*.yaml` | `docs/multigear_baseline_models/three_architectures_comparison.md` | `test_mcpm.py` |
| mgcf | `mgcf` | `mgcf/` | `multigear_baseline_models.yaml` | `docs/multigear_baseline_models/three_architectures_comparison.md` (frontier follow-up section) | `test_mgcf.py` |
| mrwt | `mrwt` | `mrwt/` | `multigear_baseline_models.yaml`, `ablations/mrwt_*.yaml` | `docs/multigear_baseline_models/three_architectures_comparison.md` | `test_mrwt.py` |
| (cross-cutting) | -- | `_shared/` (infra, not registrable) | `multigear_generative_comparison.yaml` (mecm/mght/mgcf transformer comparison) | `docs/tokenizer/` (tokenizer-only docs) | `test_multigear_baseline_models.py` (registry + build-from-config checks) |

No model or folder is named generically "native" -- the four MultiGear
baseline architectures (mecm/mcpm/mgcf/mrwt) are independently named and
registered; only the cross-family scaffolding they share lives in the
leading-underscore `_shared/` infra namespace.

**Checkpoints and outputs** (both gitignored) follow the same per-family
convention as the table above:

- `checkpoints/<family>/*.pt` is the single canonical location for every
  trained checkpoint -- e.g. `checkpoints/mecm/`, `checkpoints/gear_transformer/`,
  `checkpoints/transformer/` (also holds `mght` checkpoints, since `mght` is a
  transformer variant, not a separate family). There is no second
  `outputs/checkpoints/` location.
- `outputs/<family>/` holds everything else family-specific: logs, JSON
  results, screening runs, prepared-data caches used only by that family's
  scripts.
- `outputs/tokenizer/` holds tokenizer-only artifacts shared across model
  families: `spt_bench/`, `multigear_prepared/`,
  `multigear_prediction_aware_prepared/`,
  `multigear_prediction_aware_matched_prepared/`, `sentencepiece_bpe_prepared/`.
- `outputs/rfk/` holds falsification-kernel reports, which are a cross-cutting
  framework diagnostic, not tied to one model family.

## 1. TL;DR / Current Recommendation

- **Tokenizer**: SentencePiece BPE remains the best default for generative
  modeling. MultiGear (this repo's hierarchical tokenizer) beats plain byte
  BPE and is competitive with SentencePiece when paired with
  merge-compositional initialization + hierarchical output, but its pure
  Python implementation is far slower to train/encode.
- **Model**: among MultiGear baseline architectures, **MGHT**
  (MultiGear Hierarchical Transformer) is the strongest pilot result so far,
  followed by **MGCF** (non-Transformer frontier baseline). Neither yet beats
  the matched Transformer + SentencePiece baseline.
- **Gear Transformer** (attention + gear residual side-channel) does not beat
  a parameter-matched plain Transformer. The unexpected result is that
  **removing attention entirely** ("gear-only") was the best performer on a
  repeated/easy corpus -- interesting but unconfirmed on harder data.
- **Three forward-looking architectures** (MECM, MCPM, MRWT) and one research
  proposal (MPJA) are designed but only partially implemented/validated; see
  Section 4 and Section 5.

## 2. Implemented Model Registry

| Registry name | Family | Description | Status |
| --- | --- | --- | --- |
| `transformer` | baseline | RMSNorm + RoPE + SwiGLU + SDPA, parameter-matched reference | validated baseline |
| `rhca` | RHCA | rolling-frontier model: bounded carried-state windows, factorized codebook, entropy-based block commits, SDPA exact-recall tail | primary resident family |
| `opet` | OPET | `transformer` baseline with phase-enriched token embeddings + coherence auxiliary loss | exploratory |
| `gear_transformer` (alias `mlgt`) | Gear Transformer | full Transformer trunk + write/update/cross-gear-coupling/read gear side-channel | exploratory, beaten by param-matched Transformer |
| `gear_only` | Gear Transformer | same gear mechanism with causal self-attention removed | exploratory, best on repeated-corpus smoke test |
| `pure_parallel_gear` | Pure Parallel Gear | attention-free LM, fast-weight associative memory, constant-size generation cache | canonical first generation |
| `pure_parallel_gear_v3` | Bounded Hybrid Gear | strict constant-state rotor model, no attention (ablation control within the family) | exploratory |
| `hybrid_parallel_gear` | Bounded Hybrid Gear | `pure_parallel_gear_v3` plus fixed-window grouped-query local attention | exploratory |
| `bounded_transformer` | Bounded Hybrid Gear | bounded local-attention trunk with no Gear memory (ablation control) | exploratory |
| `bounded_hybrid_gear_block_additive` | Bounded Hybrid Gear | bounded-attention trunk + block-rate Gear memory, additive fusion | exploratory |
| `bounded_hybrid_gear_block_selective_film` | Bounded Hybrid Gear | same trunk, token/channel-selective FiLM fusion | exploratory |
| `bounded_hybrid_gear_block_bank_router` | Bounded Hybrid Gear | same trunk, learned bank-slot router fusion | exploratory |
| `mght` | MultiGear baseline | Transformer trunk + MultiGear input-gear embedding + hierarchical (`bias`/`factorized`) gear-aware output | best MultiGear pilot so far |
| `mecm` | MultiGear baseline | non-Transformer causal long-convolution trunk + zero-gated mesh residual | runnable first-pass baseline |
| `mcpm` | MultiGear baseline | non-Transformer surface model + zero-gated deterministic execution-trace adapter | runnable first-pass baseline |
| `mgcf` | MultiGear baseline | non-Transformer, non-Mamba; routed dilated causal branches + learned long-filter memory + gear-aware output | strongest non-Transformer pilot |
| `mrwt` | MultiGear baseline | Transformer anchor + zero-gated causal atlas/workbench residual adapters + exact fallback | strongest MultiGear baseline pilot overall (still trails SentencePiece) |

## 3. Gear Transformer Findings (attention + gear residual)

V5 adds stacked parallel banks, causal context carriers, sparse adjacent/anchor
phase meshing, gated bank-to-bank carriers, direct future-token supervision,
and true multi-rate bank execution. The efficient profile uses five gears in
each of two banks; the fast bank updates every token and the slow bank every
fourth token. Full three-bank/nine-gear configurations remain supported.

**Fair 300 + 300 update development result** on the repeated narrative corpus:
both models first receive 300 Transformer updates, then either the Transformer
or the warm-started V5 model receives the same 300 additional updates.

| model | params | valid NLL | valid bits/token | continuation seconds |
| --- | ---: | ---: | ---: | ---: |
| continued Transformer | 261,792 | **0.5787** | **0.8350** | **2.23** |
| V5 stacked parallel gears | 303,113 | 0.8521 | 1.2293 | 6.54 |

V5 forward throughput is 40.7% of the small matched Transformer and measured
training throughput is 37.7%. Multi-rate execution and skipping diagnostic
objectives on alternating steps materially improve V4's cost, but V5 remains a
research architecture rather than a production replacement.

Every tested V5 mechanism had positive held-out ablation impact in the final
run: complete gears +0.00395 NLL, phase/rotation about +0.00278, temporal
context +0.00013, future prediction +0.01937, fast bank +0.00344, and slow bank
at +0.00055. Sparse phase and explicit inter-bank coupling were positive but very
small (about +0.00004 and +0.000005 NLL), so larger-corpus confirmation is
still required. Full results and generated predictions are in
`outputs/gear_transformer/gear_transformer_v5_acceptance_results.json`.

V4 replaces the earlier sequential V3 controller with parallel local, phrase,
semantic, and discourse gear trains. It uses positive-only phase velocity,
geometric rotation of latent memory pairs, a state-dependent affine rotational
scan, hierarchical lane fusion, routing floors, staged activation, and
gear-specific learning rates.

**Matched 500-update development result** on the repeated narrative corpus:

| model | params | valid bits/token | training seconds |
| --- | ---: | ---: | ---: |
| matched Transformer | 387,552 | **0.2996** | 7.5 |
| V3 gear model | 389,299 | 0.6450 | 110.7 |
| V4 parallel gears | 379,300 | 0.4512 | 63.8 |
| V4 + lane horizon supervision | 379,300 | 0.4537 | 72.2 |

V4 materially improves over V3 but still does not beat the matched Transformer.
Removing the complete V4 gear path increased held-out NLL by about 0.016;
removing phase/rotation increased it by about 0.004. These effects are small but
no longer diagnostic noise. Lane supervision improved slow-lane utilization
but did not improve 500-step held-out loss, so its default weight remains
conservative.

The gear path augments a standard Transformer block with a phase-conditioned
side channel: write gates, a causal per-gear summary, cross-gear message
passing, and read gates back into the token stream (V2 mechanism; V1 was a
plain phase-conditioned residual adapter). Diagnostics exposed during
training: `gear_write_activity`, `gear_read_activity`, `gear_coupling_entropy`,
`gear_coupling_gate`, `gear_coupling_offdiag`, `gear_conflict`.

**Apple-to-apple result (500 steps, repeated narrative corpus, SentencePiece
BPE vocab 1,037):**

| model | params | valid bits/token | valid bits/byte | train it/s |
| --- | ---: | ---: | ---: | ---: |
| Transformer dim=64,layers=2 (same trunk) | 173,184 | 2.4600 | 0.6889 | 204.28 |
| Gear Transformer V2 | 406,326 | 2.4355 | 0.6821 | 14.46 |
| Gear Transformer V2, no aux losses | 406,326 | 2.3710 | 0.6640 | 16.00 |
| **Transformer dim=80,layers=4 (param-matched)** | 401,120 | **1.2655** | **0.3544** | 41.68 |

**Conclusion**: the param-matched Transformer wins by a large margin. Gear V2
only looks good against an unfairly small same-trunk baseline. Auxiliary
losses hurt short-run performance; disabling them helps but doesn't close the
gap. Verdict: keep Gear Transformer as an experimental branch only, don't
treat it as better until it beats a parameter-matched Transformer.

**Efficiency tuning** (compact gear side-channel, `gear_dim` defaults to
`0.75 * dim` instead of full width):

| gear_dim | params | valid bits/token | valid bits/byte | it/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 (compact) | 241,237 | 2.5034 | 0.7248 | 29.27 |
| 48 (balanced) | 302,965 | 2.4057 | 0.6965 | 21.63 |
| 64 (full-width) | 377,749 | 2.2667 | 0.6563 | 16.45 |
| baseline Transformer | 173,184 | 2.4651 | 0.7138 | 204.28 |

`gear_dim=48` is the recommended default: beats the baseline's validation
loss while being cheaper and faster than full-width gears.

**Gear-only (attention removed) result** (500 steps, repeated corpus):

| model | attention | params | valid bits/token | valid bits/byte | it/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Transformer dim=64,layers=2 | yes | 173,184 | 2.4600 | 0.6889 | 204.28 |
| Transformer dim=80,layers=4 | yes | 401,120 | 1.2655 | 0.3544 | 41.68 |
| Gear Transformer V2 | yes | 406,326 | 2.4355 | 0.6821 | 14.46 |
| **Gear-only V2** | **no** | 373,494 | **0.9745** | **0.2729** | 9.07 |

Gear-only was the best predictive model on this deliberately repetitive
corpus -- notable, but the corpus may simply reward its causal
summaries/dilated updates/multi-rate state. Needs testing on non-repetitive
and long-range-dependency corpora before treating attention removal as a real
direction.

## 4. MultiGear Tokenizer Findings

MultiGear is a five-stage hierarchical BPE-style tokenizer: grapheme clusters
-> lexical spans -> shifted 2/4/8-lexical-span windows, each stage continuing
BPE merge training from the previous stage's vocabulary. An optional Viterbi
lattice-inference mode was tested and rejected as the default (merge-rank
inference performs better). Default max-token-byte cap is 16 (swept 8/12/16/
24/32; 16 was the local optimum).

**Intrinsic FLORES-200 result** (vocab 8,192, fixed 2-layer transformer, mean
over seeds 0-2, bits per raw UTF-8 byte, lower is better):

| Tokenizer | Fixed token updates | Fixed raw-byte exposure |
| --- | ---: | ---: |
| SentencePiece BPE | **2.4727** | **2.4797** |
| MultiGear (16-byte cap) | 2.4773 | 2.5264 |
| SPT | 2.4895 | 2.5404 |
| SentencePiece Unigram | 2.5345 | 2.5282 |
| Byte BPE | 2.5720 | 2.5720 |

MultiGear beats byte BPE, trails SentencePiece BPE by a small margin (within
seed noise at fixed token updates, more clearly worse at fixed raw-byte
exposure). Training cost: ~168s for MultiGear vs ~1.1s for SentencePiece BPE
(pure-Python implementation, not production-speed).

**Exact generative downstream result** (deterministic marked-span extraction,
631K-param model, 5 paired seeds):

| Tokenizer | Exact match | Edit similarity | Mean prompt tokens |
| --- | ---: | ---: | ---: |
| SentencePiece BPE | **19.55%** | **29.52%** | 22.11 |
| Byte BPE | 15.91% | 25.42% | 21.78 |
| MultiGear (16-byte cap) | 10.55% | 19.99% | **21.07** |
| SentencePiece Unigram | 6.82% | 16.41% | 22.71 |
| SPT | 5.27% | 14.81% | 25.51 |

MultiGear used the fewest prompt tokens but that compression didn't translate
into better exact generation -- motivated the embedding-initialization fix
below.

**Merge-compositional initialization fix**: initialize each MultiGear
embedding as `(embedding(left) + embedding(right)) / sqrt(2)` from its
merge-tree children instead of an independent random row. Ten paired seeds,
four-byte task:

| Integration | Exact match | Seed std dev |
| --- | ---: | ---: |
| MultiGear, independent rows | 11.32% | 8.76 |
| **MultiGear, merge-compositional rows** | **29.41%** | **6.72** |
| SentencePiece BPE, independent rows | 18.55% | 8.92 |
| Byte BPE, independent rows | 13.36% | 9.17 |

+18.09 exact-match points over independent rows (9/10 seeds won), +10.86 over
SentencePiece BPE. Confirmed again on harder eight-byte targets (+4.73
points). **Recommendation: always use `token_embedding_init: merge_compositional`
when training a new generative model with MultiGear.**

**Final 360-degree controlled comparison** (FLORES-200, 22 languages, 3 target
lengths, 270-model discovery run + 150-model independent confirmation, 3,000
updates/model, paired 95% CIs):

| Variant | Exact match | Edit similarity | Bits/target-byte |
| --- | ---: | ---: | ---: |
| SentencePiece BPE | 7.58% | 23.25% | 2.833 |
| Byte BPE | 6.05% | 22.23% | 2.864 |
| MultiGear flat | 5.02% | 15.96% | 3.157 |
| MultiGear compositional | 13.78% | 33.05% | 2.400 |
| **MultiGear hierarchical** | **21.80%** | **45.28%** | **1.936** |
| MultiGear hierarchical + auxiliary loss | 18.18% | 41.88% | 2.041 |
| MultiGear full stack (+ segmentation dropout) | 16.52% | 39.39% | 2.125 |

Independent confirmation: hierarchy-only MultiGear beats SentencePiece BPE by
+13.10 exact-match points (CI [9.27, 16.93]), winning all 10 confirmation
seeds and positive across all 22 languages. **Auxiliary hierarchy loss and
segmentation dropout add no confirmed gain and should stay disabled.**
**Recommended config: merge-compositional init + hierarchical output, nothing
else.** This conclusion is scoped to deterministic marked-span generation on a
~631K-parameter model -- not evidence MultiGear beats SentencePiece on
translation or open-ended generation, and scaling behavior is unknown.

**Runtime**: merge-rank encoding was rewritten to use a linked token list +
lazy occurrence heap, cutting FLORES-22 tokenizer training from 170.28s to
102.42s and raising encode throughput from 0.19 to 0.77 MB/s. SentencePiece
BPE still encodes at ~10 MB/s -- a native/Rust MultiGear implementation remains
necessary for production use.

**Hard overall conclusion**: MultiGear is a real, testable improvement over
plain byte BPE and is genuinely useful when its hierarchy is exposed to the
model (compositional init + hierarchical output), but it has not yet beaten
SentencePiece BPE as the default tokenizer, and its current implementation is
not production-speed.

## 5. MultiGear Baseline Architectures: MECM, MCPM, MRWT (+ MGCF)

Three architectures were designed with different objectives, replacing
earlier rejected designs (MBOC for MECM, MCPGM for MCPM, MHDT for MRWT -- all
rejected for forcing a fixed-capacity bottleneck or lacking an exact
fallback). All three require a stronger MultiGear interface than the
canonical hierarchy: every valid child-pair construction per token, merge
rank/construction gear/byte length per edge, and a compact acyclic interval
lattice. Construction gear (where a token was learned) must never be treated
as a semantic level.

### MECM -- MultiGear Elastic Causal Mesh
No Transformer or Mamba. Stores context as an append-only typed graph
(immutable byte leaves -> reversible MultiGear span views -> reasoning-workspace
nodes), processed by (1) a parallel causal long-convolution trunk, (2) a
sparse typed causal mesh with content-addressed retrieval, and (3) a dynamic
reasoning workspace for difficult tasks. Hierarchy, retrieval, and reasoning
branches are zero-gated residuals so the base causal model is always
recoverable. Best objective: fast generation + elastic reasoning without a
fixed-size compression bottleneck.

### MCPM -- MultiGear Constructive Program Machine
No Transformer or Mamba. Dual-plane: a fast causal **surface plane** for
ordinary generation, and an append-only copy-on-write **execution plane** that
proposes/runs/verifies typed programs (`READ_SPAN`, `BIND`, `APPLY`, `QUERY`,
`ASSERT`, `SPAWN`, `JOIN`, `CALL_TOOL`, `CALL_TYPED`, `CHECK`, `REVISE`,
`ROLLBACK`, `EMIT`, `HALT`) only when useful. Verifier cascade ordered by cost:
syntax/type/sandbox -> deterministic execution/tests -> formal certificates ->
learned critics (advisory only). Best objective: executable, falsifiable
reasoning with counterexample-guided repair; highest implementation risk.

### MRWT -- MultiGear Residual Workbench Transformer
Unrestricted hybrid -- the only one of the three that keeps a Transformer.
A qualified **anchor Transformer** defines the canonical distribution; a
**MultiGear span atlas** gives reversible multiscale memory/retrieval; an
**elastic reasoning workbench** (append-only work cells, read-compute-write
rounds) adds free-form compute for hard requests; a **MultiGear draft tree**
accelerates output via exact speculative decoding. All optional mechanisms
enter through zero-gated residuals so the anchor is always exactly
recoverable. Judged the **highest-probability path to a strong general
model**, since failure degrades gracefully to "just the anchor."

### MGCF -- MultiGear Fractal Causal Field (implemented frontier follow-up)
After MGHT showed Transformer-derived hierarchy works, MGCF was added as the
first concrete **non-Transformer, non-Mamba** baseline: routed dilated causal
convolution branches, learned causal long-filter memory, MultiGear input-gear
embeddings, MultiGear child-composition input residuals, byte-length
embeddings, gear-aware output (bias or factorized). See Section 6 for pilot results.

### Comparative summary

| Property | MECM | MCPM | MRWT |
| --- | --- | --- | --- |
| Transformer-free | Yes | Yes | No |
| Primary goal | fast generation + elastic reasoning | executable reasoning + fast surface bypass | anchor-quality general LM + elastic reasoning |
| Mandatory history bottleneck | none (growing ledger) | none (growing object/branch ledger) | none, but finite anchor context/KV remain |
| Distribution-preserving acceleration | lossless speculative decoding | surface path only | draft-tree speculative decoding only |
| Implementation risk | very high | extremely high | very high |
| Probability of near-term useful result | medium | medium-low | **highest** |
| Scientific novelty potential | high | high | medium |

**Recommended development order**: (0) shared MultiGear lattice API with
shuffled-hierarchy/SentencePiece-lattice controls -> (1) MECM base
long-convolution path validated before adding reasoning nodes -> (2) MRWT
anchor + atlas + draft tree validated before adding the workbench -> (3) MCPM
surface path, then typed execution on synthetic tasks with exact verifiers.

**Final recommendation**: if funding only one, choose **MRWT** (but gate the
workbench behind the anchor matching SentencePiece/byte controls). For a
genuinely new non-Transformer contribution, choose **MECM** (gate the dynamic
graph behind the base conv path beating baselines). For reliability-over-
perplexity, choose **MCPM** (highest risk, requires measured surface
fallback).

Every architecture above carries the same non-degradation pattern: an
immutable/zero-gated base path, validation gates before scaling, and explicit
kill criteria (e.g. stop MECM if real MultiGear hierarchy doesn't beat
shuffled/SentencePiece-lattice controls; stop MRWT if the anchor isn't
competitive; stop MCPM if it can't beat ordinary tool use + search at matched
budgets). See git history of the original docs if the full gate lists are
ever needed again -- they were omitted here as procedural detail rather than
findings.

## 6. MultiGear Predictive Junction Algebra (MPJA) -- unimplemented proposal

Status: **research design only, nothing implemented, no empirical claims.**
Proposed as a stronger alternative to an earlier "hierarchical rewrite-flow"
idea (rejected: rewrite trajectories make normalized likelihood and
comparison difficult, with no convergence guarantee).

Core idea: represent a text span by the *predictive information that must
cross its boundary* (a finite latent separator `Z_s`), not by what the span
"is". MultiGear supplies the exact bottom-level byte/lexical lattice; a
learned interval hierarchy above it groups spans by predictive dependence
(not by construction gear); a small bounded number of persistent threads `K`
crosses region boundaries for long-range dependencies (entities, quotation
state, topic) without blowing up treewidth.

Key falsifiable bound: a separator with `|Z|=chi` carries at most `logâ‚‚chi` bits,
so a cut carrying `m` bits of mutual information needs `chi >= 2áµ�`. Conditional
mutual information `I(A;B|Z)` is the correct diagnostic for whether a
separator is sufficient -- not token length or compression ratio.

Gated plan: Gate 0 (predictive-separator audit: do real MultiGear spans beat
shuffled-MultiGear / SentencePiece / random spans on separator difficulty?) ->
Gate 1 (fixed small exact circuit, verify normalization) -> Gate 2 (hierarchy/
tokenizer ablations against MGHT and a matched Transformer) -> Gate 3 (adaptive
separators, one capability at a time). **Do not implement the full generator
before Gate 0 passes.** Likely outcome if it works at all: a useful exact
lexical module inside another LM, not a complete replacement.

## 7. Generative Pilots -- Empirical Results

All pilots below: capped diverse `edu_combined` subset (200K source BPE
tokens/domain x 7 domains), `batch_size=2`, `seq_len=64`, one seed unless
noted. Bits/byte is the cross-tokenizer-comparable metric; bits/token is only
comparable within the same tokenizer.

### Baseline pilot (200 steps, 5 eval batches)

| model | tokenizer | params | bits/byte | tokens/sec |
| --- | --- | ---: | ---: | ---: |
| Transformer matched | SentencePiece BPE | 3.36M | **3.3772** | 56,151 |
| MECM | SentencePiece BPE | 3.46M | 3.4777 | 31,194 |
| MECM | MultiGear | 3.46M | 3.5510 | 31,904 |
| MECM, no draft aux | MultiGear | 3.46M | 3.5480 | 32,947 |

MECM ablation signal: skipping `reasoning_mesh.layers[0]` hurts (10.9004 vs
10.8729 bits/token); disabling draft auxiliary loss helps slightly; span
atlas / active cover removals are neutral at this resolution.

### MECM gear-aware output follow-up (200 steps)

| output head | bits/byte | tokens/sec |
| --- | ---: | ---: |
| flat token head | 3.5480 | 30,092 |
| gear bias | 3.5301 | 29,848 |
| **factorized gear (gear, then within-gear)** | **3.5181** | 20,477 |

Gear-aware emission helps MECM. Recommended fast setting: `gear_output_mode:
bias`; recommended quality setting: `gear_output_mode: factorized`. Extra
`gear_aux_weight` was neutral -- keep at 0.0. Still behind the 3.3772 SentencePiece
Transformer baseline.

### MGHT -- MultiGear Hierarchical Transformer (200 steps)

| model | tokenizer | bits/byte | tokens/sec |
| --- | --- | ---: | ---: |
| Transformer matched | SentencePiece BPE | 3.3772 | 52,502 |
| Transformer matched | MultiGear | 3.4942 | 53,963 |
| MGHT bias | MultiGear | 3.4852 | 44,865 |
| **MGHT factorized** | MultiGear | **3.4822** | 27,822 |

Best MultiGear model in this pilot, clearly ahead of MECM/MRWT. Recommended:
`mght` with `hierarchy_output_mode: bias` for speed, `factorized` only when
quality matters more than throughput. **Do not invest further in MECM as the
primary generative architecture until a new non-attention trunk beats MGHT.**

### MCPM and MRWT pilot (200 steps)

| model | tokenizer | params | bits/byte | tokens/sec |
| --- | --- | ---: | ---: | ---: |
| Transformer matched | SentencePiece BPE | 3.36M | **3.3772** | 54,685 |
| MCPM | MultiGear | 4.01M | 3.5480 | 19,717 |
| MCPM, no draft aux | MultiGear | 4.01M | 3.5450 | 20,252 |
| MCPM minimal (research stack removed) | MultiGear | 1.73M | 3.5750 | 73,170 |
| **MRWT** | MultiGear | 3.39M | **3.5121** | 30,436 |
| MRWT anchor-only | MultiGear | 1.93M | 3.5271 | 53,045 |

MRWT is the strongest MultiGear baseline architecture tested, still trailing
SentencePiece. Removing the full research stack (minimal MCPM / anchor-only
MRWT) is not quality-safe in either case. Recommendation: keep full MCPM with
`draft_aux_weight: 0.0`; keep full MRWT for MultiGear pilots.

### MGCF frontier pilot (200 steps, 20 eval batches)

| model | tokenizer | bits/byte | forward speed |
| --- | --- | ---: | ---: |
| Transformer matched | SentencePiece BPE | **3.2343** | n/a |
| MGHT factorized | MultiGear | 3.3878 | 29,720 tok/s |
| **MGCF bias v2** | MultiGear | 3.4098 | 40,579 tok/s |
| MGHT bias / MGCF factorized v2 | MultiGear | 3.4105 | ~35,960 / 26,769 tok/s |
| Transformer matched | MultiGear | 3.4156 | 54,833 tok/s |

Ranking on byte-normalized loss: SentencePiece Transformer > MGHT factorized
> MGCF bias v2 > {MGHT bias, MGCF factorized v2} > MultiGear Transformer.
MGCF bias is the best MGCF setting (best quality + speed advantage over MGHT
bias). Remaining gap: MGCF lacks content-addressed sparse retrieval over
previous spans (global associative recall) -- next feature to test, before
adding a reasoning workspace that would hide this trunk limitation.

### MGCF scaled to 1,000 steps (500 eval batches)

| model | tokenizer | bits/byte | bytes/token | speed |
| --- | --- | ---: | ---: | ---: |
| **MGCF bias** | MultiGear | **3.1670** | 3.1755 | 39,072 tok/s |
| Transformer matched | SentencePiece BPE | 3.2169 | 2.9261 | 47,376 tok/s |
| MGCF | SentencePiece BPE | 3.3889 | 2.9261 | **77,842 tok/s** |

At 1,000 steps MGCF + MultiGear takes the byte-normalized lead -- but
validation token windows differ in byte coverage by tokenizer, so this isn't
a fully byte-aligned protocol yet, and generation quality is still poor.
Promising, not proof.

### MGCF + SentencePiece control (200 steps)

Confirms MGCF's poor MultiGear result isn't a trunk failure: with
SentencePiece the same trunk (3.2437 bits/byte) is close to the Transformer
baseline (3.2343) and much faster (78,166 vs 55,981 tok/s). The issue is the
MultiGear token distribution / hierarchy interface / training objective
interaction, not the trunk.

### MGCF tokenizer diagnostics + prediction-aware tokenizer

Full-validation check (same decoded text, 425,397 bytes, both tokenizers):

| model | tokenizer | bits/token | bits/text-byte |
| --- | --- | ---: | ---: |
| MGCF bias 1000 | MultiGear prediction-aware | 10.0962 | **3.1262** |
| MGCF bias 1000 | MultiGear (plain BPE inference) | 10.0771 | 3.1695 |
| MGCF 1000 | SentencePiece BPE | 9.9173 | 3.3990 |
| **Transformer 1000** | SentencePiece BPE | **9.4151** | 3.2269 |

MultiGear's higher bits/token is not a failure by itself: it uses 8.6% more
bytes/token than SentencePiece for only 1.6% more bits/token, so
byte-normalized loss still improves (3.3990 -> 3.1695). Breakdown by gear shows
gear 0 (byte/short-token fallback) is the main inefficient region (7.882
bits/byte) while higher gears are efficient (2.155-2.460 bits/byte); 1-byte
tokens cost 8.062 bits/byte vs 1.248 for 10-byte tokens. Conclusion: the
segmentation is **compression-driven, not prediction-driven** -- don't force
lower bits/token (would shrink bytes/token and erase the gain). Implemented
`MultiGearPredictionAwareTokenizer` (Viterbi cost = frequency + gear + byte
length + rare-token penalty + byte-length reward, with a hook to plug in
measured NLL later) improved matched-text MGCF from 3.1695 to **3.1262**
bits/text-byte.

## 8. Open Questions / Next Steps

1. Byte-aligned evaluation: build a strict fixed-byte-span eval protocol so
   MultiGear-vs-SentencePiece comparisons stop depending on token-window
   sampling artifacts (flagged in both the 1,000-step MGCF pilot and the
   tokenizer diagnostics).
2. Add content-addressed sparse retrieval to MGCF before adding any reasoning
   workspace on top of it.
3. Move MultiGear vocabulary construction and encoding to a native/Rust
   implementation -- it is the main practical blocker to using MultiGear at
   all, independent of the quality results above.
4. Confirm gear-only (attention-removed) Gear Transformer result on
   non-repetitive and long-range-dependency corpora before treating attention
   removal as a real direction.
5. Run MPJA Gate 0 (predictive-separator audit) before writing any more of
   the MPJA generator.
6. Phase 1/2/3 of the MECM/MRWT/MCPM development order (Section 5) are still mostly
   unstarted beyond the runnable first-pass baselines in
   `src/lmf/models/mecm/`, `mcpm/`, `mrwt/`, and `mgcf/`.

## 9. Where the Raw Data Lives

- `outputs/tokenizer/spt_bench/*.json` -- tokenizer benchmark artifacts (FLORES intrinsic,
  generation-360, compositional-init, runtime).
- `results/ablations/*/summary.md` -> deleted; regenerate via `lmf ablate`
  using the configs in `configs/ablations/`.
- `results/multigear_generative_comparison/*.json` -- MGCF tokenizer diagnostics
  and prediction-aware tokenizer results.
- `scripts/benchmark_*.py` -- reproduction scripts for every tokenizer
  benchmark in Section 4.
- `configs/multigear_recommended.yaml` -- the evidence-backed MultiGear model
  integration (merge-compositional init + hierarchical output only).
- `configs/multigear_enhanced.yaml` -- full opt-in feature stack, kept only to
  reproduce the full-stack ablation.
- `configs/multigear_generative_comparison.yaml` and `configs/multigear_baseline_models.yaml`
 -- blocks for every pilot in Section 7.
