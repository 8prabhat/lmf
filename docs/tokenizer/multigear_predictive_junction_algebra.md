# MultiGear Predictive Junction Algebra

Date: 2026-06-15

Status: unimplemented research design. None of the quality or efficiency claims
below have been empirically established.

## Decision

The previously proposed hierarchical rewrite-flow model should not be the
primary next architecture.

It has three structural problems:

1. A MultiGear merge is evidence of reusable byte compression, not evidence of
   a semantic constituent.
2. A free rewrite process creates many trajectories for the same byte string,
   making normalized likelihood and reliable comparison difficult.
3. Iterative refinement has no strong guarantee that it will converge quickly
   or preserve a calibrated language distribution.

The stronger proposal is **MultiGear Predictive Junction Algebra (MPJA)**: a
normalized hierarchical probabilistic circuit in which a text span is
represented by the predictive information that must cross its boundaries.

MultiGear supplies the exact bottom-level lattice of byte and lexical span
constructions. Learned higher levels are selected by predictive dependence, not
by construction gear.

## Core Idea

An ordinary token embedding describes what a token is. MPJA instead represents
a span by what an outside context needs to know about that span.

For a span `s`, define a finite latent separator `Z_s`. The interior bytes
`X_s` may contain much more information than `Z_s`; `Z_s` carries only the
information needed to coordinate `X_s` with bytes outside the span.

Two neighboring child spans are joined by summing over compatible separator
states. This is a junction-tree or tensor-contraction operation:

```text
left span operator  *  separator factor  *  right span operator
                           |
                           v
                    parent span operator
```

A MultiGear token is therefore not an independent class. It is a cached,
factorized span operator built from all valid constructions in the MultiGear
merge DAG.

## Three-Level Architecture

### 1. Exact byte and MultiGear lattice

The leaves are raw bytes, preserving exact round-trip behavior.

For spans up to `max_token_bytes`, the model uses the complete MultiGear merge
DAG rather than one canonical child pair. Sum nodes marginalize alternative
segmentations and merge constructions. Product nodes combine disjoint child
spans.

This bottom circuit answers:

- which MultiGear constructions explain these bytes;
- how much probability mass each construction receives;
- which construction is cheapest to use during conditional sampling.

### 2. Learned predictive junction hierarchy

MultiGear construction stops at short lexical spans. Above that level, a
learned balanced interval hierarchy groups spans according to predictive
dependence.

Each internal region has:

- a split or route variable;
- one or more separator variables;
- a normalized conditional table or low-rank neural tensor connecting parent
  and child separator states;
- an adaptive separator width.

The hierarchy is trained to place narrow separators where the left and right
regions are close to conditionally independent. Construction gear is only an
input feature; it does not decide model scale.

### 3. Bounded persistent threads

A pure tree forces every long-range dependency through a common ancestor.
Natural language contains sparse persistent dependencies such as entities,
quotation state, indentation, topic, and agreement.

MPJA therefore permits a small bounded number `K` of persistent thread
variables to cross a region boundary. `K` is fixed to preserve bounded
treewidth. The model must explicitly pay computation for every active thread.

This is the risky component. If useful language dependencies require large
`K`, tractable exact inference will not survive.

## Normalized Probability Model

Use an explicit length model `p(N)` and an exact conditional circuit
`p(X_1:N | N)`.

For a fixed hierarchy `T`, an internal region `s` with child regions `l` and
`r` has the normalized form:

```text
p_s(X_s | Z_s)
  = sum over q, Z_l, Z_r, U:
      p(q, Z_l, Z_r, U | Z_s)
      p_l(X_l | Z_l, U)
      p_r(X_r | Z_r, U)
```

where:

- `q` selects a valid split or MultiGear construction;
- `U` is a bounded separator or persistent-thread state;
- every conditional table is normalized;
- child scopes are disjoint.

Alternative valid trees may also be marginalized:

```text
p(X | N) = sum_T p(T | N) p(X | T, N)
```

This is intentional latent-variable marginalization, not accidental duplicate
probability. The model is normalized as long as route distributions are
normalized and the circuit remains smooth and decomposable.

For the first implementation, use a fixed upper hierarchy and marginalize only
the bounded MultiGear lattice. Adaptive upper trees should be added only after
the fixed model passes the quality gate.

## Generation

Generation is conditional inference followed by ancestral sampling:

1. Observe prompt bytes as evidence.
2. Run an upward sum-product pass to compute compatible separator beliefs.
3. Run a downward pass to sample unresolved route and separator states.
4. Sample independent child regions in parallel when the circuit says they are
   conditionally independent.
5. Resolve the bottom MultiGear lattice to exact bytes.

For a balanced hierarchy, the ideal parallel depth is `O(log N)`. Total work is
still at least linear in generated bytes.

This model naturally supports infilling and bidirectional constraints. It does
not require left-to-right recurrence, attention, a KV cache, diffusion steps,
or Mamba-style selective state updates.

## Theoretical Effectiveness Map

The architecture is useful only if language has low-dimensional predictive
separators under a learnable interval hierarchy. This can be tested before
building the full model.

### Separator capacity

Suppose regions `A` and `B` communicate only through a categorical separator
`Z` with `|Z| = chi`. Then:

```text
I(A; B) <= H(Z) <= log2(chi)
```

Therefore a cut carrying `m` bits of mutual information requires at least:

```text
chi >= 2^m
```

This is a hard representational lower bound, not an optimization argument.

### Factorization error

For a selected separator `Z`, the KL error created by imposing conditional
independence between two regions is:

```text
KL(
  p(A, B, Z)
  ||
  p(Z) p(A | Z) p(B | Z)
) = I(A; B | Z)
```

Conditional mutual information is therefore the correct diagnostic for whether
a proposed separator is sufficient. It is more relevant than token length,
gear number, or compression ratio.

The errors of arbitrary hierarchy cuts are not automatically additive. Measure
both per-cut conditional mutual information and final exact NLL.

### Computation

For dense separator tensors with common width `chi`, a binary contraction has
approximately `O(chi^3)` work. A balanced `N`-leaf hierarchy has:

```text
work           O(N * chi^3)
parallel depth O(log N)
memory         O(N * chi^2) during training/inference
```

Low-rank factors can reduce the constants and exponent, but they also reduce
capacity. Adaptive rank should optimize the measured quality-cost objective:

```text
J = bits_per_byte
    + lambda_work * measured_work
    + lambda_memory * peak_memory
    + lambda_depth * serial_depth
```

### MultiGear-specific value

MultiGear is beneficial only if its candidate spans reduce separator difficulty
per unit of compute. Define a diagnostic utility for a candidate span `s`:

```text
utility(s)
  = bytes_collapsed(s)
    / (estimated_separator_cost(s) + epsilon)
```

The separator cost must be estimated from predictive information or required
rank, not from construction gear.

The decisive tokenizer comparison is whether real MultiGear spans have better
utility than:

- SentencePiece BPE spans;
- ordinary byte-BPE spans;
- length-matched random spans;
- shuffled MultiGear merges.

If they do not, MPJA may still be a useful architecture, but MultiGear is not
the reason.

## Why This Improves on the Rewrite-Flow Proposal

| Property | Rewrite flow | MPJA |
| --- | --- | --- |
| Probability normalization | Difficult | Explicit probabilistic circuit |
| Alternative tokenizations | Multiple trajectories | Exact latent marginal |
| Convergence | Iterative and uncertain | Finite upward/downward passes |
| Role of MultiGear hierarchy | Assumed generation structure | Bottom candidate lattice |
| Higher semantic structure | Inherited from merges | Learned by predictive dependence |
| Capacity diagnosis | Mostly empirical | Separator MI/rank bounds |
| Conditional infilling | Natural but approximate | Exact within the circuit |
| Primary failure signal | Slow or unstable refinement | Required separator rank explodes |

## Required MultiGear Changes

The current tokenizer metadata exposes only one immediate child pair per token.
MPJA requires:

1. every valid child-pair construction for each token;
2. merge rank and construction gear for every edge;
3. byte interval lengths and lexical-boundary features;
4. a compact acyclic lattice representation;
5. optional model-aware merge utility statistics.

The tokenizer should eventually allocate vocabulary using a combination of
frequency and predictive separator utility. Frequency-only merging may spend
capacity on strings that are common but do not simplify language modeling.

## Falsifiable Development Plan

### Gate 0: predictive-separator audit

Do not implement the full generator first.

Train small probes that estimate cross-boundary predictive information or
required separator rank for MultiGear, SentencePiece BPE, byte BPE, shuffled
MultiGear, and random length-matched spans.

Stop if MultiGear does not consistently reduce separator difficulty at matched
span length and frequency.

### Gate 1: exact fixed-length circuit

Build a fixed balanced upper hierarchy for 32-64 byte sequences. Use:

- exact byte leaves;
- full MultiGear lattice only in the bottom layers;
- no adaptive tree;
- no persistent threads;
- small separator widths that permit exhaustive normalization checks.

Verify on tiny alphabets that probabilities sum to one. Then compare held-out
bits per byte, exact marked-span generation, calibration, sampling speed, and
memory.

### Gate 2: hierarchy and tokenizer ablations

Compare at matched parameters, training bytes, optimizer updates, wall-clock
budget, and hardware:

1. MPJA with real MultiGear lattice;
2. MPJA with SentencePiece lattice;
3. MPJA with shuffled MultiGear lattice;
4. MPJA with byte leaves only;
5. current hierarchical MultiGear transformer;
6. matched modern transformer baseline.

The real MultiGear lattice must beat the shuffled and SentencePiece lattices to
claim a MultiGear-specific benefit.

### Gate 3: adaptive separators

Only after Gate 2 succeeds:

- allocate separator width from estimated predictive dependence;
- learn upper-region split choices;
- add bounded persistent threads one at a time;
- test whether every added capability improves the quality-cost frontier.

## Expected Outcomes

### Best case

Language dependencies become low-rank after good span selection. MPJA obtains
competitive bits per byte, exact conditional inference, parallel hierarchical
sampling, and a direct way to spend computation only at difficult boundaries.

### Likely case

The bottom MultiGear lattice improves lexical modeling, but separator rank must
grow rapidly above short spans. MPJA becomes useful as a small exact lexical
module inside another language model, not as the complete model.

### Failure case

Cross-boundary information remains high under every tested hierarchy.
Contraction cost grows faster than the saved sequence work, and probabilistic
circuit expressivity trails autoregressive models. In that case, stop the
architecture and retain MultiGear's currently validated hierarchical output.

## Novelty Boundary

The ingredients are not individually new:

- probabilistic circuits provide exact tractable inference under structural
  constraints;
- tensor networks use bounded separator or bond dimensions;
- MERA-like models use multiscale hierarchy and boundary disentangling;
- lattice language models marginalize tokenization;
- MultiGear supplies a staged merge hierarchy.

The potentially original synthesis is:

> Use the complete MultiGear merge DAG as the exact lexical region graph of a
> structured-decomposable language circuit, then learn higher interval
> partitions and adaptive separator ranks by predictive conditional mutual
> information, with bounded persistent dependency threads.

This is a research hypothesis, not a defensible novelty or patent claim without
a formal literature and patent search.

