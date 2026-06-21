# Three MultiGear-Native Model Architectures

Date: 2026-06-16

Status: research recommendations plus runnable baselines. MECM, MCPM, MRWT,
and the later MGCF frontier baseline now have code paths in `src/lmf/models`.
The pilot results so far are early evidence, not final validation.

## Executive Decision

Recommend three architectures with different objectives:

1. **MultiGear Elastic Causal Mesh (MECM)**  
   No Transformer or Mamba. Best non-Transformer design for combined
   generation and adaptive reasoning without a mandatory compression
   bottleneck.

2. **MultiGear Constructive Program Machine (MCPM)**  
   No Transformer or Mamba. Best high-risk design for executable reasoning,
   adaptive parallel search, counterexample-guided repair, and verified answer
   generation.

3. **MultiGear Residual Workbench Transformer (MRWT)**  
   Unrestricted hybrid. Best probability of producing a strong general
   generative and reasoning language model while retaining an exact qualified
   Transformer fallback.

These should not be presented as three interchangeable backbones. They make
different assumptions and should be evaluated against different primary goals.

## Implemented Frontier Follow-Up: MGCF

After the first MGHT result, a Transformer-derived hierarchy model was no
longer sufficient for the stated research direction. The repo now includes
`mgcf`, **MultiGear Fractal Causal Field**, as the first concrete
non-Transformer, non-Mamba frontier baseline.

MGCF is intentionally narrower than the full MECM ledger proposal:

- routed dilated causal convolution branches;
- learned causal long-filter memory views;
- learned MultiGear input-gear embeddings;
- MultiGear child-composition input residuals;
- byte-length embeddings;
- gear-aware output in bias or factorized mode;
- no dense self-attention;
- no selective state-space layer.

This gives a testable trunk improvement path without prematurely adding dynamic
retrieval, graph editing, or reasoning workspace machinery.

Early 200-step result: MGCF bias now beats the plain MultiGear Transformer and
MGHT bias on the 20-batch pilot, but it does not yet beat MGHT factorized or
the SentencePiece Transformer baseline. See
`results/generative_mecm_iteration/mgcf_frontier_pilot200_summary.md`.

## Shared MultiGear Foundation

All three architectures require a stronger MultiGear interface than the current
canonical hierarchy:

1. Expose every valid child-pair construction for every token, not only the
   first pair.
2. Preserve merge rank, construction gear, byte length, lexical-span count, and
   boundary features for every edge.
3. Represent all candidate spans over an encoded byte sequence as an acyclic
   interval lattice.
4. Preserve exact byte provenance through every model operation.
5. Distinguish:
   - construction gear: where the token was learned;
   - compute gear: how much model computation the current context allocates;
   - reasoning level: the role of an object inside a reasoning graph.
6. Treat every parent span as a reversible view over its children and bytes,
   never as the only surviving representation.

Construction gear is not a reliable semantic level and must never be treated as
one.

---

## Architecture 1: MultiGear Elastic Causal Mesh

### Objective

Build one non-Transformer, non-Mamba architecture that supports fast language
model training, fast lossless-accelerated generation, and adaptive reasoning
without forcing all history through a fixed-size recurrent state, separator, or
hierarchical summary.

MECM replaces the earlier MultiGear Boundary Operator Circuit. MBOC cannot meet
these requirements because its tractability depends on the fixed separator rank
that creates its information bottleneck.

### Why MBOC Is Rejected

MBOC has useful mathematical properties, but its central trade-off is wrong for
a general reasoning and generation model:

1. **The separator is a hard bottleneck.** A separator with `chi` states carries
   at most `log2(chi)` bits. Removing this restriction removes MBOC's tractable
   exact-inference advantage.
2. **Capacity becomes expensive too quickly.** Dense contractions cost roughly
   `O(chi^3)`, so increasing capacity to avoid quality loss directly conflicts
   with fast training and prediction.
3. **Intervals do not match reasoning structure.** Logical dependencies can
   connect arbitrary distant evidence, intermediate variables, tools, and
   subgoals.
4. **Exact normalization does not imply strong reasoning.** MBOC can infer over
   a supplied factorization but cannot naturally invent or revise a reasoning
   procedure.
5. **Parallel generation requires conditional independence.** That assumption
   often fails for exact wording, code, mathematics, and long coherent text.
6. **Open-ended variable-length generation is awkward.**
7. **Its likely quality ceiling is below unrestricted neural sequence models.**

These are architectural conflicts, not implementation defects. MBOC should not
remain architecture 1.

### Core Representation: Lossless Causal Ledger

MECM stores context as an append-only typed graph. It never deletes the exact
source representation:

```text
immutable byte leaves
    ↕ reversible parent-child edges
overlapping MultiGear span views
    ↕ causal / retrieval / provenance edges
generated spans and reasoning-workspace nodes
```

Node types include:

- exact byte or canonical-token leaves;
- MultiGear candidate span views;
- emitted output spans;
- entities, variables, facts, hypotheses, subgoals, and tool results;
- optional summary nodes that can always be bypassed.

MultiGear parents are pointers and learned computational shortcuts over their
children. They never replace the underlying leaves. If a parent representation
is insufficient, the model can refine it to children or retrieve the exact byte
nodes.

The ledger grows with the problem and therefore avoids one fixed-capacity
history state. Practical memory and retrieval limits still create resource
constraints; no finite architecture is literally bottleneck-free.

Use a two-tier ledger:

- keep the active cover, active reasoning graph, and retrieval index on the
  accelerator;
- keep immutable bytes, interval metadata, and cold nodes in cheaper storage;
- reload or deterministically recompute cold representations when retrieved.

This preserves information without requiring every historical embedding to
remain resident in expensive memory.

### Causal Probability Semantics

MECM remains an ordinary normalized causal language model over canonical
MultiGear output events `y_t`, with byte fallback:

```text
p(y_1:T) = product_t p(y_t | y_<t)
```

Every `y_t` owns an exact immutable byte interval, and decoding those intervals
defines the output bytes. Additional MultiGear views, retrieval links, and
reasoning nodes only provide features for predicting the next canonical event;
they never create additional target events or silently change the probability
factorization.

All graph construction used for predicting position `t` must be a deterministic
or sampled function of `y_<t` only:

- retrieval indices contain only completed prefix nodes;
- an edge may not expose a source whose byte interval ends after the target
  position;
- reasoning and tool-result nodes become available only after they are created;
- parallel teacher-forced training must enforce these causal edge rules.

This rule is necessary for valid likelihood measurement and exact speculative
verification. Violating it would create future leakage and invalidate every
quality claim.

### Elastic Active Cover

The model does not process every stored node equally. It maintains an active
execution cover of the current context:

```text
easy region       → one wide MultiGear span
uncertain region  → several children
exact/reasoning region → lexical spans or raw bytes
```

The execution cover is a non-overlapping ordered partition, which gives the
causal long-convolution trunk one unambiguous sequence. Additional candidate
span views may overlap in the side mesh, but they cannot independently count
the same bytes as multiple sequential inputs.

The canonical MultiGear encoding is the base execution cover. The elastic
router is optional; disabling it restores exactly that canonical cover. This is
required for the zero-gate equivalence and fallback claims.

Refinement decisions use composition error, predictive entropy, retrieval
disagreement, task type, and verifier failures. Construction gear is only a
feature.

This makes hierarchy an optional compute accelerator rather than an information
bottleneck.

### Computation

MECM combines three non-Transformer, non-Mamba mechanisms.
It contains no dense self-attention block and no selective state-space layer.
Content-addressed retrieval is an explicit sparse index lookup followed by
typed edge processing, not a fully connected token mixer.

#### 1. Parallel causal long-convolution trunk

A gated causal long-convolution network processes the active sequential cover.
This supplies broad language-model context with parallel teacher-forced
training. Hyena-like implicit filters are an initial implementation candidate,
not a required final form.

#### 2. Sparse typed causal mesh

Every active node receives a small set of typed edges:

- recent sequential neighbors;
- MultiGear parent-child and sibling edges;
- content-addressed retrieval edges into the immutable ledger;
- explicit reasoning and provenance edges.

Edge-gated residual message passing runs only over this sparse mesh. Retrieval
fan-out and graph-update depth are elastic. Difficult regions may widen toward
the full ledger; easy regions stay sparse.

Sparse message passing can itself over-squash long-range information. MECM
mitigates rather than eliminates that risk through direct retrieval edges,
MultiGear skip paths, reversible refinement to leaves, and adaptive fan-out.

#### 3. Dynamic reasoning workspace

For difficult tasks, the model appends free-form typed scratch nodes and edges:

```text
CREATE_VARIABLE
LINK_EVIDENCE
INTRODUCE_SUBGOAL
DERIVE
CALL_TOOL
CHECK
REVISE
STOP
```

Only the active reasoning subgraph receives repeated updates. Ordinary language
modeling does not pay this recurrent reasoning cost.

### Training

Training is divided so that general language pretraining remains parallel:

1. **Fast causal pretraining:** construct the ledger and candidate MultiGear
   views from teacher-forced text; train the long-convolution trunk and sparse
   causal mesh at all positions in parallel.
2. **Elastic-cover training:** train refine/coarsen decisions using next-output
   loss, composition error, and a measured compute penalty.
3. **Retrieval training:** learn causal retrieval against next-span influence,
   evidence links, and hard negatives. Always retain local and hierarchy edges
   so early retriever errors cannot disconnect the sequence.
4. **Reasoning training:** train graph-edit actions on executable traces,
   verifier feedback, and outcome-based search.
5. **Hierarchical drafting:** train MultiGear parents and child sequences as
   draft proposals while the causal target distribution remains the primary
   objective.

Use zero-initialized residual gates for hierarchy, retrieval, and reasoning
branches. At initialization, the architecture behaves like its causal
long-convolution base rather than injecting untrained graph messages.

```text
h_target =
    h_base
  + alpha_hierarchy * h_hierarchy
  + alpha_retrieval * h_retrieval
  + alpha_reasoning * h_reasoning
```

With all optional `alpha` values at zero, MECM produces exactly the base-path
logits. This embeds the base model inside the larger hypothesis class.

Learned retrieval should be introduced only after the base and deterministic
MultiGear/local-edge model is stable. Parallel training can use prefix-safe
retrieval edges computed in a separate indexing pass; a naive global nearest
neighbor search over the full training sequence would leak future information.

### Generation and Prediction Speed

The target model remains causal. MultiGear hierarchy accelerates decoding
through lossless speculative generation:

1. Draft a wide MultiGear token, its child sequence, or several future spans.
2. Evaluate the proposed block under MECM's causal target distribution in
   parallel.
3. Use exact speculative acceptance/rejection.
4. Commit the accepted prefix; refine or fall back to one token/byte after a
   rejection.

Correct lossless speculative decoding preserves the target distribution
exactly. Drafting can improve speed but cannot change target-model quality.

Reasoning nodes are created only when uncertainty, task control, or a verifier
requests additional work.

### Theoretical Effectiveness and Cost

Let:

- `B` be raw byte count;
- `s` be mean bytes per active span;
- `A ≈ B / s` be active sequential nodes;
- `k` be mean sparse-mesh fan-out;
- `L` be graph-update depth;
- `d` be hidden width;
- `r` be message projection rank;
- `q` be appended reasoning nodes.

An implementation target is:

```text
parallel causal mixing:  approximately O(A log A * d)
sparse mesh updates:     O(L * k * (A + q) * d * r)
ledger storage:          O(B + (A + q) * d)
retrieval index/query:   implementation-dependent, typically sublinear search
```

These are design targets, not guaranteed realized runtimes. Sparse graph
kernels, retrieval, ledger growth, and block verification can dominate on real
hardware.

Unlike MBOC, there is no fixed `chi` separator through which every dependency
must pass. However, practical `k`, `L`, `d`, active-cover, and memory limits are
still bottlenecks. MECM's claim is **elastic fallback**, not infinite capacity.
Each node still has finite width `d`; the difference is that the model may
preserve information across additional nodes and direct edges instead of being
forced to collapse it into one state.

### Reasoning and Speed Conditions

MECM can use a dependency during one graph-update phase only when that
dependency is connected to the active target within the update depth:

```text
shortest_mesh_path(required_evidence, target) <= L
```

Retrieval and explicit reasoning edges are therefore essential rather than
optional conveniences. If a task requires `m` evidence items and item `i` is
retrieved with probability `r_i`, the union bound gives:

```text
P(all required evidence retrieved) >= 1 - sum_i (1 - r_i)
```

This lower bound can become useless quickly when many facts are needed.
Reasoning quality therefore depends on measured multi-evidence retrieval recall,
not only single-edge recall.

Free-form reasoning does not remove sequential complexity. A reasoning
procedure with `h` inherently dependent steps still requires at least `h`
ordered graph edits or update phases unless the procedure itself can be
parallelized.

Speculative generation improves latency only when:

```text
draft_cost + block_verification_cost + rejection_overhead
    < accepted_tokens * target_single_step_cost
```

Likewise, fast training requires the reduction from `B` bytes to `A` active
nodes and sparse `k`-edge updates to exceed routing, indexing, and graph-kernel
overhead. These conditions must be measured; asymptotic notation alone is
insufficient.

### Non-Degradation Contract

No architecture can guarantee equal or better benchmark quality after
optimization. MECM can provide narrower, testable safeguards:

1. **Representational safeguard:** exact leaves and causal edges remain
   reachable; macro spans never irreversibly replace them.
2. **Hypothesis-class safeguard:** optional hierarchy, retrieval, and reasoning
   branches use residual gates and may be set to zero, so the base causal model
   remains representable.
3. **Decoding safeguard:** exact speculative verification preserves the target
   output distribution.
4. **Runtime safeguard:** when routing confidence is low or acceleration is
   slower, bypass hierarchy and use the base causal path.
5. **Training safeguard:** reject any enhancement that worsens the
   quality-versus-compute Pareto frontier in controlled evaluation.

These safeguards prevent forced degradation; they do not prove that training
will discover a superior model. They apply relative to MECM's causal
long-convolution base, not relative to an arbitrary stronger Transformer or
future architecture.

### Expected Strengths

- No mandatory fixed separator, recurrent state, or irreversible summary.
- Parallel general-language training.
- Fast typical-case sparse inference with dense/refined fallback.
- One architecture supports generation, retrieval, and adaptive reasoning.
- Lossless speculative decoding can accelerate output without changing quality.
- Exact byte provenance supports code, multilingual text, tools, and auditing.
- MultiGear hierarchy directly controls compute and draft granularity.

### Expected Weaknesses

- Append-only memory grows with context unless explicitly archived.
- Recomputing or loading cold nodes can introduce unpredictable latency.
- Sparse retrieval may miss crucial evidence.
- GNN-style updates can over-squash, over-smooth, or be inefficient on GPUs.
- Long convolutions may still trail strong attention models on associative
  recall and unrestricted reasoning.
- Dynamic graph actions complicate batching and optimization.
- Reasoning-workspace training requires expensive traces or verifiers.
- The full architecture is more complex than its individual components.

### Kill Criteria

Stop or simplify MECM if any of the following holds under matched parameters,
training bytes, hardware, and wall-clock budget:

- the base long-convolution path materially trails a matched strong baseline;
- real MultiGear hierarchy does not beat shuffled hierarchy and
  SentencePiece-lattice controls;
- graph/retrieval branches fail to improve quality at matched inference cost;
- reasoning workspace fails to improve size generalization at matched
  test-time compute;
- speculative acceptance is too low to reduce measured end-to-end latency;
- active-cover refinement frequently falls back to leaves, eliminating the
  intended efficiency gain.

### Required Validation Gates

1. **Causal correctness:** changing any future byte must not change logits for
   an earlier position. Test every edge builder and router for leakage.
2. **Zero-gate equivalence:** with optional branch gates at zero, logits and
   gradients through the base path must match the standalone base model.
3. **Base-path competitiveness:** the causal long-convolution model must pass
   matched-parameter, matched-training-byte, and matched-wall-clock comparisons
   before graph complexity is added.
4. **Hierarchy attribution:** real MultiGear views must beat shuffled hierarchy,
   SentencePiece spans, and length-matched random spans.
5. **Retrieval recall:** measure whether required evidence is retrieved before
   claiming long-context or reasoning gains.
6. **Lossless decoding:** speculative output must be statistically equivalent
   to target-only decoding and must reduce measured end-to-end latency.
7. **Reasoning scaling:** evaluate accuracy as problem size and allowed
   reasoning nodes increase; compare against equal-compute scratchpad and search
   baselines.

### Novelty Boundary

Long convolutions, dynamic graph memory, message passing, scratchpads,
hierarchical tokenization, residual gating, and lossless speculative decoding
already exist.

The potentially novel contribution is:

> Maintain MultiGear spans as reversible overlapping views in an append-only
> exact-byte causal ledger; dynamically select an elastic active cover; combine
> parallel causal long convolution with a sparse typed reasoning mesh; and use
> verified hierarchical span drafts for distribution-preserving acceleration.

---

## Architecture 2: MultiGear Constructive Program Machine

### Objective

Build a non-Transformer, non-Mamba generative model whose difficult reasoning is
performed by executing and repairing typed programs, while ordinary language
generation retains a fast causal path.

MCPM is not a proof graph with a decoder attached. It is a dual-plane machine:

1. A **surface plane** models the exact causal language distribution and
   handles ordinary generation.
2. An **execution plane** constructs, runs, checks, and repairs explicit
   programs only when additional reasoning is useful.

This split is essential. Requiring every next-token decision to pass through a
discrete proof search would make general generation slow and brittle.

### Why the Previous MCPGM Design Is Rejected

The previous Constructive Proof Graph Machine should not be implemented as
specified. Its main flaws are structural:

1. Making proof-graph construction the primary computation creates a
   sequential discrete-search bottleneck before ordinary text can be emitted.
2. It lacks a strong, explicitly defined fast path for open-ended generation.
3. GFlowNet training over long graph-edit trajectories is high variance and is
   not justified as the default controller for language modelling.
4. Local graph message passing introduces over-squashing and limited
   expressivity precisely when distant facts must interact.
5. Variable graph edits, branches, and backtracking are difficult to batch,
   reducing training throughput.
6. General text corpora provide few supervised proof-graph traces, so graph
   construction would depend heavily on synthetic data or unstable on-policy
   learning.
7. Learned verifiers can prefer plausible but invalid reasoning.
8. A formal verifier proves only the formalized statement; incorrect
   autoformalization can still produce a verified but wrong answer to the
   source question.
9. Learned graph search does not remove the worst-case exponential search
   cost.
10. Treewidth bounds apply to selected exact subsolvers, not to the controller
    that must discover and construct the useful decomposition.
11. A generated proof graph can be a post-hoc explanation rather than the
    causal computation that produced the answer.
12. The design had no explicit fallback or probability semantics preventing
    its reasoning path from degrading base generation.

MCPM replaces graph-message-passing reasoning with executable state
transitions. A branch dependency graph still exists, but only as an execution
record and sharing structure, not as the neural computation substrate.

### MCPGM-to-MCPM Design Corrections

| MCPGM flaw | MCPM correction | Expected effect | Remaining cost |
| --- | --- | --- | --- |
| Proof construction blocks every output | Independent surface bypass | Preserves fast ordinary generation | Router errors can miss useful reasoning |
| Local graph message passing | Typed execution over explicit state | Makes operations causal and inspectable | Compilation to operations can be wrong |
| One edit at a time | Propose and teacher-force instruction blocks | Improves training parallelism | Execution dependencies remain serial |
| Repeated branch prefixes | Copy-on-write branch DAG and result cache | Removes duplicated prefix work | Worst-case branch count remains exponential |
| Fixed action vocabulary | Typed extension calls plus primitive fallback | Reduces DSL expressivity bottleneck | Modules and contracts require engineering |
| Plausibility-based checking | Ordered verifier and contract cascade | Rejects more operational errors | Open-ended claims often remain unverifiable |
| Restart after failed reasoning | Counterexample-guided localized repair | Can improve sample and execution efficiency | Counterexamples may be incomplete |
| Graph narrative may be post-hoc | Condition answer on executed trace and test ablations | Tests causal use of reasoning | Trace-grounded rendering can still omit context |

### Dual-Plane Representation

#### Surface plane

The surface plane is a fast causal MultiGear language model. A practical first
implementation should use the same causal long-convolution family proposed for
MECM, but without MECM's dynamic reasoning mesh.

It maintains:

- the canonical causal next-unit distribution;
- exact byte and MultiGear-span provenance;
- reversible access from selected spans to bytes;
- a zero-initialized residual gate for optional execution results;
- a lossless speculative-generation interface.

Setting the execution gate to zero must recover the standalone surface model
exactly. This requires retaining an immutable surface checkpoint or freezing
the shared surface weights after qualification; a zero gate alone does not
prevent later joint-training drift. This is the primary non-degradation
control.

#### Execution plane

The execution plane is an append-only, copy-on-write branch DAG. Each branch
stores only its delta from a shared immutable parent state. Common prefixes,
tool results, constants, and verified subprograms can therefore be reused
without re-execution.

Cached results must be content-addressed by exact inputs, module and dependency
versions, environment, permissions, and contract. Effectful operations must be
transactional or isolated per branch; otherwise sharing can silently reuse
stale results or leak side effects across branches.

Its typed object store may contain:

```text
exact value
variable and binding
evidence reference
hypothesis
subgoal
tool result
contract
counterexample
verified result
output draft
```

Every object originating from language retains exact MultiGear span and byte
references. MultiGear provides reversible operands, hierarchical drafting, and
provenance; it must not be treated as a guaranteed semantic hierarchy.

Objects and branches may grow with the problem, so there is no mandatory
single-state or fixed-workspace bottleneck. The finite memory, execution,
branch, and latency budgets remain real practical bottlenecks.

### Extensible Executable Instruction Language

The controller proposes typed executable instructions such as:

```text
READ_SPAN
BIND
APPLY
QUERY
ASSERT
SPAWN
JOIN
CALL_TOOL
CALL_TYPED
CHECK
REVISE
ROLLBACK
EMIT
HALT
```

`CALL_TYPED(module, arguments, contract)` is the extensibility mechanism. It
allows domain-specific solvers, learned modules, or future operations to be
added without forcing all reasoning into a permanently narrow DSL. Primitive
instructions remain available as a fallback when learned macro-operations
fail.

Execution must be sandboxed, typed, resource-bounded, and deterministic where
possible. Instructions and operands retain provenance links so that a checked
result can be traced to exact evidence and executed operations.

The learned router and instruction proposer are heads over the causal
long-convolution surface activations and explicitly retrieved typed objects;
they do not use attention or a Mamba-style state-space layer. The controller
addresses objects by stable ID or typed query. Optional learned retrieval may
rank candidates, but the execution trace records the explicit objects actually
read.

### Computation and Generation

1. The surface router estimates whether the request should **BYPASS** or
   **DELIBERATE** under the current quality and latency budget.
2. BYPASS uses only the surface plane for fast ordinary language generation.
3. DELIBERATE proposes one or more instruction blocks or reusable
   macro-programs.
4. The execution plane runs deterministic operations and typed tools.
5. Uncertain choices may `SPAWN` branches; copy-on-write state shares their
   common computation.
6. A verifier cascade checks types, execution results, tests, contracts, and
   formal certificates where available.
7. Failed checks produce typed counterexamples. The controller can `REVISE` or
   `ROLLBACK` instead of restarting from an unstructured text prompt.
8. Compatible verified results are `JOIN`ed into the active state.
9. Joined results are serialized into a typed, provenance-preserving result
   channel and enter the surface generator through the gated residual path.
10. The surface plane renders the answer and, when requested, an explanation
   grounded in the causal execution trace.
11. Hierarchical MultiGear drafts may accelerate the final surface output only
    through lossless speculative verification.

The answer should be conditioned on executed results, not on a separately
generated proof narrative. Trace ablation must change the answer when the trace
is causally necessary; otherwise the execution plane is only decorative.

### Verifier Cascade and Correctness Boundary

Checks should be ordered by cost and trust:

1. Syntax, type, resource, and sandbox checks.
2. Deterministic execution and unit or property tests.
3. Formal solver or proof-checker certificates.
4. Learned critics or verifiers.

A learned verifier is advisory unless it is independently calibrated for the
target distribution. It must not override the surface answer by itself.

A sound formal checker can establish that a certificate proves a formal
statement. It cannot establish that the formal statement faithfully represents
the user's original request. If audited bounds were available, a simple
end-to-end upper bound would be:

```text
P(end-to-end wrong)
  <= epsilon_formalization
   + epsilon_checker_or_tool
   + epsilon_result_interface
```

This union bound is meaningful only if those failure classes and error rates
are actually audited. In most open-ended language tasks they are unknown. MCPM
therefore offers conditional, domain-specific verification, not universal
correctness.

### Training Strategy

Training should separate high-throughput language modelling from expensive
execution learning:

1. Pretrain the surface plane as a normal causal MultiGear language model.
2. Pretrain execution on synthetic programs, arithmetic, code, constraints,
   tools, and tasks with exact answers.
3. Teacher-force complete instruction blocks in parallel where dependencies
   permit, rather than learning only one graph edit at a time.
4. Corrupt programs, bindings, tool results, and contracts; train the model to
   use counterexamples for localized repair.
5. Fine-tune the proposal and routing policies on-policy using outcome,
   progress, execution-cost, and contract signals.
6. Learn reusable macro-programs from recurring successful instruction
   subsequences, while preserving primitive-operation fallback.
7. Distil successful expensive traces into the fast proposal and surface
   policies.
8. Keep the qualified surface fallback immutable, or re-qualify it after every
   surface-weight update; otherwise the claimed fallback is not preserved.

GFlowNet training may be tested in bounded domains with many equivalent valid
programs, but it is not the default training algorithm. Ordinary supervised
execution traces, search, and counterexample-guided repair are lower-risk
starting points.

### Theoretical Effectiveness and Limits

Let:

- `W` be total useful execution work;
- `D` be the serial critical-path length;
- `P` be the number of parallel execution workers;
- `H` be routing, scheduling, communication, and verification overhead.

Then an optimistic lower bound on execution latency is:

```text
T_parallel >= max(W / P, D) + H
```

Ignoring overhead, useful parallel speedup cannot exceed `W / D`. Inherently
serial reasoning therefore remains serial. Parallel branches help only when
they explore independent uncertainties or execute independent subprograms.

For branching factor `b` and depth `h`, worst-case search remains:

```text
O(b^h)
```

Copy-on-write state and cached verified subprograms remove repeated-prefix
work, but do not remove this exponential worst case. Learned macro-programs can
reduce effective depth when tasks reuse structure; they can also overfit and
fail out of distribution.

MCPM can improve reasoning when executable intermediate state, tests, or
counterexamples reject errors more reliably than latent or textual reasoning.
It is unlikely to help tasks whose correctness cannot be operationalized or
whose critical path is long and serial.

General generation remains fast because it can bypass execution. Difficult
reasoning is fast only when:

```text
router cost + proposal cost + execution cost + verification cost
  < cost of the slower baseline needed to reach the same quality
```

Training remains high throughput for the surface model and teacher-forced
instruction proposals. On-policy execution, branching, and verification remain
irregular and expensive.

### Bottleneck Audit

MCPM avoids a mandatory recurrent-state, fixed-workspace, and proof-graph
message-passing bottleneck:

- exact evidence remains in an append-only store;
- objects and branches can be added as required;
- typed calls extend the operation set;
- primitive instructions remain available when learned macros fail;
- the surface model remains available when formal execution is inappropriate.

It is not literally bottleneck-free. Important remaining bottlenecks are:

- converting ambiguous language into faithful executable statements;
- the controller's instruction and routing quality;
- branch, memory, tool, and latency budgets;
- incomplete or expensive verifiers;
- serial critical paths;
- the coverage and reliability of typed module interfaces.

The formalization interface is likely to be MCPM's most consequential semantic
bottleneck.

### Non-Degradation Contract

No architecture can guarantee that adding a learned reasoning system will
never degrade every task. MCPM should instead enforce testable safeguards:

1. With the execution gate zero, the surface model must be exactly
   recoverable from an immutable or re-qualified surface checkpoint.
2. Ordinary generation bypasses the execution plane unless the router predicts
   a measured quality-compute benefit.
3. Execution results enter through a zero-initialized gated residual path.
4. Learned verifier scores alone cannot override the surface path.
5. Failed, timed-out, or low-confidence execution falls back to the surface
   answer or explicitly abstains.
6. Primitive instructions remain available when a learned macro is wrong.
7. Speculative surface decoding must preserve the target distribution exactly.
8. Enhancements are retained only if they improve the measured
   quality-compute-latency frontier.

Verified formal results may intentionally change the surface distribution.
That is a conditional correctness intervention, not a distribution-preserving
acceleration.

### Expected Strengths

- Executed programs make difficult reasoning causal rather than post-hoc.
- Fast bypass path supports ordinary open-ended generation.
- Adaptive compute, branch parallelism, and reusable subprograms.
- Exact arithmetic, code, tools, tests, and formal checks where available.
- Counterexample-guided localized repair rather than full restart.
- Auditable provenance from generated claims to evidence and operations.
- Copy-on-write branches share common work.
- No mandatory single-state or fixed-workspace bottleneck.

### Expected Weaknesses

- Autoformalization and instruction compilation can be wrong.
- Long serial critical paths remain slow.
- Search remains exponential in the worst case.
- Tools and verifiers can be incomplete, expensive, or distribution-specific.
- Dynamic execution and branching are difficult to batch.
- Learned macro-programs can overfit or hide errors.
- Natural-language tasks often lack sound verification.
- Explanations can still misrepresent computation unless rendered from the
  executed trace.
- The complete dual-plane system is operationally complex.

### Validation and Kill Criteria

Do not scale MCPM unless it passes all relevant gates:

1. The zero-gated surface plane exactly matches its standalone baseline.
2. Easy and open-ended generation does not regress at matched compute and
   latency.
3. The execution plane beats ordinary search, program-aided prompting, and
   textual scratchpads at matched tool calls and test-time compute.
4. It improves generalization to larger or structurally different problems.
5. Copy-on-write sharing reduces measured executed work, not only stored
   nodes.
6. Counterexamples measurably improve repair success and sample efficiency.
7. Formal-task evaluation reports certificate correctness separately from
   formalization correctness.
8. MultiGear provenance and hierarchy beat shuffled-hierarchy,
   SentencePiece-span, and exact-byte controls.
9. The router does not invoke deliberation so often that dynamic overhead
   erases its gains.
10. Ablating the execution trace removes the claimed reasoning gain,
    demonstrating that it is causally used.

Stop the architecture if it cannot beat a strong surface model plus ordinary
tool use and search at matched quality, compute, and latency.

### Novelty Boundary

Program-aided language models, neural program synthesis, library learning,
counterexample-guided inductive synthesis, tool execution, adaptive parallel
reasoning, execution supervision, and speculative decoding already exist.

The potentially novel contribution is:

> A MultiGear-provenance-preserving dual-plane model with fast causal surface
> generation and an extensible copy-on-write executable branch DAG,
> counterexample-guided repair, learned reusable macro-programs, and
> contract-gated answer injection.

---

## Architecture 3: MultiGear Residual Workbench Transformer

### Objective

Build the architecture with the highest probability of becoming a strong
general-purpose generative and reasoning language model, without making
MultiGear routing, hierarchy compression, latent recurrence, or tool use
mandatory for every prediction.

MRWT intentionally uses a proven decoder-only Transformer as its target model.
MultiGear is an optional, reversible residual substrate around that target:

1. A qualified **anchor Transformer** defines the canonical language
   distribution and ordinary generation path.
2. A causal **MultiGear span atlas** provides reversible multiscale memory,
   retrieval landmarks, and exact evidence units.
3. An elastic **reasoning workbench** allocates extra free-form computation to
   hard requests without repeatedly transforming the whole prompt.
4. A **MultiGear draft tree** proposes variable-length canonical continuations
   that the anchor verifies with exact speculative sampling.

This is deliberately different from MCPM. MCPM makes typed program execution
the primary hard-reasoning mechanism. MRWT uses a flexible Transformer
workbench and optional tools, prioritizing general language quality over
formally executable reasoning.

Calling MRWT the highest-probability architecture is a research-risk judgment,
not an empirical result. Its anchor makes failure less catastrophic, but the
MultiGear residual mechanisms still have to earn their cost experimentally.

### Why the Previous MHDT Design Is Rejected

The previous MultiGear Hierarchical Deliberation Transformer should not be
implemented as specified:

1. Materializing a complete span lattice is expensive in memory and compute;
   the set of all possible intervals grows quadratically with sequence length,
   and even vocabulary-constrained overlapping candidates can be large.
2. Choosing the correct macro-span routing often requires global contextual
   understanding that the router is supposed to avoid computing, creating a
   circular dependency.
3. Replacing exact units with macro spans creates an information bottleneck.
   Retaining unrestricted global cross-scale access to exact units largely
   restores the cost that compression was meant to remove.
4. The claimed `O(N^2 / s^2)` attention saving omits local composition,
   routing, cross-scale communication, feed-forward computation, KV-cache
   costs, and the distribution of span lengths. Mean span length alone is not
   a reliable runtime model.
5. Sparse attention and MoE do not automatically reduce wall-clock latency;
   dispatch, load imbalance, communication, and small irregular kernels can
   dominate.
6. Recurrently transforming selected macro spans can still be expensive, can
   overthink, and does not establish that the latent states perform faithful
   reasoning.
7. A fixed set of reasoning slots is another capacity bottleneck.
8. Routing, compression, and halting alter the main model path, so MHDT had no
   exact qualified fallback when those mechanisms fail.
9. Predicting output scale, parent, children, or bytes gives the same byte
   string multiple possible generation paths unless the model defines a unique
   canonical factorization or explicitly marginalizes them.
10. `L_next_byte_or_token` is not a well-defined likelihood objective until one
    unique target event sequence is specified.
11. Byte-consistency checking is insufficient for lossless speculative
    decoding. Exact target probabilities and the appropriate acceptance and
    residual-sampling rule are required.
12. The design did not define probability semantics for stochastic routing,
    variable deliberation budgets, or tool results.
13. Introducing hierarchy routing, MoE, recurrent depth, and a hierarchical
    decoder together would make causal attribution and failure diagnosis weak.

MRWT keeps the strong target path intact and treats every new mechanism as an
independently measurable residual enhancement.

### MHDT-to-MRWT Design Corrections

| MHDT flaw | MRWT correction | Expected effect | Remaining cost |
| --- | --- | --- | --- |
| Mandatory macro-span backbone | Immutable anchor Transformer | Retains a qualified general-generation path | Anchor compute remains |
| Complete span lattice | Bounded causal span atlas | Makes hierarchy storage tractable | Candidate selection can miss useful spans |
| Macro spans replace details | Summaries point to retained exact units | Allows refinement to exact evidence | Exact storage and retrieval still cost memory |
| Local router decides global scale | Route after anchor features and observed checks | Uses stronger task context | Anchor must run before savings decisions |
| Whole-span recurrent deliberation | Elastic work cells cross-attend to selected evidence | Extra compute focuses on the reasoning state | Retrieval and work-cell selection can fail |
| Fixed reasoning slots | Append-only variable-size workbench | Avoids one fixed slot bottleneck | Finite budgets remain |
| Ambiguous hierarchical output | Unique canonical target; hierarchy drafts only | Defines valid likelihood and exact verification | Hierarchy cannot change target granularity |
| Consistency-only verification | Exact speculative acceptance and residual sampling | Preserves the selected target distribution | Speedup depends on draft acceptance and hardware |
| No exact fallback | Zero-gated residuals plus immutable anchor checkpoint | Makes fallback testable | Reasoning-mode outputs can still be worse |

### Canonical Probability Semantics

The anchor operates over one unique canonical MultiGear event sequence with an
exact byte fallback:

```text
p_B(y_1:T | x, e) = product_t p_B(y_t | x, y_<t, e)
```

Here:

- `B` is the complete fixed compute policy for the session, including the
  deterministic routing, budget, and checkpoint-selection rules;
- `e` is the pinned tool, retrieval, and execution environment;
- every canonical event maps to one exact immutable byte interval.

Alternative MultiGear spans are features and draft proposals, not additional
target events. This avoids assigning several unaccounted generation paths to
the same byte string.

The canonical encoder must be deterministic: for each byte string it chooses
one target event sequence, including exactly when byte fallback is used.

All routing, atlas construction, retrieval, and workbench updates used at
output position `t` must depend only on `x`, `y_<t`, and pinned environment
state. If branch sampling is stochastic, exact likelihood requires including
or marginalizing that randomness. In practice, likelihood evaluation should
use deterministic routing and tools.

Workbench state may be fixed for an answer segment or append new state that
affects future events only. It may not retroactively alter cached
representations of prior events. Otherwise KV-cache decoding would no longer
match full causal evaluation.

Each compute policy defines a potentially different target distribution.
Speculative decoding can preserve the selected policy's distribution exactly.
Workbench reasoning that changes the answer is not lossless relative to the
anchor-only policy.

### Anchor Transformer

The anchor is a strong decoder-only Transformer over the canonical MultiGear
sequence. Dense, grouped-query, or a proven MoE implementation may be tested,
but hierarchy routing and MoE are not mandatory architectural assumptions.

The anchor provides:

- the ordinary next-event language distribution;
- contextual states used by the budget controller;
- a full-quality path that does not require hierarchy retrieval or
  deliberation;
- exact target probabilities for speculative verification;
- the baseline against which every optional mechanism is judged.

Retain an immutable qualified anchor checkpoint. Optional atlas and workbench
features enter through zero-initialized gated residual adapters. Turning every
optional gate off must reproduce the qualified anchor within numerical
tolerance. Jointly changing shared anchor weights invalidates this claim unless
the updated anchor is re-qualified.

### Causal MultiGear Span Atlas

The span atlas is a bounded, append-only index of reversible views over exact
completed context. It contains:

- canonical MultiGear tokens and their exact bytes;
- canonical parent-child paths;
- a budgeted set of alternative spans selected by measured utility;
- span embeddings, composition error, boundary features, and provenance;
- landmarks that point to exact child blocks.

It does not materialize every possible interval. Every stored span must end
before the position that queries it, preventing future leakage.

Candidate generation must itself be bounded. It should expand canonical
merge-DAG neighbors, locally proposed boundaries, or retrieved constructions;
it must not score all `O(N^2)` intervals before selecting the bounded atlas.

The atlas has three roles:

1. **Evidence retrieval:** work cells retrieve promising spans, then refine to
   exact children before relying on them.
2. **Long-context landmarks:** queries can first select span landmarks and then
   access exact blocks.
3. **Generation drafting:** span paths provide candidate canonical
   continuations to the draft tree.

In full-quality mode, atlas summaries cannot remove the anchor's exact context
access. A separately qualified sparse-context mode may replace full attention
with landmark retrieval, but it is approximate and can change the model
distribution. It must not be presented as an exact acceleration.

### Elastic Evidence-Grounded Workbench

For difficult requests, MRWT creates an append-only set of work cells. A work
cell may represent:

```text
latent proposal
textual scratch state
subgoal or hypothesis
evidence reference
candidate answer or claim
tool request or result
counterexample
check or confidence record
```

Work cells carry stable IDs and provenance links. Their count and number of
rounds are budgeted but not architecturally fixed.

Each workbench round follows a read-compute-write pattern:

1. **Read:** work cells query span landmarks, refine selected spans to exact
   evidence, and optionally read prior work cells.
2. **Compute:** a shared Transformer block applies full interaction among the
   active work cells and cross-attends to the retrieved evidence.
3. **Branch:** independent hypotheses or solution approaches may be processed
   in parallel.
4. **Check:** tools, tests, retrieval agreement, or learned critics may provide
   signals and counterexamples.
5. **Write:** append revised work cells, candidate claims, and an anytime
   answer state.

Only work cells and selected evidence receive repeated deliberation. The full
prompt is not recurrently overwritten or reprocessed every round. This reduces
avoidable work and prevents latent recurrence from corrupting the qualified
anchor state.

Latent work cells are not assumed to be faithful reasoning traces. Learned
critics are not correctness guarantees. Claims of reasoning improvement require
evidence-grounding tests, branch and trace ablations, out-of-distribution
generalization, and comparison with equal-compute scratchpad and search
baselines.

Branching does not remove search complexity. With branching factor `b` and
reasoning depth `h`, worst-case exploration remains `O(b^h)`. Parallel workers
can reduce latency for independent branches but cannot remove total work or an
inherently serial critical path.

The final workbench result enters the anchor through a typed,
provenance-preserving, zero-gated residual channel. The channel may contain
multiple result cells and exact evidence references; forcing all reasoning
through one vector would reintroduce a bottleneck. A practical implementation
uses gated cross-attention adapters from future anchor positions to static or
append-only result cells. Their per-event latency and memory cost must be
included in workbench evaluation.

### Budget Controller and Overthinking Guard

The budget controller chooses among explicit profiles:

```text
ANCHOR_ONLY
ANCHOR_PLUS_ATLAS
ANCHOR_PLUS_WORKBENCH
ANCHOR_PLUS_CHECKED_WORKBENCH
APPROXIMATE_SPARSE_CONTEXT
```

In full-quality profiles, the controller may add computation but cannot skip
the qualified anchor path. The approximate sparse-context profile is a
separately evaluated model mode.

Routing should use anchor contextual states, request constraints, observed
retrieval agreement, check outcomes, and measured marginal improvement. Local
entropy or construction gear alone is insufficient.

The workbench emits an answer checkpoint after each round. Stop when the
quality-cost controller predicts that another round has negative expected
utility, or when the budget is exhausted. Retain earlier checkpoints because
additional recurrence can degrade predictions. A learned value model may guide
checkpoint selection, but it can also be wrong; overthinking curves must be
measured directly.

### MultiGear Draft Tree

The draft tree proposes one or more variable-length canonical continuations
using MultiGear parent-child paths, auxiliary multi-event heads, and a smaller
draft model. Each leaf expands to an exact sequence of canonical target events.

The anchor scores candidate continuations in parallel. Sampling must use a
valid speculative-decoding acceptance and residual-sampling algorithm. Merely
checking that bytes are valid or consistent does not preserve the target
distribution.

Drafting and verification must use the same fixed target compute policy,
workbench snapshot, and pinned environment. Changing any of them during
verification changes the target distribution.

MultiGear is valuable here when it raises accepted events per target-model call
without making drafting and verification slower than ordinary decoding.

### Training Strategy

Train MRWT progressively so each component has a defensible baseline:

1. Train and qualify the anchor on the primary canonical next-event
   likelihood. Compare it with SentencePiece, byte, and shuffled-MultiGear
   controls under matched data, parameters, FLOPs, and wall-clock budgets.
2. Freeze or retain the qualified anchor. Train zero-gated span-atlas adapters,
   retrieval, composition, and draft heads.
3. Train workbench cells using a mixture of pause-style tasks, textual
   scratchpads, latent cells, evidence-grounding supervision, tool traces, and
   counterexample repair. Do not assume latent cells reason faithfully.
4. Randomize work-cell counts and round budgets during training. Supervise
   anytime answer heads so useful outputs exist at several budgets.
5. Train the budget controller as a constrained policy using measured quality,
   latency, memory, and tool cost rather than FLOP proxies alone.
6. Distil successful expensive workbench trajectories into the anchor,
   drafter, or lower-budget workbench policies.
7. Introduce approximate sparse-context mode only after exact-context retrieval
   recall and downstream quality are measured.
8. Re-qualify the anchor fallback after any shared-weight update.

A representative objective is:

```text
L =
    L_anchor_likelihood
  + alpha * L_span_composition_and_retrieval
  + beta  * L_canonical_draft
  + gamma * L_workbench_outcome
  + delta * L_evidence_grounding
  + eta   * L_anytime_answer
  + zeta  * measured_compute_penalty
```

These auxiliary objectives can conflict. Stage them, isolate gradients where
needed, and retain the anchor checkpoint rather than assuming the sum improves
general language modelling.

Do not reintroduce ordinary segmentation dropout as a default; it failed to
improve the existing controlled MultiGear benchmark.

### Theoretical Effectiveness and Efficiency Limits

Let:

- `N` be canonical context length;
- `d` be hidden width;
- `L` be anchor layer count;
- `C` be the number of stored atlas spans;
- `M` be active work-cell count;
- `K` be retrieved exact evidence-unit count;
- `R` be workbench rounds.

A dense anchor prefill retains the usual approximate cost:

```text
C_anchor = O(L * (N^2 * d + N * d^2))
```

This is the cost of preserving a strong unrestricted anchor. MRWT does not
claim that hierarchy makes it disappear. Atlas composition can be near-linear
in `C` for bounded-child spans, but retrieval, indexing, and adapter costs must
be measured.

With a conventional KV cache, each ordinary decode event still reads context
state with cost that grows approximately linearly with cached context length,
and KV memory grows approximately as `O(L * N * d)`. Draft verification can
amortize target calls but does not remove the cache.

A sparse landmark mode may reduce attention work toward a form such as:

```text
O(N * window + queries * retrieved_units + C)
```

but a retrieval miss can change the answer. Exact-context fallback preserves
quality opportunity, not sparse-mode speed.

An idealized workbench round costs approximately:

```text
C_workbench =
  O(R * (M^2 * d + M * K * d + M * d^2))
```

This can be cheaper than recurrently processing the complete prompt when
`M << N` and `K << N`. If the task requires most context or a large workbench,
the advantage disappears. Increasing workbench rounds does not guarantee
improved reasoning and can cause overthinking.

For an output of `T` canonical events, let `A` be accepted draft events per
target verification call. An optimistic target-call count is:

```text
target calls approximately T / (1 + E[A])
```

Wall-clock speedup also depends on draft cost, parallel scoring cost, rejection
frequency, batch size, memory bandwidth, and implementation. It can be less
than one even when drafts are often accepted.

There is no theorem that additional work cells improve reasoning. They increase
available computation, addressable intermediate state, and access to exact
evidence. Improvement requires the retrieval, update, checking, and selection
policies to be informative. General natural-language reasoning usually has no
sound verifier.

Training remains high throughput for the anchor because ordinary next-event
pretraining is parallel across positions. Atlas composition and draft-head
training can also be batched. Dynamic workbench trajectories, on-policy
routing, tools, and variable rounds are irregular and slower; train them on a
controlled subset, bucket examples by budget, and report the throughput loss
rather than hiding it in FLOP estimates.

### Bottleneck Audit

MRWT avoids making hierarchy compression, a fixed recurrent state, or a fixed
number of reasoning slots mandatory:

- the qualified anchor remains available;
- exact canonical events and bytes are retained;
- atlas summaries can refine to exact children;
- work cells and evidence references may grow with the task;
- several result cells can condition the answer;
- MultiGear spans are reversible features and drafts, not semantic truth.

It is not bottleneck-free. Remaining bottlenecks include:

- finite anchor context and KV-cache capacity;
- fixed hidden width and layer capacity;
- atlas candidate and retrieval budgets;
- work-cell count, evidence count, and round budgets;
- the budget controller and learned checkpoint selector;
- residual-channel capacity;
- residual cross-attention latency and cache semantics;
- unavailable or unreliable tools and checks;
- hardware inefficiency from dynamic routing and branching.

### Non-Degradation Contract

No optional reasoning or hierarchy mechanism can universally guarantee better
answers. MRWT should enforce narrower, testable safeguards:

1. Store an immutable qualified anchor checkpoint.
2. With optional residual gates disabled, reproduce anchor logits within
   numerical tolerance.
3. In full-quality mode, atlas summaries and workbench routing cannot remove
   exact anchor inputs or base computation.
4. Timeouts, failed checks, or low-confidence workbench execution can return
   the anchor-only answer before emission, or restart anchor-only generation
   from the last unchanged canonical prefix. It cannot silently roll back text
   already emitted. Fallback availability does not guarantee that the
   controller will choose it correctly.
5. Evaluate reasoning-mode answer selection against the anchor on every target
   domain; it may still degrade because it intentionally changes the
   distribution.
6. Treat approximate sparse-context mode as a separate qualified model, not an
   exact acceleration.
7. Use exact speculative sampling for distribution-preserving generation
   acceleration.
8. Retain components only when they improve the measured
   quality-compute-latency-memory frontier.

The strongest guarantee is exact recoverability of the anchor path, not
universal non-degradation of workbench-enabled answers.

### Expected Strengths

- Highest-probability path to strong open-ended generation among the three
  proposals.
- Exact qualified anchor fallback and valid likelihood semantics.
- MultiGear contributes to context retrieval, evidence granularity, reasoning
  provenance, and output drafting without becoming mandatory compression.
- Elastic workbench provides adjustable free-form reasoning compute.
- Only selected evidence and work cells are recurrently processed.
- Optional tools and checks can ground difficult tasks.
- Exact speculative drafting can accelerate generation without changing the
  selected target distribution.
- Progressive training and zero-gated integration improve causal attribution.

### Expected Weaknesses

- The strong anchor retains Transformer prefill and KV-cache costs.
- Workbench reasoning is not inherently faithful or correct.
- Retrieval can miss necessary evidence.
- Dynamic work cells, branches, and tools complicate batching and serving.
- The budget controller and checkpoint selector can make harmful choices.
- Exact anchor fallback preserves quality opportunity but does not create
  efficiency.
- MultiGear hierarchy may fail to add value over learned chunking,
  SentencePiece, landmarks, or ordinary retrieval.
- Maintaining and qualifying several compute profiles is operationally
  expensive.
- Scientific novelty is limited because every broad component has substantial
  prior art.

### Validation and Kill Criteria

Do not scale MRWT unless it passes the following gates:

1. Anchor-only logits match the qualified checkpoint when optional gates are
   disabled.
2. The MultiGear anchor beats or matches SentencePiece and byte controls at
   matched data, parameters, training FLOPs, inference FLOPs, and wall-clock
   budgets.
3. Real MultiGear spans beat shuffled hierarchy, learned-chunk, length-matched,
   and ordinary retrieval controls.
4. Span-atlas retrieval achieves sufficient exact-evidence recall before
   claiming long-context reasoning gains.
5. Workbench mode beats equal-compute chain-of-thought, pause tokens, recurrent
   depth, sampling/search, and tool-use baselines.
6. Work-cell and evidence ablations remove the claimed reasoning gain,
   demonstrating causal use rather than decorative traces.
7. Accuracy-versus-round curves identify and control overthinking.
8. Approximate sparse-context mode is reported separately from full-quality
   mode.
9. Speculative output is statistically equivalent to target-only sampling and
   improves measured end-to-end latency.
10. Every optional module moves the quality-compute-latency-memory Pareto
    frontier on held-out and out-of-distribution tasks.

Stop the architecture if the anchor is not competitive or if the residual
modules cannot beat simpler adapters, retrieval, scratchpad reasoning, and
speculative decoding at matched budgets.

### Novelty Boundary

Strong decoder-only Transformers, byte-patch models, dynamic chunking,
hierarchical tokenization, landmark retrieval, pause tokens, latent reasoning,
recurrent-depth Transformers, adaptive computation, tool use, multi-token
prediction, and speculative decoding already exist.

The potentially novel contribution is:

> Preserve a qualified canonical Transformer as an exact anchor while using a
> reversible causal MultiGear span atlas as evidence memory and draft
> structure; allocate difficult requests an elastic evidence-grounded
> workbench; admit workbench results through zero-gated residual channels; and
> accelerate output through exact target-verified hierarchical drafts.

---

## Comparative Recommendation

| Property | MECM | MCPM | MRWT |
| --- | --- | --- | --- |
| Transformer-free | Yes | Yes | No |
| Mamba-free | Yes | Yes | Yes by default |
| Primary goal | Fast generation plus elastic reasoning | Executable reasoning plus fast surface generation | Anchor-quality general LM plus elastic free-form reasoning |
| Open-ended generation expectation | Medium | Medium through surface bypass | High |
| Structured reasoning expectation | Medium-high | High | High but generally unverified |
| No mandatory history bottleneck | Yes, with growing ledger | Yes, with growing object and branch ledger | No fixed compressed state; finite anchor context and KV remain |
| Exact conditioning | Approximate plus tools | Through execution, tools, and contracts where available | Approximate; exact anchor inputs retained |
| Explicit verification | Optional/core for hard tasks | Core verifier cascade | Optional verifier |
| Distribution-preserving acceleration | Lossless speculative decoding | Surface path only | Draft-tree speculative decoding only |
| Main theoretical variable | Active-cover compression, retrieval recall, graph depth | Critical path, branch efficiency, and verifier coverage | Anchor cost, retrieval recall, workbench efficiency, and draft acceptance |
| Implementation risk | Very high | Extremely high | Very high |
| Probability of useful near-term result | Medium | Medium-low | Highest |
| Scientific novelty potential | High | High | Medium |

## Recommended Development Order

### Phase 0: shared foundation

Implement the complete MultiGear merge-DAG and interval-lattice API without
materializing every possible interval. Add shuffled-hierarchy and
SentencePiece-lattice controls.

### Phase 1: MECM base-path and hierarchy experiment

Build a causal long-convolution baseline, then add reversible MultiGear views,
zero-gated hierarchy edges, and exact hierarchical speculative drafting. Do not
add dynamic reasoning nodes until the base path matches its baselines.

### Phase 2: MRWT anchor, atlas, and draft experiment

Qualify a strong canonical MultiGear Transformer anchor first. Then add the
zero-gated bounded span atlas and exact MultiGear draft tree. Add the elastic
workbench only after the anchor and draft path pass matched quality and runtime
gates. This remains the highest-probability path to strong general quality and
provides a demanding control for MECM.

### Phase 3: MCPM executable reasoning prototype

Start with the standalone surface path, then add typed execution on synthetic
arithmetic, constraint problems, and small programs with exact verifiers.
Measure ordinary generation before enabling the router. Do not begin with
unrestricted natural-language reasoning.

## Final Recommendation

If only one architecture can be funded, choose **MRWT**, but do not fund the
workbench until its canonical MultiGear anchor matches strong SentencePiece and
byte controls and its residual atlas passes exact-fallback tests.

If the objective is a genuinely new non-Transformer research contribution,
choose **MECM**, but require the base long-convolution path to pass quality and
runtime gates before adding the dynamic graph.

If the objective is reliable reasoning rather than language-model perplexity,
choose **MCPM**, but require executable contracts, a measured surface fallback,
and separate evaluation of formalization and verifier correctness. It remains
the highest-risk architecture.

## Primary Research Basis

The recommendations were checked against the following adjacent work:

- [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955)
- [Byte Latent Transformer](https://aclanthology.org/2025.acl-long.453/)
- [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185)
- [SpaceByte: Towards Deleting Tokenization from Large Language Modeling](https://arxiv.org/abs/2404.14408)
- [From Characters to Tokens: Dynamic Grouping with Hierarchical BPE](https://aclanthology.org/2025.findings-emnlp.595/)
- [Landmark Attention: Random-Access Infinite Context Length for Transformers](https://arxiv.org/abs/2305.16300)
- [Think before you speak: Training Language Models With Pause Tokens](https://arxiv.org/abs/2310.02226)
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- [Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought](https://arxiv.org/abs/2512.21711)
- [Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers](https://arxiv.org/abs/2604.07822)
- [Better and Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171)
- [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)
- [GFlowNet Foundations](https://arxiv.org/abs/2111.09266)
- [DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning](https://arxiv.org/abs/2006.08381)
- [Learning Libraries of Subroutines for Neurally-Guided Bayesian Program Induction](https://papers.nips.cc/paper/8006-learning-libraries-of-subroutines-for-neurallyguided-bayesian-program-induction)
- [Neuro Symbolic Reasoning for Planning: Counterexample Guided Inductive Synthesis using Large Language Models and Satisfiability Solving](https://arxiv.org/abs/2309.16436)
- [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732)
- [Code as Policies: Language Model Programs for Embodied Control](https://arxiv.org/abs/2209.07753)
- [Learning Adaptive Parallel Reasoning with Language Models](https://arxiv.org/abs/2504.15466)
- [Do LLMs Game Formalization? Evaluating Faithfulness in Logical Reasoning](https://arxiv.org/abs/2604.19459)
- [Faithful Autoformalization via Roundtrip Verification and Repair](https://arxiv.org/abs/2604.25031)
- [A Generalist Neural Algorithmic Learner](https://proceedings.mlr.press/v198/ibarz22a.html)
- [Recursive Algorithmic Reasoning](https://proceedings.mlr.press/v231/jurss24a.html)
- [Deep Equilibrium Algorithmic Reasoner](https://arxiv.org/abs/2402.06445)
- [Hyena Hierarchy](https://arxiv.org/abs/2302.10866)
- [Near Linear Time Inference for Long Convolution Sequence Models](https://arxiv.org/abs/2410.12982)
- [On the Bottleneck of Graph Neural Networks and its Practical Implications](https://arxiv.org/abs/2006.05205)
- [DRew: Dynamically Rewired Message Passing with Delay](https://proceedings.mlr.press/v202/gutteridge23a.html)
- [GNNAutoScale](https://proceedings.mlr.press/v139/fey21a.html)
- [ReZero is All You Need](https://arxiv.org/abs/2003.04887)
- [Accelerating LLM Inference with Lossless Speculative Decoding Algorithms](https://arxiv.org/abs/2502.05202)
- [Should You Marginalize over Possible Tokenizations?](https://aclanthology.org/2023.acl-short.1/)
- [Tree-Structured Diffusion Language Model](https://arxiv.org/abs/2604.03537)

These sources establish substantial prior art around every broad ingredient.
The novelty statements above therefore apply only to the specific MultiGear
integrations, not to hierarchy, graph reasoning, program execution,
counterexample-guided synthesis, branch search, recurrent depth, probabilistic
memory, landmark retrieval, latent workspaces, long convolutions, adaptive
computation, speculative decoding, or dynamic chunking in general.
