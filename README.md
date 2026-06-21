# Language Model Foundry (`lmf`)

A clean-break research framework for sequence-model experiments. It is designed to
host **many** language-model families side by side behind a small set of stable
interfaces, so that adding a new architecture never requires touching the training,
data, evaluation, or CLI layers.

## Layout

```
src/lmf/
  core/         framework-agnostic contracts: interfaces, registry, config, device/precision, seeding
  data/         tokenizers, corpora, batching, background prefetch
  training/     base Trainer (loop/optim/logging), callbacks, versioned checkpoints
  evaluation/   metrics (BPT, repetition) and structural benchmarks (long-context, tokens/settle)
  models/
    transformer/    parameter-matched baseline family + MGHT
    rhca/           rolling-frontier family (config, codebook, memory, dynamics, settle, model, trainer)
    gear_transformer/ Transformer trunk + parallel phase-conditioned gear side-channel
    pure_parallel_gear/ canonical attention-free persistent-rotor LM
    bounded_hybrid_gear/ bounded local-attention trunk + scan-based Gear memory, switchable fusion
    mecm/ mcpm/ mgcf/ mrwt/ MultiGear baseline architectures (mecm/mcpm share scaffolding via _shared/)
    gru/            recurrent control baseline for Pure Gear studies
    opet/            OPET phase-enriched embedding family
    _shared/        cross-family infrastructure (not a registrable architecture)
  experiments/  falsification kernels (RFK gates)
  cli/          single `lmf` entrypoint: train | eval | generate | rfk
configs/        one YAML per experiment; merged over a base + environment overlay
scripts/        thin wrappers around the CLI for train / evaluate / generate / rfk
tests/          unit + smoke tests
docs/RESEARCH_NOTES.md  condensed research log: every architecture decision,
                tokenizer experiment, and pilot result produced in this repo
```

## Design principles

* **Single Responsibility / Interface Segregation** -- `core/interfaces.py` defines narrow
  Protocols (`LanguageModel`, `Generative`, `Trainable`, `Corpus`, `Tokenizer`). A family
  implements only what it needs.
* **Open/Closed via registries** -- models, corpora, and trainers self-register through
  decorators (`core/registry.py`). New families are added without editing dispatch code.
* **Dependency Inversion** -- the `Trainer`, evaluation, and CLI depend on the interfaces,
  never on a concrete model.
* **DRY** -- precision/device policy, checkpoint IO, batching, and the optimizer/logging loop
  live once in the framework and are reused by every family.

## Quickstart

```bash
pip install -e ".[dev]"
lmf rfk --config configs/rhca.yaml --block smoke      # falsification gates
lmf train --config configs/rhca.yaml --block smoke    # tiny smoke training run
lmf eval  --config configs/rhca.yaml --block smoke
pytest
```

## Models

Every model below self-registers under `src/lmf/models/` and is selected via
`model.name` in a config. See `docs/RESEARCH_NOTES.md` for the evidence and
recommendations behind each one.

| Registry name | Family | What it is | Config |
| --- | --- | --- | --- |
| `transformer` | baseline | RMSNorm + RoPE + SwiGLU + SDPA decoder-only Transformer, parameter-matched reference for every other family | `configs/transformer_baseline.yaml` |
| `rhca` | RHCA | rolling-frontier model: bounded carried-state windows, factorized codebook, unshared deep macro steps, entropy-based block commits, SDPA exact-recall tail | `configs/rhca.yaml` |
| `opet` | OPET | `transformer` baseline with phase-enriched token embeddings (`OPETEmbedding`) and a coherence auxiliary loss | `configs/opet_baseline.yaml` |
| `gear_transformer` (alias `mlgt`) | Stacked Parallel Gear Transformer V5 | Transformer trunk plus multi-rate banks of 5–20 positive-velocity rotating memories with causal and inter-bank carriers | `configs/gear_transformer.yaml` |
| `gear_only` | Gear Transformer | the same gear mechanism with causal self-attention removed entirely | `configs/gear_transformer.yaml` |
| `pure_parallel_gear` | Pure Parallel Gear | attention-free LM whose only cross-token state is independently rotating rotor banks with explicit noncommutative sentence-boundary clutches and a constant-size generation cache | `configs/pure_parallel_gear.yaml` |
| `pure_parallel_gear_v3`, `hybrid_parallel_gear`, `bounded_transformer`, `bounded_hybrid_gear_block_additive`, `bounded_hybrid_gear_block_selective_film`, `bounded_hybrid_gear_block_bank_router` | Bounded Hybrid Gear | bounded local-attention trunk plus scan-based Gear memory at token rate or block rate, with switchable fusion (additive / selective-FiLM / bank-router) | `configs/bounded_hybrid_gear*.yaml` |
| `gru_lm` | GRU control | parameter-matched recurrent control used to distinguish gear-specific gains from generic recurrence | benchmark-generated |
| `mght` | MultiGear baseline | `transformer` plus a learned MultiGear input-gear embedding and hierarchical (`bias`/`factorized`) gear-aware output head | `configs/multigear_generative_comparison.yaml` |
| `mecm` | MultiGear baseline | non-Transformer causal long-convolution trunk with a zero-gated mesh residual | `configs/multigear_baseline_models.yaml` |
| `mcpm` | MultiGear baseline | non-Transformer surface model with a zero-gated deterministic execution-trace adapter | `configs/multigear_baseline_models.yaml` |
| `mgcf` | MultiGear baseline | non-Transformer, non-Mamba MultiGear Fractal Causal Field: routed dilated causal branches, learned causal long-filter memory, MultiGear child composition, gear-aware output | `configs/multigear_baseline_models.yaml` |
| `mrwt` | MultiGear baseline | Transformer anchor with zero-gated causal atlas/workbench residual adapters and an exact anchor fallback | `configs/multigear_baseline_models.yaml` |

`rhca` is the framework's primary resident family. The MultiGear baseline models
(`mecm`, `mcpm`, `mgcf`, `mrwt`, `mght`) and the Gear Transformer family are
research baselines exploring whether MultiGear hierarchy or a gear side-channel
can beat a matched Transformer + SentencePiece BPE -- as of the latest pilots
(`docs/RESEARCH_NOTES.md`), none of them have, though `mght` and `mgcf` are the
closest.

Pure Parallel Gear’s architecture contract, mathematical mechanism, data
preparation, gated scale workflow, and honest stopping rules are documented in
[`docs/pure_parallel_gear/pure_parallel_gear.md`](docs/pure_parallel_gear/pure_parallel_gear.md).
Verified implementation status and the still-unrun compute stages are recorded
in [`docs/pure_parallel_gear/pure_parallel_gear_implementation_report.md`](docs/pure_parallel_gear/pure_parallel_gear_implementation_report.md).

The canonical family deliberately contains no attention, Q/K/V projections,
token-history retrieval, routing over previous tokens, or KV cache. SentencePiece
BPE is shared unchanged with the Transformer and GRU controls.

### Generative gear mechanism

`gear_transformer` treats gears as multi-scale generation controllers.
Configured `gear_speeds` are angular advances in radians/token and must be
strictly decreasing from fast to slow. Learned multiplicative modulation keeps
every phase advance positive, so a slow gear can decelerate without reversing.

The efficient default uses two stacked banks. Each bank contains five
vectorized gears grouped into local, phrase, semantic, and discourse lanes.
The lower bank updates every token and the upper bank updates every fourth
token while preserving elapsed phase. The architecture also supports the full
three-bank/nine-gear form. At each active position the gear path:

1. updates a vectorized causal context carrier and receives a gated carrier
   from the preceding bank;
2. predicts drive-only phases, then applies sparse adjacent/anchor
   sinusoidal gear-ratio correction;
3. selects phase-preferred latent slots;
4. geometrically rotates paired dimensions of each persistent gear memory;
5. applies a state-dependent recurrent update using a chunk-stable
   closed-form affine rotation scan;
6. fuses gears inside each train, then fuses the four trains with routing
   floors that prevent slow-lane starvation;
7. supervises banks and lanes at progressively longer prediction horizons; and
8. optionally adds a late-ramped future predictor before the tied token head.

`num_gears=0` remains available only as an ablation. Active gear stacks require
5–20 gears. Training initially uses the Transformer path, then ramps gear
residuals, phase/rotation, auxiliary lane losses, and finally future logits.
Gear parameters use a configurable higher learning rate.
Expensive auxiliary and future objectives can run at configurable intervals;
the LM path and gear memories still update every scheduled bank step.

For fastest convergence, build the gear model with the same trunk width/depth
as a trained Transformer and call
`gear_model.initialize_trunk_from_transformer(transformer_model)`. This copies
only compatible embedding, attention, feed-forward, normalization, and LM-head
weights; clocks, slots, lanes, and rotating memories remain independently
initialized.

The reproducible V5 acceptance benchmark includes an equal-update Transformer
continuation, component ablations, generated predictions, and forward/training
throughput:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_gear_transformer_v5.py
```

```bash
lmf train    --config configs/multigear_baseline_models.yaml --block smoke_mecm --steps 10
lmf train    --config configs/multigear_baseline_models.yaml --block smoke_mcpm --steps 10
lmf train    --config configs/multigear_baseline_models.yaml --block smoke_mgcf --steps 10
lmf train    --config configs/multigear_baseline_models.yaml --block smoke_mrwt --steps 10
lmf eval     --config configs/multigear_baseline_models.yaml --block smoke_mrwt --n-batches 2
lmf generate --config configs/multigear_baseline_models.yaml --block smoke_mecm --max-new-tokens 32
```

Full research variants add ablation-visible modules around those safe paths:

```bash
lmf train --config configs/multigear_baseline_models.yaml --block full_smoke_mecm --steps 10
lmf train --config configs/multigear_baseline_models.yaml --block full_smoke_mcpm --steps 10
lmf train --config configs/multigear_baseline_models.yaml --block full_smoke_mrwt --steps 10
```

The matching one-at-a-time ablation specs are:

```bash
lmf ablate --config configs/ablations/multigear_full_mecm.yaml --dry-run
lmf ablate --config configs/ablations/multigear_full_mcpm.yaml --dry-run
lmf ablate --config configs/ablations/multigear_full_mrwt.yaml --dry-run
```

Remove `--dry-run` to execute the ablation. The full modules are deliberately
named for structural ablation, for example `span_atlas.scales.skip[0]`,
`active_cover.bypass`, `execution_workbench.rounds.skip[0]`,
`contract_verifier.bypass`, `budget_controller.bypass`, and
`workbench_rounds.skip[0]`.

For the already-downloaded pre-tokenized corpus:

```bash
lmf train --config configs/multigear_baseline_models.yaml --block edu_mecm
lmf train --config configs/multigear_baseline_models.yaml --block edu_mcpm
lmf train --config configs/multigear_baseline_models.yaml --block edu_mrwt
lmf train --config configs/multigear_baseline_models.yaml --block edu_full_mecm
lmf train --config configs/multigear_baseline_models.yaml --block edu_full_mcpm
lmf train --config configs/multigear_baseline_models.yaml --block edu_full_mrwt
```

`edu_combined` samples the large `train_bpe32768_v2.bin` shards with numpy
memmap and loads bounded valid/test `.pt` tensors. If the sibling Quanthelion
package is available, the shared tokenizer is used for prompt encoding and text
decoding; otherwise generation still works with token-id fallback.

## MultiGear tokenizer

The evidence-backed MultiGear generative integration uses merge-compositional
initialization and hierarchical gear/token output:

```bash
lmf train --config configs/multigear_recommended.yaml --block smoke
```

For real runs, train the slow MultiGear tokenizer once and materialize token ids
before model training:

```bash
lmf pretokenize-multigear \
  --source /path/to/raw_text_file_or_directory \
  --output-root /path/to/multigear_prepared \
  --tokenizer-name multigear32768_v1 \
  --vocab-size 32768

lmf train \
  --config configs/multigear_pretokenized.yaml \
  --block transformer_smoke \
  --set data.root=/path/to/multigear_prepared \
  --set data.tokenizer_name=multigear32768_v1
```

The preprocessing command writes the same disk format used by `edu_combined`:
`shared_tokenizer_<name>.pt`, memory-mapped `train_<name>.bin`, and
`valid_<name>.pt` / `test_<name>.pt`. Training then samples token windows from
disk without retraining or re-encoding the tokenizer.

To derive a diverse MultiGear subset from the existing `edu_combined` BPE shards:

```bash
lmf pretokenize-edu-multigear \
  --source-root "/path/to/edu_combined" \
  --output-root outputs/multigear_prepared \
  --source-tokenizer-name bpe32768_v2 \
  --tokenizer-name multigear_edu_subset_v1 \
  --vocab-size 4096 \
  --fraction 0.10 \
  --max-bpe-tokens-per-domain 200000

lmf ablate --config configs/ablations/mecm_multigear_pretok_smoke.yaml --force
```

Omit `--max-bpe-tokens-per-domain` only for a literal 10% run. On the current
`edu_combined` corpus that is about 4.35B source BPE tokens before MultiGear
re-tokenization, so the capped command is the practical smoke path.

For an apple-to-apple generative baseline, materialize a SentencePiece BPE view
of the same sampled text and evaluate byte-normalized loss:

```bash
lmf pretokenize-edu-sentencepiece-bpe \
  --source-root "/path/to/edu_combined" \
  --output-root outputs/sentencepiece_bpe_prepared \
  --source-tokenizer-name bpe32768_v2 \
  --tokenizer-name sentencepiece_bpe_edu_subset_v1 \
  --vocab-size 4096 \
  --fraction 0.10 \
  --max-bpe-tokens-per-domain 200000

lmf train --config configs/multigear_generative_comparison.yaml \
  --block transformer_sentencepiece_matched_smoke \
  --steps 200 \
  --checkpoint outputs/checkpoints/transformer_sentencepiece_matched_pilot200.pt \
  --set trainer.total_steps=200 \
  --set trainer.warmup_steps=20 \
  --set run.steps=200

lmf eval --config configs/multigear_generative_comparison.yaml \
  --block transformer_sentencepiece_matched_smoke \
  --checkpoint outputs/checkpoints/transformer_sentencepiece_matched_pilot200.pt \
  --n-batches 5 \
  --set trainer.total_steps=200 \
  --set trainer.warmup_steps=20 \
  --set run.steps=200
```

`lmf eval` reports `bits_per_byte` whenever the corpus tokenizer has a lossless
decoder. Use that metric, not raw bits/token, when comparing tokenizers.

All pilot results, tokenizer benchmarks, and architecture decisions referenced
above (MECM, MCPM, MGCF, MRWT, MGHT, the Gear Transformer family, and the
unimplemented MultiGear Predictive Junction Algebra proposal) are written up in
[`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md).

## RHCA training profile

RHCA training intentionally keeps only the final `max_train_windows` carried
frontier windows differentiable per optimizer step. Earlier context is folded
into prefill, so retained activation memory no longer grows with sequence length.
The default is two windows. Increase it only when partial-commit carry quality is
worth the additional training compute and memory.
