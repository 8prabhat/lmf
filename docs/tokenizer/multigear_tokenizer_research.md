# MultiGear Tokenizer: Research Record

Date: 2026-06-14

## Status

MultiGear is an exploratory tokenizer, not a demonstrated replacement for BPE
or SentencePiece. It translates the proposed multiple-gear intuition into a
falsifiable algorithm and evaluates it with matched-vocabulary language models.

The first result is mixed:

- The five-stage vocabulary with merge-rank inference beat matched byte BPE in
  the 512-vocabulary smoke LM run: 3.1133 versus 3.1773 bits/byte.
- SPT was better in that tiny-vocabulary smoke run: 3.0549 bits/byte.
- Viterbi gear shifting was worse than both: 3.1949 bits/byte.
- MultiGear is currently much slower to train and encode than the optimized
  baselines.
- On the 22-language FLORES-200 intrinsic benchmark at vocabulary 8,192,
  MultiGear reaches 3.7780 bytes/token versus 3.2727 for byte BPE, but takes
  170.28 seconds to train versus 1.43 seconds.
- A 100-step, one-seed FLORES LM screen rejects the uncapped/default 48-byte
  setting (2.9985 bits/byte). A predeclared max-token-length ablation improves
  the 16-byte setting to 2.9754 bits/byte, versus 2.9849 for byte BPE and
  2.9704 for SPT. This is not yet a multi-seed result.
- Bracketing the cap confirms a local optimum in the tested set: 8 bytes gives
  3.0472 bits/byte, 12 gives 2.9856, 16 gives 2.9754, and 24/32 give 2.9985.
  MultiGear therefore defaults to a configurable 16-byte cap.

The final three-seed evaluation supports MultiGear as an improvement over this
repository's ordinary byte BPE, but not as a replacement for SentencePiece BPE.

## Mechanical Intuition to Algorithm

Literal gear rotation has no established connection to language segmentation.
The useful interpretation is hierarchical scale:

| Mechanical idea | Tokenizer interpretation |
| --- | --- |
| Five gears with decreasing speed | Five progressively wider text scopes |
| Each gear connects to many gears | Every position has multiple candidate token edges |
| Gear moves up/down in a slot | Inference may switch token scale at each position |
| Gear rotates within a slot | Wider training windows use shifted alignments |

The five scopes are:

1. Unicode grapheme clusters.
2. Lexical spans.
3. Shifted windows of two lexical spans.
4. Shifted windows of four lexical spans.
5. Shifted windows of eight lexical spans.

Each stage continues BPE training from the vocabulary and merge hierarchy
learned by the previous stages. This is the core MultiGear tokenizer.

An optional Viterbi inference mode treats all vocabulary pieces as a token
lattice. It learns token unigram costs and a five-by-five gear transition model.
The first downstream result rejects this as the default; merge-rank inference
currently performs better.

## What Is Being Fused

- BPE supplies bottom-up, frequency-driven vocabulary construction.
- A staged boundary curriculum permits both subwords and short multiword
  expressions.
- Unigram-style dynamic programming supplies the optional token-lattice path.
- Raw byte tokens guarantee total, exact-roundtrip encoding for unseen text.

## Why This Design Is Plausible

- [BPE for subword units](https://aclanthology.org/P16-1162/) established
  bottom-up frequent-pair merging as a strong open-vocabulary baseline.
- [SentencePiece](https://aclanthology.org/D18-2012/) and
  [Subword Regularization](https://aclanthology.org/P18-1007/) establish raw
  sentence training and probabilistic mixtures of segmentations.
- [SuperBPE](https://arxiv.org/abs/2503.13423) reports gains from a curriculum
  that first learns subwords and then permits tokens across whitespace.
- [SaGe](https://aclanthology.org/2023.eacl-main.45/) shows that contextual
  signals during vocabulary construction can matter.

## Why It May Fail

- [Tokenization Is More Than Compression](https://aclanthology.org/2024.emnlp-main.40/)
  shows that minimizing token count alone does not imply better language
  modeling.
- [Greed is All You Need](https://aclanthology.org/2024.acl-short.73/) finds
  that greedy inference is often surprisingly strong. This is consistent with
  MultiGear's first Viterbi failure.
- Multiword tokens can waste vocabulary on memorized phrases, increase rare
  types, and make next-token prediction harder.
- A static tokenizer cannot reproduce the learned, entropy-driven dynamic
  patching of [Byte Latent Transformer](https://aclanthology.org/2025.acl-long.453/).
- The current pure-Python staged trainer and encoder are not production-speed.

## Evaluation Gates

All comparisons must use the same train/evaluation text split and vocabulary
budget. Language-model comparisons report bits per raw UTF-8 byte.

1. Exactness: `decode(encode(text)) == text`, including unseen scripts and emoji.
2. Vocabulary budget and determinism.
3. Held-out bytes/token, vocabulary utilization, rare-token mass, and speed.
4. Fixed-token-update transformer bits/byte.
5. Fixed-raw-byte-exposure transformer bits/byte.
6. Multiple seeds before claiming an improvement.

Compression is diagnostic only. Gate 4 or 5 must improve to claim language
modeling impact.

## Final FLORES-200 Result

Vocabulary size is 8,192. The language model is the repository's fixed
two-layer transformer. Values are mean bits per raw UTF-8 byte over seeds
0, 1, and 2; lower is better.

| Tokenizer | Fixed token updates | Fixed raw-byte exposure |
| --- | ---: | ---: |
| SentencePiece BPE | **2.4727** | **2.4797** |
| MultiGear, 16-byte cap | 2.4773 | 2.5264 |
| SPT | 2.4895 | 2.5404 |
| SentencePiece Unigram | 2.5345 | 2.5282 |
| Byte BPE | 2.5720 | 2.5720 |

Interpretation:

- MultiGear beats byte BPE by 0.0947 bits/byte at fixed token updates and
  0.0457 bits/byte at fixed raw-byte exposure on average.
- MultiGear and SPT have high paired-seed variance; MultiGear's mean is better,
  but three seeds are insufficient for a strong claim between them.
- MultiGear trails SentencePiece BPE by only 0.0047 bits/byte in the fixed-token
  mean, which is small relative to the paired seed variation in this
  three-seed sample. It is consistently worse by 0.0467 bits/byte when
  raw-byte exposure is matched.
- MultiGear takes roughly 168 seconds to train on this corpus, versus roughly
  1.1 seconds for SentencePiece BPE. Its current Python implementation is not a
  production-speed tokenizer.

Hard conclusion: the multi-gear idea produced a real, testable tokenizer and a
meaningful improvement over ordinary byte BPE. It did not produce a tokenizer
that is clearly better than the strongest existing baseline. SentencePiece BPE
remains the recommended tokenizer for generative modeling in this benchmark.

## Exact Generative Downstream Result

Date: 2026-06-15.

Open-ended continuation overlap is not a reliable tokenizer metric because many
valid generations differ from a single reference. The generative comparison
therefore uses deterministic multilingual marked-span extraction: a causal
transformer sees a FLORES sentence containing an explicitly marked short span
and must greedily generate that exact span followed by EOS. Exact match is the
primary metric.

The comparison holds vocabulary size (8,192), model shape and parameter count
(631,104), all 21,934 admitted training examples, all 220 held-out examples,
padded sequence length (80), batch order, optimizer, 3,000 updates, decoding
budget, and seeds 0--4 fixed. Only examples that fit every tokenizer are
admitted. Values are mean exact-match percentages over the five paired seeds.

| Tokenizer | Exact match | Edit similarity | Mean eval prompt tokens |
| --- | ---: | ---: | ---: |
| SentencePiece BPE | **19.55%** | **29.52%** | 22.11 |
| Byte BPE | 15.91% | 25.42% | 21.78 |
| MultiGear, 16-byte cap | 10.55% | 19.99% | **21.07** |
| SentencePiece Unigram | 6.82% | 16.41% | 22.71 |
| SPT | 5.27% | 14.81% | 25.51 |

MultiGear uses the fewest prompt tokens, but that compression does not translate
to better exact generation. Relative to byte BPE, MultiGear is -5.36 exact-match
points with a paired-seed 95% confidence interval of [-14.83, 4.11]. Relative
to SentencePiece BPE, it is -9.00 points with an interval of [-22.72, 4.72].
Both intervals cross zero because MultiGear is highly seed-sensitive, so the
reliable conclusion is not that MultiGear is definitively worse; it is that
there is no evidence it improves this downstream task.

Operationally, MultiGear is clearly inefficient in the current implementation:
tokenizer training takes about 160 seconds in this task run versus 0.97 seconds
for SentencePiece BPE and 1.83 seconds for byte BPE. The separate intrinsic run
measures encoding at 0.19 MB/s for MultiGear versus 10.07 MB/s for SentencePiece
BPE and 5.36 MB/s for byte BPE.

This exact task deliberately tests short extractive generation, not open-ended
semantic quality. It supports a reliable narrow comparison without relying on
an approximate judge, but it should not be generalized to translation or
long-form generation.

## Generative Efficiency Deep Dive

Date: 2026-06-15.

The first generation result exposed two distinct efficiency problems:

1. **Downstream learnability.** MultiGear builds tokens hierarchically, but the
   transformer discarded that structure and initialized every vocabulary row as
   an independent random class. This is especially wasteful for sparse or wide
   tokens: the tokenizer already knows their children, but the model must
   relearn them from scratch.
2. **Tokenizer runtime.** Merge-rank encoding repeatedly rescanned the full token
   list for each selected pair, making prompt encoding and tokenizer experiments
   unnecessarily slow.

The initial hypothesis that wide gears simply consumed too much vocabulary was
not supported. A controlled one-seed screen moved 90--100% of learned-token
capacity into grapheme and lexical gears. Exact match fell from 30.45% for
default MultiGear to 19.09%, 12.73%, and 11.36% for the local-heavy variants.
Wide tokens are uncommon in answers, but their prompt compression remains
useful. Removing them trades away input efficiency without fixing token
learnability.

### Merge-Tree Compositional Initialization

The tested improvement initializes each learned MultiGear embedding from its
merge-tree children in rank order:

`embedding(parent) = (embedding(left) + embedding(right)) / sqrt(2)`

The division preserves initialization variance. Special-token rows remain
independently initialized. This changes no parameter count, model shape,
training examples, batches, optimizer, or training time.

On the four-byte exact generation task, ten paired seeds give:

| Model/tokenizer integration | Exact match | Seed standard deviation |
| --- | ---: | ---: |
| MultiGear, independent token rows | 11.32% | 8.76 |
| MultiGear, merge-compositional rows | **29.41%** | **6.72** |
| SentencePiece BPE, independent rows | 18.55% | 8.92 |
| Byte BPE, independent rows | 13.36% | 9.17 |

Compositional initialization improves MultiGear by 18.09 exact-match points,
wins 9 of 10 paired seeds, and has a paired-seed 95% confidence interval of
[7.49, 28.69]. Enhanced MultiGear beats SentencePiece BPE by 10.86 points,
95% CI [2.32, 19.41], and byte BPE by 16.05 points, 95% CI [5.97, 26.12].

A harder eight-byte-target confirmation also improves consistently across all
five seeds: exact match rises from 1.00% to 5.73%, paired difference 4.73 points
with 95% CI [2.75, 6.71]. Edit similarity rises from 12.14% to 23.67%.

Practical recommendation: use `token_embedding_init: merge_compositional` when
training a new generative transformer with MultiGear. Keep it opt-in until it is
validated on translation and long-form generation.

### Runtime Improvement

Merge application now uses a linked token list and lazy occurrence heap while
preserving the previous merge-rank semantics exactly. On the same FLORES-22
intrinsic benchmark:

| Runtime metric | Previous | Improved |
| --- | ---: | ---: |
| Tokenizer training | 170.28 s | **102.42 s** |
| Encoding throughput | 0.19 MB/s | **0.77 MB/s** |

Compression and vocabulary-use metrics are unchanged. This is a meaningful
Python-level improvement, but SentencePiece BPE still encodes at 10.07 MB/s.
A native/Rust trainer and encoder remains necessary for production efficiency.

### Implemented Enhancement Stack

The remaining model-side recommendations are implemented as opt-in features:

1. `token_embedding_init: merge_compositional` initializes learned token rows
   from their merge-tree children. This is the only enhancement currently
   supported by a full downstream result.
2. `hierarchy_aux_weight` adds auxiliary hierarchy supervision for wide output
   tokens. `hierarchy_aux_target: bytes` predicts the token's raw byte sequence
   through a small set of positional slots and scores only the 256 byte rows.
   `hierarchy_aux_target: children` instead predicts immediate merge children.
3. `segmentation_dropout_prob` occasionally decomposes wide input/target tokens
   into canonical children during training. Attention and loss masks are copied
   to decomposed children; fixed-length overflow preserves supervised spans.
4. `hierarchical_output: true` trains the factorization
   `P(gear | hidden) * P(token | gear, hidden)`. Training scores only the target
   gear's token subset. Generation chooses a gear first and scores only that
   subset, reducing flat-vocabulary output work.
5. `multigear_text` is a normal config-driven corpus that trains MultiGear on the
   train split only and wires hierarchy metadata into supported models.

The complete runnable preset is `configs/multigear_enhanced.yaml`. Its relevant
configuration is:

```yaml
data:
  name: multigear_text
  text_file: /path/to/train.txt
  max_vocab: 8192
  tokenizer_kwargs:
    max_token_bytes: 16

model:
  name: transformer
  token_embedding_init: merge_compositional
  hierarchical_output: true
  hierarchy_gears: 6
  hierarchy_aux_weight: 0.10
  hierarchy_aux_min_gear: 2
  hierarchy_aux_target: bytes
  hierarchy_aux_max_bytes: 16

trainer:
  segmentation_dropout_prob: 0.10
  segmentation_dropout_min_gear: 2
  segmentation_dropout_max_depth: 1
```

The full stack passes unit, train, evaluation, and generation smoke tests. A
subsequent 270-model broad evaluation and independent 150-model confirmation
found that compositional initialization plus hierarchical output is the
strongest supported configuration. Auxiliary hierarchy loss and segmentation
dropout provide no confirmed quality gain while adding training cost. The
complete methodology and confidence intervals are recorded in
`docs/tokenizer/multigear_generation_360.md`.

Remaining recommendations:

1. Move vocabulary construction and encoding to a native implementation. The
   optimized Python encoder is still far behind production tokenizers.
2. Do not change default gear fractions based on the current evidence. The
   controlled local-heavy screen made generation worse.
3. Use `configs/multigear_recommended.yaml` for the evidence-backed model
   integration. Keep `configs/multigear_enhanced.yaml` only to reproduce the
   full-stack ablation.

## Reproducible Commands

```bash
.venv/bin/python -m pytest -q

.venv/bin/lmf spt-bench \
  --config configs/spt_bench.yaml \
  --block smoke

.venv/bin/python scripts/benchmark_multilingual_tokenizers.py \
  --flores-root /path/to/flores200_dataset \
  --vocab-size 8192 \
  --tokenizers multigear spt byte_bpe sentencepiece_bpe sentencepiece_unigram \
  --out outputs/tokenizer/spt_bench/multilingual_flores22_8k_multigear_final.json

.venv/bin/python scripts/benchmark_tokenizer_lm_impact.py \
  --flores-root /path/to/flores200_dataset \
  --vocab-size 8192 \
  --tokenizers multigear spt byte_bpe sentencepiece_bpe sentencepiece_unigram \
  --seeds 0 1 2 \
  --steps 300 \
  --out outputs/tokenizer/spt_bench/tokenizer_lm_impact_flores22_8k_multigear_final.json

.venv/bin/python scripts/benchmark_tokenizer_generation.py \
  --flores-root /path/to/flores200_dataset \
  --vocab-size 8192 \
  --tokenizers multigear spt byte_bpe sentencepiece_bpe sentencepiece_unigram \
  --seeds 0 1 2 3 4 \
  --steps 3000 \
  --seq-len 80 \
  --context-bytes 48 \
  --target-bytes 4 \
  --eval-per-language 10 \
  --out outputs/tokenizer/spt_bench/tokenizer_generation_flores22_8k_exact_span_final.json

.venv/bin/python scripts/benchmark_multigear_compositional_init.py \
  --flores-root /path/to/flores200_dataset \
  --seeds 0 1 2 3 4 \
  --out outputs/tokenizer/spt_bench/multigear_compositional_init_flores22_8k.json

.venv/bin/python scripts/benchmark_multigear_compositional_init.py \
  --flores-root /path/to/flores200_dataset \
  --seeds 5 6 7 8 9 \
  --out outputs/tokenizer/spt_bench/multigear_compositional_init_flores22_8k_seeds5_9.json

.venv/bin/lmf train \
  --config configs/multigear_recommended.yaml \
  --block smoke

.venv/bin/python scripts/benchmark_tokenizer_generation_360.py \
  --flores-root /path/to/flores200_dataset \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --target-bytes 4 8 16 \
  --steps 3000 \
  --eval-per-language 200 \
  --out outputs/tokenizer/spt_bench/tokenizer_generation_360_flores22_8k.json

.venv/bin/python scripts/benchmark_multigear_compositional_init.py \
  --flores-root /path/to/flores200_dataset \
  --seeds 0 1 2 3 4 \
  --hierarchical-output \
  --hierarchy-aux-weight 0.10 \
  --hierarchy-aux-target bytes \
  --segmentation-dropout-prob 0.10 \
  --out outputs/tokenizer/spt_bench/multigear_enhanced_generation.json
```

## Implementation Notes

`MultiGearTokenizer` lives in `src/lmf/data/tokenizers.py`. It intentionally
shares the repository's small `train`/`encode`/`decode`/`vocab_size` contract,
works with `SpecialTokenTokenizer`, and is included in tokenizer fingerprints
so incompatible checkpoints are rejected.
