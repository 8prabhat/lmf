# MultiGear Generative Downstream 360-Degree Test

Date: 2026-06-15

## Honest Conclusion

For this benchmark, MultiGear is useful only when its hierarchy is exposed to
the model.

- Flat MultiGear is not better than SentencePiece BPE. Its macro exact-match
  difference is -2.56 points, paired 95% CI [-5.47, 0.35].
- An independent confirmation run verifies merge-compositional initialization
  at exactly the same parameter count: +7.30 exact-match points over
  SentencePiece BPE, paired 95% CI [2.95, 11.66].
- The independent confirmation also verifies compositional initialization plus
  hierarchical output: +13.10 exact-match points over SentencePiece BPE,
  paired 95% CI [9.27, 16.93], winning all ten paired seeds.
- Hierarchical output improves over compositional initialization alone by 5.80
  exact-match points, confirmation CI [2.53, 9.07].
- Auxiliary hierarchy loss and segmentation dropout provide no confirmed
  quality gain. The full stack differs from hierarchy-only MultiGear by -2.48
  exact-match points, confirmation CI [-5.75, 0.80], while training is 4.30
  seconds slower, CI [4.06, 4.54].

The recommendation is therefore:

`merge-compositional initialization + hierarchical output`, with auxiliary
hierarchy loss and segmentation dropout disabled.

This conclusion applies to deterministic multilingual marked-span generation
with the tested small transformer. It is not evidence that MultiGear beats
SentencePiece on translation, instruction following, or open-ended generation.

## Experimental Controls

- FLORES-200 `dev` training split and disjoint `devtest` evaluation split.
- 22 languages and three predeclared target lengths: 4, 8, and 16 UTF-8 bytes.
- 21,934 shared training examples and 4,400 evenly sampled held-out examples
  per task.
- The broad discovery comparison trains 270 models over paired seeds 0--9.
- A predeclared independent confirmation trains another 150 models over paired
  seeds 10--19 for SentencePiece BPE and the four relevant MultiGear stages.
- Every model receives 3,000 updates; 420 models are trained in total.
- Vocabulary size 8,192; batch size 8; padded sequence length 80.
- Same task examples, sampled batch order, optimizer, learning-rate schedule,
  shared-parameter initialization, model shape, and greedy decode budget.
- Correct two-sided Student-t 95% confidence intervals with seed as the
  independent unit. Examples within one trained model are not falsely treated
  as independent.
- Exact match is the primary metric. Edit similarity and teacher-forced
  bits/target-byte are supporting metrics.

Flat models and compositional MultiGear have exactly 631,104 parameters.
Hierarchical output adds 384 parameters (+0.061%). The auxiliary/full variants
add 1,408 parameters (+0.223%). Hierarchical comparisons are integrated-system
comparisons, not strict tokenizer-only comparisons.

## Quality Results

The broad comparison values below are macro means across the three tasks and
discovery seeds 0--9.

| Variant | Exact match | Edit similarity | Bits/target-byte |
| --- | ---: | ---: | ---: |
| SentencePiece BPE | 7.58% | 23.25% | 2.833 |
| Byte BPE | 6.05% | 22.23% | 2.864 |
| SentencePiece unigram | 3.14% | 17.66% | 3.023 |
| SPT | 2.63% | 23.04% | 2.975 |
| MultiGear flat | 5.02% | 15.96% | 3.157 |
| MultiGear compositional | 13.78% | 33.05% | 2.400 |
| MultiGear hierarchical | **21.80%** | **45.28%** | **1.936** |
| MultiGear hierarchical + auxiliary | 18.18% | 41.88% | 2.041 |
| MultiGear full stack | 16.52% | 39.39% | 2.125 |

The independent confirmation verifies hierarchy-only MultiGear against
SentencePiece BPE at every tested difficulty:

| Target length | SentencePiece BPE | MultiGear hierarchical | Paired difference, 95% CI |
| --- | ---: | ---: | ---: |
| 4 bytes | 15.08% | **39.63%** | +24.55 [17.37, 31.72] |
| 8 bytes | 1.38% | **10.78%** | +9.40 [5.56, 13.23] |
| 16 bytes | 0.04% | **5.40%** | +5.37 [1.41, 9.32] |

In confirmation, the hierarchy-only configuration's macro gain over
SentencePiece BPE is +13.10 exact-match points, CI [9.27, 16.93]. Its
teacher-forced loss is lower by 0.848 bits/target-byte, CI [-1.023, -0.673].
The exact-match gain is positive in all ten confirmation seeds. Its mean gain
is positive for all 22 tested languages in both the broad run and independent
confirmation.

## Efficiency Results

MultiGear produces the shortest evaluation prompts, but its current Python
tokenizer is operationally inefficient.

| Metric | SentencePiece BPE | MultiGear |
| --- | ---: | ---: |
| Mean 4-byte-task prompt tokens | 21.22 | **20.21** |
| Tokenizer training, one run | **0.98 s** | 102.09 s |
| Median encode throughput, 7 repeats | **9.82 MB/s** | 0.77 MB/s |
| Median decode throughput, 7 repeats | 16.60 MB/s | **63.28 MB/s** |

The hierarchy-only model trains in 31.29 seconds per run versus 32.32 seconds
for SentencePiece BPE, but generation takes 0.923 seconds versus 0.852 seconds.
Training benefits from scoring only the target gear's vocabulary subset.
Generation is slower because the current implementation dispatches gear subsets
through Python loops. More importantly, MultiGear prompt encoding is 12.7 times
slower than SentencePiece BPE, which dominates end-to-end first-pass latency.

The Rust conversion should prioritize vocabulary construction and encoding.
Until then, MultiGear is a research-quality downstream improvement, not a
production-efficient tokenizer.

## Limitations

- The task is extractive deterministic generation, selected because exact match
  is meaningful without an approximate judge.
- The model has roughly 631K parameters. Scaling behavior is unknown.
- Evaluation uses one multilingual corpus family and a train/devtest domain
  match.
- Byte-defined target lengths represent different character counts across
  scripts, although every tokenizer sees identical examples and gains were
  positive across all tested languages.
- Runtime repeats characterize this machine and implementation, not other
  hardware or a future Rust implementation.

## Artifacts

- `outputs/tokenizer/spt_bench/tokenizer_generation_360_flores22_8k.json`
- `outputs/tokenizer/spt_bench/tokenizer_generation_360_hierarchy_confirmation_seeds10_19.json`
- `outputs/tokenizer/spt_bench/tokenizer_runtime_repeated_flores22_8k.json`
- `scripts/benchmark_tokenizer_generation_360.py`
- `scripts/benchmark_tokenizer_runtime_repeated.py`
- `configs/multigear_recommended.yaml`

## Reproduce

```bash
.venv/bin/python scripts/benchmark_tokenizer_generation_360.py \
  --flores-root /path/to/flores200_dataset \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --target-bytes 4 8 16 \
  --steps 3000 \
  --eval-per-language 200 \
  --out outputs/tokenizer/spt_bench/tokenizer_generation_360_flores22_8k.json

.venv/bin/python scripts/benchmark_tokenizer_generation_360.py \
  --flores-root /path/to/flores200_dataset \
  --seeds 10 11 12 13 14 15 16 17 18 19 \
  --target-bytes 4 8 16 \
  --steps 3000 \
  --eval-per-language 200 \
  --variants sentencepiece_bpe multigear_compositional \
    multigear_hierarchical multigear_hierarchical_aux multigear_full \
  --out outputs/tokenizer/spt_bench/tokenizer_generation_360_hierarchy_confirmation_seeds10_19.json
```
