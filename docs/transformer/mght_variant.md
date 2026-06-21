# MGHT: MultiGear Hierarchical Transformer

`mght` is a named variant of the `transformer` family, not a separate model
folder: it builds the same `CachedTransformerLM` trunk
(`src/lmf/models/transformer/model.py`) and adds a learned MultiGear
input-gear embedding plus a hierarchical (`bias` or `factorized`) gear-aware
output head. It is documented separately here because it is the strongest
MultiGear-tokenizer pilot result so far, and because its config and results
live alongside the other MultiGear baseline models (mecm/mcpm/mgcf/mrwt) even
though its code does not.

## Result summary (200-step pilot, see `docs/RESEARCH_NOTES.md` Section 7)

| model | tokenizer | bits/byte | tokens/sec |
| --- | --- | ---: | ---: |
| Transformer matched | SentencePiece BPE | 3.3772 | 52,502 |
| Transformer matched | MultiGear | 3.4942 | 53,963 |
| MGHT bias | MultiGear | 3.4852 | 44,865 |
| **MGHT factorized** | MultiGear | **3.4822** | 27,822 |

MGHT is the best MultiGear-tokenizer model tested so far, ahead of MECM and
MRWT, though it still trails the SentencePiece BPE Transformer baseline.
Recommended setting: `hierarchy_output_mode: bias` for speed; `factorized`
only when quality matters more than throughput.

## Where it lives

- Code: `build_multigear_hierarchical_transformer()` in
  `src/lmf/models/transformer/model.py`, registered as `mght`.
- Config: `configs/multigear_generative_comparison.yaml`.
- Results: `results/multigear_generative_comparison/mght_architecture_pilot200_summary.md`.
