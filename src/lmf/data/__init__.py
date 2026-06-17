"""Data layer: batches, tokenizers, corpora, prefetch."""

from __future__ import annotations

from .batch import TrainingBatch, lm_batch
from .corpora import (
    InMemoryTextCorpus,
    EduCombinedCorpus,
    MultiGearTextCorpus,
    NumericFallbackTokenizer,
    ProceduralCorpus,
    WikiTextCorpus,
    build_corpus,
    build_wikitext103,
    corpus_fingerprint,
    tokenizer_fingerprint,
)
from .prefetch import PrefetchCorpus
from .pretokenize import (
    materialize_multigear_dataset,
    materialize_multigear_from_edu_combined,
    materialize_prediction_aware_multigear_dataset,
    materialize_prediction_aware_multigear_from_edu_combined,
    materialize_sentencepiece_bpe_dataset,
    materialize_sentencepiece_bpe_from_edu_combined,
)
from .tokenizers import (
    DEFAULT_SPECIAL_TOKENS,
    ByteBPETokenizer,
    FastBPETokenizer,
    MultiGearPredictionAwareTokenizer,
    MultiGearTokenizer,
    SentencePieceTokenizer,
    SpecialTokenTokenizer,
    SurprisalPhaseTokenizer,
    build_bpe_tokenizer,
)

__all__ = [
    "TrainingBatch",
    "lm_batch",
    "ByteBPETokenizer",
    "FastBPETokenizer",
    "MultiGearTokenizer",
    "MultiGearPredictionAwareTokenizer",
    "SentencePieceTokenizer",
    "SurprisalPhaseTokenizer",
    "SpecialTokenTokenizer",
    "DEFAULT_SPECIAL_TOKENS",
    "build_bpe_tokenizer",
    "ProceduralCorpus",
    "InMemoryTextCorpus",
    "EduCombinedCorpus",
    "NumericFallbackTokenizer",
    "MultiGearTextCorpus",
    "WikiTextCorpus",
    "build_corpus",
    "build_wikitext103",
    "tokenizer_fingerprint",
    "corpus_fingerprint",
    "PrefetchCorpus",
    "materialize_multigear_dataset",
    "materialize_multigear_from_edu_combined",
    "materialize_prediction_aware_multigear_dataset",
    "materialize_prediction_aware_multigear_from_edu_combined",
    "materialize_sentencepiece_bpe_dataset",
    "materialize_sentencepiece_bpe_from_edu_combined",
]
