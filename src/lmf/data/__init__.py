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
from .contiguous_lanes import ContiguousDocumentLaneCorpus
from .prefetch import PrefetchCorpus
from .pure_gear import (
    PairedDocumentManifestCorpus,
    build_document_index,
    build_exhaustive_evaluation_manifest,
    build_paired_training_manifest,
)
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
from .sentence_boundaries import (
    BOUNDARY_DETECTOR_VERSION,
    SentenceBoundaryDetector,
    boundary_detector_hash,
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
    "BOUNDARY_DETECTOR_VERSION",
    "SentenceBoundaryDetector",
    "boundary_detector_hash",
    "ProceduralCorpus",
    "InMemoryTextCorpus",
    "EduCombinedCorpus",
    "NumericFallbackTokenizer",
    "MultiGearTextCorpus",
    "ContiguousDocumentLaneCorpus",
    "WikiTextCorpus",
    "build_corpus",
    "build_wikitext103",
    "tokenizer_fingerprint",
    "corpus_fingerprint",
    "PrefetchCorpus",
    "PairedDocumentManifestCorpus",
    "build_document_index",
    "build_paired_training_manifest",
    "build_exhaustive_evaluation_manifest",
    "materialize_multigear_dataset",
    "materialize_multigear_from_edu_combined",
    "materialize_prediction_aware_multigear_dataset",
    "materialize_prediction_aware_multigear_from_edu_combined",
    "materialize_sentencepiece_bpe_dataset",
    "materialize_sentencepiece_bpe_from_edu_combined",
]
