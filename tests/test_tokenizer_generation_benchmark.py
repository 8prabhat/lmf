"""Focused checks for the exact conditional-generation tokenizer benchmark."""

from __future__ import annotations

import pytest
import torch

from lmf.data.tokenizers import ByteBPETokenizer, SpecialTokenTokenizer
from scripts.benchmark_tokenizer_generation import (
    MULTIGEAR_VARIANTS,
    TASK_SPECIAL_TOKENS,
    _SpanTaskCorpus,
    _common_prefix_fraction,
    _encode_examples,
    _make_span_example,
    _normalized_edit_similarity,
    _shared_fit_indices,
)


def _tokenizer():
    base = ByteBPETokenizer(max_vocab=300)
    base.train(["alpha beta gamma delta epsilon " * 20])
    return SpecialTokenTokenizer(base, TASK_SPECIAL_TOKENS)


def test_span_example_marks_exact_target():
    example = _make_span_example("eng_Latn", "alpha beta gamma delta", 64, 8)
    assert example.target
    assert "<|span_start|>" + example.target + "<|span_end|>" in example.prompt
    assert example.prompt.endswith("<|answer|>")


def test_multigear_allocation_variants_are_valid_and_localized():
    for settings in MULTIGEAR_VARIANTS.values():
        fractions = settings.get("gear_fractions", (0.14, 0.38, 0.22, 0.16, 0.10))
        assert len(fractions) == 5
        assert sum(fractions) > 0.0
    assert sum(MULTIGEAR_VARIANTS["multigear_local90"]["gear_fractions"][:2]) == pytest.approx(0.9)
    assert sum(MULTIGEAR_VARIANTS["multigear_local100"]["gear_fractions"][:2]) == pytest.approx(1.0)


def test_similarity_metrics_have_exact_endpoints():
    assert _normalized_edit_similarity("same", "same") == 1.0
    assert _normalized_edit_similarity("", "same") == 0.0
    assert _common_prefix_fraction("same", "same") == 1.0
    assert _common_prefix_fraction("x", "same") == 0.0


def test_task_corpus_masks_only_answer_and_reproduces_sampling():
    tokenizer = _tokenizer()
    examples = [_make_span_example("eng_Latn", text, 64, 8) for text in (
        "alpha beta gamma delta",
        "epsilon alpha beta gamma",
    )]
    encoded = _encode_examples(tokenizer, examples)
    indices = _shared_fit_indices({"first": encoded, "second": encoded}, seq_len=64)
    corpus = _SpanTaskCorpus(tokenizer, encoded, indices, seq_len=64, seed=7)
    state = corpus.sampler_state()
    first = corpus.sample_batch(2, 64)
    corpus.load_sampler_state(state)
    second = corpus.sample_batch(2, 64)
    assert torch.equal(first.tokens, second.tokens)
    assert torch.equal(first.loss_mask, second.loss_mask)
    assert bool((first.loss_mask & ~first.attention_mask).any()) is False
    assert int(first.loss_mask.sum()) > 0
