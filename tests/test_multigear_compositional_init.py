"""Merge-tree initialization preserves shape and is deterministic."""

from __future__ import annotations

import math

import torch

from lmf.core.build import initialize_token_embeddings
from lmf.data import MultiGearTokenizer, SpecialTokenTokenizer
from lmf.models.transformer import CachedTransformerLM, TransformerConfig
from scripts.benchmark_multigear_compositional_init import initialize_from_multigear_merges


def test_multigear_compositional_init_uses_child_rows():
    base = MultiGearTokenizer(max_vocab=320)
    base.train(["alpha beta gamma delta " * 30])
    tokenizer = SpecialTokenTokenizer(base)
    model = CachedTransformerLM(TransformerConfig(vocab_size=tokenizer.vocab_size, dim=16))
    before = model.token.weight.detach().clone()
    initialize_from_multigear_merges(model, tokenizer)
    first_pair = base._merges[0]
    token_id = base._merge_ids[first_pair]
    expected = (before[first_pair[0]] + before[first_pair[1]]) / math.sqrt(2.0)
    assert torch.allclose(model.token.weight[token_id], expected)
    assert model.token.weight.shape == before.shape


def test_special_token_wrapper_leaves_special_rows_unchanged():
    base = MultiGearTokenizer(max_vocab=320)
    base.train(["alpha beta gamma delta " * 30])
    tokenizer = SpecialTokenTokenizer(base)
    weight = torch.randn(tokenizer.vocab_size, 8)
    special_before = weight[base.vocab_size:].clone()
    tokenizer.initialize_embeddings_from_merges(weight)
    assert torch.equal(weight[base.vocab_size:], special_before)


def test_build_helper_applies_merge_compositional_strategy():
    base = MultiGearTokenizer(max_vocab=320)
    base.train(["alpha beta gamma delta " * 30])
    tokenizer = SpecialTokenTokenizer(base)
    model = CachedTransformerLM(TransformerConfig(vocab_size=tokenizer.vocab_size, dim=16))
    before = model.token.weight.detach().clone()
    assert initialize_token_embeddings(model, tokenizer, "merge_compositional")
    assert not torch.equal(model.token.weight, before)
    assert initialize_token_embeddings(model, tokenizer, "independent") is False
