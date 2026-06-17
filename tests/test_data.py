"""Corpus correctness: fresh batches (F2), disjoint distinct splits (F5),
reproducible sampler state (F4)."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from lmf.data import (
    EduCombinedCorpus,
    InMemoryTextCorpus,
    MultiGearTokenizer,
    ProceduralCorpus,
    SpecialTokenTokenizer,
    SurprisalPhaseTokenizer,
    materialize_multigear_dataset,
    materialize_multigear_from_edu_combined,
    materialize_prediction_aware_multigear_dataset,
    materialize_prediction_aware_multigear_from_edu_combined,
    materialize_sentencepiece_bpe_dataset,
    materialize_sentencepiece_bpe_from_edu_combined,
    tokenizer_fingerprint,
)


def test_procedural_returns_fresh_batches():
    corpus = ProceduralCorpus(vocab_size=32, seed=0)
    a = corpus.sample_tokenized(4, 40, "train")
    b = corpus.sample_tokenized(4, 40, "train")
    # The generator advances, so consecutive draws must differ (finding 2).
    assert not torch.equal(a, b)


def test_procedural_splits_are_distinct():
    corpus = ProceduralCorpus(vocab_size=32, seed=0)
    tr = corpus.sample_tokenized(4, 40, "train")
    va = corpus.sample_tokenized(4, 40, "valid")
    assert not torch.equal(tr, va)


def test_procedural_sampler_state_roundtrip():
    corpus = ProceduralCorpus(vocab_size=32, seed=0)
    state = corpus.sampler_state()
    first = corpus.sample_tokenized(4, 40, "train")
    corpus.load_sampler_state(state)
    again = corpus.sample_tokenized(4, 40, "train")
    assert torch.equal(first, again)   # restoring state reproduces the batch (finding 4)


def test_inmemory_valid_and_test_are_distinct():
    text = ("the quick brown fox jumps over the lazy dog. " * 200)
    corpus = InMemoryTextCorpus(text, max_vocab=300, seed=0)
    # Repeated text can make the values identical; the storage must still be distinct.
    assert corpus._splits["valid"].data_ptr() != corpus._splits["test"].data_ptr()
    assert len(corpus._splits["test"]) > 0


def test_spt_fingerprint_includes_boundary_model_settings():
    tok = SurprisalPhaseTokenizer(max_vocab=300)
    tok.train(["the quick brown fox jumps over the lazy dog " * 20])
    changed = deepcopy(tok)
    changed.threshold = 0.95
    assert tokenizer_fingerprint(tok) != tokenizer_fingerprint(changed)


def test_multigear_fingerprint_includes_inference_model_settings():
    tok = MultiGearTokenizer(max_vocab=300)
    tok.train(["the quick brown fox jumps over the lazy dog " * 20])
    changed = deepcopy(tok)
    changed.inference = "viterbi"
    assert tokenizer_fingerprint(tok) != tokenizer_fingerprint(changed)


def test_materialize_multigear_dataset_loads_from_disk(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text(
        (
            "alpha beta gamma delta epsilon zeta eta theta. "
            "reasoning models need stable tokenized training data. "
        )
        * 120,
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    report = materialize_multigear_dataset(
        [source],
        output,
        tokenizer_name="multigear_test",
        vocab_size=320,
        tokenizer_kwargs={"max_token_bytes": 12},
    )
    assert report["vocab_size"] > 320
    corpus = EduCombinedCorpus(str(output), tokenizer_name="multigear_test", seed=7)
    assert isinstance(corpus.tokenizer, SpecialTokenTokenizer)
    batch = corpus.sample_tokenized(2, 24, "train")
    valid = corpus.sample_tokenized(1, 16, "valid")
    test = corpus.sample_tokenized(1, 16, "test")
    assert batch.shape == (2, 24)
    assert valid.shape == test.shape == (1, 16)
    assert int(batch.max()) < corpus.vocab_size
    assert corpus.decode_text(batch[0, :8])


def test_materialize_sentencepiece_bpe_dataset_loads_from_disk(tmp_path):
    pytest.importorskip("sentencepiece")
    source = tmp_path / "sample.txt"
    source.write_text(
        (
            "alpha beta gamma delta epsilon zeta eta theta. "
            "sentencepiece bpe is the transformer baseline tokenizer. "
        )
        * 120,
        encoding="utf-8",
    )
    output = tmp_path / "sp_prepared"
    report = materialize_sentencepiece_bpe_dataset(
        [source],
        output,
        tokenizer_name="sp_test",
        vocab_size=320,
    )
    assert report["base_vocab_size"] <= 320
    corpus = EduCombinedCorpus(str(output), tokenizer_name="sp_test", seed=7)
    assert isinstance(corpus.tokenizer, SpecialTokenTokenizer)
    batch = corpus.sample_tokenized(2, 24, "train")
    assert batch.shape == (2, 24)
    assert int(batch.max()) < corpus.vocab_size
    assert corpus.decode_text(batch[0, :8])


def test_materialize_prediction_aware_multigear_dataset_loads_from_disk(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text(
        (
            "alpha beta gamma delta epsilon zeta eta theta. "
            "prediction aware segmentation should avoid rare brittle tokens. "
        )
        * 120,
        encoding="utf-8",
    )
    output = tmp_path / "pa_prepared"
    report = materialize_prediction_aware_multigear_dataset(
        [source],
        output,
        tokenizer_name="pa_test",
        vocab_size=340,
        tokenizer_kwargs={"max_token_bytes": 12},
    )
    assert report["format"] == "lmf_multigear_prediction_aware_tokenizer_v1"
    corpus = EduCombinedCorpus(str(output), tokenizer_name="pa_test", seed=7)
    assert isinstance(corpus.tokenizer, SpecialTokenTokenizer)
    batch = corpus.sample_tokenized(2, 24, "train")
    assert batch.shape == (2, 24)
    assert int(batch.max()) < corpus.vocab_size


def test_materialize_multigear_from_edu_combined_samples_all_domains(tmp_path):
    source_a = tmp_path / "domain_a.txt"
    source_b = tmp_path / "domain_b.txt"
    source_a.write_text(("alpha beta gamma delta. " * 120), encoding="utf-8")
    source_b.write_text(("math proof theorem lemma. " * 120), encoding="utf-8")
    source_root = tmp_path / "source"
    materialize_multigear_dataset(
        [source_a, source_b],
        source_root,
        tokenizer_name="source_tok",
        vocab_size=320,
        tokenizer_kwargs={"max_token_bytes": 12},
    )
    output = tmp_path / "sampled"
    report = materialize_multigear_from_edu_combined(
        source_root,
        output,
        source_tokenizer_name="source_tok",
        tokenizer_name="sampled_tok",
        vocab_size=320,
        tokenizer_kwargs={"max_token_bytes": 12},
        fraction=0.10,
        max_bpe_tokens_per_domain=128,
        window_tokens=32,
        seed=11,
    )
    assert {domain["domain"] for domain in report["domains"]} == {"domain_a", "domain_b"}
    corpus = EduCombinedCorpus(str(output), tokenizer_name="sampled_tok", seed=3)
    assert corpus.sample_tokenized(2, 16, "train").shape == (2, 16)


def test_materialize_prediction_aware_multigear_from_edu_combined_samples_all_domains(tmp_path):
    source_a = tmp_path / "domain_a.txt"
    source_b = tmp_path / "domain_b.txt"
    source_a.write_text(("alpha beta gamma delta. " * 120), encoding="utf-8")
    source_b.write_text(("math proof theorem lemma. " * 120), encoding="utf-8")
    source_root = tmp_path / "source"
    materialize_multigear_dataset(
        [source_a, source_b],
        source_root,
        tokenizer_name="source_tok",
        vocab_size=320,
        tokenizer_kwargs={"max_token_bytes": 12},
    )
    output = tmp_path / "pa_sampled"
    report = materialize_prediction_aware_multigear_from_edu_combined(
        source_root,
        output,
        source_tokenizer_name="source_tok",
        tokenizer_name="pa_sampled_tok",
        vocab_size=340,
        tokenizer_kwargs={"max_token_bytes": 12},
        fraction=0.10,
        max_bpe_tokens_per_domain=128,
        window_tokens=32,
        seed=11,
    )
    assert {domain["domain"] for domain in report["domains"]} == {"domain_a", "domain_b"}
    corpus = EduCombinedCorpus(str(output), tokenizer_name="pa_sampled_tok", seed=3)
    assert corpus.sample_tokenized(2, 16, "train").shape == (2, 16)


def test_materialize_sentencepiece_bpe_from_edu_combined_samples_all_domains(tmp_path):
    pytest.importorskip("sentencepiece")
    source_a = tmp_path / "domain_a.txt"
    source_b = tmp_path / "domain_b.txt"
    source_a.write_text(("alpha beta gamma delta. " * 120), encoding="utf-8")
    source_b.write_text(("math proof theorem lemma. " * 120), encoding="utf-8")
    source_root = tmp_path / "source"
    materialize_multigear_dataset(
        [source_a, source_b],
        source_root,
        tokenizer_name="source_tok",
        vocab_size=320,
        tokenizer_kwargs={"max_token_bytes": 12},
    )
    output = tmp_path / "sp_sampled"
    report = materialize_sentencepiece_bpe_from_edu_combined(
        source_root,
        output,
        source_tokenizer_name="source_tok",
        tokenizer_name="sp_sampled_tok",
        vocab_size=320,
        fraction=0.10,
        max_bpe_tokens_per_domain=128,
        window_tokens=32,
        seed=11,
    )
    assert {domain["domain"] for domain in report["domains"]} == {"domain_a", "domain_b"}
    corpus = EduCombinedCorpus(str(output), tokenizer_name="sp_sampled_tok", seed=3)
    assert corpus.sample_tokenized(2, 16, "train").shape == (2, 16)
