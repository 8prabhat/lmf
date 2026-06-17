"""Focused invariants for the multi-scale gear tokenizer."""

from __future__ import annotations

from lmf.data import MultiGearPredictionAwareTokenizer, MultiGearTokenizer, SpecialTokenTokenizer
from lmf.data.tokenizers import MultiGearPeridictionAwareToeknizer

_TEXT = (
    "the quick brown fox turns the first gear. "
    "the second gear turns more slowly. "
    "the quick brown fox turns both gears. "
) * 8


def test_multigear_roundtrip_for_both_inference_modes():
    for inference in ("bpe", "viterbi"):
        tokenizer = MultiGearTokenizer(max_vocab=340, inference=inference)
        tokenizer.train([_TEXT])
        text = _TEXT + " العربية 中文 नमस्ते 👩🏽\u200d💻"
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_multigear_defaults_to_merge_rank_inference():
    assert MultiGearTokenizer(max_vocab=300).inference == "bpe"


def test_multigear_uses_five_progressively_wider_stages():
    tokenizer = MultiGearTokenizer(max_vocab=400)
    tokenizer.train([_TEXT])
    assert len(tokenizer._stage_vocab_sizes) == 5
    assert tokenizer._stage_vocab_sizes == sorted(tokenizer._stage_vocab_sizes)
    assert set(tokenizer._token_gears.values()).issubset(set(range(5)))
    assert any(gear > 1 for gear in tokenizer._token_gears.values())


def test_shifted_wide_gears_can_learn_cross_space_tokens():
    tokenizer = MultiGearTokenizer(max_vocab=430, inference="bpe", max_token_bytes=32)
    tokenizer.train(["alpha beta gamma delta " * 40])
    pieces = tokenizer._vocab.values()
    assert any(b" " in piece.strip() for piece in pieces)


def test_multigear_training_and_encoding_are_deterministic():
    first = MultiGearTokenizer(max_vocab=340)
    second = MultiGearTokenizer(max_vocab=340)
    first.train([_TEXT])
    second.train([_TEXT])
    assert first._vocab == second._vocab
    assert first._merges == second._merges
    assert first.encode(_TEXT) == second.encode(_TEXT)


def test_multigear_respects_vocab_and_token_length_budgets():
    tokenizer = MultiGearTokenizer(max_vocab=350, max_token_bytes=12)
    tokenizer.train([_TEXT])
    assert tokenizer.vocab_size <= 350
    assert max(map(len, tokenizer._vocab.values())) <= 12


def test_multigear_supports_special_tokens():
    base = MultiGearTokenizer(max_vocab=340)
    base.train([_TEXT])
    tokenizer = SpecialTokenTokenizer(base)
    text = "<|bos|>" + _TEXT + "<|eos|>"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_fast_merge_application_matches_reference():
    tokenizer = MultiGearTokenizer(max_vocab=430)
    tokenizer.train([_TEXT * 4])
    samples = [
        list(text.encode("utf-8"))
        for text in (
            _TEXT,
            "the quick brown fox turns both gears.",
            " العربية 中文 नमस्ते 👩🏽\u200d💻 " * 5,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    ]
    for ids in samples:
        assert tokenizer._apply_merge_rules(ids) == tokenizer._apply_merge_rules_reference(ids)


def test_prediction_aware_multigear_roundtrips_and_preserves_hierarchy():
    tokenizer = MultiGearPredictionAwareTokenizer(max_vocab=380, max_token_bytes=16)
    tokenizer.train([_TEXT])
    text = _TEXT + "JP Morgan conference in Australia, then India. العربية 中文"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.inference == "prediction_aware"
    assert tokenizer._prediction_costs
    hierarchy = tokenizer.token_hierarchy()
    assert hierarchy["gear_count"] == 5
    assert len(hierarchy["token_gears"]) == tokenizer.vocab_size


def test_prediction_aware_alias_keeps_request_spelling_working():
    tokenizer = MultiGearPeridictionAwareToeknizer(max_vocab=340)
    tokenizer.train([_TEXT])
    assert tokenizer.decode(tokenizer.encode("the quick gear")) == "the quick gear"


def test_prediction_aware_costs_can_be_model_calibrated():
    tokenizer = MultiGearPredictionAwareTokenizer(max_vocab=340)
    tokenizer.train([_TEXT])
    costs = list(tokenizer._prediction_costs)
    costs[ord(" ")] += 1.0
    tokenizer.set_prediction_costs(costs)
    assert tokenizer._prediction_costs[ord(" ")] == costs[ord(" ")]
