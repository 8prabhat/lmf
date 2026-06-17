"""Surprisal-Driven Phase Tokenizer (SPT): a standalone, architecture-agnostic
tokenizer alongside the BPE tokenizers, sharing their train/encode/decode/
vocab_size contract."""

from __future__ import annotations

from lmf.data import SpecialTokenTokenizer, SurprisalPhaseTokenizer

_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox jumps over the lazy dog again and again. "
    "pack my box with five dozen liquor jugs."
)


def test_train_encode_decode_roundtrip():
    tok = SurprisalPhaseTokenizer(max_vocab=300)
    tok.train([_TEXT])
    ids = tok.encode(_TEXT)
    assert tok.decode(ids) == _TEXT


def test_vocab_size_respects_budget():
    tok = SurprisalPhaseTokenizer(max_vocab=300)
    tok.train([_TEXT])
    assert tok.vocab_size <= 300
    assert tok.vocab_size >= 256


def test_learned_merge_tokens_respect_length_cap():
    tok = SurprisalPhaseTokenizer(max_vocab=350, max_token_bytes=8)
    tok.train([_TEXT * 20])
    merge_token_ids = set(tok._merge_ids.values())
    assert all(len(tok._vocab[token_id]) <= 8 for token_id in merge_token_ids)


def test_training_is_deterministic():
    first = SurprisalPhaseTokenizer(max_vocab=300)
    second = SurprisalPhaseTokenizer(max_vocab=300)
    first.train([_TEXT])
    second.train([_TEXT])
    assert first._vocab == second._vocab
    assert first.encode(_TEXT) == second.encode(_TEXT)


def test_direct_surprisal_cutoff_matches_phase_scores():
    tok = SurprisalPhaseTokenizer(max_vocab=300, unicode_safe=False, boundary_unit="byte")
    tok.train([_TEXT])
    data = _TEXT.encode("utf-8")
    scores = tok._boundary_scores(data)
    expected = []
    start = 0
    for i in range(1, len(data)):
        if scores[i] > tok.threshold:
            expected.append(data[start:i])
            start = i
    expected.append(data[start:])
    assert tok._segments(data) == expected


def test_phase_segments_do_not_split_grapheme_clusters():
    text = "नमस्ते café e\u0301 👩🏽\u200d💻 中文 العربية"
    tok = SurprisalPhaseTokenizer(max_vocab=400)
    tok.train([text * 8])
    segments = tok._segments_text(text)
    assert b"".join(segments).decode("utf-8") == text
    assert all(segment.decode("utf-8") for segment in segments)


def test_pretokenization_prevents_multiword_tokens():
    text = ("one word follows another word. one word follows another word. " * 20)
    tok = SurprisalPhaseTokenizer(max_vocab=400, threshold=1.0, pretokenize=True)
    tok.train([text])
    pieces = [tok.decode([token_id]) for token_id in tok.encode(text)]
    assert all(len(piece.strip().split()) <= 1 for piece in pieces)


def test_pretokenization_is_lossless():
    text = "can't Ελληνικά_123 中文...\\n👩🏽\u200d💻  العربية"
    assert "".join(SurprisalPhaseTokenizer._pretokenized_text(text)) == text


def test_auto_pretokenization_uses_lexical_boundaries_for_monolingual_text():
    tok = SurprisalPhaseTokenizer(max_vocab=300)
    tok.train(["a mostly monolingual Latin-script corpus " * 20])
    assert tok._effective_pretokenize


def test_auto_pretokenization_avoids_one_script_policy_for_multilingual_text():
    tok = SurprisalPhaseTokenizer(max_vocab=300)
    tok.train(["English text " * 10, "العربية " * 10, "中文文本" * 10])
    assert not tok._effective_pretokenize


def test_balance_texts_applies_to_boundary_statistics():
    texts = ["a" * 1000, "z" * 10]
    unbalanced = SurprisalPhaseTokenizer(max_vocab=300)
    balanced = SurprisalPhaseTokenizer(max_vocab=300, balance_texts=True)
    unbalanced.train(texts)
    balanced.train(texts)
    unbalanced_gap = abs(
        unbalanced._unigram_surprisal[ord("z")] - unbalanced._unigram_surprisal[ord("a")]
    )
    balanced_gap = abs(
        balanced._unigram_surprisal[ord("z")] - balanced._unigram_surprisal[ord("a")]
    )
    assert balanced_gap < unbalanced_gap


def test_optional_grapheme_vocabulary_emits_atomic_clusters():
    text = "中文 العربية नमस्ते 👩🏽\u200d💻 " * 8
    tok = SurprisalPhaseTokenizer(
        max_vocab=400,
        threshold=1.0,
        grapheme_vocab_fraction=1.0,
    )
    tok.train([text])
    zh, wen = "中".encode("utf-8"), "文".encode("utf-8")
    assert zh in tok._encoder and wen in tok._encoder
    assert tok.decode([tok._encoder[zh]]) == "中"
    assert tok.decode([tok._encoder[wen]]) == "文"


def test_phase_merges_improve_heldout_compression():
    text = (
        "the river carries water through the valley. "
        "the river carries silt through the valley. "
        "the railway carries people through the valley. "
    ) * 12
    split = int(len(text) * 0.8)
    train, heldout = text[:split], text[split:]

    whole_only = SurprisalPhaseTokenizer(max_vocab=320, phase_merges=False)
    merged = SurprisalPhaseTokenizer(max_vocab=320)
    whole_only.train([train])
    merged.train([train])

    assert len(merged.encode(heldout)) < len(whole_only.encode(heldout))


def test_long_phase_region_trains_and_roundtrips():
    text = ("the same long phase region repeats without forced boundaries " * 200)
    tok = SurprisalPhaseTokenizer(max_vocab=300, threshold=1.0)
    tok.train([text])
    assert tok.decode(tok.encode(text)) == text


def test_segment_cache_is_bounded():
    tok = SurprisalPhaseTokenizer(max_vocab=300, segment_cache_size=2)
    tok.train([_TEXT])
    for text in ("first unseen text", "second unseen text", "third unseen text"):
        tok.encode(text)
    assert len(tok._segment_cache) <= 2


def test_whole_segment_mode_does_not_overwrite_grapheme_tokens():
    text = "中文 العربية नमस्ते " * 8
    tok = SurprisalPhaseTokenizer(
        max_vocab=400,
        phase_merges=False,
        grapheme_vocab_fraction=0.5,
    )
    tok.train([text])
    assert len(tok._vocab.values()) == len(set(tok._vocab.values()))
    assert tok.decode(tok.encode(text)) == text


def test_repeated_text_compresses_below_byte_count():
    tok = SurprisalPhaseTokenizer(max_vocab=300)
    tok.train([_TEXT])
    ids = tok.encode(_TEXT)
    assert len(ids) < len(_TEXT.encode("utf-8"))


def test_empty_text_roundtrip():
    tok = SurprisalPhaseTokenizer(max_vocab=300)
    tok.train([_TEXT])
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_unseen_bytes_fall_back():
    tok = SurprisalPhaseTokenizer(max_vocab=300)
    tok.train([_TEXT])
    other = "éèê unseen characters \U0001F600"
    ids = tok.encode(other)
    assert tok.decode(ids) == other


def test_works_with_special_token_wrapper():
    base = SurprisalPhaseTokenizer(max_vocab=300)
    base.train([_TEXT])
    wrapped = SpecialTokenTokenizer(base)
    ids = wrapped.encode("<|bos|>" + _TEXT + "<|eos|>")
    assert wrapped.decode(ids) == "<|bos|>" + _TEXT + "<|eos|>"
    assert wrapped.vocab_size == base.vocab_size + len(wrapped.special_tokens)
