"""Byte-level BPE tokenizers, an oscillation-based alternative, + a stable
special-token adapter.

* ``ByteBPETokenizer`` — pure-Python, no dependency, exact-roundtrip BPE over a
  256-byte base alphabet (Sennrich 2015).
* ``FastBPETokenizer`` — interface-compatible Rust-backed BPE (HuggingFace
  ``tokenizers``) for large corpora; falls back to the pure-Python one if the
  library is absent.
* ``SurprisalPhaseTokenizer`` — architecture-agnostic, boundary-constrained BPE
  that limits pair merges using a phase score over grapheme-transition
  surprisal, with exact byte fallback.
* ``MultiGearTokenizer`` — five-scale staged BPE vocabulary learning with
  optional Unigram-style Viterbi inference over the resulting token lattice.
* ``MultiGearPredictionAwareTokenizer`` — MultiGear vocabulary learning with
  frequency/gear/length-aware lattice inference that avoids hard-to-predict
  rare pieces while still preserving byte compression.
* ``SpecialTokenTokenizer`` — wraps a trained tokenizer and reserves stable ids
  for ``<|bos|>``/``<|eos|>``/task-control tokens above the base vocabulary.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

try:
    import regex as _unicode_regex
except ImportError:  # pragma: no cover - exercised when the optional package is absent
    _unicode_regex = None


class ByteBPETokenizer:
    """BPE over a 256-byte base vocabulary plus learned merges."""

    def __init__(self, max_vocab: int = 8192) -> None:
        self.max_vocab = int(max_vocab)
        self._merges: list[tuple[int, int]] = []
        self._vocab: dict[int, bytes] = {}
        self._encoder: dict[bytes, int] = {}
        self._initialized = False

    def train(self, texts: list[str]) -> None:
        self._vocab = {i: bytes([i]) for i in range(256)}
        self._encoder = {bytes([i]): i for i in range(256)}
        next_id = 256
        self._merges = []
        word_freq: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for word in re.findall(r" ?\S+|\s+", text):
                word_freq[tuple(word.encode("utf-8", errors="replace"))] += 1
        word_tokens: dict[tuple[int, ...], list[int]] = {w: list(w) for w in word_freq}
        for _ in range(self.max_vocab - 256):
            pair_freq: Counter[tuple[int, int]] = Counter()
            for word, toks in word_tokens.items():
                freq = word_freq[word]
                for a, b in zip(toks, toks[1:]):
                    pair_freq[(a, b)] += freq
            if not pair_freq:
                break
            best = max(pair_freq, key=lambda p: pair_freq[p])
            merged = self._vocab[best[0]] + self._vocab[best[1]]
            self._vocab[next_id] = merged
            self._encoder[merged] = next_id
            self._merges.append(best)
            for word in word_tokens:
                toks = word_tokens[word]
                new: list[int] = []
                i = 0
                while i < len(toks):
                    if i < len(toks) - 1 and (toks[i], toks[i + 1]) == best:
                        new.append(next_id)
                        i += 2
                    else:
                        new.append(toks[i])
                        i += 1
                word_tokens[word] = new
            next_id += 1
        self._initialized = True

    def encode(self, text: str) -> list[int]:
        if not self._initialized:
            raise RuntimeError("call train() before encode()")
        ids = list(text.encode("utf-8", errors="replace"))
        for a, b in self._merges:
            merged_id = self._encoder[self._vocab[a] + self._vocab[b]]
            new: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    new.append(merged_id)
                    i += 2
                else:
                    new.append(ids[i])
                    i += 1
            ids = new
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._vocab.get(i, b"?") for i in ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)


class FastBPETokenizer:
    """GPT-2 style byte-level BPE backed by the Rust ``tokenizers`` engine."""

    def __init__(self, max_vocab: int = 8192) -> None:
        self.max_vocab = int(max_vocab)
        self._tok = None
        self._initialized = False

    def train(self, texts: list[str]) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as BLDecoder

        self._tok = Tokenizer(BPE(unk_token=None))
        self._tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._tok.decoder = BLDecoder()
        trainer = BpeTrainer(
            vocab_size=self.max_vocab,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        )
        self._tok.train_from_iterator(texts, trainer=trainer)
        self._initialized = True

    def encode(self, text: str) -> list[int]:
        if not self._initialized:
            raise RuntimeError("call train() before encode()")
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        if not self._initialized:
            raise RuntimeError("call train() before decode()")
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.max_vocab if not self._initialized else self._tok.get_vocab_size()


class SentencePieceTokenizer:
    """Importable adapter around a serialized SentencePiece model."""

    def __init__(self, model_proto: bytes | None = None, model_file: str | None = None) -> None:
        import sentencepiece as spm

        if (model_proto is None) == (model_file is None):
            raise ValueError("provide exactly one of model_proto or model_file")
        self.processor = spm.SentencePieceProcessor()
        if model_file is not None:
            self.processor.load(model_file)
            self._model_proto = bytes(self.processor.serialized_model_proto())
        else:
            assert model_proto is not None
            self._model_proto = bytes(model_proto)
            self.processor.LoadFromSerializedProto(self._model_proto)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode([int(token_id) for token_id in ids])

    def __getstate__(self) -> dict:
        return {"model_proto": self._model_proto}

    def __setstate__(self, state: dict) -> None:
        import sentencepiece as spm

        self._model_proto = bytes(state["model_proto"])
        self.processor = spm.SentencePieceProcessor()
        self.processor.LoadFromSerializedProto(self._model_proto)


class SurprisalPhaseTokenizer:
    """Standalone tokenizer driven by an oscillation/phase boundary score.

    SPT is a BPE variant, not a replacement for pair merging. It first makes a
    local pass over grapheme-transition *surprisal* by default, then limits
    iterative pair merges to the resulting regions. It is not tied to any
    model architecture (drop-in
    ``train``/``encode``/``decode``/``vocab_size`` contract):

    1. Estimate a grapheme-level bigram model ``P(c_i | c_{i-1})`` from the
       corpus and the surprisal ``s_i = -log P(c_i | c_{i-1})``. Byte-level
       boundaries remain available as a compatibility option.
    2. Squash the corpus-normalized surprisal ``z_i`` to a phase
       ``omega_i = pi * sigmoid(z_i)`` in ``(0, pi)``.
    3. Boundary score ``B_i = (1 - cos(omega_i)) / 2`` in ``[0, 1]``: a
       low-surprisal (expected) continuation keeps ``omega_i`` near 0 and
       ``B_i`` near 0 (no split); a high-surprisal transition pushes
       ``omega_i`` toward ``pi`` and ``B_i`` toward 1 (split).
    4. Respect Unicode-grapheme boundaries and corpus-adaptive lexical
       boundaries, segment wherever ``B_i > threshold``, then learn
       hierarchical pair merges *inside* those regions. Learned tokens are
       length-limited to avoid memorizing long phrases.

    Encoding repeats steps 1-3 with the trained bigram table, then applies the
    learned merge ranks inside each boundary-delimited segment, falling back to
    raw bytes for anything unseen.

    Like ``ByteBPETokenizer``/``FastBPETokenizer``, every byte sequence has a
    representation (256 single-byte tokens are always present), so
    ``decode(encode(text)) == text`` holds for *any* input -- including
    symbols, control characters, and emoji that never appeared in training.
    This is an exactness guarantee SentencePiece does not provide unless
    ``byte_fallback=True`` is set explicitly (it is not the default).
    """

    def __init__(
        self,
        max_vocab: int = 8192,
        threshold: float = 0.7,
        phase_merges: bool = True,
        unicode_safe: bool = True,
        pretokenize: bool | str = "auto",
        boundary_unit: str = "grapheme",
        grapheme_vocab_fraction: float = 0.15,
        balance_texts: bool = False,
        max_token_bytes: int | None = None,
        segment_cache_size: int = 16384,
    ) -> None:
        self.max_vocab = int(max_vocab)
        self.threshold = float(threshold)
        self.phase_merges = bool(phase_merges)
        self.unicode_safe = bool(unicode_safe)
        if pretokenize not in {True, False, "auto"}:
            raise ValueError("pretokenize must be True, False, or 'auto'")
        self.pretokenize = pretokenize
        self._effective_pretokenize = False
        self.boundary_unit = str(boundary_unit)
        self.grapheme_vocab_fraction = float(grapheme_vocab_fraction)
        self.balance_texts = bool(balance_texts)
        # Scale the learned-merge-token length cap with the vocab budget: a
        # fixed small cap (e.g. 12 bytes) starves larger-vocab tokenizers of
        # useful long tokens for repetitive/structured text (numbers, code,
        # templated phrases), while a fixed large cap lets a small-vocab
        # tokenizer memorize whole phrases. ``max_vocab // 256`` keeps the cap
        # proportional, with a floor of 12 bytes.
        self.max_token_bytes = (
            int(max_token_bytes) if max_token_bytes is not None else max(12, self.max_vocab // 256)
        )
        self.segment_cache_size = int(segment_cache_size)
        if self.max_vocab < 256:
            raise ValueError("max_vocab must be at least 256 for byte fallback")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 <= self.grapheme_vocab_fraction <= 1.0:
            raise ValueError("grapheme_vocab_fraction must be between 0 and 1")
        if self.boundary_unit not in {"byte", "grapheme"}:
            raise ValueError("boundary_unit must be 'byte' or 'grapheme'")
        if self.max_token_bytes < 1:
            raise ValueError("max_token_bytes must be positive")
        if self.segment_cache_size < 0:
            raise ValueError("segment_cache_size must be non-negative")
        self._merges: list[tuple[int, int]] = []
        self._merge_ids: dict[tuple[int, int], int] = {}
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._segment_cache: OrderedDict[bytes, tuple[int, ...]] = OrderedDict()
        self._vocab: dict[int, bytes] = {}
        self._encoder: dict[bytes, int] = {}
        self._bigram_surprisal: dict[tuple[int, int], float] = {}
        self._unigram_surprisal: list[float] = [0.0] * 256
        self._grapheme_bigram_surprisal: dict[tuple[str, str], float] = {}
        self._grapheme_unigram_surprisal: dict[str, float] = {}
        self._unknown_grapheme_surprisal = 0.0
        self._surprisal_mean = 0.0
        self._surprisal_std = 1.0
        self._max_token_len = 1
        self._initialized = False
        # Diagnostics for tuning max_token_bytes/segment_cache_size without
        # re-running full ablations: how many merge candidates were skipped
        # for exceeding the byte cap, and how many cache entries were evicted.
        self._capped_merge_skips = 0
        self._cache_evictions = 0

    def _surprisal(self, prev: int | None, cur: int) -> float:
        if prev is None:
            return self._unigram_surprisal[cur]
        return self._bigram_surprisal.get((prev, cur), self._unigram_surprisal[cur])

    def _boundary_scores(self, data: bytes) -> list[float]:
        scores: list[float] = []
        prev: int | None = None
        for b in data:
            z = (self._surprisal(prev, b) - self._surprisal_mean) / self._surprisal_std
            omega = math.pi / (1.0 + math.exp(-z))  # pi * sigmoid(z)
            scores.append((1.0 - math.cos(omega)) / 2.0)
            prev = b
        return scores

    def _surprisal_cutoff(self) -> float:
        """Return the surprisal cutoff equivalent to the configured phase threshold."""
        if self.threshold <= 0.0:
            return -math.inf
        if self.threshold >= 1.0:
            return math.inf
        omega = math.acos(1.0 - 2.0 * self.threshold)
        phase = omega / math.pi
        z_cutoff = math.log(phase / (1.0 - phase))
        return self._surprisal_mean + z_cutoff * self._surprisal_std

    @staticmethod
    def _fallback_graphemes(text: str) -> list[str]:
        """Approximate extended grapheme clusters without an optional dependency."""
        clusters: list[str] = []
        regional_run = 0
        for char in text:
            codepoint = ord(char)
            is_mark = bool(unicodedata.combining(char)) or unicodedata.category(char).startswith("M")
            is_variation = 0xFE00 <= codepoint <= 0xFE0F or 0xE0100 <= codepoint <= 0xE01EF
            is_modifier = 0x1F3FB <= codepoint <= 0x1F3FF
            is_regional = 0x1F1E6 <= codepoint <= 0x1F1FF
            joins_previous = (
                bool(clusters)
                and (
                    is_mark
                    or is_variation
                    or is_modifier
                    or clusters[-1].endswith("\u200d")
                    or char == "\u200d"
                    or (is_regional and regional_run % 2 == 1)
                )
            )
            if joins_previous:
                clusters[-1] += char
            else:
                clusters.append(char)
            regional_run = regional_run + 1 if is_regional else 0
        return clusters

    @classmethod
    def _graphemes(cls, text: str) -> list[str]:
        if _unicode_regex is not None:
            return _unicode_regex.findall(r"\X", text)
        return cls._fallback_graphemes(text)

    @classmethod
    def _grapheme_byte_boundaries(cls, text: str) -> set[int]:
        boundaries: set[int] = set()
        offset = 0
        for cluster in cls._graphemes(text):
            offset += len(cluster.encode("utf-8", errors="replace"))
            boundaries.add(offset)
        return boundaries

    @staticmethod
    def _pretokenized_text(text: str) -> list[str]:
        """Split into reusable lexical/punctuation spans without changing text."""
        if _unicode_regex is not None:
            pattern = (
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
                r" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"
            )
            return _unicode_regex.findall(pattern, text)
        pattern = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
            r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+| ?_+|\s+"
        )
        return re.findall(pattern, text)

    @staticmethod
    def _script_bucket(char: str) -> str:
        name = unicodedata.name(char, "")
        if name.startswith(("CJK ", "IDEOGRAPHIC ")):
            return "HAN"
        return name.split(" ", 1)[0] if name else "UNKNOWN"

    @classmethod
    def _should_pretokenize(cls, texts: list[str]) -> bool:
        scripts: Counter[str] = Counter()
        for text in texts:
            for char in text:
                if unicodedata.category(char).startswith("L"):
                    scripts[cls._script_bucket(char)] += 1
        total = sum(scripts.values())
        return bool(total and scripts.most_common(1)[0][1] / total >= 0.8)

    def _segments(self, data: bytes, allowed_boundaries: set[int] | None = None) -> list[bytes]:
        if not data:
            return []
        cutoff = self._surprisal_cutoff()
        segments: list[bytes] = []
        start = 0
        prev = data[0]
        for i in range(1, len(data)):
            cur = data[i]
            if ((allowed_boundaries is None or i in allowed_boundaries)
                    and self._surprisal(prev, cur) > cutoff):
                segments.append(data[start:i])
                start = i
            prev = cur
        segments.append(data[start:])
        return segments

    def _segments_text(self, text: str) -> list[bytes]:
        # "auto" is decided per text (not once for the whole corpus): a mixed
        # corpus may contain single-script runs (e.g. English paragraphs)
        # alongside multilingual ones, and lexical pretokenization helps the
        # former while hurting the latter (see _should_pretokenize).
        if self.pretokenize == "auto":
            pretokenize_this = self._should_pretokenize([text])
        else:
            pretokenize_this = bool(self.pretokenize)
        pieces = self._pretokenized_text(text) if pretokenize_this else [text]
        segments: list[bytes] = []
        for piece in pieces:
            if self.boundary_unit == "grapheme":
                graphemes = self._graphemes(piece)
                if not graphemes:
                    continue
                cutoff = self._surprisal_cutoff()
                current = [graphemes[0]]
                prev = graphemes[0]
                for grapheme in graphemes[1:]:
                    surprisal = self._grapheme_bigram_surprisal.get(
                        (prev, grapheme),
                        self._grapheme_unigram_surprisal.get(
                            grapheme, self._unknown_grapheme_surprisal
                        ),
                    )
                    if surprisal > cutoff:
                        segments.append("".join(current).encode("utf-8", errors="replace"))
                        current = []
                    current.append(grapheme)
                    prev = grapheme
                segments.append("".join(current).encode("utf-8", errors="replace"))
                continue
            data = piece.encode("utf-8", errors="replace")
            allowed = self._grapheme_byte_boundaries(piece) if self.unicode_safe else None
            segments.extend(self._segments(data, allowed))
        return segments

    def _apply_merge_rules(self, ids: list[int]) -> list[int]:
        while len(ids) > 1:
            ranked = (
                (self._merge_ranks[pair], pair)
                for pair in zip(ids, ids[1:])
                if pair in self._merge_ranks
            )
            chosen = min(ranked, default=None)
            if chosen is None:
                break
            best = chosen[1]
            merged_id = self._merge_ids[best]
            updated: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    updated.append(merged_id)
                    i += 2
                else:
                    updated.append(ids[i])
                    i += 1
            ids = updated
        return ids

    def _initial_ids_for_segment(self, segment: bytes) -> list[int]:
        if not self.unicode_safe:
            return list(segment)
        try:
            text = segment.decode("utf-8")
        except UnicodeDecodeError:
            return list(segment)
        ids: list[int] = []
        for grapheme in self._graphemes(text):
            encoded = grapheme.encode("utf-8", errors="replace")
            token_id = self._encoder.get(encoded)
            if token_id is not None:
                ids.append(token_id)
            else:
                ids.extend(encoded)
        return ids

    def _train_phase_merges(
        self,
        segment_freq: Counter[tuple[int, ...]],
        target_vocab_size: int | None = None,
    ) -> None:
        target_vocab_size = min(self.max_vocab, target_vocab_size or self.max_vocab)
        segments = list(segment_freq)
        frequencies = [segment_freq[segment] for segment in segments]
        tokenized = [
            self._apply_merge_rules(self._initial_ids_for_segment(bytes(segment)))
            for segment in segments
        ]
        pair_freq: Counter[tuple[int, int]] = Counter()
        pair_segments: dict[tuple[int, int], set[int]] = defaultdict(set)

        # Pair occurrences point to compact integer segment ids. Storing the
        # full segment tuple here re-hashes potentially huge byte sequences for
        # every pair occurrence and makes long phase regions pathologically slow.
        for segment_id, (tokens, freq) in enumerate(zip(tokenized, frequencies)):
            for pair in zip(tokens, tokens[1:]):
                pair_freq[pair] += freq
                pair_segments[pair].add(segment_id)

        def _heap_entry(pair: tuple[int, int], freq: int) -> tuple:
            # Tie-break on the *resulting* token length (ascending), not raw
            # byte/token ids: when many pairs tie at the same frequency (common
            # for rare multi-byte sequences, e.g. non-Latin scripts), preferring
            # shorter merges spreads merges breadth-first across the corpus
            # instead of "unrolling" one rare segment end-to-end and burning the
            # vocab budget memorizing it.
            merged_len = len(self._vocab[pair[0]]) + len(self._vocab[pair[1]])
            return (-freq, merged_len, pair[0], pair[1], pair)

        heap = [_heap_entry(pair, freq) for pair, freq in pair_freq.items()]
        heapify(heap)
        next_id = len(self._vocab)
        while next_id < target_vocab_size and pair_freq:
            best = None
            while heap:
                neg_freq, _, _, _, candidate = heappop(heap)
                if pair_freq.get(candidate, 0) == -neg_freq:
                    best = candidate
                    break
            if best is None:
                break
            merged = self._vocab[best[0]] + self._vocab[best[1]]
            if len(merged) > self.max_token_bytes:
                self._capped_merge_skips += 1
                pair_freq.pop(best, None)
                pair_segments.pop(best, None)
                continue
            merged_id = self._encoder.get(merged)
            if merged_id is None:
                merged_id = next_id
                self._vocab[merged_id] = merged
                self._encoder[merged] = merged_id
                next_id += 1
            self._merge_ids[best] = merged_id
            self._merges.append(best)

            changed_pairs: set[tuple[int, int]] = set()
            for segment_id in list(pair_segments.get(best, ())):
                tokens = tokenized[segment_id]
                freq = frequencies[segment_id]
                for pair in zip(tokens, tokens[1:]):
                    pair_freq[pair] -= freq
                    pair_segments[pair].discard(segment_id)
                    changed_pairs.add(pair)
                    if pair_freq[pair] <= 0:
                        del pair_freq[pair]
                        pair_segments.pop(pair, None)

                updated: list[int] = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                        updated.append(merged_id)
                        i += 2
                    else:
                        updated.append(tokens[i])
                        i += 1
                tokenized[segment_id] = updated

                for pair in zip(updated, updated[1:]):
                    pair_freq[pair] += freq
                    pair_segments[pair].add(segment_id)
                    changed_pairs.add(pair)

            for pair in changed_pairs:
                freq = pair_freq.get(pair, 0)
                if freq > 0:
                    heappush(heap, _heap_entry(pair, freq))

            self._merge_ranks[best] = len(self._merge_ranks)

    def train(self, texts: list[str]) -> None:
        self._effective_pretokenize = (
            self._should_pretokenize(texts) if self.pretokenize == "auto" else self.pretokenize
        )
        byte_counts = [
            max(len(text.encode("utf-8", errors="replace")), 1) for text in texts
        ]
        average_bytes = sum(byte_counts) / max(len(byte_counts), 1)
        text_weights = [
            average_bytes / byte_count if self.balance_texts else 1.0
            for byte_count in byte_counts
        ]

        unigram: Counter[int] = Counter()
        bigram: Counter[tuple[int, int]] = Counter()
        for text, weight in zip(texts, text_weights):
            data = text.encode("utf-8", errors="replace")
            prev: int | None = None
            for b in data:
                unigram[b] += weight
                if prev is not None:
                    bigram[(prev, b)] += weight
                prev = b
        total = sum(unigram.values()) or 1
        self._unigram_surprisal = [
            -math.log((unigram.get(b, 0) + 1) / (total + 256)) for b in range(256)
        ]
        prev_totals: Counter[int] = Counter()
        for (a, b), c in bigram.items():
            prev_totals[a] += c
        self._bigram_surprisal = {
            (a, b): -math.log(c / prev_totals[a]) for (a, b), c in bigram.items()
        }

        grapheme_unigram: Counter[str] = Counter()
        grapheme_bigram: Counter[tuple[str, str]] = Counter()
        for text, weight in zip(texts, text_weights):
            prev_grapheme: str | None = None
            for grapheme in self._graphemes(text):
                grapheme_unigram[grapheme] += weight
                if prev_grapheme is not None:
                    grapheme_bigram[(prev_grapheme, grapheme)] += weight
                prev_grapheme = grapheme
        grapheme_total = sum(grapheme_unigram.values()) or 1.0
        grapheme_vocab_size = max(len(grapheme_unigram), 1)
        self._unknown_grapheme_surprisal = -math.log(
            1.0 / (grapheme_total + grapheme_vocab_size)
        )
        self._grapheme_unigram_surprisal = {
            grapheme: -math.log((count + 1.0) / (grapheme_total + grapheme_vocab_size))
            for grapheme, count in grapheme_unigram.items()
        }
        grapheme_prev_totals: Counter[str] = Counter()
        for (prev_grapheme, _), count in grapheme_bigram.items():
            grapheme_prev_totals[prev_grapheme] += count
        self._grapheme_bigram_surprisal = {
            pair: -math.log(count / grapheme_prev_totals[pair[0]])
            for pair, count in grapheme_bigram.items()
        }

        def _observed_surprisals():
            for text, weight in zip(texts, text_weights):
                if self.boundary_unit == "grapheme":
                    prev_grapheme = None
                    for grapheme in self._graphemes(text):
                        surprisal = (
                            self._grapheme_unigram_surprisal.get(
                                grapheme, self._unknown_grapheme_surprisal
                            )
                            if prev_grapheme is None
                            else self._grapheme_bigram_surprisal.get(
                                (prev_grapheme, grapheme),
                                self._grapheme_unigram_surprisal.get(
                                    grapheme, self._unknown_grapheme_surprisal
                                ),
                            )
                        )
                        yield surprisal, weight
                        prev_grapheme = grapheme
                else:
                    prev = None
                    for b in text.encode("utf-8", errors="replace"):
                        yield self._surprisal(prev, b), weight
                        prev = b

        surprisal_sum = 0.0
        surprisal_weight = 0.0
        for value, weight in _observed_surprisals():
            surprisal_sum += value * weight
            surprisal_weight += weight
        if surprisal_weight:
            mean = surprisal_sum / surprisal_weight
            var_sum = 0.0
            for value, weight in _observed_surprisals():
                var_sum += weight * (value - mean) ** 2
            var = var_sum / surprisal_weight
            self._surprisal_mean = mean
            self._surprisal_std = math.sqrt(var) or 1.0
        else:
            self._surprisal_mean = 0.0
            self._surprisal_std = 1.0

        self._vocab = {i: bytes([i]) for i in range(256)}
        self._encoder = {bytes([i]): i for i in range(256)}
        self._merges = []
        self._merge_ids = {}
        self._merge_ranks = {}
        self._segment_cache = OrderedDict()
        self._capped_merge_skips = 0
        self._cache_evictions = 0

        grapheme_freq: Counter[tuple[int, ...]] = Counter()
        for text, weight in zip(texts, text_weights):
            for grapheme in self._graphemes(text):
                encoded = tuple(grapheme.encode("utf-8", errors="replace"))
                if len(encoded) > 1:
                    grapheme_freq[encoded] += weight
        grapheme_budget = int((self.max_vocab - 256) * self.grapheme_vocab_fraction)
        for encoded, _ in grapheme_freq.most_common(grapheme_budget):
            value = bytes(encoded)
            if value in self._encoder:
                continue
            token_id = len(self._vocab)
            self._vocab[token_id] = value
            self._encoder[value] = token_id

        seg_freq: Counter[tuple[int, ...]] = Counter()
        for text, weight in zip(texts, text_weights):
            for seg in self._segments_text(text):
                if seg:
                    seg_freq[tuple(seg)] += weight

        if self.phase_merges:
            self._train_phase_merges(seg_freq)
        else:
            next_id = len(self._vocab)
            for seg, _ in seg_freq.most_common():
                if len(seg) <= 1:
                    continue
                value = bytes(seg)
                if value in self._encoder:
                    continue
                self._vocab[next_id] = value
                self._encoder[value] = next_id
                next_id += 1
                if next_id >= self.max_vocab:
                    break

        self._max_token_len = max((len(s) for s in self._vocab.values()), default=1)
        self._initialized = True

    def _encode_segment(self, seg: bytes) -> list[int]:
        if self.phase_merges:
            cached = self._segment_cache.get(seg)
            if cached is not None:
                self._segment_cache.move_to_end(seg)
                return list(cached)
            ids = self._apply_merge_rules(self._initial_ids_for_segment(seg))
            if self.segment_cache_size:
                self._segment_cache[seg] = tuple(ids)
                self._segment_cache.move_to_end(seg)
                while len(self._segment_cache) > self.segment_cache_size:
                    self._segment_cache.popitem(last=False)
                    self._cache_evictions += 1
            return ids

        ids: list[int] = []
        i, n = 0, len(seg)
        while i < n:
            max_len = min(self._max_token_len, n - i)
            for length in range(max_len, 1, -1):
                chunk = seg[i:i + length]
                if chunk in self._encoder:
                    ids.append(self._encoder[chunk])
                    i += length
                    break
            else:
                ids.append(seg[i])  # byte fallback: ids 0-255 are single bytes
                i += 1
        return ids

    def encode(self, text: str) -> list[int]:
        if not self._initialized:
            raise RuntimeError("call train() before encode()")
        ids: list[int] = []
        for seg in self._segments_text(text):
            ids.extend(self._encode_segment(seg))
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._vocab.get(i, b"?") for i in ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) if self._initialized else self.max_vocab


class MultiGearTokenizer:
    """Lossless multi-scale tokenizer with staged BPE and lattice inference.

    The five gears are progressively wider views of the same byte stream:

    * gear 0 learns inside Unicode grapheme clusters;
    * gear 1 learns inside lexical spans;
    * gears 2-4 learn across shifted windows of 2, 4, and 8 lexical spans.

    Every wider gear starts from the vocabulary and merge rules learned by the
    faster gears. Shifted windows let a wider gear learn across every boundary,
    rather than only boundaries aligned to one fixed chunking. At inference,
    ``"bpe"`` applies the learned merge ranks, while ``"viterbi"`` selects a
    path through all vocabulary pieces using learned token and gear-transition
    costs. Raw bytes are always present, so encoding is total and lossless.

    This is deliberately a falsifiable fusion of established mechanisms:
    staged BPE vocabulary construction, cross-whitespace pieces, and
    Unigram-style dynamic-programming segmentation. The gear transition model
    is a small regularizer, not a claim that literal oscillation improves text.
    """

    def __init__(
        self,
        max_vocab: int = 8192,
        inference: str = "bpe",
        gear_spans: tuple[int, ...] = (0, 1, 2, 4, 8),
        gear_fractions: tuple[float, ...] = (0.14, 0.38, 0.22, 0.16, 0.10),
        max_token_bytes: int | None = None,
        transition_weight: float = 0.15,
        unigram_iterations: int = 2,
        chunk_bytes: int = 65536,
        prune_fraction: float = 0.0,
    ) -> None:
        self.max_vocab = int(max_vocab)
        self.inference = str(inference)
        self.gear_spans = tuple(int(span) for span in gear_spans)
        self.gear_fractions = tuple(float(value) for value in gear_fractions)
        self.max_token_bytes = (
            int(max_token_bytes) if max_token_bytes is not None else 16
        )
        self.transition_weight = float(transition_weight)
        self.unigram_iterations = int(unigram_iterations)
        self.chunk_bytes = int(chunk_bytes)
        self.prune_fraction = float(prune_fraction)
        if self.max_vocab < 256:
            raise ValueError("max_vocab must be at least 256 for byte fallback")
        if self.inference not in {"bpe", "viterbi"}:
            raise ValueError("inference must be 'bpe' or 'viterbi'")
        if len(self.gear_spans) != 5 or len(self.gear_fractions) != 5:
            raise ValueError("MultiGearTokenizer requires exactly five gears")
        if self.gear_spans[0] != 0 or any(span < 1 for span in self.gear_spans[1:]):
            raise ValueError("gear_spans must start with 0 then contain positive spans")
        if any(value < 0.0 for value in self.gear_fractions):
            raise ValueError("gear_fractions must be non-negative")
        if sum(self.gear_fractions) <= 0.0:
            raise ValueError("gear_fractions must have positive mass")
        if self.max_token_bytes < 1:
            raise ValueError("max_token_bytes must be positive")
        if self.transition_weight < 0.0:
            raise ValueError("transition_weight must be non-negative")
        if self.unigram_iterations < 0:
            raise ValueError("unigram_iterations must be non-negative")
        if self.chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        if not 0.0 <= self.prune_fraction < 1.0:
            raise ValueError("prune_fraction must be in [0.0, 1.0)")

        self._vocab: dict[int, bytes] = {}
        self._encoder: dict[bytes, int] = {}
        self._merges: list[tuple[int, int]] = []
        self._merge_ids: dict[tuple[int, int], int] = {}
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._merge_gears: dict[tuple[int, int], int] = {}
        self._token_gears: dict[int, int] = {}
        self._token_costs: list[float] = []
        self._transition_costs: list[list[float]] = [[0.0] * 5 for _ in range(5)]
        self._trie: dict = {}
        self._initialized = False
        self._stage_vocab_sizes: list[int] = []
        self._capped_merge_skips = 0

    @staticmethod
    def _lexical_pieces(text: str) -> list[str]:
        return SurprisalPhaseTokenizer._pretokenized_text(text)

    @staticmethod
    def _graphemes(text: str) -> list[str]:
        return SurprisalPhaseTokenizer._graphemes(text)

    def _gear_segments(self, text: str, gear: int):
        if gear == 0:
            for grapheme in self._graphemes(text):
                yield grapheme.encode("utf-8", errors="replace"), 1.0
            return

        pieces = self._lexical_pieces(text)
        span = self.gear_spans[gear]
        if span == 1:
            for piece in pieces:
                yield piece.encode("utf-8", errors="replace"), 1.0
            return

        # Rotate the wider window through every possible alignment. Weighting
        # each rotation by 1/span keeps the total corpus mass roughly stable.
        weight = 1.0 / span
        for offset in range(span):
            for start in range(offset, len(pieces) - span + 1, span):
                yield "".join(pieces[start:start + span]).encode("utf-8", errors="replace"), weight

    def _apply_merge_rules_reference(self, ids: list[int]) -> list[int]:
        """Simple reference implementation used to define merge-rank semantics."""
        while len(ids) > 1:
            chosen = min(
                (
                    (self._merge_ranks[pair], pair)
                    for pair in zip(ids, ids[1:])
                    if pair in self._merge_ranks
                ),
                default=None,
            )
            if chosen is None:
                break
            best = chosen[1]
            merged_id = self._merge_ids[best]
            updated: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    updated.append(merged_id)
                    i += 2
                else:
                    updated.append(ids[i])
                    i += 1
            ids = updated
        return ids

    def _apply_merge_rules(self, ids: list[int]) -> list[int]:
        """Apply ranked pair merges without repeatedly rescanning the full list.

        Semantics match ``_apply_merge_rules_reference``: choose the lowest-rank
        pair type, merge all of its currently present non-overlapping
        occurrences left-to-right, then choose again. A linked token list plus
        lazy occurrence heap makes the work proportional to affected pairs.
        """
        if len(ids) < 2:
            return ids
        tokens = list(ids)
        size = len(tokens)
        previous = [index - 1 for index in range(size)]
        following = [index + 1 for index in range(size)]
        following[-1] = -1
        alive = [True] * size
        occurrences: dict[tuple[int, int], set[int]] = defaultdict(set)
        heap: list[tuple[int, int, int, tuple[int, int]]] = []

        def pair_at(left: int) -> tuple[int, int] | None:
            if left < 0 or not alive[left]:
                return None
            right = following[left]
            if right < 0 or not alive[right]:
                return None
            return tokens[left], tokens[right]

        def remove_occurrence(left: int) -> None:
            pair = pair_at(left)
            if pair is not None:
                occurrences[pair].discard(left)

        def add_occurrence(left: int) -> None:
            pair = pair_at(left)
            rank = self._merge_ranks.get(pair) if pair is not None else None
            if pair is None or rank is None:
                return
            occurrences[pair].add(left)
            heappush(heap, (rank, pair[0], pair[1], pair))

        for left in range(size - 1):
            add_occurrence(left)

        while heap:
            _, _, _, chosen = heappop(heap)
            starts = occurrences.get(chosen)
            if not starts:
                continue
            valid_starts = sorted(left for left in starts if pair_at(left) == chosen)
            if not valid_starts:
                occurrences.pop(chosen, None)
                continue

            merged_id = self._merge_ids[chosen]
            for left in valid_starts:
                if pair_at(left) != chosen:
                    starts.discard(left)
                    continue
                right = following[left]
                before = previous[left]
                after = following[right]
                remove_occurrence(before)
                remove_occurrence(left)
                remove_occurrence(right)

                tokens[left] = merged_id
                following[left] = after
                if after >= 0:
                    previous[after] = left
                alive[right] = False
                following[right] = -1
                previous[right] = -1

                add_occurrence(before)
                add_occurrence(left)

        output = []
        position = 0
        while position >= 0:
            output.append(tokens[position])
            position = following[position]
        return output

    def _train_stage(
        self,
        segment_freq: Counter[tuple[int, ...]],
        target_vocab_size: int,
        gear: int,
    ) -> None:
        segments = list(segment_freq)
        frequencies = [segment_freq[segment] for segment in segments]
        tokenized = [self._apply_merge_rules(list(segment)) for segment in segments]
        pair_freq: Counter[tuple[int, int]] = Counter()
        pair_segments: dict[tuple[int, int], set[int]] = defaultdict(set)
        for segment_id, (tokens, freq) in enumerate(zip(tokenized, frequencies)):
            for pair in zip(tokens, tokens[1:]):
                pair_freq[pair] += freq
                pair_segments[pair].add(segment_id)

        def heap_entry(pair: tuple[int, int], freq: float) -> tuple:
            merged_len = len(self._vocab[pair[0]]) + len(self._vocab[pair[1]])
            return (-freq, merged_len, pair[0], pair[1], pair)

        heap = [heap_entry(pair, freq) for pair, freq in pair_freq.items()]
        heapify(heap)
        while len(self._vocab) < target_vocab_size and pair_freq:
            best = None
            while heap:
                neg_freq, _, _, _, candidate = heappop(heap)
                if math.isclose(pair_freq.get(candidate, 0.0), -neg_freq):
                    best = candidate
                    break
            if best is None:
                break

            merged = self._vocab[best[0]] + self._vocab[best[1]]
            if len(merged) > self.max_token_bytes:
                self._capped_merge_skips += 1
                pair_freq.pop(best, None)
                pair_segments.pop(best, None)
                continue

            merged_id = self._encoder.get(merged)
            if merged_id is None:
                merged_id = len(self._vocab)
                self._vocab[merged_id] = merged
                self._encoder[merged] = merged_id
                self._token_gears[merged_id] = gear
            self._merge_ids[best] = merged_id
            self._merge_ranks[best] = len(self._merge_ranks)
            self._merge_gears[best] = gear
            self._merges.append(best)

            changed_pairs: set[tuple[int, int]] = set()
            for segment_id in list(pair_segments.get(best, ())):
                tokens = tokenized[segment_id]
                freq = frequencies[segment_id]
                for pair in zip(tokens, tokens[1:]):
                    pair_freq[pair] -= freq
                    pair_segments[pair].discard(segment_id)
                    changed_pairs.add(pair)
                    if pair_freq[pair] <= 1e-12:
                        pair_freq.pop(pair, None)
                        pair_segments.pop(pair, None)

                updated: list[int] = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                        updated.append(merged_id)
                        i += 2
                    else:
                        updated.append(tokens[i])
                        i += 1
                tokenized[segment_id] = updated
                for pair in zip(updated, updated[1:]):
                    pair_freq[pair] += freq
                    pair_segments[pair].add(segment_id)
                    changed_pairs.add(pair)

            for pair in changed_pairs:
                freq = pair_freq.get(pair, 0.0)
                if freq > 0.0:
                    heappush(heap, heap_entry(pair, freq))

    def _prune_low_usage_tokens(self, texts: list[str], prune_fraction: float) -> None:
        """Drop the least-used learned merges, freeing vocab slots for the rest.

        Merges are pruned in reverse training order (latest first) so a token
        is only removed once nothing kept depends on it as a merge component
        -- the dependency direction always points from later merges to
        earlier ones, so a single reverse pass suffices.
        """
        if prune_fraction <= 0.0:
            return

        token_counts: Counter[int] = Counter()
        for text in texts:
            token_counts.update(self._encode_internal(text, "bpe"))

        mergeable_ids = [tid for tid in self._vocab if tid >= 256]
        num_to_prune = int(round(prune_fraction * len(mergeable_ids)))
        if num_to_prune <= 0:
            return
        prune_set = set(
            sorted(mergeable_ids, key=lambda tid: token_counts.get(tid, 0))[:num_to_prune]
        )

        needed: set[int] = set()
        kept_merges_rev: list[tuple[int, int]] = []
        pruned_ids: set[int] = set()
        for pair in reversed(self._merges):
            merged_id = self._merge_ids[pair]
            if merged_id in prune_set and merged_id not in needed:
                pruned_ids.add(merged_id)
                continue
            kept_merges_rev.append(pair)
            for component in pair:
                if component >= 256:
                    needed.add(component)

        if not pruned_ids:
            return

        old_merge_ids = self._merge_ids
        old_merge_gears = self._merge_gears
        self._merges = list(reversed(kept_merges_rev))

        for token_id in pruned_ids:
            piece = self._vocab.pop(token_id)
            self._encoder.pop(piece, None)
            self._token_gears.pop(token_id, None)

        remaining_merged_ids = sorted(tid for tid in self._vocab if tid >= 256)
        remap = {old: 256 + i for i, old in enumerate(remaining_merged_ids)}

        def remapped(token_id: int) -> int:
            return remap.get(token_id, token_id)

        new_vocab = {i: self._vocab[i] for i in range(256)}
        new_token_gears = {i: 0 for i in range(256)}
        for old, new in remap.items():
            new_vocab[new] = self._vocab[old]
            new_token_gears[new] = self._token_gears[old]
        self._vocab = new_vocab
        self._encoder = {piece: tid for tid, piece in self._vocab.items()}
        self._token_gears = new_token_gears

        new_merges: list[tuple[int, int]] = []
        new_merge_ids: dict[tuple[int, int], int] = {}
        new_merge_ranks: dict[tuple[int, int], int] = {}
        new_merge_gears: dict[tuple[int, int], int] = {}
        for rank, pair in enumerate(self._merges):
            new_pair = (remapped(pair[0]), remapped(pair[1]))
            new_merges.append(new_pair)
            new_merge_ids[new_pair] = remapped(old_merge_ids[pair])
            new_merge_ranks[new_pair] = rank
            new_merge_gears[new_pair] = old_merge_gears[pair]
        self._merges = new_merges
        self._merge_ids = new_merge_ids
        self._merge_ranks = new_merge_ranks
        self._merge_gears = new_merge_gears

    def _build_trie(self) -> None:
        self._trie = {}
        for token_id, piece in self._vocab.items():
            node = self._trie
            for byte in piece:
                node = node.setdefault(byte, {})
            node[-1] = token_id

    def _encoding_chunks(self, text: str):
        for line in text.splitlines(keepends=True):
            data = line.encode("utf-8", errors="replace")
            for start in range(0, len(data), self.chunk_bytes):
                yield data[start:start + self.chunk_bytes]
        if text and not text.splitlines(keepends=True):
            yield text.encode("utf-8", errors="replace")

    def _encode_bpe_bytes(self, data: bytes) -> list[int]:
        return self._apply_merge_rules(list(data))

    def _matches(self, data: bytes, start: int):
        node = self._trie
        stop = min(len(data), start + self.max_token_bytes)
        for end in range(start, stop):
            node = node.get(data[end])
            if node is None:
                break
            token_id = node.get(-1)
            if token_id is not None:
                yield end + 1, token_id

    def _viterbi_encode_bytes(self, data: bytes, use_transitions: bool = True) -> list[int]:
        if not data:
            return []
        gears = len(self.gear_spans)
        inf = math.inf
        costs = [[inf] * gears for _ in range(len(data) + 1)]
        backs: list[list[tuple[int, int, int] | None]] = [
            [None] * gears for _ in range(len(data) + 1)
        ]
        for start in range(len(data)):
            at_start = start == 0
            if not at_start and all(math.isinf(value) for value in costs[start]):
                continue
            for end, token_id in self._matches(data, start):
                gear = self._token_gears[token_id]
                token_cost = self._token_costs[token_id]
                if at_start:
                    candidate_cost = token_cost
                    previous_gear = -1
                else:
                    previous_gear, previous_cost = min(
                        enumerate(costs[start]),
                        key=lambda item: (
                            item[1]
                            + (
                                self.transition_weight
                                * self._transition_costs[item[0]][gear]
                                if use_transitions
                                else 0.0
                            ),
                            item[0],
                        ),
                    )
                    candidate_cost = previous_cost + token_cost
                    if use_transitions:
                        candidate_cost += (
                            self.transition_weight
                            * self._transition_costs[previous_gear][gear]
                        )
                # Prefer a longer piece only when objective costs are effectively tied.
                candidate_cost -= 1e-12 * len(self._vocab[token_id])
                if candidate_cost < costs[end][gear]:
                    costs[end][gear] = candidate_cost
                    backs[end][gear] = (start, previous_gear, token_id)

        gear = min(range(gears), key=lambda value: (costs[len(data)][value], value))
        if math.isinf(costs[len(data)][gear]):
            raise RuntimeError("byte fallback invariant violated")
        ids: list[int] = []
        position = len(data)
        while position:
            back = backs[position][gear]
            if back is None:
                raise RuntimeError("incomplete Viterbi backtrace")
            position, previous_gear, token_id = back
            ids.append(token_id)
            gear = previous_gear
        ids.reverse()
        return ids

    def _encode_internal(
        self,
        text: str,
        inference: str,
        use_transitions: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        for data in self._encoding_chunks(text):
            if inference == "bpe":
                ids.extend(self._encode_bpe_bytes(data))
            else:
                ids.extend(self._viterbi_encode_bytes(data, use_transitions))
        return ids

    def _update_lattice_costs(self, texts: list[str], inference: str, use_transitions: bool) -> None:
        token_counts: Counter[int] = Counter()
        transition_counts = [[1.0] * 5 for _ in range(5)]
        for text in texts:
            ids = self._encode_internal(text, inference, use_transitions)
            token_counts.update(ids)
            for previous, current in zip(ids, ids[1:]):
                transition_counts[self._token_gears[previous]][self._token_gears[current]] += 1.0

        alpha = 0.1
        total = sum(token_counts.values()) + alpha * len(self._vocab)
        self._token_costs = [
            -math.log((token_counts.get(token_id, 0) + alpha) / max(total, 1.0))
            for token_id in range(len(self._vocab))
        ]
        self._transition_costs = []
        for row in transition_counts:
            row_total = sum(row)
            self._transition_costs.append([-math.log(value / row_total) for value in row])

    def train(self, texts: list[str]) -> None:
        self._vocab = {i: bytes([i]) for i in range(256)}
        self._encoder = {bytes([i]): i for i in range(256)}
        self._merges = []
        self._merge_ids = {}
        self._merge_ranks = {}
        self._merge_gears = {}
        self._token_gears = {i: 0 for i in range(256)}
        self._stage_vocab_sizes = []
        self._capped_merge_skips = 0

        fraction_total = sum(self.gear_fractions)
        cumulative = 0.0
        nonbyte_budget = self.max_vocab - 256
        for gear, fraction in enumerate(self.gear_fractions):
            cumulative += fraction / fraction_total
            target = (
                self.max_vocab
                if gear == 4
                else min(self.max_vocab, 256 + round(nonbyte_budget * cumulative))
            )
            segment_freq: Counter[tuple[int, ...]] = Counter()
            for text in texts:
                for segment, weight in self._gear_segments(text, gear):
                    if segment:
                        segment_freq[tuple(segment)] += weight
            self._train_stage(segment_freq, target, gear)
            self._stage_vocab_sizes.append(len(self._vocab))

        self._build_trie()
        self._prune_low_usage_tokens(texts, self.prune_fraction)
        self._build_trie()
        self._token_costs = [0.0] * len(self._vocab)
        self._transition_costs = [[0.0] * 5 for _ in range(5)]
        self._update_lattice_costs(texts, "bpe", use_transitions=False)
        if self.inference == "viterbi":
            for iteration in range(self.unigram_iterations):
                self._update_lattice_costs(
                    texts,
                    "viterbi",
                    use_transitions=iteration > 0,
                )
        self._initialized = True

    def encode(self, text: str) -> list[int]:
        if not self._initialized:
            raise RuntimeError("call train() before encode()")
        return self._encode_internal(text, self.inference)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._vocab.get(token_id, b"?") for token_id in ids).decode(
            "utf-8", errors="replace"
        )

    def initialize_embeddings_from_merges(self, weight) -> None:
        """Initialize learned token rows compositionally from merge-tree children.

        MultiGear's hierarchy is useful structure that a downstream model would
        otherwise discard by independently initializing every vocabulary row.
        Processing merges in rank order ensures child rows are initialized before
        their parents. Dividing by sqrt(2) preserves the variance of independent
        child rows; duplicate token constructions keep their earliest hierarchy.

        ``weight`` may include rows for wrapper-added special tokens. Those rows
        are intentionally left unchanged.
        """
        if not self._initialized:
            raise RuntimeError("call train() before initializing embeddings")
        if len(weight) < self.vocab_size:
            raise ValueError(
                f"embedding rows={len(weight)} smaller than tokenizer vocab={self.vocab_size}"
            )
        detached = weight.detach()
        initialized: set[int] = set()
        for pair in self._merges:
            token_id = self._merge_ids[pair]
            if token_id in initialized:
                continue
            detached[token_id].copy_(
                (detached[pair[0]] + detached[pair[1]]) / math.sqrt(2.0)
            )
            initialized.add(token_id)

    def token_hierarchy(self) -> dict:
        """Return canonical gear and immediate-child metadata for downstream models."""
        if not self._initialized:
            raise RuntimeError("call train() before requesting token hierarchy")
        children = [[-1, -1] for _ in range(self.vocab_size)]
        for pair in self._merges:
            token_id = self._merge_ids[pair]
            if children[token_id][0] < 0:
                children[token_id] = [pair[0], pair[1]]
        return {
            "gear_count": len(self.gear_spans),
            "token_gears": [self._token_gears[token_id] for token_id in range(self.vocab_size)],
            "token_children": children,
            "token_bytes": [list(self._vocab[token_id]) for token_id in range(self.vocab_size)],
        }

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) if self._initialized else self.max_vocab


class MultiGearPredictionAwareTokenizer(MultiGearTokenizer):
    """MultiGear tokenizer with predictability-aware lattice segmentation.

    This keeps MultiGear's staged vocabulary construction, but changes inference
    from merge-rank BPE to a Viterbi path whose token costs approximate
    downstream next-token difficulty. The default cost is model-free and fitted
    from the training text:

    * frequent pieces are cheaper;
    * very short gear-0 pieces are penalized when an alternative exists;
    * long rare pieces are penalized so compression does not create brittle
      low-frequency labels;
    * a small byte-length reward keeps the tokenizer from collapsing back to
      byte-like segmentation.

    ``set_prediction_costs`` can replace the proxy costs with costs measured
    from a trained model later. Encoding remains exact and lossless because the
    byte vocabulary is always present.
    """

    def __init__(
        self,
        max_vocab: int = 8192,
        inference: str = "prediction_aware",
        gear_spans: tuple[int, ...] = (0, 1, 2, 4, 8),
        gear_fractions: tuple[float, ...] = (0.14, 0.38, 0.22, 0.16, 0.10),
        max_token_bytes: int | None = None,
        transition_weight: float = 0.15,
        unigram_iterations: int = 2,
        chunk_bytes: int = 65536,
        prune_fraction: float = 0.0,
        prediction_alpha: float = 0.25,
        byte_reward: float = 0.28,
        gear0_penalty: float = 0.70,
        rare_threshold: int = 3,
        rare_penalty: float = 0.45,
        long_rare_penalty: float = 0.25,
        unseen_penalty: float = 0.60,
        prediction_transition_weight: float | None = None,
    ) -> None:
        if inference not in {"bpe", "viterbi", "prediction_aware"}:
            raise ValueError("inference must be 'bpe', 'viterbi', or 'prediction_aware'")
        super().__init__(
            max_vocab=max_vocab,
            inference="bpe" if inference == "prediction_aware" else inference,
            gear_spans=gear_spans,
            gear_fractions=gear_fractions,
            max_token_bytes=max_token_bytes,
            transition_weight=transition_weight,
            unigram_iterations=unigram_iterations,
            chunk_bytes=chunk_bytes,
            prune_fraction=prune_fraction,
        )
        self.inference = inference
        self.prediction_alpha = float(prediction_alpha)
        self.byte_reward = float(byte_reward)
        self.gear0_penalty = float(gear0_penalty)
        self.rare_threshold = int(rare_threshold)
        self.rare_penalty = float(rare_penalty)
        self.long_rare_penalty = float(long_rare_penalty)
        self.unseen_penalty = float(unseen_penalty)
        self.prediction_transition_weight = (
            self.transition_weight
            if prediction_transition_weight is None
            else float(prediction_transition_weight)
        )
        if self.prediction_alpha <= 0.0:
            raise ValueError("prediction_alpha must be positive")
        if self.byte_reward < 0.0:
            raise ValueError("byte_reward must be non-negative")
        if self.gear0_penalty < 0.0:
            raise ValueError("gear0_penalty must be non-negative")
        if self.rare_threshold < 0:
            raise ValueError("rare_threshold must be non-negative")
        if self.rare_penalty < 0.0:
            raise ValueError("rare_penalty must be non-negative")
        if self.long_rare_penalty < 0.0:
            raise ValueError("long_rare_penalty must be non-negative")
        if self.unseen_penalty < 0.0:
            raise ValueError("unseen_penalty must be non-negative")
        if self.prediction_transition_weight < 0.0:
            raise ValueError("prediction_transition_weight must be non-negative")
        self._prediction_costs: list[float] = []
        self._prediction_token_counts: list[int] = []

    def _fit_prediction_costs(self, texts: list[str]) -> None:
        token_counts: Counter[int] = Counter()
        for text in texts:
            token_counts.update(super()._encode_internal(text, "bpe", use_transitions=False))

        vocab_size = len(self._vocab)
        alpha = self.prediction_alpha
        total = sum(token_counts.values()) + alpha * vocab_size
        rare_threshold = max(0, self.rare_threshold)
        costs: list[float] = []
        counts: list[int] = []
        for token_id in range(vocab_size):
            count = int(token_counts.get(token_id, 0))
            counts.append(count)
            piece = self._vocab[token_id]
            byte_len = max(1, len(piece))
            gear = self._token_gears[token_id]
            cost = -math.log((count + alpha) / max(total, 1.0))
            cost -= self.byte_reward * math.log1p(byte_len)
            if gear == 0:
                cost += self.gear0_penalty / math.sqrt(byte_len)
            if rare_threshold > 0 and count < rare_threshold:
                rarity = (rare_threshold - count) / rare_threshold
                cost += self.rare_penalty * rarity
                if byte_len >= 4:
                    cost += self.long_rare_penalty * rarity * math.log(byte_len)
            if count == 0:
                cost += self.unseen_penalty
            # Raw bytes are the lossless fallback. Keep them available but do
            # not let smoothing make unseen learned pieces cheaper than bytes.
            if token_id < 256:
                cost = min(cost, -math.log((count + alpha) / max(total, 1.0)) + self.gear0_penalty)
            costs.append(cost)
        self._prediction_costs = costs
        self._prediction_token_counts = counts

    def set_prediction_costs(self, costs: list[float]) -> None:
        """Install externally measured token prediction costs.

        This is the calibration hook for a later two-pass tokenizer/model loop:
        train a small model, estimate per-token NLL, then retokenize with those
        costs while keeping the same vocabulary.
        """
        if not self._initialized:
            raise RuntimeError("call train() before setting prediction costs")
        if len(costs) != len(self._vocab):
            raise ValueError(f"expected {len(self._vocab)} costs, got {len(costs)}")
        if any(not math.isfinite(float(value)) for value in costs):
            raise ValueError("prediction costs must be finite")
        self._prediction_costs = [float(value) for value in costs]

    def train(self, texts: list[str]) -> None:
        requested_inference = self.inference
        self.inference = "bpe" if requested_inference == "prediction_aware" else requested_inference
        super().train(texts)
        self._fit_prediction_costs(texts)
        self.inference = requested_inference
        self._initialized = True

    def _prediction_aware_encode_bytes(self, data: bytes, use_transitions: bool = True) -> list[int]:
        if not data:
            return []
        if not self._prediction_costs:
            raise RuntimeError("prediction costs are not initialized")
        gears = len(self.gear_spans)
        inf = math.inf
        costs = [[inf] * gears for _ in range(len(data) + 1)]
        backs: list[list[tuple[int, int, int] | None]] = [
            [None] * gears for _ in range(len(data) + 1)
        ]
        for start in range(len(data)):
            at_start = start == 0
            if not at_start and all(math.isinf(value) for value in costs[start]):
                continue
            for end, token_id in self._matches(data, start):
                gear = self._token_gears[token_id]
                token_cost = self._prediction_costs[token_id]
                if at_start:
                    candidate_cost = token_cost
                    previous_gear = -1
                else:
                    previous_gear, previous_cost = min(
                        enumerate(costs[start]),
                        key=lambda item: (
                            item[1]
                            + (
                                self.prediction_transition_weight
                                * self._transition_costs[item[0]][gear]
                                if use_transitions
                                else 0.0
                            ),
                            item[0],
                        ),
                    )
                    candidate_cost = previous_cost + token_cost
                    if use_transitions:
                        candidate_cost += (
                            self.prediction_transition_weight
                            * self._transition_costs[previous_gear][gear]
                        )
                candidate_cost -= 1e-12 * len(self._vocab[token_id])
                if candidate_cost < costs[end][gear]:
                    costs[end][gear] = candidate_cost
                    backs[end][gear] = (start, previous_gear, token_id)

        gear = min(range(gears), key=lambda value: (costs[len(data)][value], value))
        if math.isinf(costs[len(data)][gear]):
            raise RuntimeError("byte fallback invariant violated")
        ids: list[int] = []
        position = len(data)
        while position:
            back = backs[position][gear]
            if back is None:
                raise RuntimeError("incomplete prediction-aware backtrace")
            position, previous_gear, token_id = back
            ids.append(token_id)
            gear = previous_gear
        ids.reverse()
        return ids

    def _encode_internal(
        self,
        text: str,
        inference: str,
        use_transitions: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        for data in self._encoding_chunks(text):
            if inference == "bpe":
                ids.extend(self._encode_bpe_bytes(data))
            elif inference == "viterbi":
                ids.extend(self._viterbi_encode_bytes(data, use_transitions))
            elif inference == "prediction_aware":
                ids.extend(self._prediction_aware_encode_bytes(data, use_transitions))
            else:
                raise ValueError(f"unknown inference mode {inference!r}")
        return ids

    def encode(self, text: str) -> list[int]:
        if not self._initialized:
            raise RuntimeError("call train() before encode()")
        return self._encode_internal(text, self.inference)


# Keep a misspelled compatibility alias because the experiment/request used
# this spelling in several notes. New code should use the correctly spelled
# ``MultiGearPredictionAwareTokenizer``.
MultiGearPeridictionAwareToeknizer = MultiGearPredictionAwareTokenizer


def build_bpe_tokenizer(max_vocab: int):
    """Prefer the Rust engine; fall back to pure-Python BPE."""
    try:
        import tokenizers  # noqa: F401

        return FastBPETokenizer(max_vocab=max_vocab)
    except ImportError:
        return ByteBPETokenizer(max_vocab=max_vocab)


DEFAULT_SPECIAL_TOKENS = (
    "<|bos|>", "<|eos|>", "<|pad|>",
    "<|context|>", "<|question|>", "<|answer|>", "<|no_answer|>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end_turn|>",
    "<|tool_call|>", "<|tool_result|>",
)


@dataclass
class SpecialTokenTokenizer:
    """Reserve stable ids above the base vocabulary for control tokens."""

    base: object
    special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS

    def __post_init__(self) -> None:
        start = int(self.base.vocab_size)
        self.special_to_id = {t: start + i for i, t in enumerate(self.special_tokens)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return int(self.base.vocab_size) + len(self.special_tokens)

    def encode(self, text: str) -> list[int]:
        parts: list[int] = []
        remaining = text
        while remaining:
            matches = [(remaining.find(t), t) for t in self.special_tokens if remaining.find(t) >= 0]
            if not matches:
                parts.extend(self.base.encode(remaining))
                break
            position, token = min(matches)
            if position:
                parts.extend(self.base.encode(remaining[:position]))
            parts.append(self.special_to_id[token])
            remaining = remaining[position + len(token):]
        return parts

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        regular: list[int] = []
        for tid in ids:
            if tid in self.id_to_special:
                if regular:
                    out.append(self.base.decode(regular))
                    regular = []
                out.append(self.id_to_special[tid])
            else:
                regular.append(tid)
        if regular:
            out.append(self.base.decode(regular))
        return "".join(out)

    def initialize_embeddings_from_merges(self, weight) -> None:
        initializer = getattr(self.base, "initialize_embeddings_from_merges", None)
        if initializer is None:
            raise TypeError(f"{type(self.base).__name__} has no merge-tree initializer")
        initializer(weight)

    def token_hierarchy(self) -> dict:
        hierarchy = getattr(self.base, "token_hierarchy", None)
        if hierarchy is None:
            raise TypeError(f"{type(self.base).__name__} has no token hierarchy")
        metadata = hierarchy()
        special_gear = int(metadata["gear_count"])
        special_count = len(self.special_tokens)
        return {
            "gear_count": special_gear + 1,
            "token_gears": list(metadata["token_gears"]) + [special_gear] * special_count,
            "token_children": list(metadata["token_children"]) + [[-1, -1]] * special_count,
            "token_bytes": list(metadata["token_bytes"]) + [[] for _ in range(special_count)],
        }
