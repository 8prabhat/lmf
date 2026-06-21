"""Frozen sentence-boundary policy shared by data preparation and generation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import torch


BOUNDARY_DETECTOR_VERSION = "pure-gear-sentence-boundary-v1"
_CLOSERS = "\"'”’)]}"
_ABBREVIATIONS = frozenset(
    {
        "a.m.",
        "p.m.",
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "fig.",
        "eq.",
        "no.",
        "inc.",
        "ltd.",
        "u.s.",
        "u.k.",
    }
)


def boundary_detector_hash(max_sentence_tokens: int = 128) -> str:
    payload = {
        "version": BOUNDARY_DETECTOR_VERSION,
        "max_sentence_tokens": int(max_sentence_tokens),
        "abbreviations": sorted(_ABBREVIATIONS),
        "closers": _CLOSERS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


@dataclass
class SentenceBoundaryDetector:
    tokenizer: Any
    max_sentence_tokens: int = 128
    eos_id: int | None = None

    def __post_init__(self) -> None:
        self.max_sentence_tokens = int(self.max_sentence_tokens)
        if self.max_sentence_tokens < 2:
            raise ValueError("max_sentence_tokens must be at least 2")
        if self.eos_id is None:
            special = getattr(self.tokenizer, "special_to_id", {})
            self.eos_id = special.get("<|eos|>")

    @property
    def version(self) -> str:
        return BOUNDARY_DETECTOR_VERSION

    @property
    def fingerprint(self) -> str:
        return boundary_detector_hash(self.max_sentence_tokens)

    def _decode(self, token_ids: list[int]) -> str:
        try:
            return str(self.tokenizer.decode([int(value) for value in token_ids]))
        except Exception:
            return ""

    @staticmethod
    def _terminal_text(text: str) -> bool:
        stripped = text.rstrip()
        while stripped and stripped[-1] in _CLOSERS:
            stripped = stripped[:-1].rstrip()
        if not stripped or stripped[-1] not in ".!?":
            return False
        if stripped[-1] in "!?":
            return True
        lowered = stripped.lower()
        if any(lowered.endswith(value) for value in _ABBREVIATIONS):
            return False
        if re.search(r"\d\.\d+$", stripped):
            return False
        if re.search(r"(?:^|\s)[A-Z]\.$", stripped):
            return False
        return True

    def classify(self, token_ids: list[int]) -> tuple[bool, bool]:
        """Return (is_boundary, is_forced) for one current sentence."""
        if not token_ids:
            return False, False
        if self.eos_id is not None and int(token_ids[-1]) == int(self.eos_id):
            return True, False
        if self._terminal_text(self._decode(token_ids)):
            return True, False
        forced = len(token_ids) >= self.max_sentence_tokens
        return forced, forced

    def is_boundary(self, token_ids: list[int]) -> bool:
        return self.classify(token_ids)[0]

    def classify_incremental(
        self,
        token_ids: list[int],
        *,
        tail_tokens: int = 64,
    ) -> tuple[bool, bool]:
        """Classify generation state with bounded, context-independent work.

        Boundary rules depend only on EOS, token count, and the decoded text
        suffix. Training still uses the full frozen scan; incremental decoding
        examines a conservative 64-token suffix instead of repeatedly decoding
        the entire sentence.
        """
        if not token_ids:
            return False, False
        if self.eos_id is not None and int(token_ids[-1]) == int(self.eos_id):
            return True, False
        if self._terminal_text(self._decode(token_ids[-tail_tokens:])):
            return True, False
        forced = len(token_ids) >= self.max_sentence_tokens
        return forced, forced

    def is_boundary_incremental(self, token_ids: list[int]) -> bool:
        return self.classify_incremental(token_ids)[0]

    def scan_tokens(
        self,
        token_ids: list[int],
        segment_ids: list[int] | None = None,
        *,
        close_final: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = len(token_ids)
        if segment_ids is None:
            segment_ids = [0] * count
        if len(segment_ids) != count:
            raise ValueError("segment_ids must match token_ids")
        sentence_ids = torch.full((count,), -1, dtype=torch.long)
        sentence_end = torch.zeros(count, dtype=torch.bool)
        forced = torch.zeros(count, dtype=torch.bool)
        current: list[int] = []
        sentence = 0
        previous_segment = None
        for index, (token_id, segment) in enumerate(
            zip(token_ids, segment_ids)
        ):
            if segment < 0:
                continue
            if previous_segment is not None and segment != previous_segment:
                if current:
                    sentence_end[index - 1] = True
                current = []
                sentence += 1
            current.append(int(token_id))
            sentence_ids[index] = sentence
            is_boundary, is_forced = self.classify(current)
            if is_boundary:
                sentence_end[index] = True
                forced[index] = is_forced
                current = []
                sentence += 1
            previous_segment = segment
        if close_final and count and current:
            # A document/packed-span boundary is always a mechanical boundary.
            sentence_end[count - 1] = True
        return sentence_ids, sentence_end, forced

    def trailing_sentence(self, token_ids: list[int]) -> list[int]:
        current: list[int] = []
        for token_id in token_ids:
            current.append(int(token_id))
            if self.is_boundary(current):
                current = []
        return current
