"""Contiguous document-lane corpus for stateful recurrent training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ..core.registry import CORPORA
from .batch import TrainingBatch
from .corpora import EduCombinedCorpus
from .pure_gear import _domain_probabilities, _load_documents
from .sentence_boundaries import SentenceBoundaryDetector


@CORPORA.register("pure_gear_contiguous_lanes")
class ContiguousDocumentLaneCorpus:
    """Yield deterministic per-lane continuations across successive batches.

    A lane may cross a document boundary inside a row. Every new document
    receives a fresh segment id, so carried model state resets exactly at that
    boundary while remaining continuous between adjacent windows of one
    document.
    """

    def __init__(
        self,
        corpus_root: str,
        index_root: str,
        tokenizer_name: str,
        domains: Iterable[str],
        *,
        seed: int = 0,
        max_sentence_tokens: int = 128,
        whole_windows: bool = False,
    ) -> None:
        self.corpus_root = Path(corpus_root).expanduser()
        self.index_root = Path(index_root).expanduser()
        self.tokenizer_name = str(tokenizer_name)
        self.domains = tuple(str(value) for value in domains)
        self.seed = int(seed)
        self.whole_windows = bool(whole_windows)
        self._documents = _load_documents(
            self.corpus_root,
            self.index_root,
            self.tokenizer_name,
            self.domains,
        )
        if not any(len(domain.train_documents) for domain in self._documents):
            raise ValueError("contiguous lane corpus has no training documents")
        self._domain_probabilities = _domain_probabilities(self._documents)
        loader = EduCombinedCorpus(
            root=str(self.corpus_root),
            tokenizer_name=self.tokenizer_name,
            domains=list(self.domains),
            seed=seed,
            load_tokenizer=True,
        )
        self.tokenizer = loader.tokenizer
        self.vocab_size = loader.vocab_size
        self.detector = SentenceBoundaryDetector(
            self.tokenizer,
            max_sentence_tokens=max_sentence_tokens,
        )
        self._rng = np.random.default_rng(self.seed)
        self._lanes: list[dict[str, int]] = []
        self._next_segment_id = 0

    def _new_document(self) -> dict[str, int]:
        domain_index = int(
            self._rng.choice(
                len(self._documents),
                p=self._domain_probabilities,
            )
        )
        domain = self._documents[domain_index]
        if len(domain.train_documents) == 0:
            available = [
                index
                for index, candidate in enumerate(self._documents)
                if len(candidate.train_documents)
            ]
            domain_index = available[int(self._rng.integers(len(available)))]
            domain = self._documents[domain_index]
        selected = int(self._rng.integers(len(domain.train_documents)))
        document = int(domain.train_documents[selected])
        segment_id = self._next_segment_id
        self._next_segment_id += 1
        return {
            "domain": domain_index,
            "document": document,
            "offset": 0,
            "segment_id": segment_id,
        }

    def _ensure_lanes(self, batch: int) -> None:
        if len(self._lanes) != batch:
            self._lanes = [self._new_document() for _ in range(batch)]

    def _lane_row(
        self,
        lane_index: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        lane = self._lanes[lane_index]
        initial_segment = lane["segment_id"]
        initial_offset = lane["offset"]
        if self.whole_windows:
            for _ in range(10_000):
                domain = self._documents[lane["domain"]]
                start = int(domain.starts[lane["document"]])
                end = int(domain.ends[lane["document"]])
                position = start + lane["offset"]
                if end - position >= seq_len:
                    values = np.asarray(
                        domain.tokens[position : position + seq_len],
                        dtype=np.int64,
                    ).tolist()
                    lane["offset"] += seq_len
                    tokens = [int(value) for value in values]
                    segments = [lane["segment_id"]] * seq_len
                    token_tensor = torch.tensor(tokens, dtype=torch.long)
                    segment_tensor = torch.tensor(
                        segments, dtype=torch.long
                    )
                    _, sentence_end, _ = self.detector.scan_tokens(
                        tokens, segments
                    )
                    loss = torch.ones(seq_len, dtype=torch.bool)
                    loss[0] = False
                    continuation = (
                        lane["segment_id"] == initial_segment
                        and initial_offset > 0
                    )
                    return (
                        token_tensor,
                        segment_tensor,
                        sentence_end,
                        loss,
                        torch.tensor(continuation, dtype=torch.bool),
                    )
                lane = self._new_document()
                self._lanes[lane_index] = lane
            raise RuntimeError(
                "unable to find a document long enough for a whole window"
            )
        tokens: list[int] = []
        segments: list[int] = []
        while len(tokens) < seq_len:
            domain = self._documents[lane["domain"]]
            start = int(domain.starts[lane["document"]])
            end = int(domain.ends[lane["document"]])
            position = start + lane["offset"]
            remaining = end - position
            if remaining <= 0:
                lane = self._new_document()
                self._lanes[lane_index] = lane
                continue
            take = min(seq_len - len(tokens), remaining)
            values = np.asarray(
                domain.tokens[position : position + take],
                dtype=np.int64,
            ).tolist()
            tokens.extend(int(value) for value in values)
            segments.extend([lane["segment_id"]] * take)
            lane["offset"] += take
            if lane["offset"] >= end - start and len(tokens) < seq_len:
                lane = self._new_document()
                self._lanes[lane_index] = lane
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        segment_tensor = torch.tensor(segments, dtype=torch.long)
        _, sentence_end, _ = self.detector.scan_tokens(tokens, segments)
        loss = torch.ones(seq_len, dtype=torch.bool)
        loss[0] = False
        loss[1:] &= segment_tensor[1:] == segment_tensor[:-1]
        continuation = segments[0] == initial_segment and initial_offset > 0
        return (
            token_tensor,
            segment_tensor,
            sentence_end,
            loss,
            torch.tensor(continuation, dtype=torch.bool),
        )

    def sample_batch(
        self,
        batch: int,
        seq_len: int,
        split: str = "train",
    ) -> TrainingBatch:
        if split != "train":
            raise ValueError("contiguous lanes are a training-only corpus")
        batch = int(batch)
        seq_len = int(seq_len)
        self._ensure_lanes(batch)
        rows = [self._lane_row(index, seq_len) for index in range(batch)]
        tokens, segments, sentence_end, loss, continuation = (
            torch.stack(values) for values in zip(*rows)
        )
        attention = torch.ones_like(tokens, dtype=torch.bool)
        return TrainingBatch(
            tokens,
            attention,
            loss,
            metadata={
                "segment_ids": segments,
                "sentence_end_mask": sentence_end,
                "lane_ids": torch.arange(batch),
                "lane_continuation": continuation,
                "contiguous_lanes": True,
                "single_segment_rows": self.whole_windows,
            },
        )

    def sample_tokenized(
        self,
        batch: int,
        seq_len: int,
        split: str = "train",
    ) -> torch.Tensor:
        return self.sample_batch(batch, seq_len, split).tokens

    def sampler_state(self) -> dict:
        return {
            "rng": self._rng.bit_generator.state,
            "lanes": [dict(lane) for lane in self._lanes],
            "next_segment_id": self._next_segment_id,
        }

    def load_sampler_state(self, state: dict) -> None:
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state["rng"]
        self._lanes = [dict(lane) for lane in state["lanes"]]
        self._next_segment_id = int(state["next_segment_id"])
