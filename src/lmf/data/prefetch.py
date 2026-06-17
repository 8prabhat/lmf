"""Background batch prefetch (review S7).

CPU-side sampling and the host->device copy serialise with compute. A single
one-batch-ahead prefetch thread overlaps them with the previous step's backward
pass. Unified memory makes the copy itself cheap; the win is removing the Python
sampling latency from the critical path.

The prefetcher is transparent: it wraps any object exposing ``sample_batch`` (or
``sample_tokenized``) and forwards every other attribute, so it is a drop-in for
a raw corpus.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

import torch

from .batch import TrainingBatch, lm_batch


def _sample(corpus: Any, batch: int, seq_len: int, split: str) -> TrainingBatch:
    if hasattr(corpus, "sample_batch"):
        return corpus.sample_batch(batch, seq_len, split)
    return lm_batch(corpus.sample_tokenized(batch, seq_len, split))


class PrefetchCorpus:
    """One-batch-ahead async prefetch wrapper for a fixed (batch, seq_len, split)."""

    def __init__(self, corpus: Any, batch: int, seq_len: int, split: str = "train",
                 device: torch.device | None = None, depth: int = 2) -> None:
        self._corpus = corpus
        self._batch = batch
        self._seq_len = seq_len
        self._split = split
        self._device = device
        self._q: "queue.Queue[TrainingBatch]" = queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                b = _sample(self._corpus, self._batch, self._seq_len, self._split)
                if self._device is not None:
                    b = b.to(self._device)
                self._q.put(b, timeout=1.0)
            except queue.Full:
                continue
            except Exception:
                # Surface sampling errors on the main thread's next() instead of
                # dying silently in the background.
                self._stop.set()
                raise

    def next(self) -> TrainingBatch:
        return self._q.get()

    def close(self) -> None:
        self._stop.set()

    def __getattr__(self, name: str) -> Any:
        # Forward fingerprint/vocab/tokenizer access to the wrapped corpus.
        return getattr(self._corpus, name)
