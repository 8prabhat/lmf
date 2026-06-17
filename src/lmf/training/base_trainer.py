"""The one optimization loop every family reuses (DRY + Dependency Inversion).

``BaseTrainer`` depends only on the ``Trainable`` contract (a ``training_step``
returning a dict with a ``"total"`` loss). It owns the optimizer, LR schedule,
precision policy, MPS memory governor, gradient clipping, logging, and the
callback hooks. Concrete families subclass only to override two small hooks:

* ``batch_metadata(step)`` — per-step training metadata (e.g. a scheduled-sampling
  ramp), passed through to ``training_step``;
* ``evaluate_bpt(...)`` — the family's held-out metric (RHCA and the transformer
  score BPT differently, so this is the natural extension point).
"""

from __future__ import annotations

import inspect
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from ..core.device import MemoryGovernor, PrecisionPolicy, resolve_device
from ..core.seeding import capture_rng_state, restore_rng_state
from ..data.batch import TrainingBatch, lm_batch
from ..data.corpora import corpus_fingerprint, tokenizer_fingerprint
from ..data.prefetch import PrefetchCorpus
from .checkpoints import load_checkpoint, save_checkpoint


class BaseTrainer:
    def __init__(self, model, corpus, *, device: str = "auto", precision: str = "bf16",
                 lr: float = 3e-4, warmup_steps: int = 200, total_steps: int = 20000,
                 weight_decay: float = 0.01, grad_accum_steps: int = 1,
                 memory_evict_every: int = 50, prefetch: bool = False,
                 batch_size: int | None = None, seq_len: int | None = None) -> None:
        self.device = resolve_device(device)
        self.precision_policy = PrecisionPolicy(precision)
        self.raw_model = self.precision_policy.cast_model(model, self.device)
        self.model = self.raw_model
        self.corpus = corpus
        self.lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.optimizer = torch.optim.AdamW(
            self.raw_model.parameters(), lr=lr, weight_decay=weight_decay,
            fused=self.device.type == "cuda")
        self.memory = MemoryGovernor(self.device, every=memory_evict_every)
        self.step = 0
        self._supports_loss_term_scales = "loss_term_scales" in inspect.signature(
            self.raw_model.training_step).parameters
        self.loss_term_scales: dict[str, float] | None = None
        self._prefetcher: PrefetchCorpus | None = None
        if prefetch and batch_size and seq_len:
            self._prefetcher = PrefetchCorpus(corpus, batch_size, seq_len, "train", self.device)

    # ---- overridable family hooks -------------------------------------------
    def batch_metadata(self, step: int) -> dict[str, Any]:
        return {}

    def _metric_bpt(self, batch_size, seq_len, n_batches, split) -> float:
        from ..evaluation.metrics import bits_per_token
        return bits_per_token(self.raw_model, self.corpus, batch_size, seq_len, n_batches, split)

    def evaluate_bpt(self, batch_size: int, seq_len: int, n_batches: int = 10,
                     split: str = "valid") -> float:
        # Freeze RNG + corpus sampler so a mid-training eval never perturbs the
        # training data stream or global RNG (review finding 4).
        with self.frozen_sampling():
            return self._metric_bpt(batch_size, seq_len, n_batches, split)

    @contextmanager
    def frozen_sampling(self):
        """Snapshot/restore global RNG and the corpus sampler around a side-task."""
        rng = capture_rng_state(self.device)
        sampler = (self.corpus.sampler_state()
                   if hasattr(self.corpus, "sampler_state") else None)
        was_training = self.model.training
        try:
            yield
        finally:
            restore_rng_state(rng)
            if sampler is not None and hasattr(self.corpus, "load_sampler_state"):
                self.corpus.load_sampler_state(sampler)
            self.model.train(was_training)

    # ---- internals ----------------------------------------------------------
    def _lr(self) -> float:
        if self.step < self.warmup_steps:
            return self.lr * (self.step + 1) / max(1, self.warmup_steps)
        progress = (self.step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def _sample_batch(self, batch_size: int, seq_len: int) -> TrainingBatch:
        if self._prefetcher is not None:
            return self._prefetcher.next()
        if hasattr(self.corpus, "sample_batch"):
            return self.corpus.sample_batch(batch_size, seq_len, "train").to(self.device)
        return lm_batch(self.corpus.sample_tokenized(batch_size, seq_len, "train")).to(self.device)

    # ---- training loop ------------------------------------------------------
    def train_steps(self, n_steps: int, batch_size: int, seq_len: int,
                    log_every: int = 50, callbacks: list | None = None) -> list[dict[str, float]]:
        self.model.train()
        callbacks = callbacks or []
        records: list[dict[str, float]] = []
        started = time.perf_counter()
        for _ in range(n_steps):
            for group in self.optimizer.param_groups:
                group["lr"] = self._lr()
            self.optimizer.zero_grad(set_to_none=True)
            meta = self.batch_metadata(self.step)
            accum: dict[str, float] = {}
            for _ in range(self.grad_accum_steps):
                batch = self._sample_batch(batch_size, seq_len)
                step_meta = {
                    **meta,
                    **batch.metadata,
                    "loss_mask": batch.loss_mask,
                    "attention_mask": batch.attention_mask,
                }
                with self.precision_policy.autocast(self.device):
                    if self._supports_loss_term_scales:
                        losses = self.model.training_step(
                            batch.tokens, step_meta, loss_term_scales=self.loss_term_scales)
                    else:
                        losses = self.model.training_step(batch.tokens, step_meta)
                    total = losses["total"]
                (total / self.grad_accum_steps).backward()
                for k, v in losses.items():
                    if torch.is_tensor(v):
                        accum[k] = accum.get(k, 0.0) + float(v.detach()) / self.grad_accum_steps
            if not math.isfinite(accum.get("total", 0.0)):
                raise FloatingPointError(f"non-finite loss at step {self.step}: {accum}")
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), 1.0)
            self.optimizer.step()
            self.step += 1
            self.memory.maybe_evict(self.step)
            record = {**accum, "lr": self._lr(), "step": self.step}
            for cb in callbacks:
                cb.on_step_end(self, self.step, record)
            records.append(record)
            if log_every and self.step % log_every == 0:
                elapsed = time.perf_counter() - started
                parts = "  ".join(f"{k}={accum[k]:.4f}" for k in
                                  ("commit_token", "routing_balance") if k in accum)
                print(f"step {self.step:>6}/{self.total_steps}  total={accum.get('total', 0):.4f}"
                      f"  {parts}  {self.step / max(elapsed, 1e-9):.2f} it/s", flush=True)
        return records

    # ---- checkpoint passthroughs --------------------------------------------
    def _fingerprints(self) -> dict:
        fp = {"corpus": corpus_fingerprint(self.corpus)}
        tok = getattr(self.corpus, "tokenizer", None)
        if tok is not None:
            fp["tokenizer"] = tokenizer_fingerprint(tok)
        return fp

    def save_checkpoint(self, path: str | Path) -> None:
        save_checkpoint(
            path, self.raw_model, self.optimizer, self.step,
            rng=capture_rng_state(self.device),
            sampler_state=(self.corpus.sampler_state()
                           if hasattr(self.corpus, "sampler_state") else None),
            fingerprints=self._fingerprints())

    def load_checkpoint(self, path: str | Path, strict: bool = True,
                        resume_rng: bool = True) -> dict:
        ckpt = load_checkpoint(path, self.raw_model, self.optimizer, self.device, strict,
                               expected_fingerprints=self._fingerprints() if strict else None)
        self.step = int(ckpt["step"])
        if resume_rng:
            if ckpt.get("rng") is not None:
                restore_rng_state(ckpt["rng"])
            if (ckpt.get("sampler_state") is not None
                    and hasattr(self.corpus, "load_sampler_state")):
                self.corpus.load_sampler_state(ckpt["sampler_state"])
        return ckpt
