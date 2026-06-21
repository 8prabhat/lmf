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
                 batch_size: int | None = None, seq_len: int | None = None,
                 betas: tuple[float, float] = (0.9, 0.999),
                 schedule_mode: str = "steps",
                 warmup_tokens: int | None = None,
                 total_training_tokens: int | None = None,
                 warmup_seconds: float | None = None,
                 total_seconds: float | None = None) -> None:
        self.device = resolve_device(device)
        self.precision_policy = PrecisionPolicy(precision)
        self.raw_model = self.precision_policy.cast_model(model, self.device)
        self.model = self.raw_model
        self.corpus = corpus
        self.lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.schedule_mode = str(schedule_mode)
        if self.schedule_mode not in {"steps", "tokens", "time"}:
            raise ValueError("schedule_mode must be 'steps', 'tokens', or 'time'")
        self.total_training_tokens = (
            None
            if total_training_tokens is None
            else int(total_training_tokens)
        )
        self.warmup_tokens = (
            int(warmup_tokens)
            if warmup_tokens is not None
            else (
                max(1, self.total_training_tokens // 10)
                if self.total_training_tokens is not None
                else None
            )
        )
        self.total_seconds = (
            None if total_seconds is None else float(total_seconds)
        )
        self.warmup_seconds = (
            float(warmup_seconds)
            if warmup_seconds is not None
            else (
                0.1 * self.total_seconds
                if self.total_seconds is not None
                else None
            )
        )
        if self.schedule_mode == "tokens" and not self.total_training_tokens:
            raise ValueError("token schedule requires total_training_tokens")
        if self.schedule_mode == "time" and not self.total_seconds:
            raise ValueError("time schedule requires total_seconds")
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.optimizer = torch.optim.AdamW(
            self.optimizer_param_groups(self.raw_model, weight_decay),
            lr=lr, weight_decay=weight_decay,
            betas=(float(betas[0]), float(betas[1])),
            fused=self.device.type == "cuda")
        self.memory = MemoryGovernor(self.device, every=memory_evict_every)
        self.step = 0
        self.tokens_seen = 0
        self.supervised_tokens_seen = 0
        self.optimization_seconds = 0.0
        self._wall_clock_origin: float | None = None
        self._supports_loss_term_scales = "loss_term_scales" in inspect.signature(
            self.raw_model.training_step).parameters
        self.loss_term_scales: dict[str, float] | None = None
        self._prefetcher: PrefetchCorpus | None = None
        if prefetch and batch_size and seq_len:
            self._prefetcher = PrefetchCorpus(corpus, batch_size, seq_len, "train", self.device)

    # ---- overridable family hooks -------------------------------------------
    def batch_metadata(self, step: int) -> dict[str, Any]:
        return {}

    def effective_seq_len(self, requested_seq_len: int, step: int) -> int:
        """Hook for family-specific sequence-length curricula."""
        return int(requested_seq_len)

    def param_group_lr_multiplier(self, group: dict, step: int) -> float:
        """Hook for staged optimization of named parameter groups."""
        return float(group.get("lr_multiplier", 1.0))

    def optimizer_param_groups(self, model, weight_decay: float):
        """Hook for family-specific decay and learning-rate groups."""
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]

    def clip_gradients(self):
        """Hook for family-specific clipping before the global safety bound."""
        return torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), 1.0)

    def validate_step_metrics(self, metrics: dict[str, float]) -> None:
        """Hook for architecture-specific fail-fast stability checks."""
        return None

    def checkpoint_metadata(self) -> dict[str, Any]:
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

    def evaluate_lm_metrics(
        self,
        batch_size: int,
        seq_len: int,
        n_batches: int = 10,
        split: str = "valid",
    ) -> dict[str, float]:
        """Return architecture-appropriate LM metrics without moving RNG streams."""
        from ..evaluation.metrics import lm_metrics

        with self.frozen_sampling():
            return lm_metrics(
                self.raw_model,
                self.corpus,
                batch_size,
                seq_len,
                n_batches,
                split,
            )

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
        if self.schedule_mode == "tokens":
            current = float(self.supervised_tokens_seen)
            warmup = float(self.warmup_tokens or 0)
            total = float(self.total_training_tokens or 1)
        elif self.schedule_mode == "time":
            # Validation, checkpointing, logging and pauses must not consume a
            # training-time budget or advance the LR schedule.
            current = self.optimization_seconds
            warmup = float(self.warmup_seconds or 0.0)
            total = float(self.total_seconds or 1.0)
        else:
            current = float(self.step)
            warmup = float(self.warmup_steps)
            total = float(self.total_steps)
        if current < warmup:
            return self.lr * (current + 1.0) / max(1.0, warmup)
        progress = (current - warmup) / max(1.0, total - warmup)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def _sample_batch(self, batch_size: int, seq_len: int) -> TrainingBatch:
        if self._prefetcher is not None:
            return self._prefetcher.next()
        if hasattr(self.corpus, "sample_batch"):
            return self.corpus.sample_batch(batch_size, seq_len, "train").to(self.device)
        return lm_batch(self.corpus.sample_tokenized(batch_size, seq_len, "train")).to(self.device)

    # ---- training loop ------------------------------------------------------
    def train_steps(self, n_steps: int, batch_size: int, seq_len: int,
                    log_every: int = 50, callbacks: list | None = None,
                    max_seconds: float | None = None) -> list[dict[str, float]]:
        self.model.train()
        callbacks = callbacks or []
        records: list[dict[str, float]] = []
        started = time.perf_counter()
        optimization_origin = self.optimization_seconds
        if self._wall_clock_origin is None:
            self._wall_clock_origin = started
        for _ in range(n_steps):
            if (
                max_seconds is not None
                and self.optimization_seconds - optimization_origin
                >= float(max_seconds)
            ):
                break
            step_started = time.perf_counter()
            base_lr = self._lr()
            for group in self.optimizer.param_groups:
                group["lr"] = base_lr * self.param_group_lr_multiplier(
                    group,
                    self.step,
                )
            self.optimizer.zero_grad(set_to_none=True)
            meta = self.batch_metadata(self.step)
            accum: dict[str, float] = {}
            for _ in range(self.grad_accum_steps):
                batch = self._sample_batch(
                    batch_size,
                    self.effective_seq_len(seq_len, self.step),
                )
                self.tokens_seen += int(batch.tokens.numel())
                supervised = (
                    batch.loss_mask[:, 1:].bool()
                    & batch.attention_mask[:, 1:].bool()
                )
                self.supervised_tokens_seen += int(supervised.sum())
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
            self.validate_step_metrics(accum)
            # A zero-LR group is semantically frozen. Clearing its gradients also
            # prevents Adam moments from accumulating behind the freeze and
            # causing an unadvertised jump when the group is enabled.
            for group in self.optimizer.param_groups:
                if float(group["lr"]) == 0.0:
                    for parameter in group["params"]:
                        parameter.grad = None
            grad_norm = self.clip_gradients()
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    f"non-finite gradient norm at step {self.step}: {grad_norm}"
                )
            self.optimizer.step()
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
            self.optimization_seconds += time.perf_counter() - step_started
            self.step += 1
            self.memory.maybe_evict(self.step)
            record = {
                **accum,
                "lr": base_lr,
                "grad_norm": float(grad_norm),
                "step": self.step,
                "tokens_seen": self.tokens_seen,
                "supervised_tokens_seen": self.supervised_tokens_seen,
            }
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
            extra={
                "trainer_tokens_seen": self.tokens_seen,
                "trainer_supervised_tokens_seen": self.supervised_tokens_seen,
                "trainer_optimization_seconds": self.optimization_seconds,
                **self.checkpoint_metadata(),
            },
            rng=capture_rng_state(self.device),
            sampler_state=(self.corpus.sampler_state()
                           if hasattr(self.corpus, "sampler_state") else None),
            fingerprints=self._fingerprints())

    def load_checkpoint(self, path: str | Path, strict: bool = True,
                        resume_rng: bool = True) -> dict:
        ckpt = load_checkpoint(path, self.raw_model, self.optimizer, self.device, strict,
                               expected_fingerprints=self._fingerprints() if strict else None)
        self.step = int(ckpt["step"])
        extra = ckpt.get("extra", {})
        self.tokens_seen = int(extra.get("trainer_tokens_seen", 0))
        self.supervised_tokens_seen = int(
            extra.get("trainer_supervised_tokens_seen", 0)
        )
        self.optimization_seconds = float(
            extra.get("trainer_optimization_seconds", 0.0)
        )
        if resume_rng:
            if ckpt.get("rng") is not None:
                restore_rng_state(ckpt["rng"])
            if (ckpt.get("sampler_state") is not None
                    and hasattr(self.corpus, "load_sampler_state")):
                self.corpus.load_sampler_state(ckpt["sampler_state"])
        return ckpt
