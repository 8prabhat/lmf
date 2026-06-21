"""Stateful truncated-BPTT training for Gear V3 and bounded controls."""

from __future__ import annotations

import math
import time

import torch

from ...core.registry import TRAINERS
from ...training.base_trainer import BaseTrainer


class PureParallelGearV3Trainer(BaseTrainer):
    """Carry model state across contiguous lanes and detach every TBPTT group."""

    def __init__(
        self,
        model,
        corpus,
        *,
        stateful: bool = True,
        tbptt_chunks: int = 2,
        **kwargs,
    ) -> None:
        self.stateful = bool(stateful)
        self.tbptt_chunks = max(1, int(tbptt_chunks))
        self._stream_cache = None
        self._stream_batch_size: int | None = None
        kwargs.setdefault("betas", (0.9, 0.95))
        super().__init__(model, corpus, **kwargs)

    def _progress(self) -> float:
        if self.schedule_mode == "time" and self.total_seconds:
            return min(1.0, self.optimization_seconds / self.total_seconds)
        if self.total_training_tokens:
            return min(
                1.0,
                self.supervised_tokens_seen / self.total_training_tokens,
            )
        return min(1.0, self.step / max(1, self.total_steps))

    def batch_metadata(self, step: int) -> dict:
        del step
        decay = float(
            getattr(self.raw_model.config, "future_aux_decay_fraction", 0.8)
        )
        return {
            "training_progress": self._progress(),
            "future_aux_scale": max(
                0.0,
                1.0 - self._progress() / max(decay, 1e-9),
            ),
        }

    def checkpoint_metadata(self) -> dict:
        return {
            "v3_stateful": self.stateful,
            "v3_tbptt_chunks": self.tbptt_chunks,
            "v3_stream_cache": (
                None
                if self._stream_cache is None
                else self._stream_cache.detach()
            ),
            "v3_stream_batch_size": self._stream_batch_size,
        }

    def load_checkpoint(self, *args, **kwargs):
        checkpoint = super().load_checkpoint(*args, **kwargs)
        extra = checkpoint.get("extra", {})
        cache = extra.get("v3_stream_cache")
        self._stream_cache = (
            None if cache is None else cache.to(self.device)
        )
        self._stream_batch_size = extra.get("v3_stream_batch_size")
        return checkpoint

    def train_steps(
        self,
        n_steps: int,
        batch_size: int,
        seq_len: int,
        log_every: int = 50,
        callbacks: list | None = None,
        max_seconds: float | None = None,
    ) -> list[dict[str, float]]:
        if not self.stateful:
            return super().train_steps(
                n_steps,
                batch_size,
                seq_len,
                log_every,
                callbacks,
                max_seconds,
            )
        if not hasattr(self.model, "stream_training_step"):
            raise TypeError("stateful V3 training requires stream_training_step")
        callbacks = callbacks or []
        records: list[dict[str, float]] = []
        started = time.perf_counter()
        optimization_origin = self.optimization_seconds
        self.model.train()
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
                    group, self.step
                )
            self.optimizer.zero_grad(set_to_none=True)
            if self._stream_batch_size != batch_size:
                self._stream_cache = None
                self._stream_batch_size = batch_size
            cache = self._stream_cache
            accumulated: dict[str, float] = {}

            for _accumulation in range(self.grad_accum_steps):
                group_loss = None
                for _chunk in range(self.tbptt_chunks):
                    batch = self._sample_batch(
                        batch_size,
                        self.effective_seq_len(seq_len, self.step),
                    )
                    if not batch.metadata.get("contiguous_lanes", False):
                        raise RuntimeError(
                            "stateful training requires a contiguous-lane corpus"
                        )
                    self.tokens_seen += int(batch.tokens.numel())
                    supervised = (
                        batch.loss_mask[:, 1:].bool()
                        & batch.attention_mask[:, 1:].bool()
                    )
                    self.supervised_tokens_seen += int(supervised.sum())
                    metadata = {
                        **self.batch_metadata(self.step),
                        **batch.metadata,
                        "loss_mask": batch.loss_mask,
                        "attention_mask": batch.attention_mask,
                    }
                    with self.precision_policy.autocast(self.device):
                        losses, cache = self.model.stream_training_step(
                            batch.tokens,
                            cache=cache,
                            detach_cache=False,
                            task_metadata=metadata,
                            loss_term_scales=self.loss_term_scales,
                        )
                    scaled = losses["total"] / (
                        self.grad_accum_steps * self.tbptt_chunks
                    )
                    group_loss = (
                        scaled if group_loss is None else group_loss + scaled
                    )
                    for name, value in losses.items():
                        if torch.is_tensor(value):
                            accumulated[name] = accumulated.get(name, 0.0) + (
                                float(value.detach())
                                / (self.grad_accum_steps * self.tbptt_chunks)
                            )
                assert group_loss is not None
                group_loss.backward()
                cache = cache.detach()

            if not math.isfinite(accumulated.get("total", 0.0)):
                raise FloatingPointError(
                    f"non-finite V3 loss at step {self.step}: {accumulated}"
                )
            self.validate_step_metrics(accumulated)
            grad_norm = self.clip_gradients()
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    f"non-finite V3 gradient at step {self.step}: {grad_norm}"
                )
            self.optimizer.step()
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
            self.optimization_seconds += time.perf_counter() - step_started
            self._stream_cache = cache
            self.step += 1
            self.memory.maybe_evict(self.step)
            record = {
                **accumulated,
                "lr": base_lr,
                "grad_norm": float(grad_norm),
                "step": self.step,
                "tokens_seen": self.tokens_seen,
                "supervised_tokens_seen": self.supervised_tokens_seen,
            }
            for callback in callbacks:
                callback.on_step_end(self, self.step, record)
            records.append(record)
            if log_every and self.step % log_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"step {self.step:>6}/{self.total_steps} "
                    f"total={accumulated.get('total', 0.0):.4f} "
                    f"{self.step / max(elapsed, 1e-9):.2f} it/s",
                    flush=True,
                )
        return records


@TRAINERS.register("pure_parallel_gear_v3")
def build_pure_parallel_gear_v3_trainer(model, corpus, **kwargs):
    return PureParallelGearV3Trainer(model, corpus, **kwargs)


@TRAINERS.register("hybrid_parallel_gear")
def build_hybrid_parallel_gear_trainer(model, corpus, **kwargs):
    return PureParallelGearV3Trainer(model, corpus, **kwargs)


@TRAINERS.register("bounded_transformer")
def build_bounded_transformer_trainer(model, corpus, **kwargs):
    return PureParallelGearV3Trainer(model, corpus, **kwargs)


@TRAINERS.register("block_hybrid_gear_v4")
def build_block_hybrid_gear_v4_trainer(model, corpus, **kwargs):
    return PureParallelGearV3Trainer(model, corpus, **kwargs)


@TRAINERS.register("selective_hybrid_gear_v42")
def build_selective_hybrid_gear_v42_trainer(model, corpus, **kwargs):
    return PureParallelGearV3Trainer(model, corpus, **kwargs)


@TRAINERS.register("gear_bank_router_v43")
def build_gear_bank_router_v43_trainer(model, corpus, **kwargs):
    return PureParallelGearV3Trainer(model, corpus, **kwargs)
