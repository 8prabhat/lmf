"""Training policy for the canonical Pure Parallel Gear model."""

from __future__ import annotations

import time

from ...core.registry import TRAINERS
from ...evaluation.metrics import transformer_bits_per_token
from ...training.base_trainer import BaseTrainer


class PureParallelGearTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        corpus,
        *,
        context_lengths=(128, 256, 512, 1024, 2048, 4096),
        context_fractions=(0.10, 0.15, 0.20, 0.20, 0.20, 0.15),
        total_training_tokens: int | None = None,
        dynamics_lr_multiplier: float = 1.0,
        dynamics_grad_clip: float = 1.0,
        gradient_explosion_threshold: float = 20.0,
        max_consecutive_gradient_skips: int = 10,
        **kwargs,
    ) -> None:
        self.context_lengths = tuple(int(value) for value in context_lengths)
        self.context_fractions = tuple(float(value) for value in context_fractions)
        if len(self.context_lengths) != len(self.context_fractions):
            raise ValueError("context lengths and fractions must have equal length")
        if abs(sum(self.context_fractions) - 1.0) > 1e-6:
            raise ValueError("context_fractions must sum to one")
        self.dynamics_lr_multiplier = float(dynamics_lr_multiplier)
        self.dynamics_grad_clip = float(dynamics_grad_clip)
        # The omega/angle/load "dynamics" parameters drive a recurrence
        # (settle() updates omega once per chunk, reused by every token's
        # phase = cumsum(delta + omega) within the next chunk) that is
        # chained across every sentence/clutch boundary in a sequence --
        # an RNN-style backprop-through-time chain. Short sequences only
        # chain a few boundaries together and stay well-conditioned; long
        # sequences (>=1024 tokens, dozens of chained boundaries) can blow
        # the gradient up by many orders of magnitude in a handful of
        # steps. dynamics_grad_clip alone can't prevent this -- it rescales
        # an already-huge-but-finite gradient, it can't stop the next
        # step's backward pass from independently producing another one,
        # and once a single tensor entry overflows to a literal inf no
        # rescale can recover it. Detecting the early, still-finite blowup
        # and skipping that one optimizer step (zero grad, no update) is
        # the standard mitigation for an occasional pathological batch;
        # repeated skips would mean the instability is persistent, not
        # transient, so that still escalates to a hard failure.
        self.gradient_explosion_threshold = float(gradient_explosion_threshold)
        self.max_consecutive_gradient_skips = int(max_consecutive_gradient_skips)
        self._consecutive_gradient_skips = 0
        self.total_gradient_skips = 0
        self._stability_failures = {
            "rotor_energy": 0,
            "omega_saturation": 0,
            "clutch_collapse": 0,
            "dead_gear_fraction": 0,
            "memory_energy": 0,
        }
        if total_training_tokens is not None:
            kwargs.setdefault("schedule_mode", "tokens")
            kwargs.setdefault("warmup_tokens", max(1, total_training_tokens // 10))
        kwargs.setdefault("betas", (0.9, 0.95))
        super().__init__(
            model,
            corpus,
            total_training_tokens=total_training_tokens,
            **kwargs,
        )

    @staticmethod
    def _is_dynamics(name: str) -> bool:
        return any(
            key in name
            for key in (
                "base_omega",
                "initial_phase",
                "pair_kernel",
                "cross_kernel",
                "intra_gate",
                "cross_gate",
                "load_response",
                "omega_response",
                "angle_projection",
                "clutch_projection",
                "torque_projection",
                "retention_projection",
            )
        )

    @staticmethod
    def _slow_lr_dynamics(name: str) -> bool:
        """Parameters that need a conservative LR for stable mechanics.

        Token-conditioned angle, clutch, and torque projections are the write
        path and must learn at the normal model LR.  Applying the dynamics
        multiplier to them made the core sequence mechanism learn four times
        slower than the token-local FFN.
        """
        return any(
            key in name
            for key in (
                "base_omega",
                "initial_phase",
                "pair_kernel",
                "cross_kernel",
                "intra_gate",
                "cross_gate",
                "load_response",
                "omega_response",
            )
        )

    @staticmethod
    def _no_decay(name: str) -> bool:
        return (
            name.endswith(".bias")
            or "norm" in name
            or PureParallelGearTrainer._is_dynamics(name)
            or name.endswith("residual")
        )

    def optimizer_param_groups(self, model, weight_decay: float):
        groups: dict[tuple[bool, bool, bool], list] = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                key = (
                    self._slow_lr_dynamics(name),
                    self._no_decay(name),
                    self._is_dynamics(name),
                )
                groups.setdefault(key, []).append(parameter)
        output = []
        for (slow_lr, no_decay, dynamics), parameters in groups.items():
            if not parameters:
                continue
            output.append(
                {
                    "params": parameters,
                    "weight_decay": 0.0 if no_decay else float(weight_decay),
                    "lr_multiplier": (
                        self.dynamics_lr_multiplier if slow_lr else 1.0
                    ),
                    "gear_dynamics": dynamics,
                    "slow_gear_dynamics": slow_lr,
                }
            )
        return output

    def clip_gradients(self):
        import torch

        all_params = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if all_params:
            # max_norm=inf makes this a pure measurement: clip_grad_norm_'s
            # rescale factor is min(1, max_norm / total_norm), which is 1
            # whenever total_norm is finite, so this never alters the
            # gradients -- it only computes the combined norm so we can
            # detect a pathological step before it reaches the optimizer.
            total_norm = float(
                torch.nn.utils.clip_grad_norm_(all_params, float("inf"))
            )
            non_finite = not torch.isfinite(torch.tensor(total_norm))
            if non_finite:
                # Already past fp32's range -- there is no sane direction
                # left to step in, so this step must be skipped outright.
                for parameter in all_params:
                    parameter.grad = None
                self._consecutive_gradient_skips += 1
                self.total_gradient_skips += 1
                if (
                    self._consecutive_gradient_skips
                    >= self.max_consecutive_gradient_skips
                ):
                    raise FloatingPointError(
                        "persistent gradient explosion: "
                        f"{self._consecutive_gradient_skips} consecutive "
                        "non-finite gradients"
                    )
                return torch.tensor(0.0)
            if total_norm > self.gradient_explosion_threshold:
                # Still finite but clearly past the omega-recurrence's
                # well-conditioned range. Clipping (not skipping) lets the
                # optimizer keep taking a small, bounded step that walks
                # the runaway dynamics parameters back toward stability --
                # zeroing the gradient here just freezes them in the bad
                # state and the very next step re-triggers identically,
                # which empirically only escalates to a hard failure.
                torch.nn.utils.clip_grad_norm_(
                    all_params, self.gradient_explosion_threshold
                )
            self._consecutive_gradient_skips = 0
        dynamics = [
            parameter
            for group in self.optimizer.param_groups
            if group.get("gear_dynamics", False)
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if dynamics:
            torch.nn.utils.clip_grad_norm_(dynamics, self.dynamics_grad_clip)
        return super().clip_gradients()

    def validate_step_metrics(self, metrics: dict[str, float]) -> None:
        thresholds = {
            "rotor_energy": 1.0,
            "omega_saturation": 0.02,
            "clutch_collapse": 0.25,
            # memory_energy is itself a clamped (radius - limit)^2 penalty
            # (zero whenever the accumulator stays within
            # fast_weight_energy_limit), so a nonzero value already means
            # the matrix is over its soft limit; 25.0 allows a margin of 5
            # over the limit before treating it as real divergence rather
            # than a noisy step. No-op metric (always 0.0) when the model
            # doesn't use fast-weight memory.
            "memory_energy": 25.0,
        }
        if self._progress() >= 0.10:
            thresholds["dead_gear_fraction"] = 0.50
        for name, threshold in thresholds.items():
            failed = float(metrics.get(name, 0.0)) > threshold
            self._stability_failures[name] = (
                self._stability_failures[name] + 1 if failed else 0
            )
            if self._stability_failures[name] >= 3:
                snapshot = {
                    key: round(float(value), 6)
                    for key, value in metrics.items()
                    if key in thresholds or key in ("clutch_balance", "total")
                }
                raise FloatingPointError(
                    f"persistent Pure Gear instability: {name}="
                    f"{metrics[name]:.6g} exceeded {threshold} for three steps "
                    f"at step={self.step} progress={self._progress():.3f} "
                    f"recent_metrics={snapshot}"
                )

    def checkpoint_metadata(self) -> dict:
        moment_dtypes = sorted(
            {
                str(value.dtype)
                for state in self.optimizer.state.values()
                for value in state.values()
                if hasattr(value, "dtype") and value.is_floating_point()
            }
        )
        manifest = getattr(self.corpus, "manifest", {})
        return {
            "parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in self.raw_model.parameters()}
            ),
            "optimizer_moment_dtypes": moment_dtypes,
            "boundary_detector_hash": manifest.get("boundary_detector_hash"),
            "boundary_detector_version": manifest.get(
                "boundary_detector_version"
            ),
        }

    def _progress(self) -> float:
        if self.schedule_mode == "time" and self.total_seconds:
            return min(
                1.0,
                self.optimization_seconds / max(self.total_seconds, 1e-9),
            )
        if self.total_training_tokens:
            return min(
                1.0,
                self.supervised_tokens_seen / max(1, self.total_training_tokens),
            )
        return min(1.0, self.step / max(1, self.total_steps))

    def effective_seq_len(self, requested_seq_len: int, step: int) -> int:
        progress = self._progress()
        cumulative = 0.0
        for length, fraction in zip(self.context_lengths, self.context_fractions):
            cumulative += fraction
            if progress < cumulative:
                return min(int(requested_seq_len), length)
        return min(int(requested_seq_len), self.context_lengths[-1])

    def batch_metadata(self, step: int) -> dict:
        progress = self._progress()
        decay = self.raw_model.config.regularizer_decay_fraction
        floor = self.raw_model.config.minimum_regularizer_scale
        position = min(1.0, progress / max(decay, 1e-9))
        return {
            "training_progress": progress,
            "regularizer_scale": floor + (1.0 - floor) * (1.0 - position),
        }

    def _metric_bpt(self, batch_size, seq_len, n_batches, split) -> float:
        return transformer_bits_per_token(
            self.raw_model, self.corpus, batch_size, seq_len, n_batches, split
        )


@TRAINERS.register("pure_parallel_gear")
def build_pure_parallel_gear_trainer(
    model,
    corpus,
    **kwargs,
) -> PureParallelGearTrainer:
    return PureParallelGearTrainer(model, corpus, **kwargs)
