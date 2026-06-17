"""Trainer registrations for the MultiGear-native model families."""

from __future__ import annotations

from ...core.registry import TRAINERS
from ...training.base_trainer import BaseTrainer


class NativeLMTrainer(BaseTrainer):
    """Generic trainer for MECM, MCPM, and MRWT."""

    def _metric_bpt(self, batch_size: int, seq_len: int, n_batches: int = 10,
                    split: str = "valid") -> float:
        from ...evaluation.metrics import transformer_bits_per_token

        return transformer_bits_per_token(
            self.raw_model, self.corpus, batch_size, seq_len, n_batches, split
        )


def _drop_irrelevant(kwargs: dict) -> dict:
    for key in (
        "segment_len",
        "scheduled_sampling_start",
        "scheduled_sampling_end",
        "scheduled_sampling_ramp_steps",
        "calibrate_every",
        "calibrate_precision_target",
        "max_train_windows",
        "segmentation_dropout_prob",
        "segmentation_dropout_min_gear",
        "segmentation_dropout_max_depth",
    ):
        kwargs.pop(key, None)
    return kwargs


@TRAINERS.register("mecm")
def build_mecm_trainer(model, corpus, **kwargs) -> NativeLMTrainer:
    return NativeLMTrainer(model, corpus, **_drop_irrelevant(kwargs))


@TRAINERS.register("mcpm")
def build_mcpm_trainer(model, corpus, **kwargs) -> NativeLMTrainer:
    return NativeLMTrainer(model, corpus, **_drop_irrelevant(kwargs))


@TRAINERS.register("mgcf")
def build_mgcf_trainer(model, corpus, **kwargs) -> NativeLMTrainer:
    return NativeLMTrainer(model, corpus, **_drop_irrelevant(kwargs))


@TRAINERS.register("mrwt")
def build_mrwt_trainer(model, corpus, **kwargs) -> NativeLMTrainer:
    return NativeLMTrainer(model, corpus, **_drop_irrelevant(kwargs))
