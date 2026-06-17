"""RHCA trainer — the family-specific overrides on top of BaseTrainer.

RHCA bounds the number of retained settle graphs per optimizer step.
"""

from __future__ import annotations

from ...core.registry import TRAINERS
from ...training.base_trainer import BaseTrainer


class RHCATrainer(BaseTrainer):
    def __init__(self, model, corpus, *,
                 segment_len: int | None = None,
                 max_train_windows: int = 2,
                 **kwargs) -> None:
        super().__init__(model, corpus, **kwargs)
        self.segment_len = segment_len
        self.max_train_windows = max(1, int(max_train_windows))

    def batch_metadata(self, step: int) -> dict:
        return {"segment_len": self.segment_len, "max_train_windows": self.max_train_windows}

    def _metric_bpt(self, batch_size: int, seq_len: int, n_batches: int = 10,
                    split: str = "valid") -> float:
        from ...evaluation.metrics import rhca_bits_per_token
        return rhca_bits_per_token(self.raw_model, self.corpus, batch_size, seq_len, n_batches, split)


@TRAINERS.register("rhca")
def build_rhca_trainer(model, corpus, **kwargs) -> RHCATrainer:
    return RHCATrainer(model, corpus, **kwargs)
