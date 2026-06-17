"""OPET trainer -- reuses the base loop; scores BPT the transformer way."""

from __future__ import annotations

from ...core.registry import TRAINERS
from ...training.base_trainer import BaseTrainer


class OPETTrainer(BaseTrainer):
    def _metric_bpt(self, batch_size: int, seq_len: int, n_batches: int = 10,
                    split: str = "valid") -> float:
        from ...evaluation.metrics import transformer_bits_per_token
        return transformer_bits_per_token(
            self.raw_model, self.corpus, batch_size, seq_len, n_batches, split)


@TRAINERS.register("opet")
def build_opet_trainer(model, corpus, **kwargs) -> OPETTrainer:
    return OPETTrainer(model, corpus, **kwargs)
