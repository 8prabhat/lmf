"""Transformer trainer — the baseline reuses the base loop almost verbatim."""

from __future__ import annotations

import torch

from ...core.registry import TRAINERS
from ...data.batch import TrainingBatch
from ...training.base_trainer import BaseTrainer


class TransformerTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        corpus,
        *,
        segmentation_dropout_prob: float = 0.0,
        segmentation_dropout_min_gear: int = 2,
        segmentation_dropout_max_depth: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(model, corpus, **kwargs)
        self.segmentation_dropout_prob = float(segmentation_dropout_prob)
        self.segmentation_dropout_min_gear = int(segmentation_dropout_min_gear)
        self.segmentation_dropout_max_depth = int(segmentation_dropout_max_depth)
        if not 0.0 <= self.segmentation_dropout_prob <= 1.0:
            raise ValueError("segmentation_dropout_prob must be in [0.0, 1.0]")
        if self.segmentation_dropout_min_gear < 0:
            raise ValueError("segmentation_dropout_min_gear must be non-negative")
        if self.segmentation_dropout_max_depth < 1:
            raise ValueError("segmentation_dropout_max_depth must be positive")
        self._token_hierarchy = None
        if self.segmentation_dropout_prob > 0.0:
            tokenizer = getattr(corpus, "tokenizer", None)
            hierarchy = getattr(tokenizer, "token_hierarchy", None)
            if hierarchy is None:
                raise TypeError("segmentation dropout requires a hierarchical tokenizer")
            self._token_hierarchy = hierarchy()

    def _decompose_token(self, token_id: int, depth: int) -> list[int]:
        if self._token_hierarchy is None or depth <= 0:
            return [token_id]
        gears = self._token_hierarchy["token_gears"]
        children = self._token_hierarchy["token_children"]
        if token_id >= len(children) or gears[token_id] < self.segmentation_dropout_min_gear:
            return [token_id]
        left, right = children[token_id]
        if left < 0:
            return [token_id]
        return (
            self._decompose_token(left, depth - 1)
            + self._decompose_token(right, depth - 1)
        )

    @staticmethod
    def _supervised_window(loss_mask: list[bool], seq_len: int) -> int:
        supervised = [index for index, value in enumerate(loss_mask) if value]
        if not supervised or len(loss_mask) <= seq_len:
            return 0
        first, last = supervised[0], supervised[-1]
        if last - first + 1 > seq_len:
            return first
        return max(0, last - seq_len + 1)

    def _apply_segmentation_dropout(self, batch: TrainingBatch) -> TrainingBatch:
        if self.segmentation_dropout_prob <= 0.0:
            return batch
        random_values = torch.rand(batch.tokens.shape).tolist()
        pad_id = 0
        tokenizer = getattr(self.corpus, "tokenizer", None)
        special_to_id = getattr(tokenizer, "special_to_id", {})
        if "<|pad|>" in special_to_id:
            pad_id = int(special_to_id["<|pad|>"])
        seq_len = batch.tokens.shape[1]
        rows = []
        attention_rows = []
        loss_rows = []
        replacements = 0
        for row, attention, loss, random_row in zip(
            batch.tokens.detach().cpu().tolist(),
            batch.attention_mask.detach().cpu().tolist(),
            batch.loss_mask.detach().cpu().tolist(),
            random_values,
        ):
            expanded_tokens: list[int] = []
            expanded_attention: list[bool] = []
            expanded_loss: list[bool] = []
            for token_id, attended, supervised, draw in zip(row, attention, loss, random_row):
                if not attended:
                    continue
                pieces = [token_id]
                if draw < self.segmentation_dropout_prob:
                    pieces = self._decompose_token(
                        token_id, self.segmentation_dropout_max_depth
                    )
                    replacements += int(len(pieces) > 1)
                expanded_tokens.extend(pieces)
                expanded_attention.extend([True] * len(pieces))
                expanded_loss.extend([bool(supervised)] * len(pieces))
            start = self._supervised_window(expanded_loss, seq_len)
            expanded_tokens = expanded_tokens[start:start + seq_len]
            expanded_attention = expanded_attention[start:start + seq_len]
            expanded_loss = expanded_loss[start:start + seq_len]
            padding = seq_len - len(expanded_tokens)
            rows.append(expanded_tokens + [pad_id] * padding)
            attention_rows.append(expanded_attention + [False] * padding)
            loss_rows.append(expanded_loss + [False] * padding)
        return TrainingBatch(
            torch.tensor(rows, dtype=batch.tokens.dtype, device=batch.tokens.device),
            torch.tensor(
                attention_rows, dtype=batch.attention_mask.dtype, device=batch.tokens.device
            ),
            torch.tensor(loss_rows, dtype=batch.loss_mask.dtype, device=batch.tokens.device),
            task=batch.task,
            metadata={**batch.metadata, "segmentation_dropout_replacements": replacements},
        )

    def _sample_batch(self, batch_size: int, seq_len: int) -> TrainingBatch:
        return self._apply_segmentation_dropout(super()._sample_batch(batch_size, seq_len))

    def _metric_bpt(self, batch_size: int, seq_len: int, n_batches: int = 10,
                    split: str = "valid") -> float:
        from ...evaluation.metrics import transformer_bits_per_token
        return transformer_bits_per_token(
            self.raw_model, self.corpus, batch_size, seq_len, n_batches, split)


@TRAINERS.register("transformer")
def build_transformer_trainer(model, corpus, **kwargs) -> TransformerTrainer:
    # Drop RHCA-only or legacy RHCA trainer knobs if present.
    for key in ("segment_len", "scheduled_sampling_start", "scheduled_sampling_end",
                "scheduled_sampling_ramp_steps", "calibrate_every", "calibrate_precision_target",
                "max_train_windows"):
        kwargs.pop(key, None)
    return TransformerTrainer(model, corpus, **kwargs)


@TRAINERS.register("mght")
def build_multigear_hierarchical_transformer_trainer(
    model, corpus, **kwargs
) -> TransformerTrainer:
    return build_transformer_trainer(model, corpus, **kwargs)
