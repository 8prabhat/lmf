"""The single batch container shared by every model family and the trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrainingBatch:
    """Tokens plus attention/loss masks.

    ``attention_mask`` marks real (non-pad) positions; ``loss_mask`` marks
    positions that contribute to the objective. Most LM corpora set both to all
    ones; task corpora (QA, instruction) override ``loss_mask`` to score only the
    answer span.
    """

    tokens: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    task: str = "language_modeling"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "TrainingBatch":
        def move(value):
            if torch.is_tensor(value):
                return value.to(device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, list):
                return [move(item) for item in value]
            return value

        return TrainingBatch(
            self.tokens.to(device),
            self.attention_mask.to(device),
            self.loss_mask.to(device),
            self.task,
            move(self.metadata),
        )

    @property
    def supervised_tokens(self) -> int:
        return int(self.loss_mask.sum())


def lm_batch(tokens: torch.Tensor, task: str = "language_modeling") -> TrainingBatch:
    """Build an all-supervised language-modeling batch from raw token ids."""
    mask = torch.ones_like(tokens, dtype=torch.bool)
    return TrainingBatch(tokens, mask, mask, task)
