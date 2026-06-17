"""Rolling-frontier state containers and sampling config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    deterministic: bool = False
    commit_entropy_threshold: float | None = None

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


@dataclass
class GenerationState:
    """The full O(1) rolling state — independent of sequence length."""

    memory: torch.Tensor       # B x slots x D
    frontier: torch.Tensor     # B x P x H x D
    tail: torch.Tensor         # B x tail_size x D  (verbatim recent embeddings)
    tail_ids: torch.Tensor     # B x tail_size
    active_hypotheses: torch.Tensor
    finished: torch.Tensor
    committed_count: torch.Tensor

    def detach(self) -> "GenerationState":
        """Detach every tensor — the TBPTT boundary between carried segments."""
        return GenerationState(
            **{k: (v.detach() if torch.is_tensor(v) else v) for k, v in vars(self).items()}
        )


@dataclass
class AdvanceResult:
    state: GenerationState
    committed_token_ids: torch.Tensor
    commit_mask: torch.Tensor
    commit_entropy: torch.Tensor
    selected_hypothesis: torch.Tensor
    elapsed_seconds: float


@dataclass
class GenerationResult:
    token_ids: torch.Tensor
    generated_lengths: torch.Tensor
    cycles: int
    tokens_per_second: float
    tokens_per_settle: float
    diagnostics: list[dict[str, Any]]
