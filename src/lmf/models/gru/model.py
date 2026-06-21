"""Simple recurrent control baseline for Pure Parallel Gear studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS


@dataclass(frozen=True)
class GRULMConfig:
    vocab_size: int
    dim: int = 192
    hidden_dim: int | None = None
    layers: int = 2
    dropout: float = 0.0
    max_seq_len: int = 4096

    def __post_init__(self) -> None:
        if self.hidden_dim is None:
            object.__setattr__(self, "hidden_dim", self.dim)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GRULM(nn.Module):
    def __init__(self, config: GRULMConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.dim)
        self.gru = nn.GRU(
            config.dim,
            int(config.hidden_dim),
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )
        self.output = nn.Linear(int(config.hidden_dim), config.dim, bias=False)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(
        self,
        token_ids: torch.Tensor,
        cache: torch.Tensor | None = None,
        use_cache: bool = False,
        segment_ids: torch.Tensor | None = None,
        **_,
    ):
        embedded = self.token(token_ids)
        if segment_ids is None or cache is not None:
            hidden, next_cache = self.gru(embedded, cache)
        else:
            # Packed-segment boundaries are control data.  Reading one MPS
            # scalar at a time with int(segment_ids[...]) synchronizes the
            # accelerator for every token and makes the recurrent baseline
            # artificially slow.
            control_segments = segment_ids.detach().to(
                device="cpu", dtype=torch.long
            )
            rows = []
            final = []
            for row in range(token_ids.shape[0]):
                pieces = []
                state = None
                start = 0
                while start < token_ids.shape[1]:
                    segment = int(control_segments[row, start])
                    end = start + 1
                    while (
                        end < token_ids.shape[1]
                        and int(control_segments[row, end]) == segment
                    ):
                        end += 1
                    piece, state = self.gru(
                        embedded[row : row + 1, start:end],
                        None,
                    )
                    pieces.append(piece)
                    start = end
                rows.append(torch.cat(pieces, dim=1))
                final.append(state)
            hidden = torch.cat(rows, dim=0)
            next_cache = torch.cat(final, dim=1)
        return self.head(self.output(hidden)), (next_cache if use_cache else None)

    def training_step(self, tokens, task_metadata=None, loss_term_scales=None):
        metadata = task_metadata or {}
        logits, _ = self(tokens, segment_ids=metadata.get("segment_ids"))
        targets = tokens[:, 1:]
        losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        valid = torch.ones_like(targets, dtype=torch.bool)
        if metadata.get("loss_mask") is not None:
            valid &= metadata["loss_mask"][:, 1:].bool()
        if metadata.get("attention_mask") is not None:
            valid &= metadata["attention_mask"][:, 1:].bool()
        if metadata.get("segment_ids") is not None:
            valid &= metadata["segment_ids"][:, 1:] == metadata["segment_ids"][:, :-1]
        language_modeling = losses[valid].mean()
        return {"language_modeling": language_modeling, "total": language_modeling}

    @staticmethod
    def _sample_token(logits, config):
        if config is None or config.deterministic:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(float(config.temperature), 1e-5)
        if config.top_k > 0:
            threshold = logits.topk(
                min(config.top_k, logits.shape[-1]), dim=-1
            ).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if config.top_p < 1.0:
            values, indices = logits.sort(dim=-1, descending=True)
            remove = values.softmax(-1).cumsum(-1) > config.top_p
            remove[..., 0] = False
            values = values.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                -1, indices, values
            )
        return torch.multinomial(logits.softmax(-1), 1)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, sampling_config=None):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return torch.empty(
                prompt.shape[0], 0, dtype=torch.long, device=prompt.device
            )
        logits, cache = self(prompt, use_cache=True)
        token = self._sample_token(logits[:, -1], sampling_config)
        output = []
        for index in range(max_new_tokens):
            output.append(token)
            if index + 1 == max_new_tokens:
                break
            logits, cache = self(token, cache=cache, use_cache=True)
            token = self._sample_token(logits[:, -1], sampling_config)
        return torch.cat(output, dim=1)

    def architecture_manifest(self):
        return {
            "name": "GRULM",
            "config": self.config.to_dict(),
            "parameters": {"total": sum(p.numel() for p in self.parameters())},
        }


@MODELS.register("gru_lm")
def build_gru_lm(model_cfg: dict, vocab_size: int | None = None) -> GRULM:
    config = dict(model_cfg)
    if vocab_size is not None:
        config["vocab_size"] = vocab_size
    return GRULM(GRULMConfig(**config))
