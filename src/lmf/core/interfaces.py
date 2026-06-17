"""Narrow, segregated contracts shared across all model families.

These Protocols are the only coupling point between the framework layers
(training / evaluation / CLI) and concrete model families. A family implements
the subset it needs and nothing more (Interface Segregation Principle), and the
framework depends on these abstractions rather than on any concrete class
(Dependency Inversion Principle).
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer contract — allows drop-in substitution."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...


@runtime_checkable
class Corpus(Protocol):
    """A source of tokenized training/eval batches.

    A corpus must at minimum produce raw token windows. The optional
    ``sample_batch`` hook lets a corpus attach attention/loss masks; the
    framework falls back to all-ones masks when it is absent.
    """

    vocab_size: int

    def sample_tokenized(self, batch: int, seq_len: int, split: str = "train") -> torch.Tensor: ...


@runtime_checkable
class LanguageModel(Protocol):
    """Anything that turns a fixed-size config into trainable next-token behaviour."""

    def parameters(self) -> Any: ...
    def to(self, *args: Any, **kwargs: Any) -> "LanguageModel": ...
    def train(self, mode: bool = True) -> "LanguageModel": ...
    def eval(self) -> "LanguageModel": ...


@runtime_checkable
class Trainable(Protocol):
    """A model that exposes a single differentiable training objective.

    ``training_step`` returns a dict of named loss tensors; ``"total"`` is the
    scalar the optimizer descends. This uniform contract is what lets one base
    Trainer drive every family.
    """

    def training_step(
        self, tokens: torch.Tensor, task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]: ...


@runtime_checkable
class Generative(Protocol):
    """A model that can autoregressively (or block-wise) generate continuations."""

    @torch.no_grad()
    def generate(
        self, prompt_tokens: torch.Tensor, max_new_tokens: int, **kwargs: Any
    ) -> Any: ...


@runtime_checkable
class Manifestable(Protocol):
    """A model that can describe its own architecture for fingerprinting/checkpoints."""

    def architecture_manifest(self) -> dict[str, Any]: ...


@runtime_checkable
class AblationAddressable(Protocol):
    """Optional hook for model-specific named ablation points.

    ``lmf.ablation.points.discover_points`` works on any ``nn.Module`` via
    generic introspection (``named_modules``, ``nn.ModuleList`` discovery)
    without this Protocol. A family may *additionally* implement
    ``ablation_points()`` to expose named points the generic mechanism can't
    derive (e.g. composite/multi-step ablations); those points are merged into
    the generically-discovered set, overriding on name collision.
    """

    def ablation_points(self) -> dict[str, Callable[[nn.Module], contextlib.AbstractContextManager[None]]]: ...
