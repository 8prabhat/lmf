"""Language Model Foundry — a SOLID/DRY framework for sequence-model research.

Importing :mod:`lmf` wires the framework together and registers the built-in model
families (RHCA + transformer) so they are available by name from the registries.
"""

from __future__ import annotations

from .core import (
    CORPORA,
    MODELS,
    TRAINERS,
    ExperimentConfig,
    PrecisionPolicy,
    load_config,
    resolve_device,
    seed_everything,
)
from . import data, evaluation, models, training  # noqa: F401  (registration side effects)

__version__ = "0.1.0"

__all__ = [
    "MODELS",
    "TRAINERS",
    "CORPORA",
    "ExperimentConfig",
    "load_config",
    "PrecisionPolicy",
    "resolve_device",
    "seed_everything",
    "data",
    "models",
    "training",
    "evaluation",
]
