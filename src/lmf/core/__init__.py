"""Framework-agnostic core: contracts, registries, config, device/precision, seeding."""

from __future__ import annotations

from .config import ExperimentConfig, load_config
from .device import MemoryGovernor, PrecisionPolicy, resolve_device, sync
from .registry import CORPORA, MODELS, TRAINERS, Registry
from .seeding import capture_rng_state, preserve_rng, restore_rng_state, seed_everything

__all__ = [
    "ExperimentConfig",
    "load_config",
    "PrecisionPolicy",
    "MemoryGovernor",
    "resolve_device",
    "sync",
    "Registry",
    "MODELS",
    "TRAINERS",
    "CORPORA",
    "seed_everything",
    "preserve_rng",
    "capture_rng_state",
    "restore_rng_state",
]
