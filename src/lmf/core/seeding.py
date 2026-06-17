"""Centralised RNG control and a sampler-state guard.

Diagnostics and calibration draw batches mid-training; without care they would
perturb the training data stream and break reproducibility. ``preserve_rng``
snapshots and restores every relevant RNG so a diagnostic pass is side-effect
free.
"""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state(device: torch.device | None = None) -> dict:
    """Snapshot every relevant RNG (torch/python/numpy/cuda/mps) for checkpoints."""
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": (torch.mps.get_rng_state()
                if (device is not None and device.type == "mps"
                    and hasattr(torch.mps, "get_rng_state")) else None),
    }


def restore_rng_state(state: dict) -> None:
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu())
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if state.get("mps") is not None and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(state["mps"].cpu())


@contextmanager
def preserve_rng(device: torch.device):
    """Restore torch/python/numpy/cuda/mps RNG state on exit."""
    state = capture_rng_state(device)
    try:
        yield
    finally:
        restore_rng_state(state)
