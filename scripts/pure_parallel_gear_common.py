"""Shared checkpoint and statistics utilities for Pure Gear evaluations."""

from __future__ import annotations

import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

import torch

from lmf.models.gru import GRULM, GRULMConfig
from lmf.models.pure_parallel_gear import (
    PureParallelGearConfig,
    PureParallelGearLM,
)
from lmf.models.transformer import CachedTransformerLM, TransformerConfig


def load_model(path: Path, device: str) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    manifest = checkpoint.get("manifest", {})
    name = manifest.get("name")
    config = checkpoint["config"]
    if name == "PureParallelGear":
        version = int(manifest.get("version", 1))
        if version < 2:
            raise RuntimeError(
                "PureParallelGear architecture version 1 is incompatible "
                "with the affine-retention/radial-state architecture version 2"
            )
        model = PureParallelGearLM(PureParallelGearConfig(**config))
    elif name == "CachedTransformerLM":
        model = CachedTransformerLM(TransformerConfig(**config))
    elif name == "GRULM":
        model = GRULM(GRULMConfig(**config))
    elif name in {"PureParallelRotatingGearLM", "PureParallelPredictiveGearV2"}:
        raise RuntimeError(
            "legacy V1/V2 gear checkpoints are intentionally unsupported"
        )
    else:
        raise ValueError(f"unsupported checkpoint architecture: {name!r}")
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def cache_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if hasattr(value, "__dict__"):
        return sum(cache_bytes(item) for item in vars(value).values())
    if isinstance(value, dict):
        return sum(cache_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(cache_bytes(item) for item in value)
    return 0


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = fraction * (len(ordered) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def paired_bootstrap_interval(
    differences: list[float],
    *,
    seed: int,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float]:
    if not differences:
        raise ValueError("bootstrap requires at least one paired difference")
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [differences[rng.randrange(len(differences))] for _ in differences]
        means.append(statistics.fmean(draw))
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": statistics.fmean(differences),
        "lower": percentile(means, alpha),
        "upper": percentile(means, 1.0 - alpha),
        "samples": samples,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return adjusted
