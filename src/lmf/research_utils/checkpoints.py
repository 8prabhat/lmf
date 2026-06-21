"""Load a Pure Gear family checkpoint for post-hoc evaluation, given only a
path -- distinct from ``lmf.training.checkpoints``, which resumes training
into an already-constructed model and verifies its architecture fingerprint.
Evaluation/benchmark scripts instead need to construct the right model class
from the checkpoint's own saved manifest, with no training state assumed."""

from __future__ import annotations

from pathlib import Path

import torch

from ..models.gru import GRULM, GRULMConfig
from ..models.pure_parallel_gear import PureParallelGearConfig, PureParallelGearLM
from ..models.transformer import CachedTransformerLM, TransformerConfig


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
