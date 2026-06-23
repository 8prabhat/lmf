"""Versioned, fingerprinted checkpoint IO (model-agnostic).

A checkpoint records the model state, optimizer state, step, the model's config
and architecture manifest, plus an architecture fingerprint. Loading verifies the
fingerprint so a checkpoint can never be silently restored into an incompatible
architecture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SCHEMA_VERSION = 1


def architecture_fingerprint(model) -> str:
    manifest = model.architecture_manifest() if hasattr(model, "architecture_manifest") else {}
    shapes = {n: list(t.shape) for n, t in model.state_dict().items()}
    payload = {"manifest": manifest, "state_shapes": shapes}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_checkpoint(path: str | Path, model, optimizer, step: int,
                    extra: dict[str, Any] | None = None,
                    rng: dict | None = None, sampler_state: dict | None = None,
                    fingerprints: dict | None = None) -> None:
    """Persist enough to resume training bit-for-bit (review finding 4).

    Beyond model/optimizer/step, this records the RNG snapshot, the corpus
    sampler state, and tokenizer/corpus fingerprints so a resumed run produces the
    same next batch and rejects a mismatched corpus or tokenizer.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": model.config.to_dict() if hasattr(model.config, "to_dict") else dict(vars(model.config)),
        "manifest": model.architecture_manifest() if hasattr(model, "architecture_manifest") else {},
        "architecture_fingerprint": architecture_fingerprint(model),
        "runtime_commit_threshold": getattr(model, "_runtime_commit_threshold", None),
        "rng": rng,
        "sampler_state": sampler_state,
        "fingerprints": fingerprints or {},
        "extra": extra or {},
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)


def load_checkpoint(path: str | Path, model, optimizer=None,
                    map_location="cpu", strict: bool = True,
                    expected_fingerprints: dict | None = None) -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    saved_name = ckpt.get("manifest", {}).get("name")
    current_name = (
        model.architecture_manifest().get("name")
        if hasattr(model, "architecture_manifest")
        else None
    )
    saved_version = ckpt.get("manifest", {}).get("version")
    current_version = (
        model.architecture_manifest().get("version")
        if hasattr(model, "architecture_manifest")
        else None
    )
    if current_name == "PureParallelGear" and saved_name in {
        "PureParallelRotatingGearLM",
        "PureParallelPredictiveGearV2",
    }:
        raise RuntimeError(
            "legacy Pure Parallel Gear V1/V2 checkpoints are intentionally "
            "incompatible with the canonical PureParallelGear architecture"
        )
    gear_architecture_names = {
        "PureParallelGear",
        "PureParallelGearV3",
        "HybridParallelGear",
        "BoundedTransformer",
        "BlockHybridGearV4",
        "SelectiveHybridGearV42",
        "GearBankRouterV43",
        "BoundedHybridGearBlockAdditive",
        "BoundedHybridGearBlockSelectiveFiLM",
        "BoundedHybridGearBlockBankRouter",
    }
    if (
        current_name in gear_architecture_names
        and saved_name is not None
        and saved_name != current_name
    ) or (
        saved_name in gear_architecture_names
        and current_name is not None
        and current_name != saved_name
    ):
        raise RuntimeError(
            "Pure Gear V2, V3, V4, hybrid, and bounded-Transformer checkpoints "
            "are intentionally architecture-specific and cannot be cross-loaded"
        )
    if (
        current_name in gear_architecture_names
        and saved_name == current_name
        and saved_version is not None
        and current_version is not None
        and saved_version != current_version
    ):
        raise RuntimeError(
            "Pure Gear checkpoint version mismatch "
            f"({saved_version} != {current_version}); cross-version loading "
            "is intentionally disabled"
        )
    if ckpt.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema {ckpt.get('schema_version')!r}")
    if strict:
        current = architecture_fingerprint(model)
        if ckpt.get("architecture_fingerprint") != current:
            raise RuntimeError(
                "checkpoint architecture does not match the current model "
                f"({ckpt.get('architecture_fingerprint')} != {current})")
        for key, value in (expected_fingerprints or {}).items():
            saved = ckpt.get("fingerprints", {}).get(key)
            if saved is not None and value is not None and saved != value:
                raise RuntimeError(
                    f"checkpoint {key} fingerprint mismatch ({saved} != {value}); "
                    "the corpus or tokenizer differs from the one trained on")
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if hasattr(model, "_runtime_commit_threshold"):
        model._runtime_commit_threshold = ckpt.get("runtime_commit_threshold")
    return ckpt
