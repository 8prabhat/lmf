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
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if hasattr(model, "_runtime_commit_threshold"):
        model._runtime_commit_threshold = ckpt.get("runtime_commit_threshold")
    return ckpt
