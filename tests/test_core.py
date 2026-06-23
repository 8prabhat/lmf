"""Core framework: registry, config merge, precision policy."""

from __future__ import annotations

import pytest
import torch

from lmf import models as _models  # noqa: F401
from lmf.core.build import build
from lmf.core.config import (
    ExperimentConfig,
    apply_overrides,
    deep_merge,
    load_config,
)
from lmf.core.device import PrecisionPolicy, resolve_device, sync
from lmf.core.hashing import file_sha256, git_tree_sha256, json_sha256
from lmf.core.io import atomic_write_json
from lmf.core.registry import Registry


def test_registry_register_and_create():
    reg = Registry("widget")

    @reg.register("foo")
    def _foo(x):
        return x * 2

    assert "foo" in reg
    assert reg.create("foo", 3) == 6
    with pytest.raises(KeyError):
        reg.get("missing")


def test_registry_rejects_duplicate():
    reg = Registry("widget")
    reg.register("a")(lambda: 1)
    with pytest.raises(KeyError):
        reg.register("a")(lambda: 2)


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    overlay = {"a": {"y": 20, "z": 30}, "c": 4}
    merged = deep_merge(base, overlay)
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3, "c": 4}
    assert base["a"]["y"] == 2  # original untouched


def test_apply_overrides_typed():
    cfg = {"model": {"dim": 1}}
    apply_overrides(cfg, ["model.dim=128", "trainer.lr=0.001", "model.flag=True"])
    assert cfg["model"]["dim"] == 128
    assert cfg["trainer"]["lr"] == 0.001
    assert cfg["model"]["flag"] is True


def test_load_config_block_merge(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "base:\n  device: cpu\n  model: {dim: 1}\n"
        "smoke:\n  model: {dim: 8}\n  trainer: {lr: 0.1}\n")
    cfg = load_config(p, block="smoke")
    assert cfg.get("device") == "cpu"
    assert cfg.model["dim"] == 8
    assert cfg.trainer["lr"] == 0.1


def test_precision_policy_dtype():
    assert PrecisionPolicy("bf16").dtype == torch.bfloat16
    assert PrecisionPolicy("fp32").dtype == torch.float32
    with pytest.raises(ValueError):
        PrecisionPolicy("int8")


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_sync_accepts_str_or_device_and_is_noop_on_cpu():
    sync("cpu")
    sync(resolve_device("cpu"))


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "f.txt"
    p.write_bytes(b"hello world")
    assert file_sha256(p) == hashlib.sha256(b"hello world").hexdigest()


def test_json_sha256_is_order_independent():
    assert json_sha256({"a": 1, "b": 2}) == json_sha256({"b": 2, "a": 1})


def test_json_sha256_differs_for_different_values():
    assert json_sha256({"a": 1}) != json_sha256({"a": 2})


def test_git_tree_sha256_returns_stable_hex_digest():
    first = git_tree_sha256(paths=("pyproject.toml",))
    second = git_tree_sha256(paths=("pyproject.toml",))
    assert first == second
    assert len(first) == 64


def test_atomic_write_json_round_trips(tmp_path):
    import json

    p = tmp_path / "nested" / "out.json"
    atomic_write_json(p, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2, 3]}
    assert not p.with_suffix(p.suffix + ".tmp").exists()


def test_build_rejects_silent_model_corpus_vocabulary_mismatch():
    config = ExperimentConfig(
        {
            "device": "cpu",
            "precision": "fp32",
            "data": {"name": "procedural", "vocab_size": 64},
            "model": {
                "name": "transformer",
                "vocab_size": 32,
                "dim": 16,
                "layers": 1,
                "heads": 2,
            },
            "trainer": {"name": "transformer", "total_steps": 1},
        },
        "test",
    )
    with pytest.raises(ValueError, match="silent vocabulary replacement"):
        build(config)
