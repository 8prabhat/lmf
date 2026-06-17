"""Core framework: registry, config merge, precision policy."""

from __future__ import annotations

import pytest
import torch

from lmf.core.config import apply_overrides, deep_merge, load_config
from lmf.core.device import PrecisionPolicy, resolve_device
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
