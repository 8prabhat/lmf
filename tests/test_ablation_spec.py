"""Ablation spec parsing."""

from __future__ import annotations

import pytest
import yaml

from lmf.ablation import load_ablation_spec
from lmf.ablation.spec import AxisSpec


def _write(tmp_path, doc):
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_load_grid_spec(tmp_path):
    doc = {"ablation": {
        "name": "grid_sweep", "base_config": "configs/transformer_baseline.yaml",
        "base_block": "smoke", "mode": "grid",
        "axes": [{"path": "model.dim", "values": [64, 128]}],
        "seeds": [0, 1],
    }}
    spec = load_ablation_spec(_write(tmp_path, doc))
    assert spec.name == "grid_sweep"
    assert spec.mode == "grid"
    assert spec.axes == [AxisSpec(path="model.dim", values=[64, 128])]
    assert spec.seeds == [0, 1]


def test_load_one_at_a_time_spec(tmp_path):
    doc = {"ablation": {
        "name": "oat", "base_config": "cfg.yaml", "mode": "one_at_a_time",
        "axes": [{"path": "model.dim", "values": [64, 128, 256]},
                 {"path": "model.layers", "values": [2, 4]}],
    }}
    spec = load_ablation_spec(_write(tmp_path, doc))
    assert spec.mode == "one_at_a_time"
    assert len(spec.axes) == 2
    assert spec.seeds == [0]


def test_load_named_variants_spec(tmp_path):
    doc = {"ablation": {
        "name": "variants", "base_config": "cfg.yaml", "mode": "named_variants",
        "variants": [
            {"name": "no_routing", "overrides": {"model": {"routing_balance_weight": 0.0}}},
            {"name": "wide", "overrides": {"model": {"dim": 256}}},
        ],
    }}
    spec = load_ablation_spec(_write(tmp_path, doc))
    assert spec.mode == "named_variants"
    assert [v.name for v in spec.variants] == ["no_routing", "wide"]
    assert spec.variants[0].overrides == {"model": {"routing_balance_weight": 0.0}}


def test_load_pairwise_spec(tmp_path):
    doc = {"ablation": {
        "name": "pairwise_sweep", "base_config": "cfg.yaml", "mode": "pairwise",
        "axes": [{"path": "model.dim", "values": [64, 128]},
                 {"path": "model.layers", "values": [2, 4]},
                 {"path": "model.heads", "values": [2, 4]}],
    }}
    spec = load_ablation_spec(_write(tmp_path, doc))
    assert spec.mode == "pairwise"
    assert len(spec.axes) == 3


def test_axis_spec_kind_and_target():
    config_axis = AxisSpec(path="model.dim", values=[64])
    structural_axis = AxisSpec(path="structural:blocks.skip[1]", values=[True, False])
    loss_term_axis = AxisSpec(path="loss_term:routing_balance", values=[1.0, 0.0])

    assert config_axis.kind == "config" and config_axis.target == "model.dim"
    assert structural_axis.kind == "structural" and structural_axis.target == "blocks.skip[1]"
    assert loss_term_axis.kind == "loss_term" and loss_term_axis.target == "routing_balance"


def test_missing_required_field_raises(tmp_path):
    doc = {"ablation": {"base_config": "cfg.yaml"}}  # missing "name"
    with pytest.raises(KeyError):
        load_ablation_spec(_write(tmp_path, doc))
