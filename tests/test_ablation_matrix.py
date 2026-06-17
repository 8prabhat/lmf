"""Ablation cell-matrix expansion."""

from __future__ import annotations

import pytest

from lmf.ablation.matrix import build_matrix, cell_id_for
from lmf.ablation.spec import AblationSpec, AxisSpec, VariantSpec

BASE = {"model": {"name": "transformer", "dim": 64, "layers": 2}}


def test_grid_mode_with_dedup():
    spec = AblationSpec(
        name="grid", base_config="x.yaml", mode="grid",
        axes=[AxisSpec(path="model.dim", values=[64, 128]),
              AxisSpec(path="model.layers", values=[2, 4])],
        seeds=[0],
    )
    cells = build_matrix(spec, BASE)
    # baseline + (64,2)[dedup->alias] + (64,4) + (128,2) + (128,4) = 4 cells
    assert len(cells) == 4
    baseline = next(c for c in cells if c.cell_id == "baseline__seed0")
    assert baseline.aliases  # (64,2) combo aliased onto baseline
    assert {c.is_baseline for c in cells} == {True, False}


def test_one_at_a_time_mode():
    spec = AblationSpec(
        name="oat", base_config="x.yaml", mode="one_at_a_time",
        axes=[AxisSpec(path="model.dim", values=[64, 128]),
              AxisSpec(path="model.layers", values=[2, 4])],
        seeds=[0],
    )
    cells = build_matrix(spec, BASE)
    # baseline (with 2 aliases for the no-op values) + dim=128 + layers=4
    assert len(cells) == 3
    non_baseline_ids = {c.cell_id for c in cells if not c.is_baseline}
    assert non_baseline_ids == {"model.dim=128__seed0", "model.layers=4__seed0"}
    baseline = next(c for c in cells if c.is_baseline)
    assert len(baseline.aliases) == 2


def test_named_variants_always_injects_baseline():
    spec = AblationSpec(
        name="variants", base_config="x.yaml", mode="named_variants",
        variants=[
            VariantSpec(name="wide", overrides={"model": {"dim": 256}}),
            VariantSpec(name="deep", overrides={"model": {"layers": 8}}),
        ],
        seeds=[0],
    )
    cells = build_matrix(spec, BASE)
    cell_ids = {c.cell_id for c in cells}
    assert "baseline__seed0" in cell_ids
    assert "variant=wide__seed0" in cell_ids
    assert "variant=deep__seed0" in cell_ids
    assert len(cells) == 3


def test_named_variants_combined_with_axes():
    spec = AblationSpec(
        name="variants_axes", base_config="x.yaml", mode="named_variants",
        variants=[VariantSpec(name="no_routing", overrides={})],
        axes=[AxisSpec(path="loss_term:routing_balance", values=[1.0, 0.0])],
        seeds=[0],
    )
    cells = build_matrix(spec, BASE)
    variant_cells = [c for c in cells if c.variant_name == "no_routing"]
    assert len(variant_cells) == 2
    scales = {c.loss_term_scales.get("routing_balance") for c in variant_cells}
    assert scales == {1.0, 0.0}


def test_pairwise_mode():
    spec = AblationSpec(
        name="pairwise", base_config="x.yaml", mode="pairwise",
        axes=[AxisSpec(path="model.dim", values=[64, 128]),
              AxisSpec(path="model.layers", values=[2, 4])],
        seeds=[0],
    )
    cells = build_matrix(spec, BASE)
    # baseline + (64,2)[dedup] + (64,4) + (128,2) + (128,4) = 4
    assert len(cells) == 4


def test_seeds_multiply_cells():
    spec = AblationSpec(
        name="seeded", base_config="x.yaml", mode="one_at_a_time",
        axes=[AxisSpec(path="model.dim", values=[128])],
        seeds=[0, 1],
    )
    cells = build_matrix(spec, BASE)
    seeds = sorted({c.seed for c in cells})
    assert seeds == [0, 1]
    assert sum(1 for c in cells if c.is_baseline) == 2


def test_invalid_axis_path_raises():
    spec = AblationSpec(
        name="bad_axis", base_config="x.yaml", mode="one_at_a_time",
        axes=[AxisSpec(path="model.dim", values=[64])],  # 64 == base value, no-op
        seeds=[0],
    )
    with pytest.raises(ValueError):
        build_matrix(spec, BASE)


def test_structural_and_loss_term_axes_do_not_enter_dotted_overrides():
    spec = AblationSpec(
        name="structural", base_config="x.yaml", mode="one_at_a_time",
        axes=[AxisSpec(path="structural:blocks.skip[1]", values=[True, False]),
              AxisSpec(path="loss_term:routing_balance", values=[0.0])],
        seeds=[0],
    )
    cells = build_matrix(spec, BASE)
    for cell in cells:
        if not cell.is_baseline:
            assert cell.dotted_overrides == []
    structural_cell = next(c for c in cells if c.structural_points)
    assert structural_cell.structural_points["blocks.skip[1]"] in (True, False)
    loss_term_cell = next(c for c in cells if c.loss_term_scales)
    assert loss_term_cell.loss_term_scales["routing_balance"] == 0.0


def test_cell_id_for():
    assert cell_id_for({}, None, 0) == "baseline__seed0"
    assert cell_id_for({"model.dim": 256}, None, 1) == "model.dim=256__seed1"
    assert cell_id_for({}, "wide", 0) == "variant=wide__seed0"
    assert cell_id_for({"model.dim": 0.1}, None, 0) == "model.dim=0p1__seed0"
