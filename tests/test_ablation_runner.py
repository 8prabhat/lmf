"""End-to-end run_cell on a tiny transformer config."""

from __future__ import annotations

import yaml

from lmf.ablation.matrix import build_matrix
from lmf.ablation.runner import run_cell
from lmf.ablation.spec import AblationSpec, AxisSpec
from lmf.core.config import load_config

TINY_CONFIG = {
    "default_block": "smoke",
    "base": {"seed": 0, "device": "cpu", "precision": "fp32",
             "data": {"name": "procedural", "vocab_size": 64}},
    "smoke": {
        "model": {"name": "transformer", "vocab_size": 64, "dim": 16, "layers": 1, "heads": 2,
                  "max_seq_len": 64},
        "trainer": {"name": "transformer", "lr": 3.0e-3, "warmup_steps": 1, "total_steps": 10},
        "run": {"batch_size": 2, "seq_len": 16, "steps": 2},
    },
}


def _write_config(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(TINY_CONFIG))
    return path


def test_run_cell_ok(tmp_path):
    config_path = _write_config(tmp_path)
    spec = AblationSpec(
        name="tiny", base_config=str(config_path), base_block="smoke", mode="one_at_a_time",
        seeds=[0], run={"steps": 2, "batch_size": 2, "seq_len": 16}, eval={"n_batches": 1})
    base_cfg = load_config(spec.base_config, spec.base_block, spec.base_env)
    base_raw = dict(base_cfg.raw)
    base_raw["__block__"] = base_cfg.block

    cells = build_matrix(spec, base_cfg.raw)
    assert len(cells) == 1  # baseline only, no axes

    result = run_cell(cells[0], base_raw, spec)
    assert result.status == "ok"
    assert "bits_per_token" in result.metrics
    assert result.params_total > 0
    assert result.architecture_fingerprint


def test_run_cell_failed_on_invalid_override(tmp_path):
    config_path = _write_config(tmp_path)
    spec = AblationSpec(
        name="tiny_bad", base_config=str(config_path), base_block="smoke", mode="one_at_a_time",
        axes=[AxisSpec(path="model.heads", values=[4])],  # dim=16 % heads=4 == 0... pick bad combo
        seeds=[0], run={"steps": 2, "batch_size": 2, "seq_len": 16}, eval={"n_batches": 1})
    # Force an invalid combo directly: dim=16, heads not dividing dim.
    spec = AblationSpec(
        name="tiny_bad", base_config=str(config_path), base_block="smoke", mode="one_at_a_time",
        axes=[AxisSpec(path="model.heads", values=[5])],
        seeds=[0], run={"steps": 2, "batch_size": 2, "seq_len": 16}, eval={"n_batches": 1})
    base_cfg = load_config(spec.base_config, spec.base_block, spec.base_env)
    base_raw = dict(base_cfg.raw)
    base_raw["__block__"] = base_cfg.block

    cells = build_matrix(spec, base_cfg.raw)
    bad_cell = next(c for c in cells if not c.is_baseline)

    result = run_cell(bad_cell, base_raw, spec)
    assert result.status == "failed"
    assert "dim must be divisible by heads" in result.error
