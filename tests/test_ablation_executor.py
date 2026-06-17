"""End-to-end run_ablation orchestration + resume + report."""

from __future__ import annotations

import yaml

from lmf.ablation.executor import run_ablation
from lmf.ablation.report import build_report, write_report
from lmf.ablation.spec import AblationSpec, AxisSpec

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


def _spec(config_path, results_dir):
    return AblationSpec(
        name="tiny_sweep", base_config=str(config_path), base_block="smoke", mode="one_at_a_time",
        axes=[AxisSpec(path="model.dim", values=[32])],
        seeds=[0], run={"steps": 2, "batch_size": 2, "seq_len": 16}, eval={"n_batches": 1},
        results_dir=str(results_dir))


def test_run_ablation_dry_run(tmp_path):
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(TINY_CONFIG))
    spec = _spec(config_path, tmp_path / "results")

    out = run_ablation(spec, dry_run=True)
    assert out["dry_run"] is True
    assert out["n_cells"] == 2  # baseline + dim=32
    cell_ids = {c["cell_id"] for c in out["cells"]}
    assert cell_ids == {"baseline__seed0", "model.dim=32__seed0"}


def test_run_ablation_end_to_end_and_resume(tmp_path):
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(TINY_CONFIG))
    results_dir = tmp_path / "results"
    spec = _spec(config_path, results_dir)

    out1 = run_ablation(spec)
    assert out1["n_cells"] == 2
    assert out1["n_run"] == 2
    assert out1["n_skipped"] == 0

    # Second run resumes: everything already has a result, nothing re-run.
    out2 = run_ablation(spec)
    assert out2["n_run"] == 0
    assert out2["n_skipped"] == 2

    report = build_report(results_dir, spec=spec)
    assert report["baseline"] is not None
    cell_ids = {c["cell_id"] for c in report["cells"]}
    assert cell_ids == {"baseline", "model.dim=32"}

    md_path = write_report(results_dir, report, fmt="md")
    assert md_path.exists()
    assert "model.dim=32" in md_path.read_text()


def test_run_ablation_force_reruns(tmp_path):
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(TINY_CONFIG))
    results_dir = tmp_path / "results"
    spec = _spec(config_path, results_dir)

    run_ablation(spec)
    out = run_ablation(spec, force=True)
    assert out["n_run"] == 2
    assert out["n_skipped"] == 0
