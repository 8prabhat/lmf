"""Report aggregation: multi-seed cells must be combined into one statistical
group, keyed by cell_id with the ``__seed{N}`` suffix stripped."""

from __future__ import annotations

from lmf.ablation.report import build_report
from lmf.ablation.storage import CellResult, write_result


def _result(cell_id, seed, bpt) -> CellResult:
    return CellResult(cell_id=f"{cell_id}__seed{seed}", seed=seed, status="ok",
                       axis_values={} if cell_id == "baseline" else {"model.dim": 32},
                       metrics={"bits_per_token": bpt})


def test_multi_seed_cells_aggregate_into_one_group(tmp_path):
    for seed, bpt in [(0, 10.0), (1, 12.0), (2, 11.0)]:
        write_result(tmp_path, _result("baseline", seed, bpt))
    for seed, bpt in [(0, 8.0), (1, 9.0), (2, 7.0)]:
        write_result(tmp_path, _result("model.dim=32", seed, bpt))

    report = build_report(tmp_path)

    assert report["baseline"]["n"] == 3
    assert report["baseline"]["mean"] == 11.0

    cells = {c["cell_id"]: c for c in report["cells"]}
    assert set(cells) == {"baseline", "model.dim=32"}

    cell = cells["model.dim=32"]
    summary = cell["metrics_summary"]["bits_per_token"]
    assert summary["n"] == 3
    assert summary["mean"] == 8.0

    vs = cell["vs_baseline"]
    assert vs is not None
    assert vs["cell"]["n"] == 3
    assert vs["baseline"]["n"] == 3
