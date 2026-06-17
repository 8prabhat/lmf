"""Ablation result storage: atomic write/load, resume detection."""

from __future__ import annotations

from lmf.ablation.storage import CellResult, has_result, load_result, load_results, result_path, write_result


def _result(cell_id="baseline", seed=0, status="ok", bpt=1.23) -> CellResult:
    return CellResult(cell_id=cell_id, seed=seed, status=status, metrics={"bits_per_token": bpt})


def test_write_and_load_roundtrip(tmp_path):
    result = _result()
    path = write_result(tmp_path, result)
    assert path == result_path(tmp_path, "baseline", 0)
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()

    loaded = load_result(tmp_path, "baseline", 0)
    assert loaded == result


def test_load_result_missing_returns_none(tmp_path):
    assert load_result(tmp_path, "nope", 0) is None


def test_load_result_corrupt_returns_none(tmp_path):
    path = result_path(tmp_path, "broken", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert load_result(tmp_path, "broken", 0) is None


def test_load_results_skips_tmp_and_corrupt(tmp_path):
    write_result(tmp_path, _result(cell_id="a", seed=0))
    write_result(tmp_path, _result(cell_id="b", seed=0))
    (tmp_path / "cells" / "b__seed0.json.tmp").write_text("partial")
    (tmp_path / "cells" / "corrupt__seed0.json").write_text("{bad")

    results = load_results(tmp_path)
    cell_ids = {r.cell_id for r in results}
    assert cell_ids == {"a", "b"}


def test_has_result_resume_semantics(tmp_path):
    assert has_result(tmp_path, "baseline", 0) is False

    write_result(tmp_path, _result(status="failed"))
    assert has_result(tmp_path, "baseline", 0) is False  # failed cells are retried

    write_result(tmp_path, _result(status="ok"))
    assert has_result(tmp_path, "baseline", 0) is True
