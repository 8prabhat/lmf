"""Aggregate CellResults into summary.json / summary.md / summary.csv (+ optional plots)."""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

from .spec import AblationSpec
from .stats import compare_to_baseline, mean_std_stderr
from .storage import CellResult, load_results

_SEED_SUFFIX = re.compile(r"__seed-?\d+$")


def _group_id(cell_id: str) -> str:
    """Strip the ``__seed{N}`` suffix so all seeds of one cell aggregate together."""
    return _SEED_SUFFIX.sub("", cell_id)


def _group_by_cell(results: list[CellResult]) -> dict[str, list[CellResult]]:
    """Group results across seeds: ``"model.dim=32__seed0"`` and ``"...__seed1"``
    both fall under group id ``"model.dim=32"``."""
    groups: dict[str, list[CellResult]] = {}
    for r in results:
        groups.setdefault(_group_id(r.cell_id), []).append(r)
    return groups


def build_report(results_dir: str | Path, spec: AblationSpec | None = None) -> dict[str, Any]:
    results = load_results(results_dir)
    groups = _group_by_cell(results)
    metric = spec.metric if spec is not None else "bits_per_token"

    def _values(group: list[CellResult]) -> list[float]:
        return [r.metrics[metric] for r in group if r.status == "ok" and metric in r.metrics]

    def _is_baseline(cell_id: str) -> bool:
        return cell_id == "baseline"

    baseline_values: list[float] = []
    for cell_id, group in groups.items():
        if _is_baseline(cell_id):
            baseline_values.extend(_values(group))
    baseline_summary = mean_std_stderr(baseline_values) if baseline_values else None

    cells: list[dict[str, Any]] = []
    for cell_id, group in groups.items():
        values = _values(group)
        status_counts: dict[str, int] = {}
        for r in group:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
        entry: dict[str, Any] = {
            "cell_id": cell_id,
            "axis_values": group[0].axis_values,
            "metrics_summary": {metric: mean_std_stderr(values)} if values else {},
            "status_counts": status_counts,
            "vs_baseline": None,
        }
        if not _is_baseline(cell_id) and values and baseline_values:
            entry["vs_baseline"] = compare_to_baseline(values, baseline_values)
        cells.append(entry)

    # Stable order: baseline first, then by cell_id.
    cells.sort(key=lambda c: (not _is_baseline(c["cell_id"]), c["cell_id"]))

    axis_importance: list[dict[str, Any]] = []
    if spec is not None and spec.mode == "one_at_a_time":
        ranked = [c for c in cells if not _is_baseline(c["cell_id"]) and c["vs_baseline"]]
        ranked.sort(key=lambda c: abs(c["vs_baseline"]["delta_mean"]), reverse=True)
        axis_importance = [
            {"cell_id": c["cell_id"], "axis_values": c["axis_values"],
             "delta_mean": c["vs_baseline"]["delta_mean"]}
            for c in ranked
        ]

    return {
        "results_dir": str(results_dir),
        "metric": metric,
        "baseline": baseline_summary,
        "cells": cells,
        "axis_importance": axis_importance,
    }


def _write_markdown(report: dict[str, Any]) -> str:
    metric = report["metric"]
    lines = [f"# Ablation report: {report['results_dir']}", "", f"Metric: `{metric}`", ""]
    lines.append("| cell_id | " + metric + " (mean +/- stderr) | delta vs baseline | cohen's d | p_value | status |")
    lines.append("|---|---|---|---|---|---|")
    for cell in report["cells"]:
        summary = cell["metrics_summary"].get(metric)
        mean_str = f"{summary['mean']:.4f} +/- {summary['stderr']:.4f}" if summary else "-"
        vs = cell["vs_baseline"]
        if vs:
            delta_str = f"{vs['delta_mean']:+.4f}"
            d_str = f"{vs['cohens_d']:.3f}"
            p_str = f"{vs['welch_t_test']['p_value']:.4f}"
        else:
            delta_str = d_str = p_str = "-"
        status_str = ", ".join(f"{k}={v}" for k, v in cell["status_counts"].items())
        lines.append(f"| {cell['cell_id']} | {mean_str} | {delta_str} | {d_str} | {p_str} | {status_str} |")

    if report["axis_importance"]:
        lines += ["", "## Axis importance (|delta| vs baseline, descending)", ""]
        for entry in report["axis_importance"]:
            lines.append(f"- `{entry['cell_id']}` ({entry['axis_values']}): delta={entry['delta_mean']:+.4f}")

    return "\n".join(lines) + "\n"


def _write_csv(report: dict[str, Any]) -> str:
    metric = report["metric"]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["cell_id", f"{metric}_mean", f"{metric}_stderr", "delta_vs_baseline",
                      "cohens_d", "p_value", "status_counts"])
    for cell in report["cells"]:
        summary = cell["metrics_summary"].get(metric)
        vs = cell["vs_baseline"]
        writer.writerow([
            cell["cell_id"],
            summary["mean"] if summary else "",
            summary["stderr"] if summary else "",
            vs["delta_mean"] if vs else "",
            vs["cohens_d"] if vs else "",
            vs["welch_t_test"]["p_value"] if vs else "",
            json.dumps(cell["status_counts"]),
        ])
    return buf.getvalue()


def write_report(results_dir: str | Path, report: dict[str, Any], fmt: str = "md",
                 out: str | Path | None = None) -> Path:
    if fmt == "json":
        content = json.dumps(report, indent=2, default=str)
        default_name = "summary.json"
    elif fmt == "md":
        content = _write_markdown(report)
        default_name = "summary.md"
    elif fmt == "csv":
        content = _write_csv(report)
        default_name = "summary.csv"
    else:
        raise ValueError(f"unknown report format {fmt!r}")

    path = Path(out) if out else Path(results_dir) / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def maybe_plot(results_dir: str | Path, report: dict[str, Any], spec: AblationSpec | None = None) -> list[Path]:
    """Best-effort matplotlib plots; returns ``[]`` if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    metric = report["metric"]
    cells = [c for c in report["cells"] if c["cell_id"] != "baseline" and c["vs_baseline"]]
    if not cells:
        return []

    out_dir = Path(results_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [c["cell_id"] for c in cells]
    deltas = [c["vs_baseline"]["delta_mean"] for c in cells]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.6), 4))
    ax.bar(labels, deltas)
    ax.set_ylabel(f"delta {metric} vs baseline")
    ax.set_title(f"Ablation: {report['results_dir']}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    path = out_dir / "delta_vs_baseline.png"
    fig.savefig(path)
    plt.close(fig)
    return [path]
