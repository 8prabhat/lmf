"""Sequential (default) or CPU-process-pool orchestration over a cell list."""

from __future__ import annotations

import signal
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..core.config import load_config
from .matrix import build_matrix
from .runner import run_cell
from .spec import AblationSpec
from .storage import has_result, write_result


def run_ablation(spec: AblationSpec, *, resume: bool = True, force: bool = False,
                 workers: int = 1, only: list[str] | None = None,
                 max_cells: int | None = None, dry_run: bool = False) -> dict:
    base_cfg = load_config(spec.base_config, spec.base_block, spec.base_env)
    base_raw = dict(base_cfg.raw)
    base_raw["__block__"] = base_cfg.block

    cells = build_matrix(spec, base_cfg.raw)
    if only:
        wanted = set(only)
        cells = [c for c in cells if c.cell_id in wanted]
    if max_cells is not None:
        cells = cells[:max_cells]

    if dry_run:
        return {
            "dry_run": True,
            "n_cells": len(cells),
            "cells": [
                {"cell_id": c.cell_id, "axis_values": c.axis_values, "overrides": c.overrides,
                 "structural_points": c.structural_points, "loss_term_scales": c.loss_term_scales,
                 "seed": c.seed, "is_baseline": c.is_baseline, "variant_name": c.variant_name,
                 "aliases": c.aliases}
                for c in cells
            ],
        }

    results_dir = spec.results_dir or f"results/{spec.name}"

    to_run = []
    n_skipped = 0
    for cell in cells:
        if resume and not force and has_result(results_dir, cell.cell_id, cell.seed):
            n_skipped += 1
            continue
        to_run.append(cell)

    n_run = 0
    interrupted = False

    def _handle_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        if workers <= 1:
            for i, cell in enumerate(to_run):
                if interrupted:
                    break
                result = run_cell(cell, base_raw, spec)
                write_result(results_dir, result)
                n_run += 1
                _print_progress(i + 1, len(to_run), result)
        else:
            if base_raw.get("device", "auto") not in ("cpu",):
                raise ValueError("workers > 1 requires base config device: cpu (process-pool safety)")
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_cell, cell, base_raw, spec): cell for cell in to_run}
                for i, future in enumerate(as_completed(futures)):
                    if interrupted:
                        break
                    result = future.result()
                    write_result(results_dir, result)
                    n_run += 1
                    _print_progress(i + 1, len(to_run), result)
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    return {"results_dir": results_dir, "n_cells": len(cells), "n_run": n_run, "n_skipped": n_skipped}


def _print_progress(i: int, total: int, result) -> None:
    bpt = result.metrics.get("bits_per_token")
    bpt_str = f"{bpt:.4f}" if bpt is not None else "-"
    print(f"[{i}/{total}] {result.cell_id} seed={result.seed} status={result.status} bpt={bpt_str}",
          flush=True)
