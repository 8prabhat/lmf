"""Config-driven RFK runner."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ...core.config import load_config
from ...core.seeding import seed_everything
from .kernels import KERNELS


def run(config_path: str, block: str = "smoke", only: list[str] | None = None,
        out_dir: str = "outputs/rfk") -> dict:
    cfg = load_config(config_path, block=block).raw
    seed_everything(int(cfg.get("seed", 0)))
    torch.manual_seed(int(cfg.get("seed", 0)))
    names = only or list(KERNELS)
    report = {"config": config_path, "block": block, "results": []}
    for name in names:
        started = time.perf_counter()
        try:
            result = KERNELS[name](cfg)
        except Exception as exc:  # a kernel that raises is a failure, not a crash
            result = {"pass": False, "error": repr(exc)}
        result["status"] = "PASS" if result.get("pass") else "FAIL"
        result["name"] = name
        result["seconds"] = round(time.perf_counter() - started, 3)
        report["results"].append(result)
        print(f"[{result['status']}] {name} ({result['seconds']}s) "
              f"{ {k: v for k, v in result.items() if k not in {'name', 'status', 'seconds', 'pass'}} }")
    report["summary"] = {
        "passed": sum(bool(r["pass"]) for r in report["results"]),
        "total": len(report["results"]),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rfk_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\n{report['summary']['passed']}/{report['summary']['total']} kernels passed -> {path}")
    return report
