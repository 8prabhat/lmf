#!/usr/bin/env python
"""Run Gear Transformer ablations.

Examples:
    python scripts/ablate_gear_transformer.py --dry-run
    python scripts/ablate_gear_transformer.py --spec configs/ablations/gear_transformer_core.yaml --max-cells 3
"""

from __future__ import annotations

import argparse
import sys

from lmf.cli.main import main


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        default="configs/ablations/gear_transformer_vs_baselines.yaml",
        help="ablation spec YAML",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse(sys.argv[1:])
    command = ["ablate", "--config", args.spec, "--workers", str(args.workers)]
    if args.dry_run:
        command.append("--dry-run")
    if args.force:
        command.append("--force")
    if args.max_cells is not None:
        command.extend(["--max-cells", str(args.max_cells)])
    if args.only:
        command.append("--only")
        command.extend(args.only)
    main(command)
