#!/usr/bin/env python3
"""Apply paired quality gates to a completed Bounded Hybrid Gear stage."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from lmf.ablation.stats import analytic_confidence_interval as paired_interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-margin", type=float, required=True)
    parser.add_argument("--require-gear-value", action="store_true")
    parser.add_argument(
        "--candidate",
        default="bounded_hybrid_gear_block_additive",
    )
    return parser.parse_args()


def indexed_runs(report: dict, name: str) -> dict[int, dict]:
    runs = report["runs"][name]
    indexed = {int(run["seed"]): run for run in runs}
    if len(indexed) != len(runs):
        raise ValueError(f"{name} contains duplicate seed results")
    return indexed


def main() -> None:
    args = parse_args()
    report = json.loads(args.results.read_text())
    v4 = indexed_runs(report, args.candidate)
    expected_seeds = {int(seed) for seed in report.get("seeds", ())}
    if len(expected_seeds) < 3:
        raise ValueError("paired quality gates require at least three seeds")
    for name in (
        args.candidate,
        "bounded_transformer",
        "full_transformer",
    ):
        found = set(indexed_runs(report, name))
        if found != expected_seeds:
            raise ValueError(
                f"{name} seed set {sorted(found)} does not match "
                f"expected {sorted(expected_seeds)}"
            )
    comparisons = {}
    for baseline in ("bounded_transformer", "full_transformer"):
        control = indexed_runs(report, baseline)
        shared = sorted(set(v4) & set(control))
        absolute = [
            v4[seed]["validation"]["macro_domain_nll"]
            - control[seed]["validation"]["macro_domain_nll"]
            for seed in shared
        ]
        relative = [
            v4[seed]["validation"]["macro_domain_nll"]
            / control[seed]["validation"]["macro_domain_nll"]
            - 1.0
            for seed in shared
        ]
        comparisons[baseline] = {
            "seeds": shared,
            "absolute_nll_difference": paired_interval(absolute),
            "relative_nll_difference": paired_interval(relative),
        }

    domains = {}
    for baseline in ("bounded_transformer", "full_transformer"):
        control = indexed_runs(report, baseline)
        domain_differences = {}
        for domain in next(iter(v4.values()))["validation"]["per_domain_nll"]:
            differences = [
                v4[seed]["validation"]["per_domain_nll"][domain]
                / control[seed]["validation"]["per_domain_nll"][domain]
                - 1.0
                for seed in sorted(set(v4) & set(control))
            ]
            domain_differences[domain] = paired_interval(differences)
        domains[baseline] = domain_differences

    full_speed = statistics.fmean(
        run["tokens_per_second"]
        for run in report["runs"]["full_transformer"]
    )
    v4_speed = statistics.fmean(
        run["tokens_per_second"]
        for run in report["runs"][args.candidate]
    )
    checks = {
        "throughput_at_least_half_transformer": v4_speed / full_speed >= 0.5,
        "within_quality_margin_bounded": comparisons[
            "bounded_transformer"
        ]["relative_nll_difference"]["upper"] <= args.quality_margin,
        "within_quality_margin_full": comparisons[
            "full_transformer"
        ]["relative_nll_difference"]["upper"] <= args.quality_margin,
        "no_domain_over_2pct_worse_bounded": max(
            value["mean"]
            for value in domains["bounded_transformer"].values()
        )
        <= 0.02,
        "no_domain_over_2pct_worse_full": max(
            value["mean"]
            for value in domains["full_transformer"].values()
        )
        <= 0.02,
        "gear_removal_measurably_hurts": comparisons[
            "bounded_transformer"
        ]["absolute_nll_difference"]["upper"]
        < 0.0,
    }
    required = [
        "throughput_at_least_half_transformer",
        "within_quality_margin_bounded",
        "within_quality_margin_full",
    ]
    if args.require_gear_value:
        required.append("gear_removal_measurably_hurts")
    output = {
        "source": str(args.results),
        "candidate": args.candidate,
        "quality_margin": args.quality_margin,
        "require_gear_value": args.require_gear_value,
        "throughput_ratio": v4_speed / full_speed,
        "comparisons": comparisons,
        "domain_relative_differences": domains,
        "checks": checks,
        "required_checks": required,
        "passed": all(checks[name] for name in required),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
