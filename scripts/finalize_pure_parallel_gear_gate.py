#!/usr/bin/env python
"""Freeze evidence, compute uncertainty, and issue an auditable stage decision."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from lmf.ablation.stats import holm_adjust, paired_bootstrap_interval, sign_test_paired
from lmf.core.hashing import file_sha256
from lmf.research_utils import bar_chart_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("3m", "15m", "50m"), required=True)
    parser.add_argument("--scale-results", type=Path, required=True)
    parser.add_argument("--evaluation-results", type=Path, required=True)
    parser.add_argument("--generation-results", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--blind-ratings", type=Path, action="append", default=[])
    parser.add_argument("--ablation-results", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261200)
    return parser.parse_args()


def seed_training_evidence(scale: dict[str, Any], seed: int) -> dict[str, Any]:
    nll_differences = []
    gru_differences = []
    throughput_ratios = []
    equal_time_differences = []
    domain_regressions: dict[str, list[float]] = {}
    time_to_target_ratios = []

    def first_time_at_or_below(run: dict[str, Any], target: float) -> float:
        for row in run.get("validation_history", []):
            if row["metrics"]["macro_domain_nll"] <= target:
                return float(row["elapsed_seconds"])
        return float("inf")

    for run in scale["runs"]:
        equal_token = run["equal_token"]
        gear = equal_token["gear"]
        transformer = equal_token["transformer"]
        gru = equal_token["gru"]
        gear_nll = gear["validation"]["macro_domain_nll"]
        transformer_nll = transformer["validation"]["macro_domain_nll"]
        nll_differences.append(gear_nll - transformer_nll)
        gru_differences.append(gear_nll - gru["validation"]["macro_domain_nll"])
        throughput_ratios.append(
            gear["tokens_per_second"] / transformer["tokens_per_second"]
        )
        equal_time_differences.append(
            run["equal_time"]["gear"]["validation"]["macro_domain_nll"]
            - run["equal_time"]["transformer"]["validation"]["macro_domain_nll"]
        )
        target = transformer["validation"]["macro_domain_nll"]
        gear_time = first_time_at_or_below(gear, target)
        transformer_time = first_time_at_or_below(transformer, target)
        time_to_target_ratios.append(
            gear_time / max(transformer_time, 1e-9)
            if math.isfinite(gear_time)
            else 1.0e12
        )
        for domain, baseline in transformer["validation"][
            "per_domain_nll"
        ].items():
            domain_regressions.setdefault(domain, []).append(
                gear["validation"]["per_domain_nll"][domain] / baseline - 1.0
            )
    interval = paired_bootstrap_interval(
        nll_differences, seed=seed, samples=20_000
    )
    relative_upper = interval["upper"] / statistics.fmean(
        run["equal_token"]["transformer"]["validation"]["macro_domain_nll"]
        for run in scale["runs"]
    )
    return {
        "nll_difference": interval,
        "relative_nll_upper_95pct": relative_upper,
        "gear_minus_gru_nll": paired_bootstrap_interval(
            gru_differences, seed=seed + 1, samples=20_000
        ),
        "equal_time_nll_difference": paired_bootstrap_interval(
            equal_time_differences, seed=seed + 2, samples=20_000
        ),
        "minimum_throughput_ratio": min(throughput_ratios),
        "mean_throughput_ratio": statistics.fmean(throughput_ratios),
        "domain_mean_regressions": {
            domain: statistics.fmean(values)
            for domain, values in domain_regressions.items()
        },
        "nll_sign_test_p": sign_test_paired(nll_differences),
        "time_to_transformer_target_ratios": time_to_target_ratios,
        "maximum_time_to_target_ratio": max(time_to_target_ratios),
    }


def task_evidence(evaluation: dict[str, Any]) -> dict[str, Any]:
    models = evaluation["models"]
    gear = models["gear"]["predictive_tasks"]
    transformer = models["transformer"]["predictive_tasks"]
    regressions = {}
    for task in gear:
        values = []
        for distance, metrics in gear[task].items():
            values.append(
                metrics["accuracy"] - transformer[task][distance]["accuracy"]
            )
        regressions[task] = min(values)
    return {
        "minimum_accuracy_margin_by_task": regressions,
        "worst_accuracy_margin": min(regressions.values()),
    }


def _rating_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return path.stem, payload
    return str(payload.get("rater_id", path.stem)), list(payload["ratings"])


def blind_evidence(
    rating_paths: list[Path],
    key_path: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    if len(rating_paths) != 3:
        return {
            "complete": False,
            "reason": "exactly three independent rating files are required",
        }
    key = {
        row["id"]: row for row in json.loads(key_path.read_text())["blind_key"]
    }
    dimensions = (
        "coherence",
        "relevance",
        "grammar",
        "factual_consistency",
        "non_repetition",
        "useful_novelty",
    )
    raters = []
    by_item: dict[str, list[str]] = {}
    gear_scores = []
    dimension_scores = {name: [] for name in dimensions}
    for path in rating_paths:
        rater, rows = _rating_rows(path)
        raters.append(rater)
        seen = set()
        for row in rows:
            item_id = row["id"]
            if item_id in seen or item_id not in key:
                raise ValueError(f"invalid or duplicate blind item {item_id!r}")
            seen.add(item_id)
            preference = str(row["preference"]).upper()
            if preference not in {"A", "B", "TIE"}:
                raise ValueError("preference must be A, B, or tie")
            if preference == "TIE":
                winner = "tie"
                gear_scores.append(0.5)
            else:
                winner = key[item_id][preference]
                gear_scores.append(1.0 if winner == "gear" else 0.0)
            scores = row.get("dimensions", row.get("scores"))
            if not isinstance(scores, dict) or any(
                name not in scores for name in dimensions
            ):
                raise ValueError(
                    "every blind rating must score all six declared dimensions"
                )
            for name in dimensions:
                choice = str(scores[name]).upper()
                if choice == "TIE":
                    dimension_scores[name].append(0.5)
                elif choice in {"A", "B"}:
                    dimension_scores[name].append(
                        1.0 if key[item_id][choice] == "gear" else 0.0
                    )
                else:
                    raise ValueError(
                        f"dimension {name} must be A, B, or tie"
                    )
            by_item.setdefault(item_id, []).append(winner)
    if len(set(raters)) != 3:
        raise ValueError("blind ratings must have three distinct rater IDs")
    if set(by_item) != set(key) or any(
        len(rows) != 3 for rows in by_item.values()
    ):
        return {
            "complete": False,
            "reason": "every blind item must be rated by all three raters",
            "raters": raters,
        }
    interval = paired_bootstrap_interval(
        [value - 0.5 for value in gear_scores],
        seed=seed,
        samples=20_000,
    )
    categories = ("gear", "transformer", "tie")
    agreement = 0.0
    category_totals = {category: 0 for category in categories}
    for ratings in by_item.values():
        counts = {category: ratings.count(category) for category in categories}
        agreement += (
            sum(count * count for count in counts.values()) - 3
        ) / (3 * 2)
        for category, count in counts.items():
            category_totals[category] += count
    observed = agreement / max(len(by_item), 1)
    total = 3 * len(by_item)
    expected = sum(
        (count / max(total, 1)) ** 2 for count in category_totals.values()
    )
    kappa = (
        (observed - expected) / (1.0 - expected)
        if expected < 1.0
        else (1.0 if observed == 1.0 else 0.0)
    )
    return {
        "complete": True,
        "raters": raters,
        "items": len(by_item),
        "gear_preference": {
            "mean": interval["mean"] + 0.5,
            "lower_95pct": interval["lower"] + 0.5,
            "upper_95pct": interval["upper"] + 0.5,
        },
        "fleiss_kappa": kappa,
        "category_totals": category_totals,
        "dimension_gear_preference": {
            name: statistics.fmean(values)
            for name, values in dimension_scores.items()
        },
    }


def efficiency_evidence(evaluation: dict[str, Any]) -> dict[str, Any]:
    models = evaluation["models"]
    gear = models["gear"]
    transformer = models["transformer"]
    gear_contexts = {
        row["length"]: row
        for row in gear["efficiency"]["contexts"]
        if not row.get("failed")
    }
    transformer_contexts = {
        row["length"]: row
        for row in transformer["efficiency"]["contexts"]
        if not row.get("failed")
    }
    common = sorted(set(gear_contexts) & set(transformer_contexts))
    long = max(common)
    memory_ratio = None
    gear_memory = gear["training_memory"]
    transformer_memory = transformer["training_memory"]
    if (
        gear_memory.get("available")
        and transformer_memory.get("available")
        and gear_memory.get("peak_available", False)
        and transformer_memory.get("peak_available", False)
    ):
        memory_ratio = gear_memory["peak_proxy_bytes"] / max(
            transformer_memory["peak_proxy_bytes"], 1
        )
    return {
        "long_context": long,
        "incremental_speedup": transformer_contexts[long][
            "incremental_p50_seconds"
        ]
        / gear_contexts[long]["incremental_p50_seconds"],
        "cache_ratio": gear_contexts[long]["cache_bytes"]
        / max(transformer_contexts[long]["cache_bytes"], 1),
        "training_memory_ratio": memory_ratio,
    }


def generation_evidence(generation: dict[str, Any]) -> dict[str, Any]:
    aggregate = generation["aggregate"]
    output = {}
    for length in aggregate["gear"]:
        gear = aggregate["gear"][length]
        transformer = aggregate["transformer"][length]
        output[length] = {
            "reference_nll_difference": gear["reference_nll"]
            - transformer["reference_nll"],
            "repeated_4gram_difference": gear["decoding"]["t09_p095"][
                "repeated_4gram_fraction"
            ]
            - transformer["decoding"]["t09_p095"][
                "repeated_4gram_fraction"
            ],
            "distinct_3_difference": gear["decoding"]["t09_p095"]["distinct_3"]
            - transformer["decoding"]["t09_p095"]["distinct_3"],
        }
    return output


def integrity_evidence(
    scale: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    expected = int(scale["protocol"]["seeds"])
    gradients = evaluation["models"]["gear"]["gradient_health"]
    profiler = evaluation["models"]["gear"]["profiler_contract"]
    return {
        "training_complete": bool(scale.get("training_complete")),
        "expected_seeds": expected,
        "completed_seeds": len(scale.get("runs", [])),
        "failed_gradient_parameters": sorted(
            set(gradients.get("missing", ()))
            | set(gradients.get("nonfinite", ()))
        ),
        "profiler_passed": (
            bool(profiler.get("available")) and bool(profiler.get("passed"))
        ),
    }


def ablation_evidence(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"complete": False, "reason": "no retrained ablation artifact"}
    rows = payload.get("component_evidence", {})
    return {
        "complete": bool(payload.get("complete")) and bool(rows),
        "components": rows,
        "supported_components": sorted(
            name for name, value in rows.items() if value["retain_component"]
        ),
        "unsupported_components": sorted(
            name for name, value in rows.items() if not value["retain_component"]
        ),
    }


def decision(
    stage: str,
    training: dict[str, Any],
    tasks: dict[str, Any],
    blind: dict[str, Any],
    efficiency: dict[str, Any],
    integrity: dict[str, Any],
    ablations: dict[str, Any],
) -> dict[str, Any]:
    common = {
        "all_seed_runs_completed": (
            integrity["training_complete"]
            and integrity["completed_seeds"] == integrity["expected_seeds"]
        ),
        "no_missing_or_nonfinite_gradients": not integrity[
            "failed_gradient_parameters"
        ],
        "profiler_contract_passed": integrity["profiler_passed"],
        "training_memory_measured": efficiency[
            "training_memory_ratio"
        ]
        is not None,
        "training_throughput_at_least_half_transformer": training[
            "minimum_throughput_ratio"
        ]
        >= 0.5,
        "training_memory_at_most_1_25x": (
            efficiency["training_memory_ratio"] is not None
            and efficiency["training_memory_ratio"] <= 1.25
        ),
    }
    if stage == "3m":
        checks = {
            **common,
            "macro_nll_upper_95pct_within_3pct": training[
                "relative_nll_upper_95pct"
            ]
            <= 0.03,
            "no_domain_mean_over_5pct_worse": max(
                training["domain_mean_regressions"].values()
            )
            <= 0.05,
            "no_task_regression_over_5_points": tasks[
                "worst_accuracy_margin"
            ]
            >= -0.05,
            "retrained_ablation_program_complete": ablations["complete"],
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "quality_success_efficiency_not_achieved": False,
        }
    checks = {
        **common,
        "macro_nll_upper_95pct_within_1pct": training[
            "relative_nll_upper_95pct"
        ]
        <= 0.01,
        "no_domain_mean_over_2pct_worse": max(
            training["domain_mean_regressions"].values()
        )
        <= 0.02,
        "no_task_regression_over_3_points": tasks["worst_accuracy_margin"]
        >= -0.03,
        "gear_beats_gru": training["gear_minus_gru_nll"]["upper"] < 0.0,
        "blind_review_complete": bool(blind.get("complete")),
        "blind_preference_lower_at_least_45pct": bool(blind.get("complete"))
        and blind["gear_preference"]["lower_95pct"] >= 0.45,
    }
    significant_nll = training["nll_difference"]["upper"] < 0.0
    significant_blind = bool(blind.get("complete")) and blind[
        "gear_preference"
    ]["lower_95pct"] > 0.5
    checks["one_primary_quality_metric_significantly_better"] = (
        significant_nll or significant_blind
    )
    if stage == "50m":
        checks.update(
            {
                "time_to_target_noninferior": training[
                    "maximum_time_to_target_ratio"
                ]
                <= 1.0,
                "incremental_generation_at_least_1_5x": efficiency[
                    "incremental_speedup"
                ]
                >= 1.5,
                "cache_at_most_25pct": efficiency["cache_ratio"] <= 0.25,
                "training_memory_at_most_80pct": (
                    efficiency["training_memory_ratio"] is not None
                    and efficiency["training_memory_ratio"] <= 0.8
                ),
            }
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "quality_success_efficiency_not_achieved": (
            stage == "50m"
            and all(
                value
                for name, value in checks.items()
                if name
                not in {
                    "incremental_generation_at_least_1_5x",
                    "cache_at_most_25pct",
                    "training_memory_at_most_80pct",
                    "time_to_target_noninferior",
                }
            )
            and not all(
                checks[name]
                for name in (
                    "incremental_generation_at_least_1_5x",
                    "cache_at_most_25pct",
                    "training_memory_at_most_80pct",
                    "time_to_target_noninferior",
                )
            )
        ),
    }


def markdown_report(result: dict[str, Any]) -> str:
    gate = result["gate"]
    lines = [
        f"# Pure Parallel Gear {result['stage']} Evidence Report",
        "",
        f"Final decision: **{'PASS' if gate['passed'] else 'FAIL / INCONCLUSIVE'}**",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'} — `{name}`"
        for name, value in gate["checks"].items()
    )
    lines.extend(
        [
            "",
        "## Primary evidence",
        "",
        "![Seed-level NLL differences](seed_nll_differences.svg)",
        "",
        "![Long-context efficiency ratios](efficiency_ratios.svg)",
        "",
        "```json",
            json.dumps(
                {
                    "training": result["training"],
                    "tasks": result["tasks"],
                    "blind": result["blind"],
                    "efficiency": result["efficiency"],
                    "generation": result["generation"],
                    "integrity": result["integrity"],
                    "ablations": result["ablations"],
                    "holm_adjusted_p": result["holm_adjusted_p"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Explicit limitations",
            "",
            "- Finite gear state cannot guarantee exact arbitrary-length recall.",
            "- Coherent or novel generation cannot be proven theoretically; it is accepted only through empirical evidence.",
            "- Missing three-rater blind evidence makes the quality gate inconclusive, never passed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    scale = json.loads(args.scale_results.read_text())
    evaluation = json.loads(args.evaluation_results.read_text())
    generation = json.loads(args.generation_results.read_text())
    protocol = json.loads(args.protocol.read_text())
    ablation_payload = (
        None
        if args.ablation_results is None
        else json.loads(args.ablation_results.read_text())
    )
    if scale.get("stage") != args.stage:
        raise ValueError("scale stage does not match --stage")
    training = seed_training_evidence(scale, args.seed)
    tasks = task_evidence(evaluation)
    blind = blind_evidence(
        args.blind_ratings,
        args.blind_key,
        seed=args.seed + 10,
    )
    efficiency = efficiency_evidence(evaluation)
    generated = generation_evidence(generation)
    integrity = integrity_evidence(scale, evaluation)
    ablations = ablation_evidence(ablation_payload)
    p_values = {
        "nll": training["nll_sign_test_p"],
        "blind_preference": (
            1.0
            if not blind.get("complete")
            else sign_test_paired(
                [
                    1.0
                    if index < blind["category_totals"]["gear"]
                    else -1.0
                    for index in range(
                        blind["category_totals"]["gear"]
                        + blind["category_totals"]["transformer"]
                    )
                ]
            )
        ),
    }
    result = {
        "stage": args.stage,
        "protocol": protocol,
        "training": training,
        "tasks": tasks,
        "blind": blind,
        "efficiency": efficiency,
        "generation": generated,
        "integrity": integrity,
        "ablations": ablations,
        "holm_adjusted_p": holm_adjust(p_values),
        "source_hashes": {
            str(path): file_sha256(path)
            for path in (
                args.scale_results,
                args.evaluation_results,
                args.generation_results,
                args.blind_key,
                args.protocol,
                *args.blind_ratings,
                *(
                    ()
                    if args.ablation_results is None
                    else (args.ablation_results,)
                ),
            )
        },
    }
    result["gate"] = decision(
        args.stage,
        training,
        tasks,
        blind,
        efficiency,
        integrity,
        ablations,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nll_values = [
        run["equal_token"]["gear"]["validation"]["macro_domain_nll"]
        - run["equal_token"]["transformer"]["validation"]["macro_domain_nll"]
        for run in scale["runs"]
    ]
    bar_chart_svg(
        args.output_dir / "seed_nll_differences.svg",
        [f"seed {run['seed']}" for run in scale["runs"]],
        nll_values,
        "Gear minus Transformer macro NLL (lower is better)",
        diverging=True,
    )
    bar_chart_svg(
        args.output_dir / "efficiency_ratios.svg",
        ["generation speedup", "cache ratio", "memory ratio"],
        [
            efficiency["incremental_speedup"],
            efficiency["cache_ratio"],
            (
                efficiency["training_memory_ratio"]
                if efficiency["training_memory_ratio"] is not None
                else 0.0
            ),
        ],
        "Long-context efficiency ratios",
        diverging=True,
        baseline=1.0,
    )
    (args.output_dir / "final_evidence.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    (args.output_dir / "report.md").write_text(markdown_report(result))
    print(json.dumps(result["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
