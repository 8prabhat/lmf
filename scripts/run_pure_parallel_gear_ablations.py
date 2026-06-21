#!/usr/bin/env python
"""Retrained proxy/3M ablations; post-hoc masking is not accepted as evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

from lmf.core.io import atomic_write_json as write_json
from lmf.data import PairedDocumentManifestCorpus
from lmf.diagnostics import parameter_count as count_parameters
from lmf.models.pure_parallel_gear import (
    PureParallelGearConfig,
    PureParallelGearLM,
)

try:
    from scripts.benchmark_pure_parallel_gear import (
        STAGES,
        configs,
        gear_parameter_count,
        train_run,
    )
except ModuleNotFoundError:
    from benchmark_pure_parallel_gear import (
        STAGES,
        configs,
        gear_parameter_count,
        train_run,
    )


VARIANTS = {
    "full": {},
    "one_bank_only": {
        "num_banks": 1,
        "bank_roles": ("single_bank",),
    },
    "no_boundary_settling": {"boundary_settling": False},
    "no_cross_bank_coupling": {"cross_bank_coupling": False},
    "commuting_coupling_only": {"overlapping_coupling": False},
    "fixed_angular_velocities": {"learned_angular_velocity": False},
    "no_load_state": {"use_load_state": False},
    "no_predictor_gear": {"use_predictor_gear": False},
    "no_local_swiglu": {"use_local_swiglu": False},
    "forced_fixed_boundaries": {"boundary_policy": "fixed"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("proxy", "3m"), required=True)
    parser.add_argument("--train-manifest-template", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--seed-start", type=int, default=20261300)
    parser.add_argument("--tune-token-fraction", type=float, default=0.02)
    return parser.parse_args()


def matched_variant(
    base: PureParallelGearConfig,
    target: int,
    overrides: dict,
) -> PureParallelGearConfig:
    best = None
    for dim in range(max(16, int(0.45 * base.dim)), int(3.0 * base.dim) + 1):
        candidate = replace(
            base,
            dim=dim,
            ffn_dim=dim,
            **overrides,
        )
        low = gear_parameter_count(candidate)
        high = gear_parameter_count(replace(candidate, ffn_dim=dim + 1))
        slope = high - low
        estimate = max(dim, round(dim + (target - low) / max(slope, 1)))
        ffn_values = (
            (dim,)
            if not candidate.use_local_swiglu
            else range(max(dim, estimate - 4), estimate + 5)
        )
        for ffn in ffn_values:
            if candidate.use_local_swiglu and ffn > 8 * dim:
                continue
            value = replace(candidate, ffn_dim=ffn)
            error = abs(gear_parameter_count(value) / target - 1.0)
            score = (
                error,
                abs(dim / base.dim - 1.0),
                ffn / max(dim, 1),
            )
            if best is None or score < best[0]:
                best = (score, value)
    if best is None or best[0][0] > 0.005:
        raise RuntimeError(f"cannot match ablation {overrides}: {best}")
    actual_error = abs(
        count_parameters(PureParallelGearLM(best[1])) / target - 1.0
    )
    if actual_error > 0.005:
        raise RuntimeError(f"actual ablation parameter error is {actual_error}")
    return best[1]


def main() -> None:
    args = parse_args()
    if args.device == "mps":
        if args.qualification is None:
            raise RuntimeError("MPS ablations require --qualification")
        if not json.loads(args.qualification.read_text()).get("qualified", False):
            raise RuntimeError("engineering qualification did not pass")
    stage = STAGES[args.stage]
    first_manifest = Path(
        args.train_manifest_template.format(seed=args.seed_start)
    )
    corpus = PairedDocumentManifestCorpus(str(first_manifest), wrap=False)
    baseline = configs(args.stage, corpus.vocab_size)
    target = count_parameters(
        __import__(
            "lmf.models.transformer",
            fromlist=["CachedTransformerLM"],
        ).CachedTransformerLM(baseline["transformer"])
    )
    variants = {
        name: matched_variant(baseline["gear"], target, overrides)
        for name, overrides in VARIANTS.items()
    }
    report = {
        "stage": args.stage,
        "protocol": {
            "retrained": True,
            "parameter_tolerance": 0.005,
            "equal_lr_search_budget": True,
            "variants": list(variants),
        },
        "parameters": {
            name: count_parameters(PureParallelGearLM(config))
            for name, config in variants.items()
        },
        "runs": [],
    }
    tuning_tokens = max(
        100_000, int(stage["tokens"] * args.tune_token_fraction)
    )
    selected_lrs = {}
    for name, config in variants.items():
        trials = []
        for lr in stage["lrs"]:
            _, result = train_run(
                "gear",
                config,
                first_manifest,
                args.validation_manifest,
                args.output_dir / "lr_trials" / name / f"{lr:g}",
                seed=args.seed_start,
                lr=lr,
                device=args.device,
                precision=args.precision,
                effective_tokens=stage["effective"],
                micro_batch=args.micro_batch_size,
                total_tokens=tuning_tokens,
                max_validation_rows=args.max_validation_rows,
                eval_every_fraction=1.0,
            )
            trials.append({"lr": lr, **result})
        best = min(
            trials,
            key=lambda row: (
                row["validation"]["macro_domain_nll"],
                row["validation"]["worst_domain_nll"],
            ),
        )
        selected_lrs[name] = best["lr"]
        report.setdefault("lr_tuning", {})[name] = {
            "selected": best["lr"],
            "trials": trials,
        }
        write_json(args.output_dir / "results.partial.json", report)
    for offset in range(stage["seeds"]):
        seed = args.seed_start + offset
        manifest = Path(args.train_manifest_template.format(seed=seed))
        run = {"seed": seed, "variants": {}}
        for name, config in variants.items():
            _, result = train_run(
                "gear",
                config,
                manifest,
                args.validation_manifest,
                args.output_dir / "runs" / name,
                seed=seed,
                lr=selected_lrs[name],
                device=args.device,
                precision=args.precision,
                effective_tokens=stage["effective"],
                micro_batch=args.micro_batch_size,
                total_tokens=stage["tokens"],
                max_validation_rows=args.max_validation_rows,
            )
            run["variants"][name] = result
        report["runs"].append(run)
        write_json(args.output_dir / "results.partial.json", report)
    full = [
        run["variants"]["full"]["validation"]["macro_domain_nll"]
        for run in report["runs"]
    ]
    report["component_evidence"] = {}
    for name in variants:
        if name == "full":
            continue
        ablated = [
            run["variants"][name]["validation"]["macro_domain_nll"]
            for run in report["runs"]
        ]
        improvements = [
            ablated_value - full_value
            for full_value, ablated_value in zip(full, ablated)
        ]
        report["component_evidence"][name] = {
            "mean_full_advantage_nll": statistics.fmean(improvements),
            "full_wins_seeds": sum(value > 0 for value in improvements),
            "retain_component": (
                statistics.fmean(improvements) > 0
                and sum(value > 0 for value in improvements)
                >= max(2, len(improvements) - 1)
            ),
        }
    report["complete"] = True
    write_json(args.output_dir / "results.json", report)


if __name__ == "__main__":
    main()
