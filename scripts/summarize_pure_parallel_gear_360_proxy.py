#!/usr/bin/env python
"""Create an auditable report from the executed micro-proxy 360 artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from lmf.ablation.stats import paired_bootstrap_interval
from lmf.core.hashing import file_sha256
from lmf.research_utils import bar_chart_svg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20262300)
    return parser.parse_args()


def mean_natural(evaluation, model, length):
    return statistics.fmean(
        domain[str(length)]["nll"]
        for domain in evaluation["models"][model]["natural"].values()
        if str(length) in domain
    )


def mean_robustness(evaluation, model, key, metric="nll"):
    return statistics.fmean(
        domain["robustness"][key][metric]
        for domain in evaluation["models"][model]["natural"].values()
    )




def main():
    args = parse_args()
    training = json.loads(args.training.read_text())
    evaluation = json.loads(args.evaluation.read_text())
    generation = json.loads(args.generation.read_text())
    test = {
        name: evaluation["models"][name]["complete_manifests"]["test"]
        for name in ("transformer", "gru", "gear")
    }
    domains = sorted(test["transformer"]["per_domain_nll"])
    domain_differences = [
        test["gear"]["per_domain_nll"][domain]
        - test["transformer"]["per_domain_nll"][domain]
        for domain in domains
    ]
    nll_interval = paired_bootstrap_interval(
        domain_differences, seed=args.seed, samples=50_000
    )
    relative_domain_regressions = {
        domain: test["gear"]["per_domain_nll"][domain]
        / test["transformer"]["per_domain_nll"][domain]
        - 1.0
        for domain in domains
    }
    efficiency = {}
    for name in ("transformer", "gru", "gear"):
        efficiency[name] = {
            row["length"]: row
            for row in evaluation["models"][name]["efficiency"]["contexts"]
        }
    training_models = training["models"]
    context_rows = {}
    for length in (128, 256, 512, 1024):
        context_rows[str(length)] = {
            name: mean_natural(evaluation, name, length)
            for name in ("transformer", "gru", "gear")
        }
    robustness = {}
    for name in ("transformer", "gru", "gear"):
        base = mean_natural(evaluation, name, 512)
        robustness[name] = {
            "base_512_nll": base,
            "corruption_10pct_delta": mean_robustness(
                evaluation, name, "token_corruption_0.1"
            )
            - base,
            "irrelevant_prefix_128_delta": mean_robustness(
                evaluation, name, "irrelevant_prefix_128"
            )
            - base,
            "truncated_128_delta": mean_robustness(
                evaluation, name, "truncated_128"
            )
            - base,
            "typo_delta": mean_robustness(
                evaluation, name, "text_typo_prompt", "delta_nll"
            ),
        }
    long = 2048
    gear_long = efficiency["gear"][long]
    transformer_long = efficiency["transformer"][long]
    gear_test_nll = test["gear"]["macro_domain_nll"]
    transformer_test_nll = test["transformer"]["macro_domain_nll"]
    gru_test_nll = test["gru"]["macro_domain_nll"]
    checks = {
        "quality_within_1pct_transformer": (
            gear_test_nll / transformer_test_nll - 1.0 <= 0.01
        ),
        "no_domain_over_2pct_worse": max(
            relative_domain_regressions.values()
        )
        <= 0.02,
        "gear_beats_gru": gear_test_nll < gru_test_nll,
        "training_throughput_at_least_half_transformer": (
            training_models["gear"]["tokens_per_second"]
            / training_models["transformer"]["tokens_per_second"]
            >= 0.5
        ),
        "training_memory_at_most_1_25x_transformer": (
            evaluation["models"]["gear"]["training_memory"][
                "peak_proxy_bytes"
            ]
            / evaluation["models"]["transformer"]["training_memory"][
                "peak_proxy_bytes"
            ]
            <= 1.25
        ),
        "long_incremental_generation_at_least_1_5x": (
            gear_long["incremental_tokens_per_second"]
            / transformer_long["incremental_tokens_per_second"]
            >= 1.5
        ),
        "long_cache_at_most_25pct_transformer": (
            gear_long["cache_bytes"] / transformer_long["cache_bytes"]
            <= 0.25
        ),
        "healthy_gradients": not evaluation["models"]["gear"][
            "gradient_health"
        ]["missing"]
        and not evaluation["models"]["gear"]["gradient_health"]["nonfinite"],
        "profiler_contract_passed": evaluation["models"]["gear"][
            "profiler_contract"
        ]["passed"],
        "blind_generation_complete": False,
        "greedy_generation_non_degenerate": generation["aggregate"]["gear"][
            "256"
        ]["decoding"]["greedy"]["adjacent_repetition"]
        < 0.2,
    }
    ablations = evaluation["models"]["gear"]["ablations"]
    summary = {
        "classification": "micro_proxy_360_not_decisive",
        "verdict": "do_not_scale_yet",
        "checks": checks,
        "parameters": training["parameters"],
        "training": {
            name: {
                "seconds": training_models[name]["seconds"],
                "tokens_per_second": training_models[name][
                    "tokens_per_second"
                ],
            }
            for name in ("transformer", "gru", "gear")
        },
        "held_out_test": test,
        "gear_minus_transformer_domain_nll_bootstrap_95pct": nll_interval,
        "gear_relative_domain_regressions": relative_domain_regressions,
        "natural_context_nll": context_rows,
        "robustness": robustness,
        "long_context_efficiency": {
            "length": long,
            "gear_generation_speed_ratio": gear_long[
                "incremental_tokens_per_second"
            ]
            / transformer_long["incremental_tokens_per_second"],
            "gear_prefill_time_ratio": gear_long["prefill_p50_seconds"]
            / transformer_long["prefill_p50_seconds"],
            "gear_cache_ratio": gear_long["cache_bytes"]
            / transformer_long["cache_bytes"],
            "gear_training_memory_ratio": evaluation["models"]["gear"][
                "training_memory"
            ]["peak_proxy_bytes"]
            / evaluation["models"]["transformer"]["training_memory"][
                "peak_proxy_bytes"
            ],
        },
        "mechanism": evaluation["models"]["gear"]["mechanism"],
        "posthoc_ablations": ablations,
        "generation": generation["aggregate"],
        "generation_examples": [
            {
                "domain": row["domain"],
                "prompt": row["prompt"],
                "transformer_greedy": row["models"]["transformer"][
                    "lengths"
                ]["64"]["decoding"]["greedy"][0]["text"],
                "gru_greedy": row["models"]["gru"]["lengths"]["64"][
                    "decoding"
                ]["greedy"][0]["text"],
                "gear_greedy": row["models"]["gear"]["lengths"]["64"][
                    "decoding"
                ]["greedy"][0]["text"],
            }
            for row in generation["examples"][:3]
        ],
        "source_hashes": {
            str(path): file_sha256(path)
            for path in (args.training, args.evaluation, args.generation)
        },
        "limitations": [
            "single seed",
            "158K-parameter models",
            "about 100K supervised training tokens",
            "LR tuning subset was dominated by the first manifest domain",
            "reduced 360 sampling: 8 natural windows/domain and 2 task batches",
            "only 21 generation prompts and no independent blind ratings",
            "synthetic compositional tasks were not trained and mostly scored zero",
            "post-hoc ablations are diagnostic, not retrained causal evidence",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    bar_chart_svg(
        args.output_dir / "test_nll.svg",
        ["Transformer", "GRU", "Gear"],
        [transformer_test_nll, gru_test_nll, gear_test_nll],
        "Held-out macro-domain NLL",
        lower_is_better=True,
    )
    bar_chart_svg(
        args.output_dir / "training_throughput.svg",
        ["Transformer", "GRU", "Gear"],
        [
            training_models["transformer"]["tokens_per_second"],
            training_models["gru"]["tokens_per_second"],
            training_models["gear"]["tokens_per_second"],
        ],
        "Training supervised tokens/second",
        lower_is_better=False,
    )
    bar_chart_svg(
        args.output_dir / "long_context_cache.svg",
        ["Transformer", "GRU", "Gear"],
        [
            transformer_long["cache_bytes"],
            efficiency["gru"][long]["cache_bytes"],
            gear_long["cache_bytes"],
        ],
        "Generation cache bytes at 2,048-token prompt",
        lower_is_better=True,
    )
    lines = [
        "# Pure Parallel Gear 360° micro-proxy report",
        "",
        "**Verdict: DO NOT SCALE YET.** Gear beats the recurrent control on held-out NLL, but it is worse than the matched Transformer, far slower to train, slower at long-context generation, and its gear-specific ablations are nearly neutral at this budget.",
        "",
        "This is a single-seed 158K-parameter, ~100K-token proxy—not a decisive 3M/15M result.",
        "",
        "## Primary comparison",
        "",
        "| Metric | Transformer | GRU | Gear | Interpretation |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| Test macro NLL | {transformer_test_nll:.4f} | {gru_test_nll:.4f} | {gear_test_nll:.4f} | Gear is {(gear_test_nll/transformer_test_nll-1)*100:.2f}% worse than Transformer and {(1-gear_test_nll/gru_test_nll)*100:.2f}% better than GRU |",
        f"| Test perplexity | {test['transformer']['perplexity']:.1f} | {test['gru']['perplexity']:.1f} | {test['gear']['perplexity']:.1f} | Transformer wins |",
        f"| Top-1 accuracy | {test['transformer']['top1_accuracy']:.3%} | {test['gru']['top1_accuracy']:.3%} | {test['gear']['top1_accuracy']:.3%} | Gear is substantially worse |",
        f"| Training tokens/s | {training_models['transformer']['tokens_per_second']:.1f} | {training_models['gru']['tokens_per_second']:.1f} | {training_models['gear']['tokens_per_second']:.1f} | Gear is {training_models['transformer']['tokens_per_second']/training_models['gear']['tokens_per_second']:.1f}× slower than Transformer |",
        f"| 2K incremental tokens/s | {transformer_long['incremental_tokens_per_second']:.1f} | {efficiency['gru'][long]['incremental_tokens_per_second']:.1f} | {gear_long['incremental_tokens_per_second']:.1f} | Gear is {transformer_long['incremental_tokens_per_second']/gear_long['incremental_tokens_per_second']:.1f}× slower |",
        f"| 2K cache bytes | {transformer_long['cache_bytes']:,} | {efficiency['gru'][long]['cache_bytes']:,} | {gear_long['cache_bytes']:,} | Gear uses {(1-gear_long['cache_bytes']/transformer_long['cache_bytes'])*100:.2f}% less cache |",
        f"| Training-memory proxy | {evaluation['models']['transformer']['training_memory']['peak_proxy_bytes']:,} | {evaluation['models']['gru']['training_memory']['peak_proxy_bytes']:,} | {evaluation['models']['gear']['training_memory']['peak_proxy_bytes']:,} | Gear is close to Transformer, not 20% lower |",
        "",
        "![Test NLL](test_nll.svg)",
        "",
        "![Training throughput](training_throughput.svg)",
        "",
        "![Long-context cache](long_context_cache.svg)",
        "",
        "## Statistical and domain result",
        "",
        f"The paired seven-domain bootstrap estimate for Gear minus Transformer NLL is {nll_interval['mean']:.4f}, with a 95% interval [{nll_interval['lower']:.4f}, {nll_interval['upper']:.4f}]. Every sampled bootstrap direction is unfavorable when the lower bound is positive.",
        "",
        f"Worst relative domain regression: {max(relative_domain_regressions.values())*100:.2f}%.",
        "",
        "## Mechanism evidence",
        "",
        "Gear states are numerically healthy: no saturated velocities, no dead gears, finite gradients, active clutches, and no attention/history operation detected. However, removing boundary settling, cross-bank coupling, or learned angular velocity slightly improved NLL in the post-hoc probe. Only the local SwiGLU had a material effect. At this budget, the intended gear composition is not demonstrated as the source of predictive quality.",
        "",
        "## Generation",
        "",
        "Greedy generation failed for every model. Transformer repeated newlines; Gear repeated “of”; GRU also collapsed. Gear's 256-token greedy adjacent-repetition rate was 100%. Stochastic diversity is high, but without blind ratings—and with visibly incoherent samples—it cannot be treated as useful novelty or coherent generation.",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'} — `{name}`"
        for name, value in checks.items()
    )
    lines.extend(
        [
            "",
            "## Recommended architecture work before another scale run",
            "",
            "1. Replace Python-level sentence/row iteration with a compiled segmented rotor scan; current training speed is disqualifying.",
            "2. Strengthen the gear path or reduce the token-local SwiGLU so quality cannot be carried almost entirely by the local network.",
            "3. Add a training objective or curriculum that directly exercises boundary settling, order composition, copy, and sentence-transition state.",
            "4. Revisit predictor readout and output calibration; Gear has low top-1 accuracy and greedy collapse despite moderate NLL.",
            "5. Run retrained structural ablations before 3M. Do not rely on post-hoc masks.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in summary["limitations"])
    lines.extend(
        [
            "",
            "Finite gear state cannot guarantee arbitrary-length exact recall. Coherence and novelty were not established.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": summary["verdict"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
