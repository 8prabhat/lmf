#!/usr/bin/env python
"""Parameter-matched Transformer vs. Pure Parallel Gear comparison at ~1M params.

Apples-to-apples: both models matched to within 0.5% of a ~1,000,000-parameter
budget (verified via assert_fair_configs), same seed, same training schedule,
same corpus, same evaluation methodology. GRU is intentionally excluded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from lmf.core.io import atomic_write_json as write_json
from lmf.data import PairedDocumentManifestCorpus
from lmf.diagnostics import parameter_count as count_parameters
from lmf.models.transformer import CachedTransformerLM, TransformerConfig

try:
    from scripts.benchmark_pure_parallel_gear import (
        _configured_parameter_count,
        _context_length,
        _matched_gear,
        _trainer,
        assert_fair_configs,
        build_model,
        evaluate_manifest,
        throughput,
    )
except ModuleNotFoundError:
    from benchmark_pure_parallel_gear import (
        _configured_parameter_count,
        _context_length,
        _matched_gear,
        _trainer,
        assert_fair_configs,
        build_model,
        evaluate_manifest,
        throughput,
    )

import argparse


def _check_fairness(configs, *, allow_retrieval: bool):
    """Same parameter-matching check as assert_fair_configs, but with the
    "Pure Gear never does token-similarity/retrieval" invariant gate
    skipped when allow_retrieval is set. That gate exists to stop a
    comparison from silently smuggling in a generically-strong copy
    mechanism and attributing the win to gear's core sequence mixing --
    when fast-weight memory is deliberately enabled, the comparison is no
    longer making that "pure gear" claim in the first place, so enforcing
    the gate here would just block a comparison we're running on purpose.
    The 0.5% parameter-matching check still applies either way.
    """
    if allow_retrieval:
        models = {
            name: build_model(name, config, seed=1)
            for name, config in configs.items()
        }
        parameters = {
            name: count_parameters(model) for name, model in models.items()
        }
        baseline = parameters["transformer"]
        relative = {
            name: value / baseline - 1.0 for name, value in parameters.items()
        }
        if any(abs(value) > 0.005 for value in relative.values()):
            raise RuntimeError(f"parameter mismatch exceeds 0.5%: {relative}")
        return {
            "parameters": parameters,
            "relative_to_transformer": relative,
            "manifests": {
                name: model.architecture_manifest()
                for name, model in models.items()
            },
        }
    return assert_fair_configs(configs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--target-params", type=int, default=1_000_000)
    parser.add_argument("--transformer-dim", type=int, default=120)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=1_000_000)
    parser.add_argument("--tuning-tokens", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20262000)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--fast-weight-memory", action="store_true")
    return parser.parse_args()


def configurations(vocab_size: int, args: argparse.Namespace):
    transformer = TransformerConfig(
        vocab_size=vocab_size,
        dim=args.transformer_dim,
        layers=args.transformer_layers,
        heads=args.transformer_heads,
        max_seq_len=4096,
    )
    target = _configured_parameter_count(CachedTransformerLM, transformer)
    extra_config = {"use_fast_weight_memory": True} if args.fast_weight_memory else None
    gear = _matched_gear(
        target,
        vocab_size,
        baseline_dim=args.transformer_dim,
        layers=args.transformer_layers,
        extra_config=extra_config,
    )
    return {"transformer": transformer, "gear": gear}


def memory_gate_statistics(model, manifest_root: Path, *, batch_size: int, device: str) -> dict:
    """Per the design doc's validation plan: if val_top1 improves but the
    mean gate is pinned near 1 (not near copy_gate_target_mean), that's the
    overshadowing risk materializing -- the win would be coming from a
    generically strong copy mechanism, not gear's core sequence mixing."""
    if model.memory is None:
        return {}
    corpus = PairedDocumentManifestCorpus(str(manifest_root), wrap=False)
    seq_len = sorted(int(value) for value in corpus.manifest["rows_by_length"])[0]
    rows = min(batch_size, int(corpus.manifest["rows_by_length"][str(seq_len)]))
    batch = corpus.batch_from_indices(list(range(rows)), seq_len).to(device)
    model.eval()
    with torch.no_grad():
        _, _, _, memory_extras = model._forward_hidden(
            batch.tokens,
            token_mask=batch.attention_mask,
            segment_ids=batch.metadata["segment_ids"],
            sentence_end_mask=batch.metadata["sentence_end_mask"],
        )
    gate = memory_extras["gate"]
    memory_energy = memory_extras["memory_energy"]
    return {
        "gate_mean": float(gate.mean()),
        "gate_std": float(gate.std(unbiased=False)),
        "gate_target_mean": float(model.config.copy_gate_target_mean),
        "memory_energy_radius_mean": float(memory_energy.clamp_min(1e-8).sqrt().mean()),
        "memory_energy_radius_max": float(memory_energy.clamp_min(1e-8).sqrt().max()),
    }


def train(name, config, args, *, lr, tokens, tuning):
    corpus = PairedDocumentManifestCorpus(str(args.train_manifest), wrap=True)
    model = build_model(name, config, args.seed)
    trainer = _trainer(
        name,
        model,
        corpus,
        lr=lr,
        device=args.device,
        precision=args.precision,
        total_tokens=tokens,
    )
    started = time.perf_counter()
    while trainer.supervised_tokens_seen < tokens:
        # Tuning trials walk the same length curriculum as the final run
        # (just compressed into a smaller token budget) rather than a
        # fixed short length, so an LR that is fine at length=128 but
        # destabilizes the omega/angle recurrence at length>=1024 gets
        # penalized during selection instead of only failing later.
        progress = trainer.supervised_tokens_seen / max(tokens, 1)
        length = _context_length(progress)
        trainer.grad_accum_steps = 1
        trainer.train_steps(1, args.batch_size, length, log_every=args.log_every)
    elapsed = trainer.optimization_seconds
    wall = time.perf_counter() - started
    validation = evaluate_manifest(
        trainer.raw_model,
        args.validation_manifest,
        batch_size=args.batch_size,
        device=args.device,
        max_rows=64 if tuning else None,
    )
    return trainer, {
        "seconds": elapsed,
        "wall_seconds": wall,
        "supervised_tokens": trainer.supervised_tokens_seen,
        "tokens_per_second": trainer.supervised_tokens_seen / max(elapsed, 1e-9),
        "gradient_skips": getattr(trainer, "total_gradient_skips", 0),
        "validation": validation,
    }


def main() -> None:
    args = parse_args()
    corpus = PairedDocumentManifestCorpus(str(args.train_manifest), wrap=True)
    configs = configurations(corpus.vocab_size, args)
    fairness = _check_fairness(configs, allow_retrieval=args.fast_weight_memory)
    limitations = [
        "single seed",
        "~1M parameters",
        f"{args.tokens}-token final budget",
        "not a 3M/15M/50M gate",
    ]
    if args.fast_weight_memory:
        limitations.append(
            "gear has fast-weight associative memory enabled (token_similarity/"
            "history_retrieval=True in its manifest) -- this is no longer a "
            "'pure gear, no retrieval' comparison. A win here does not "
            "isolate gear's core sequence mixing; the matched-baseline "
            "ablation (same memory mechanism wired onto the transformer) is "
            "needed before attributing any improvement to gear specifically."
        )
    report = {
        "stage": "pure_parallel_gear_1m_comparison",
        "seed": args.seed,
        "tokens": args.tokens,
        "tuning_tokens": args.tuning_tokens,
        "models_compared": ("transformer", "gear"),
        "configs": {name: config.to_dict() for name, config in configs.items()},
        "parameters": fairness["parameters"],
        "relative_parameter_gap": fairness["relative_to_transformer"],
        "architecture_manifests": fairness["manifests"],
        "lr_tuning": {},
        "models": {},
        "limitations": limitations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "results.partial.json", report)

    learning_rates = (3e-4, 1e-3, 3e-3)
    selected = {}
    for name in ("transformer", "gear"):
        trials = []
        for lr in learning_rates:
            _, result = train(
                name, configs[name], args, lr=lr, tokens=args.tuning_tokens, tuning=True
            )
            trials.append({"lr": lr, **result})
        best = min(
            trials,
            key=lambda row: (
                row.get("gradient_skips", 0) > 0,
                row["validation"]["macro_domain_nll"],
                row["validation"]["worst_domain_nll"],
            ),
        )
        selected[name] = best["lr"]
        report["lr_tuning"][name] = {"selected_lr": best["lr"], "trials": trials}
        write_json(args.output_dir / "results.partial.json", report)

    for name in ("transformer", "gear"):
        trainer, result = train(
            name, configs[name], args, lr=selected[name], tokens=args.tokens, tuning=False
        )
        checkpoint = args.output_dir / "checkpoints" / f"{name}.pt"
        trainer.save_checkpoint(checkpoint)
        result["checkpoint"] = str(checkpoint)
        result["test"] = evaluate_manifest(
            trainer.raw_model,
            args.test_manifest,
            batch_size=args.batch_size,
            device=args.device,
        )
        result["efficiency"] = throughput(
            trainer.raw_model,
            vocab_size=corpus.vocab_size,
            seq_len=512,
            device=args.device,
            repeats=3,
        )
        if name == "gear" and args.fast_weight_memory:
            result["memory_gate_statistics"] = memory_gate_statistics(
                trainer.raw_model,
                args.test_manifest,
                batch_size=args.batch_size,
                device=args.device,
            )
        report["models"][name] = result
        write_json(args.output_dir / "results.partial.json", report)
        del trainer

    report["complete"] = True
    write_json(args.output_dir / "results.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "lr_tuning"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
