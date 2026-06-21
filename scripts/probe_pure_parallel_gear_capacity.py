#!/usr/bin/env python
"""Learning-curve probe: is gear's top1-accuracy gap undertraining or a cap?

Trains transformer/gru/gear at a shared LR for a long token budget, evaluating
on validation at regular intervals, to see whether gear's accuracy gap versus
the baselines narrows with more training or flattens early (the latter would
point to an architectural ceiling rather than a training-budget artifact).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lmf.data import PairedDocumentManifestCorpus

try:
    from scripts.benchmark_pure_parallel_gear import (
        _context_length,
        _trainer,
        build_model,
        evaluate_manifest,
    )
    from scripts.run_pure_parallel_gear_360_proxy import configurations
except ModuleNotFoundError:
    from benchmark_pure_parallel_gear import (
        _context_length,
        _trainer,
        build_model,
        evaluate_manifest,
    )
    from run_pure_parallel_gear_360_proxy import configurations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--tokens", type=int, default=500_000)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-max-rows", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20262000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument(
        "--models", nargs="+", default=["transformer", "gru", "gear"]
    )
    return parser.parse_args()


def train_with_curve(name, config, args):
    # Every model must consume the same manifest rows from cursor zero.
    corpus = PairedDocumentManifestCorpus(str(args.train_manifest), wrap=True)
    model = build_model(name, config, args.seed)
    trainer = _trainer(
        name,
        model,
        corpus,
        lr=args.lr,
        device=args.device,
        precision=args.precision,
        total_tokens=args.tokens,
    )
    curve = []
    next_eval = args.eval_every
    started = time.perf_counter()
    while trainer.supervised_tokens_seen < args.tokens:
        progress = trainer.supervised_tokens_seen / max(args.tokens, 1)
        length = _context_length(progress)
        trainer.grad_accum_steps = 1
        records = trainer.train_steps(1, args.batch_size, length, log_every=0)
        if trainer.supervised_tokens_seen >= next_eval or trainer.supervised_tokens_seen >= args.tokens:
            validation = evaluate_manifest(
                trainer.raw_model,
                args.validation_manifest,
                batch_size=args.batch_size,
                device=args.device,
                max_rows=args.eval_max_rows,
            )
            elapsed = trainer.optimization_seconds
            point = {
                "supervised_tokens": trainer.supervised_tokens_seen,
                "elapsed_seconds": elapsed,
                "wall_elapsed_seconds": time.perf_counter() - started,
                "train_language_modeling_loss": records[-1].get("language_modeling"),
                "validation_macro_domain_nll": validation["macro_domain_nll"],
                "validation_perplexity": validation["perplexity"],
                "validation_top1_accuracy": validation["top1_accuracy"],
                "validation_top5_accuracy": validation["top5_accuracy"],
            }
            curve.append(point)
            print(
                f"[{name:>11s}] tokens={point['supervised_tokens']:>8d}"
                f"  train_loss={point['train_language_modeling_loss']:.4f}"
                f"  val_nll={point['validation_macro_domain_nll']:.4f}"
                f"  val_top1={point['validation_top1_accuracy']:.5f}"
                f"  elapsed={elapsed:.1f}s",
                flush=True,
            )
            while next_eval <= trainer.supervised_tokens_seen:
                next_eval += args.eval_every
    final_validation = evaluate_manifest(
        trainer.raw_model,
        args.validation_manifest,
        batch_size=args.batch_size,
        device=args.device,
        max_rows=None,
    )
    return {
        "curve": curve,
        "final_validation": final_validation,
        "total_seconds": trainer.optimization_seconds,
        "wall_seconds": time.perf_counter() - started,
        "supervised_tokens": trainer.supervised_tokens_seen,
    }


def main() -> None:
    args = parse_args()
    corpus = PairedDocumentManifestCorpus(str(args.train_manifest), wrap=True)
    configs = configurations(corpus.vocab_size)
    report = {
        "tokens": args.tokens,
        "eval_every": args.eval_every,
        "eval_max_rows": args.eval_max_rows,
        "lr": args.lr,
        "seed": args.seed,
        "configs": {
            name: config.to_dict() for name, config in configs.items()
        },
        "architecture_manifests": {
            name: build_model(name, config, args.seed).architecture_manifest()
            for name, config in configs.items()
        },
        "models": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        print(f"=== training {name} ===", flush=True)
        report["models"][name] = train_with_curve(name, configs[name], args)
        (args.output_dir / "results.partial.json").write_text(
            json.dumps(report, indent=2, sort_keys=True)
        )
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
