#!/usr/bin/env python
"""Runnable matched micro-proxy used before the expensive gated scale study."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from lmf.data import PairedDocumentManifestCorpus
from lmf.models.transformer import CachedTransformerLM, TransformerConfig

try:
    from scripts.benchmark_pure_parallel_gear import (
        _matched_gear,
        _matched_gru,
        _trainer,
        build_model,
        count_parameters,
        evaluate_manifest,
        throughput,
        write_json,
    )
except ModuleNotFoundError:
    from benchmark_pure_parallel_gear import (
        _matched_gear,
        _matched_gru,
        _trainer,
        build_model,
        count_parameters,
        evaluate_manifest,
        throughput,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--tokens", type=int, default=100_000)
    parser.add_argument("--tuning-tokens", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20262000)
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args()


def configurations(vocab_size: int):
    transformer = TransformerConfig(
        vocab_size=vocab_size,
        dim=40,
        layers=2,
        heads=4,
        max_seq_len=4096,
    )
    target = count_parameters(CachedTransformerLM(transformer))
    return {
        "transformer": transformer,
        "gru": _matched_gru(
            target,
            vocab_size,
            baseline_dim=40,
            layers=2,
        ),
        "gear": _matched_gear(
            target,
            vocab_size,
            baseline_dim=40,
            layers=2,
        ),
    }


def context_length(progress: float) -> int:
    if progress < 0.4:
        return 128
    if progress < 0.8:
        return 256
    return 512


def train(
    name,
    config,
    args,
    *,
    lr: float,
    tokens: int,
    tuning: bool,
):
    corpus = PairedDocumentManifestCorpus(
        str(args.train_manifest), wrap=True
    )
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
        progress = trainer.supervised_tokens_seen / max(tokens, 1)
        length = 128 if tuning else context_length(progress)
        trainer.grad_accum_steps = 1
        trainer.train_steps(
            1,
            args.batch_size,
            length,
            log_every=getattr(args, "log_every", 0),
        )
    if trainer.device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    validation = evaluate_manifest(
        trainer.raw_model,
        args.validation_manifest,
        batch_size=args.batch_size,
        device=args.device,
        max_rows=32 if tuning else None,
    )
    return trainer, {
        "seconds": elapsed,
        "supervised_tokens": trainer.supervised_tokens_seen,
        "tokens_per_second": trainer.supervised_tokens_seen
        / max(elapsed, 1e-9),
        "validation": validation,
    }


def main() -> None:
    args = parse_args()
    corpus = PairedDocumentManifestCorpus(
        str(args.train_manifest), wrap=True
    )
    configs = configurations(corpus.vocab_size)
    parameters = {
        name: count_parameters(build_model(name, config, args.seed))
        for name, config in configs.items()
    }
    baseline = parameters["transformer"]
    report = {
        "stage": "micro_proxy_360",
        "seed": args.seed,
        "tokens": args.tokens,
        "tuning_tokens": args.tuning_tokens,
        "parameters": parameters,
        "configs": {
            name: config.to_dict() for name, config in configs.items()
        },
        "architecture_manifests": {
            name: build_model(name, config, args.seed).architecture_manifest()
            for name, config in configs.items()
        },
        "relative_parameter_gap": {
            name: value / baseline - 1.0
            for name, value in parameters.items()
        },
        "lr_tuning": {},
        "models": {},
        "limitations": [
            "single seed",
            "micro model",
            "100K-token default budget",
            "not a 3M/15M gate",
        ],
    }
    learning_rates = (3e-4, 1e-3, 3e-3)
    selected = {}
    for name in ("transformer", "gru", "gear"):
        trials = []
        for lr in learning_rates:
            trainer, result = train(
                name,
                configs[name],
                args,
                lr=lr,
                tokens=args.tuning_tokens,
                tuning=True,
            )
            trials.append({"lr": lr, **result})
            del trainer
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        best = min(
            trials,
            key=lambda row: (
                row["validation"]["macro_domain_nll"],
                row["validation"]["worst_domain_nll"],
            ),
        )
        selected[name] = best["lr"]
        report["lr_tuning"][name] = {
            "selected_lr": best["lr"],
            "trials": trials,
        }
        write_json(args.output_dir / "results.partial.json", report)

    for name in ("transformer", "gru", "gear"):
        trainer, result = train(
            name,
            configs[name],
            args,
            lr=selected[name],
            tokens=args.tokens,
            tuning=False,
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
        report["models"][name] = result
        write_json(args.output_dir / "results.partial.json", report)
        del trainer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    report["complete"] = True
    write_json(args.output_dir / "results.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
