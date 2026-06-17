"""Test merge-tree compositional initialization for MultiGear generation.

MultiGear tokens are created hierarchically but the downstream transformer
normally initializes every token embedding independently. This ablation keeps
the tokenizer, model shape, parameter count, examples, batches, and seeds fixed;
the only change is initializing each learned token from its merge-tree children.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import torch

from lmf.core.seeding import seed_everything
from lmf.core.build import configure_token_hierarchy, initialize_token_embeddings
from lmf.models.transformer import CachedTransformerLM, TransformerConfig, TransformerTrainer
try:
    from benchmark_tokenizer_generation import (
        DEFAULT_LANGUAGES,
        _SpanTaskCorpus,
        _encode_examples,
        _evaluate_generation,
        _load_examples,
        _load_split,
        _shared_fit_indices,
        _stratified_eval_indices,
        _train_tokenizers,
    )
except ModuleNotFoundError:
    from scripts.benchmark_tokenizer_generation import (
        DEFAULT_LANGUAGES,
        _SpanTaskCorpus,
        _encode_examples,
        _evaluate_generation,
        _load_examples,
        _load_split,
        _shared_fit_indices,
        _stratified_eval_indices,
        _train_tokenizers,
    )


@torch.no_grad()
def initialize_from_multigear_merges(model, tokenizer) -> None:
    """Initialize learned token rows from child rows while preserving variance."""
    initialize_token_embeddings(model, tokenizer, "merge_compositional")


def run(
    root: Path,
    languages: tuple[str, ...],
    seeds: tuple[int, ...],
    vocab_size: int = 8192,
    steps: int = 3000,
    batch_size: int = 8,
    seq_len: int = 80,
    dim: int = 64,
    layers: int = 2,
    heads: int = 2,
    context_bytes: int = 48,
    target_bytes: int = 4,
    eval_per_language: int = 10,
    eval_batch_size: int = 32,
    hierarchical_output: bool = False,
    hierarchy_aux_weight: float = 0.0,
    hierarchy_aux_min_gear: int = 2,
    hierarchy_aux_target: str = "bytes",
    hierarchy_aux_max_bytes: int = 16,
    segmentation_dropout_prob: float = 0.0,
    segmentation_dropout_min_gear: int = 2,
    segmentation_dropout_max_depth: int = 1,
) -> dict:
    train_text = "\n".join(_load_split(root, "dev", languages).values())
    train_examples = _load_examples(root, "dev", languages, context_bytes, target_bytes)
    eval_examples = _load_examples(root, "devtest", languages, context_bytes, target_bytes)

    with tempfile.TemporaryDirectory() as directory:
        tokenizers, train_seconds = _train_tokenizers(
            train_text, vocab_size, Path(directory), ("multigear",)
        )
        tokenizer = tokenizers["multigear"]
        train_encoded = _encode_examples(tokenizer, train_examples)
        eval_encoded = _encode_examples(tokenizer, eval_examples)
        train_fit = _shared_fit_indices({"multigear": train_encoded}, seq_len)
        eval_fit = _shared_fit_indices({"multigear": eval_encoded}, seq_len)
        eval_indices = _stratified_eval_indices(eval_examples, eval_fit, eval_per_language)
        max_new_tokens = max(len(eval_encoded[index].target_ids) for index in eval_indices)

        report = {
            "methodology": (
                "paired MultiGear model-initialization ablation; tokenizer, model shape, "
                "parameter count, examples, batches, updates, generation budget, and seeds fixed"
            ),
            "vocab_size": vocab_size,
            "languages": list(languages),
            "seeds": list(seeds),
            "steps": steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "shared_train_examples": len(train_fit),
            "shared_eval_examples": len(eval_indices),
            "tokenizer_train_seconds": train_seconds["multigear"],
            "enhancements": {
                "hierarchical_output": hierarchical_output,
                "hierarchy_aux_weight": hierarchy_aux_weight,
                "hierarchy_aux_min_gear": hierarchy_aux_min_gear,
                "hierarchy_aux_target": hierarchy_aux_target,
                "hierarchy_aux_max_bytes": hierarchy_aux_max_bytes,
                "segmentation_dropout_prob": segmentation_dropout_prob,
                "segmentation_dropout_min_gear": segmentation_dropout_min_gear,
                "segmentation_dropout_max_depth": segmentation_dropout_max_depth,
            },
            "variants": {},
        }
        parameter_counts = set()
        for variant in ("independent", "compositional"):
            per_seed = []
            for seed in seeds:
                seed_everything(seed)
                corpus = _SpanTaskCorpus(tokenizer, train_encoded, train_fit, seq_len, seed)
                model = CachedTransformerLM(
                    TransformerConfig(
                        vocab_size=vocab_size,
                        dim=dim,
                        layers=layers,
                        heads=heads,
                        max_seq_len=seq_len,
                        hierarchical_output=hierarchical_output,
                        hierarchy_gears=6,
                        hierarchy_aux_weight=hierarchy_aux_weight,
                        hierarchy_aux_min_gear=hierarchy_aux_min_gear,
                        hierarchy_aux_target=hierarchy_aux_target,
                        hierarchy_aux_max_bytes=hierarchy_aux_max_bytes,
                    )
                )
                configure_token_hierarchy(model, tokenizer)
                if variant == "compositional":
                    initialize_from_multigear_merges(model, tokenizer)
                parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
                trainer = TransformerTrainer(
                    model,
                    corpus,
                    device="auto",
                    precision="fp32",
                    lr=3e-3,
                    warmup_steps=min(30, max(1, steps // 10)),
                    total_steps=steps,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    segmentation_dropout_prob=segmentation_dropout_prob,
                    segmentation_dropout_min_gear=segmentation_dropout_min_gear,
                    segmentation_dropout_max_depth=segmentation_dropout_max_depth,
                )
                started = time.perf_counter()
                trainer.train_steps(steps, batch_size, seq_len, log_every=0)
                metrics, _ = _evaluate_generation(
                    trainer.raw_model,
                    tokenizer,
                    eval_encoded,
                    eval_examples,
                    eval_indices,
                    max_new_tokens,
                    eval_batch_size,
                )
                metrics["seed"] = seed
                metrics["model_train_seconds"] = time.perf_counter() - started
                per_seed.append(metrics)
                print(variant, seed, json.dumps(metrics), flush=True)
            exact = [row["exact_match_pct"] for row in per_seed]
            edit = [row["mean_edit_similarity_pct"] for row in per_seed]
            report["variants"][variant] = {
                "exact_match_pct_by_seed": exact,
                "edit_similarity_pct_by_seed": edit,
                "mean_exact_match_pct": statistics.mean(exact),
                "stdev_exact_match_pct": statistics.stdev(exact) if len(exact) > 1 else 0.0,
                "mean_edit_similarity_pct": statistics.mean(edit),
                "mean_model_train_seconds": statistics.mean(
                    row["model_train_seconds"] for row in per_seed
                ),
            }
        if len(parameter_counts) != 1:
            raise AssertionError(f"parameter counts differ: {sorted(parameter_counts)}")
        report["parameter_count"] = parameter_counts.pop()
        independent = report["variants"]["independent"]["exact_match_pct_by_seed"]
        compositional = report["variants"]["compositional"]["exact_match_pct_by_seed"]
        differences = [new - old for new, old in zip(compositional, independent)]
        report["paired_exact_match_difference"] = {
            "by_seed": differences,
            "mean": statistics.mean(differences),
            "stdev": statistics.stdev(differences) if len(differences) > 1 else 0.0,
        }
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flores-root", type=Path, required=True)
    parser.add_argument("--languages", nargs="*", default=DEFAULT_LANGUAGES)
    parser.add_argument("--seeds", nargs="*", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seq-len", type=int, default=80)
    parser.add_argument("--context-bytes", type=int, default=48)
    parser.add_argument("--target-bytes", type=int, default=4)
    parser.add_argument("--eval-per-language", type=int, default=10)
    parser.add_argument("--hierarchical-output", action="store_true")
    parser.add_argument("--hierarchy-aux-weight", type=float, default=0.0)
    parser.add_argument("--hierarchy-aux-min-gear", type=int, default=2)
    parser.add_argument("--hierarchy-aux-target", choices=("bytes", "children"), default="bytes")
    parser.add_argument("--hierarchy-aux-max-bytes", type=int, default=16)
    parser.add_argument("--segmentation-dropout-prob", type=float, default=0.0)
    parser.add_argument("--segmentation-dropout-min-gear", type=int, default=2)
    parser.add_argument("--segmentation-dropout-max-depth", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.flores_root,
        tuple(args.languages),
        tuple(args.seeds),
        steps=args.steps,
        seq_len=args.seq_len,
        context_bytes=args.context_bytes,
        target_bytes=args.target_bytes,
        eval_per_language=args.eval_per_language,
        hierarchical_output=args.hierarchical_output,
        hierarchy_aux_weight=args.hierarchy_aux_weight,
        hierarchy_aux_min_gear=args.hierarchy_aux_min_gear,
        hierarchy_aux_target=args.hierarchy_aux_target,
        hierarchy_aux_max_bytes=args.hierarchy_aux_max_bytes,
        segmentation_dropout_prob=args.segmentation_dropout_prob,
        segmentation_dropout_min_gear=args.segmentation_dropout_min_gear,
        segmentation_dropout_max_depth=args.segmentation_dropout_max_depth,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["variants"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
