"""Reliable multi-angle comparison of tokenizers on exact generation.

This benchmark extends ``benchmark_tokenizer_generation.py`` with:

* three predeclared target lengths;
* ten paired model seeds and a larger, evenly sampled held-out set;
* pure tokenizer baselines plus incremental MultiGear integration ablations;
* identical seeded initialization for every parameter shared by two models;
* exact match, edit similarity, teacher-forced bits/target-byte, and wall time;
* correct paired Student-t confidence intervals; and
* resumable, atomic result writes after every trained model.

The deterministic marked-span task is intentionally narrow. It supports exact
generation scoring without an approximate judge, but it is not evidence about
translation or open-ended generation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from lmf.core.build import configure_token_hierarchy, initialize_token_embeddings
from lmf.core.seeding import seed_everything
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
        _train_tokenizers,
    )


@dataclass(frozen=True)
class Variant:
    tokenizer: str
    compositional_init: bool = False
    hierarchical_output: bool = False
    hierarchy_aux_weight: float = 0.0
    segmentation_dropout_prob: float = 0.0


VARIANTS = {
    "sentencepiece_bpe": Variant("sentencepiece_bpe"),
    "byte_bpe": Variant("byte_bpe"),
    "sentencepiece_unigram": Variant("sentencepiece_unigram"),
    "spt": Variant("spt"),
    "multigear_flat": Variant("multigear"),
    "multigear_compositional": Variant("multigear", compositional_init=True),
    "multigear_hierarchical": Variant(
        "multigear", compositional_init=True, hierarchical_output=True
    ),
    "multigear_hierarchical_aux": Variant(
        "multigear",
        compositional_init=True,
        hierarchical_output=True,
        hierarchy_aux_weight=0.10,
    ),
    "multigear_full": Variant(
        "multigear",
        compositional_init=True,
        hierarchical_output=True,
        hierarchy_aux_weight=0.10,
        segmentation_dropout_prob=0.10,
    ),
}

PRIMARY_CONTRASTS = (
    ("multigear_compositional", "sentencepiece_bpe"),
    ("multigear_hierarchical", "sentencepiece_bpe"),
    ("multigear_hierarchical", "multigear_compositional"),
    ("multigear_hierarchical_aux", "multigear_hierarchical"),
    ("multigear_full", "multigear_hierarchical_aux"),
    ("multigear_full", "multigear_hierarchical"),
    ("multigear_full", "sentencepiece_bpe"),
    ("multigear_full", "byte_bpe"),
    ("multigear_full", "multigear_compositional"),
)

METRICS = {
    "exact_match_pct": "higher",
    "mean_edit_similarity_pct": "higher",
    "mean_exact_prefix_pct": "higher",
    "teacher_forced_bits_per_target_byte": "lower",
    "model_train_seconds": "lower",
    "generation_seconds": "lower",
}

# Two-sided 95% Student-t critical values indexed by degrees of freedom.
T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _critical_95(n: int) -> float | None:
    if n < 2:
        return None
    return T_CRITICAL_95.get(n - 1, 1.96)


def _mean_ci95(values: list[float]) -> dict:
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    critical = _critical_95(len(values))
    if critical is None:
        interval = [None, None]
    else:
        half = critical * stdev / math.sqrt(len(values))
        interval = [mean - half, mean + half]
    return {"mean": mean, "stdev": stdev, "ci95": interval}


def _paired_summary(candidate: list[float], reference: list[float], direction: str) -> dict:
    if len(candidate) != len(reference):
        raise ValueError("paired samples must have equal lengths")
    differences = [left - right for left, right in zip(candidate, reference)]
    summary = {"by_seed": differences, **_mean_ci95(differences)}
    if direction == "higher":
        summary["wins"] = sum(value > 0 for value in differences)
        summary["ties"] = sum(value == 0 for value in differences)
        summary["losses"] = sum(value < 0 for value in differences)
    else:
        summary["wins"] = sum(value < 0 for value in differences)
        summary["ties"] = sum(value == 0 for value in differences)
        summary["losses"] = sum(value > 0 for value in differences)
    return summary


def _evenly_spaced_eval_indices(examples, fit_indices: list[int], per_language: int) -> list[int]:
    by_language: dict[str, list[int]] = defaultdict(list)
    for index in fit_indices:
        by_language[examples[index].language].append(index)
    selected = []
    for language in sorted(by_language):
        values = by_language[language]
        if per_language <= 0 or per_language >= len(values):
            selected.extend(values)
            continue
        if per_language == 1:
            selected.append(values[len(values) // 2])
            continue
        positions = [
            round(offset * (len(values) - 1) / (per_language - 1))
            for offset in range(per_language)
        ]
        selected.extend(values[position] for position in positions)
    return selected


def _copy_shared_initialization(model, reference_state: dict[str, torch.Tensor]) -> None:
    """Make every same-shaped shared parameter identical for a paired seed."""
    with torch.no_grad():
        state = model.state_dict()
        for name, value in reference_state.items():
            if name in state and state[name].shape == value.shape:
                state[name].copy_(value)


@torch.no_grad()
def _evaluate_teacher_forced(
    model,
    tokenizer,
    encoded,
    examples,
    indices: list[int],
    seq_len: int,
    batch_size: int,
) -> dict:
    model.eval()
    device = next(model.parameters()).device
    pad_id = tokenizer.special_to_id["<|pad|>"]
    total_nats = 0.0
    total_supervised_tokens = 0
    total_target_bytes = 0
    started = time.perf_counter()
    for offset in range(0, len(indices), batch_size):
        chunk = indices[offset:offset + batch_size]
        token_rows = []
        attention_rows = []
        loss_rows = []
        for index in chunk:
            item = encoded[index]
            full = item.full_ids
            padding = seq_len - len(full)
            token_rows.append(full + (pad_id,) * padding)
            attention_rows.append((True,) * len(full) + (False,) * padding)
            loss_rows.append(
                (False,) * len(item.prompt_ids)
                + (True,) * len(item.target_ids)
                + (False,) * padding
            )
            total_target_bytes += len(examples[index].target.encode("utf-8"))
        tokens = torch.tensor(token_rows, dtype=torch.long, device=device)
        attention = torch.tensor(attention_rows, dtype=torch.bool, device=device)
        loss_mask = torch.tensor(loss_rows, dtype=torch.bool, device=device)
        supervised = int((loss_mask[:, 1:] & attention[:, 1:]).sum())
        losses = model.training_step(
            tokens, {"attention_mask": attention, "loss_mask": loss_mask}
        )
        total_nats += float(losses["language_modeling"]) * supervised
        total_supervised_tokens += supervised
    seconds = time.perf_counter() - started
    return {
        "teacher_forced_bits_per_target_byte": (
            total_nats / (math.log(2.0) * max(total_target_bytes, 1))
        ),
        "teacher_forced_nats_per_supervised_token": (
            total_nats / max(total_supervised_tokens, 1)
        ),
        "teacher_forced_seconds": seconds,
        "eval_target_bytes": total_target_bytes,
        "eval_supervised_tokens": total_supervised_tokens,
    }


def _language_metrics(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["language"]].append(row)
    return {
        language: {
            "examples": len(values),
            "exact_match_pct": 100.0 * statistics.mean(row["exact"] for row in values),
            "mean_edit_similarity_pct": (
                100.0 * statistics.mean(row["edit_similarity"] for row in values)
            ),
        }
        for language, values in sorted(grouped.items())
    }


def _atomic_write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _summarize(report: dict) -> None:
    variants = tuple(report["variants"])
    contrasts = tuple(
        (candidate, reference)
        for candidate, reference in PRIMARY_CONTRASTS
        if candidate in variants and reference in variants
    )
    for task in report["tasks"].values():
        for result in task["variants"].values():
            seeds = sorted(result["runs"], key=int)
            result["summary"] = (
                {
                    metric: _mean_ci95([result["runs"][seed][metric] for seed in seeds])
                    for metric in METRICS
                }
                if seeds else {}
            )
        comparisons = {}
        for candidate, reference in contrasts:
            comparisons[f"{candidate}_vs_{reference}"] = {}
            candidate_runs = task["variants"][candidate]["runs"]
            reference_runs = task["variants"][reference]["runs"]
            common = sorted(set(candidate_runs).intersection(reference_runs), key=int)
            if not common:
                continue
            for metric, direction in METRICS.items():
                comparisons[f"{candidate}_vs_{reference}"][metric] = _paired_summary(
                    [candidate_runs[seed][metric] for seed in common],
                    [reference_runs[seed][metric] for seed in common],
                    direction,
                )
        task["paired_primary_contrasts"] = comparisons

    aggregate = {"variants": {}, "paired_primary_contrasts": {}}
    task_names = sorted(report["tasks"])
    for variant in variants:
        complete_seeds = [
            str(seed)
            for seed in report["seeds"]
            if all(
                str(seed) in report["tasks"][task]["variants"][variant]["runs"]
                for task in task_names
            )
        ]
        aggregate["variants"][variant] = {}
        for metric in METRICS:
            by_seed = [
                statistics.mean(
                    report["tasks"][task]["variants"][variant]["runs"][seed][metric]
                    for task in task_names
                )
                for seed in complete_seeds
            ]
            aggregate["variants"][variant][metric] = {
                "by_seed": by_seed,
                **_mean_ci95(by_seed),
            } if by_seed else None
    for candidate, reference in contrasts:
        name = f"{candidate}_vs_{reference}"
        aggregate["paired_primary_contrasts"][name] = {}
        common_seeds = [
            str(seed)
            for seed in report["seeds"]
            if all(
                str(seed) in report["tasks"][task]["variants"][candidate]["runs"]
                and str(seed) in report["tasks"][task]["variants"][reference]["runs"]
                for task in task_names
            )
        ]
        for metric, direction in METRICS.items():
            if common_seeds:
                left = [
                    statistics.mean(
                        report["tasks"][task]["variants"][candidate]["runs"][seed][metric]
                        for task in task_names
                    )
                    for seed in common_seeds
                ]
                right = [
                    statistics.mean(
                        report["tasks"][task]["variants"][reference]["runs"][seed][metric]
                        for task in task_names
                    )
                    for seed in common_seeds
                ]
                aggregate["paired_primary_contrasts"][name][metric] = _paired_summary(
                    left, right, direction
                )
    report["aggregate_macro_across_tasks"] = aggregate


def run(
    root: Path,
    out: Path,
    vocab_size: int,
    languages: tuple[str, ...],
    seeds: tuple[int, ...],
    target_bytes: tuple[int, ...],
    steps: int,
    batch_size: int,
    seq_len: int,
    dim: int,
    layers: int,
    heads: int,
    context_bytes: int,
    eval_per_language: int,
    eval_batch_size: int,
    selected_variants: tuple[str, ...],
) -> dict:
    unknown = set(selected_variants).difference(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    selected = {name: VARIANTS[name] for name in selected_variants}
    train_text = "\n".join(_load_split(root, "dev", languages).values())
    report = {
        "methodology": (
            "deterministic marked-span exact generation; paired seeds; shared examples, "
            "batch order, seeded shared-parameter initialization, optimizer, update count, "
            "padded sequence length, vocabulary budget, and greedy decode budget"
        ),
        "scope_limit": (
            "This is reliable evidence for multilingual marked-span generation, not "
            "translation or open-ended generation."
        ),
        "vocab_size": vocab_size,
        "languages": list(languages),
        "seeds": list(seeds),
        "target_bytes": list(target_bytes),
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "model": {"dim": dim, "layers": layers, "heads": heads},
        "context_bytes": context_bytes,
        "eval_per_language": eval_per_language,
        "variants": {name: asdict(variant) for name, variant in selected.items()},
        "primary_contrasts": [
            [candidate, reference]
            for candidate, reference in PRIMARY_CONTRASTS
            if candidate in selected and reference in selected
        ],
        "tasks": {},
    }
    if out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        keys = (
            "vocab_size", "languages", "seeds", "target_bytes", "steps", "batch_size",
            "seq_len", "model", "context_bytes", "eval_per_language", "variants",
        )
        if any(previous.get(key) != report.get(key) for key in keys):
            raise ValueError("existing output configuration differs; choose a new --out path")
        report = previous

    with tempfile.TemporaryDirectory() as directory:
        tokenizer_names = tuple(sorted({variant.tokenizer for variant in selected.values()}))
        tokenizers, tokenizer_train_seconds = _train_tokenizers(
            train_text, vocab_size, Path(directory), tokenizer_names
        )
        train_examples_by_task = {}
        eval_examples_by_task = {}
        train_encoded_by_task = {}
        eval_encoded_by_task = {}
        for width in target_bytes:
            task_name = f"target_bytes_{width}"
            train_examples = _load_examples(root, "dev", languages, context_bytes, width)
            eval_examples = _load_examples(root, "devtest", languages, context_bytes, width)
            train_encoded = {
                name: _encode_examples(tokenizer, train_examples)
                for name, tokenizer in tokenizers.items()
            }
            eval_encoded = {
                name: _encode_examples(tokenizer, eval_examples)
                for name, tokenizer in tokenizers.items()
            }
            train_fit = _shared_fit_indices(train_encoded, seq_len)
            eval_fit = _shared_fit_indices(eval_encoded, seq_len)
            eval_indices = _evenly_spaced_eval_indices(
                eval_examples, eval_fit, eval_per_language
            )
            max_new_tokens = max(
                len(eval_encoded[name][index].target_ids)
                for name in tokenizers
                for index in eval_indices
            )
            task = report["tasks"].setdefault(task_name, {})
            task.update({
                "target_bytes": width,
                "shared_train_examples": len(train_fit),
                "shared_eval_examples": len(eval_indices),
                "max_new_tokens": max_new_tokens,
            })
            task.setdefault("variants", {})
            for variant_name, variant in selected.items():
                tokenizer = tokenizers[variant.tokenizer]
                encoded_train = train_encoded[variant.tokenizer]
                encoded_eval = eval_encoded[variant.tokenizer]
                result = task["variants"].setdefault(variant_name, {"runs": {}})
                result.update({
                    "tokenizer": variant.tokenizer,
                    "tokenizer_train_seconds": tokenizer_train_seconds[variant.tokenizer],
                    "mean_train_target_tokens_including_eos": statistics.mean(
                        len(encoded_train[index].target_ids) for index in train_fit
                    ),
                    "mean_eval_prompt_tokens": statistics.mean(
                        len(encoded_eval[index].prompt_ids) for index in eval_indices
                    ),
                    "mean_eval_target_tokens_including_eos": statistics.mean(
                        len(encoded_eval[index].target_ids) for index in eval_indices
                    ),
                })
            train_examples_by_task[task_name] = (train_examples, train_fit)
            eval_examples_by_task[task_name] = (eval_examples, eval_indices, max_new_tokens)
            train_encoded_by_task[task_name] = train_encoded
            eval_encoded_by_task[task_name] = eval_encoded

        config = TransformerConfig(
            vocab_size=vocab_size,
            dim=dim,
            layers=layers,
            heads=heads,
            max_seq_len=seq_len,
        )
        variant_names = list(selected)
        for task_offset, task_name in enumerate(sorted(report["tasks"])):
            task = report["tasks"][task_name]
            _, train_fit = train_examples_by_task[task_name]
            eval_examples, eval_indices, max_new_tokens = eval_examples_by_task[task_name]
            for seed_offset, seed in enumerate(seeds):
                seed_everything(seed)
                reference_model = CachedTransformerLM(config)
                reference_state = {
                    name: value.detach().clone()
                    for name, value in reference_model.state_dict().items()
                }
                del reference_model
                rotation = (task_offset + seed_offset) % len(variant_names)
                ordered_variants = variant_names[rotation:] + variant_names[:rotation]
                for variant_name in ordered_variants:
                    result = task["variants"][variant_name]
                    if str(seed) in result["runs"]:
                        continue
                    variant = selected[variant_name]
                    tokenizer = tokenizers[variant.tokenizer]
                    train_encoded = train_encoded_by_task[task_name][variant.tokenizer]
                    eval_encoded = eval_encoded_by_task[task_name][variant.tokenizer]
                    corpus = _SpanTaskCorpus(tokenizer, train_encoded, train_fit, seq_len, seed)
                    seed_everything(seed)
                    model = CachedTransformerLM(
                        TransformerConfig(
                            vocab_size=vocab_size,
                            dim=dim,
                            layers=layers,
                            heads=heads,
                            max_seq_len=seq_len,
                            hierarchical_output=variant.hierarchical_output,
                            hierarchy_gears=6,
                            hierarchy_aux_weight=variant.hierarchy_aux_weight,
                            hierarchy_aux_min_gear=2,
                            hierarchy_aux_target="bytes",
                            hierarchy_aux_max_bytes=16,
                        )
                    )
                    _copy_shared_initialization(model, reference_state)
                    configure_token_hierarchy(model, tokenizer)
                    if variant.compositional_init:
                        initialize_token_embeddings(model, tokenizer, "merge_compositional")
                    parameter_count = sum(parameter.numel() for parameter in model.parameters())
                    result["parameter_count"] = parameter_count
                    # Reset post-initialization RNG so augmentation is reproducible and
                    # model-construction draw counts cannot affect training.
                    seed_everything(seed + 10_000)
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
                        segmentation_dropout_prob=variant.segmentation_dropout_prob,
                        segmentation_dropout_min_gear=2,
                        segmentation_dropout_max_depth=1,
                    )
                    started = time.perf_counter()
                    trainer.train_steps(steps, batch_size, seq_len, log_every=0)
                    train_seconds = time.perf_counter() - started
                    generation, rows = _evaluate_generation(
                        trainer.raw_model,
                        tokenizer,
                        eval_encoded,
                        eval_examples,
                        eval_indices,
                        max_new_tokens,
                        eval_batch_size,
                    )
                    teacher = _evaluate_teacher_forced(
                        trainer.raw_model,
                        tokenizer,
                        eval_encoded,
                        eval_examples,
                        eval_indices,
                        seq_len,
                        eval_batch_size,
                    )
                    run_metrics = {
                        **generation,
                        **teacher,
                        "seed": seed,
                        "model_train_seconds": train_seconds,
                        "language_metrics": _language_metrics(rows),
                    }
                    result["runs"][str(seed)] = run_metrics
                    _summarize(report)
                    _atomic_write(out, report)
                    print(
                        task_name,
                        variant_name,
                        seed,
                        json.dumps({
                            key: run_metrics[key]
                            for key in (
                                "exact_match_pct",
                                "mean_edit_similarity_pct",
                                "teacher_forced_bits_per_target_byte",
                                "model_train_seconds",
                                "generation_seconds",
                            )
                        }),
                        flush=True,
                    )
    _summarize(report)
    _atomic_write(out, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flores-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--languages", nargs="*", default=DEFAULT_LANGUAGES)
    parser.add_argument("--seeds", nargs="*", type=int, default=tuple(range(10)))
    parser.add_argument("--target-bytes", nargs="*", type=int, default=(4, 8, 16))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=80)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--context-bytes", type=int, default=48)
    parser.add_argument("--eval-per-language", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--variants", nargs="*", default=tuple(VARIANTS))
    args = parser.parse_args()
    report = run(
        args.flores_root,
        args.out,
        args.vocab_size,
        tuple(args.languages),
        tuple(args.seeds),
        tuple(args.target_bytes),
        args.steps,
        args.batch_size,
        args.seq_len,
        args.dim,
        args.layers,
        args.heads,
        args.context_bytes,
        args.eval_per_language,
        args.eval_batch_size,
        tuple(args.variants),
    )
    print(json.dumps(report["aggregate_macro_across_tasks"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
