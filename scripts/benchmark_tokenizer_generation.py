"""Compare matched-vocabulary tokenizers on exact conditional generation.

The task is multilingual marked-span extraction. Each prompt contains a FLORES
sentence with one span delimited by special tokens; the model must generate the
marked span followed by EOS. Unlike open-ended continuation, exact match is a
valid primary metric because the requested output is deterministic.

Controls:

* identical tokenizer-training text and vocabulary size;
* identical task examples, model shape/parameter count, padded sequence length,
  optimizer, batch order, update count, and paired random seeds;
* answer-only loss and greedy generation with a shared token budget;
* only examples that fit every tokenizer are admitted.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from lmf.core.seeding import seed_everything
from lmf.data.batch import TrainingBatch
from lmf.data.tokenizers import (
    FastBPETokenizer,
    MultiGearTokenizer,
    SpecialTokenTokenizer,
    SurprisalPhaseTokenizer,
)
from lmf.models.transformer import CachedTransformerLM, TransformerConfig, TransformerTrainer

try:
    from benchmark_multilingual_tokenizers import DEFAULT_LANGUAGES, _load_split
except ModuleNotFoundError:
    from scripts.benchmark_multilingual_tokenizers import DEFAULT_LANGUAGES, _load_split


TASK_SPECIAL_TOKENS = (
    "<|eos|>",
    "<|pad|>",
    "<|context|>",
    "<|answer|>",
    "<|span_start|>",
    "<|span_end|>",
)

MULTIGEAR_VARIANTS = {
    "multigear": {},
    # Diagnostic variants: progressively reclaim vocabulary from phrase-scale
    # gears for output-relevant grapheme and lexical pieces.
    "multigear_local90": {
        "gear_fractions": (0.30, 0.60, 0.08, 0.02, 0.00),
    },
    "multigear_local100": {
        "gear_fractions": (0.30, 0.70, 0.00, 0.00, 0.00),
    },
    "multigear_lexical90": {
        "gear_fractions": (0.18, 0.72, 0.08, 0.02, 0.00),
    },
}


class _SentencePieceAdapter:
    def __init__(self, processor) -> None:
        self.processor = processor

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode(ids)


@dataclass(frozen=True)
class SpanExample:
    language: str
    prompt: str
    target: str

    @property
    def full_text(self) -> str:
        return self.prompt + self.target + "<|eos|>"


@dataclass(frozen=True)
class EncodedExample:
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]

    @property
    def full_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.target_ids


def _utf8_prefix(text: str, max_bytes: int) -> str:
    """Longest codepoint-safe prefix no longer than ``max_bytes`` UTF-8 bytes."""
    used = 0
    out = []
    for char in text:
        width = len(char.encode("utf-8"))
        if used + width > max_bytes:
            break
        out.append(char)
        used += width
    return "".join(out)


def _make_span_example(language: str, text: str, context_bytes: int, target_bytes: int) -> SpanExample:
    context = _utf8_prefix(text.strip(), context_bytes)
    if not context:
        raise ValueError("cannot construct a span task from empty text")
    start = max(0, len(context) // 3)
    target = _utf8_prefix(context[start:], target_bytes)
    if not target:
        target = context[start]
    end = start + len(target)
    prompt = (
        "<|context|>"
        + context[:start]
        + "<|span_start|>"
        + target
        + "<|span_end|>"
        + context[end:]
        + "<|answer|>"
    )
    return SpanExample(language, prompt, target)


def _load_examples(
    root: Path,
    split: str,
    languages: tuple[str, ...],
    context_bytes: int,
    target_bytes: int,
) -> list[SpanExample]:
    by_language = _load_split(root, split, languages)
    examples = []
    for language in languages:
        for line in by_language[language].splitlines():
            if line.strip():
                examples.append(_make_span_example(language, line, context_bytes, target_bytes))
    return examples


def _train_tokenizers(
    train_text: str,
    vocab_size: int,
    directory: Path,
    tokenizer_names: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, float]]:
    import sentencepiece as spm

    base_vocab_size = vocab_size - len(TASK_SPECIAL_TOKENS)
    if base_vocab_size <= 256:
        raise ValueError("vocab_size must leave room for byte and task-special tokens")
    selected = set(tokenizer_names)
    unknown = selected.difference(
        set(MULTIGEAR_VARIANTS)
        | {"spt", "byte_bpe", "sentencepiece_bpe", "sentencepiece_unigram"}
    )
    if unknown:
        raise ValueError(f"unknown tokenizers: {sorted(unknown)}")

    tokenizers: dict[str, object] = {}
    train_seconds: dict[str, float] = {}
    candidates = tuple(
        (name, MultiGearTokenizer(base_vocab_size, **settings))
        for name, settings in MULTIGEAR_VARIANTS.items()
    ) + (
        ("spt", SurprisalPhaseTokenizer(base_vocab_size)),
        ("byte_bpe", FastBPETokenizer(base_vocab_size)),
    )
    for name, base in candidates:
        if name not in selected:
            continue
        started = time.perf_counter()
        base.train([train_text])
        tokenizers[name] = SpecialTokenTokenizer(base, TASK_SPECIAL_TOKENS)
        train_seconds[name] = time.perf_counter() - started

    path = directory / "train.txt"
    path.write_text(train_text, encoding="utf-8")
    for model_type in ("bpe", "unigram"):
        name = f"sentencepiece_{model_type}"
        if name not in selected:
            continue
        prefix = str(directory / name)
        started = time.perf_counter()
        spm.SentencePieceTrainer.train(
            input=str(path),
            model_prefix=prefix,
            vocab_size=base_vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            byte_fallback=True,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            hard_vocab_limit=False,
            minloglevel=2,
        )
        base = _SentencePieceAdapter(spm.SentencePieceProcessor(model_file=prefix + ".model"))
        tokenizers[name] = SpecialTokenTokenizer(base, TASK_SPECIAL_TOKENS)
        train_seconds[name] = time.perf_counter() - started

    actual_sizes = {name: tokenizer.vocab_size for name, tokenizer in tokenizers.items()}
    mismatched = {name: size for name, size in actual_sizes.items() if size != vocab_size}
    if mismatched:
        raise ValueError(f"actual vocabulary sizes do not match {vocab_size}: {mismatched}")
    return tokenizers, train_seconds


def _encode_examples(tokenizer, examples: list[SpanExample]) -> list[EncodedExample]:
    eos_id = tokenizer.special_to_id["<|eos|>"]
    encoded = []
    for example in examples:
        prompt_ids = tuple(tokenizer.encode(example.prompt))
        target_ids = tuple(tokenizer.encode(example.target)) + (eos_id,)
        encoded.append(EncodedExample(prompt_ids, target_ids))
    return encoded


def _shared_fit_indices(encoded: dict[str, list[EncodedExample]], seq_len: int) -> list[int]:
    count = len(next(iter(encoded.values())))
    if any(len(values) != count for values in encoded.values()):
        raise ValueError("tokenizers were given different example counts")
    return [
        index
        for index in range(count)
        if all(len(values[index].full_ids) <= seq_len for values in encoded.values())
    ]


def _stratified_eval_indices(
    examples: list[SpanExample],
    fit_indices: list[int],
    per_language: int,
) -> list[int]:
    selected = []
    counts: dict[str, int] = defaultdict(int)
    for index in fit_indices:
        language = examples[index].language
        if counts[language] < per_language:
            selected.append(index)
            counts[language] += 1
    return selected


class _SpanTaskCorpus:
    """Fixed-example task corpus with answer-only loss and paired sampling."""

    def __init__(
        self,
        tokenizer,
        encoded: list[EncodedExample],
        indices: list[int],
        seq_len: int,
        seed: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.seq_len = seq_len
        pad_id = tokenizer.special_to_id["<|pad|>"]
        tokens = []
        attention_masks = []
        loss_masks = []
        for index in indices:
            example = encoded[index]
            full = example.full_ids
            padding = seq_len - len(full)
            tokens.append(torch.tensor(full + (pad_id,) * padding, dtype=torch.long))
            attention_masks.append(
                torch.tensor((True,) * len(full) + (False,) * padding, dtype=torch.bool)
            )
            loss_masks.append(
                torch.tensor(
                    (False,) * len(example.prompt_ids)
                    + (True,) * len(example.target_ids)
                    + (False,) * padding,
                    dtype=torch.bool,
                )
            )
        self.tokens = torch.stack(tokens)
        self.attention_masks = torch.stack(attention_masks)
        self.loss_masks = torch.stack(loss_masks)
        self._generator = torch.Generator().manual_seed(seed + 1000)

    def sample_batch(self, batch: int, seq_len: int, split: str = "train") -> TrainingBatch:
        if seq_len != self.seq_len:
            raise ValueError(f"task corpus requires seq_len={self.seq_len}, got {seq_len}")
        indices = torch.randint(0, len(self.tokens), (batch,), generator=self._generator)
        return TrainingBatch(
            self.tokens[indices],
            self.attention_masks[indices],
            self.loss_masks[indices],
            task="marked_span_extraction",
        )

    def sampler_state(self) -> dict:
        return {"train": self._generator.get_state()}

    def load_sampler_state(self, state: dict) -> None:
        self._generator.set_state(state["train"].cpu())


def _edit_distance(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def _normalized_edit_similarity(prediction: str, target: str) -> float:
    return 1.0 - _edit_distance(prediction, target) / max(len(prediction), len(target), 1)


def _common_prefix_fraction(prediction: str, target: str) -> float:
    matched = 0
    for predicted, expected in zip(prediction.encode("utf-8"), target.encode("utf-8")):
        if predicted != expected:
            break
        matched += 1
    return matched / max(len(target.encode("utf-8")), 1)


@torch.no_grad()
def _evaluate_generation(
    model,
    tokenizer,
    encoded: list[EncodedExample],
    examples: list[SpanExample],
    indices: list[int],
    max_new_tokens: int,
    eval_batch_size: int,
) -> tuple[dict, list[dict]]:
    model.eval()
    device = next(model.parameters()).device
    eos_id = tokenizer.special_to_id["<|eos|>"]
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        grouped[len(encoded[index].prompt_ids)].append(index)

    rows = []
    started = time.perf_counter()
    for prompt_len in sorted(grouped):
        group = grouped[prompt_len]
        for offset in range(0, len(group), eval_batch_size):
            chunk = group[offset:offset + eval_batch_size]
            prompts = torch.tensor(
                [encoded[index].prompt_ids for index in chunk], dtype=torch.long, device=device
            )
            generated = model.generate(prompts, max_new_tokens)
            for row_index, example_index in enumerate(chunk):
                ids = generated[row_index].tolist()
                eos_position = ids.index(eos_id) if eos_id in ids else None
                prediction_ids = ids if eos_position is None else ids[:eos_position]
                prediction = tokenizer.decode(prediction_ids)
                target = examples[example_index].target
                rows.append(
                    {
                        "index": example_index,
                        "language": examples[example_index].language,
                        "prediction": prediction,
                        "target": target,
                        "exact": prediction == target,
                        "eos": eos_position is not None,
                        "edit_similarity": _normalized_edit_similarity(prediction, target),
                        "prefix_fraction": _common_prefix_fraction(prediction, target),
                    }
                )
    seconds = time.perf_counter() - started
    metrics = {
        "examples": len(rows),
        "exact_match_pct": 100.0 * statistics.mean(row["exact"] for row in rows),
        "eos_rate_pct": 100.0 * statistics.mean(row["eos"] for row in rows),
        "mean_edit_similarity_pct": (
            100.0 * statistics.mean(row["edit_similarity"] for row in rows)
        ),
        "mean_exact_prefix_pct": 100.0 * statistics.mean(row["prefix_fraction"] for row in rows),
        "generation_seconds": seconds,
    }
    return metrics, rows


def _mean_and_stdev(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _paired_seed_summary(report: dict, reference: str = "sentencepiece_bpe") -> dict:
    tokenizers = report["tokenizers"]
    if reference not in tokenizers:
        reference = "byte_bpe" if "byte_bpe" in tokenizers else next(iter(tokenizers))
    fields = {
        "exact_match_pct_difference": "exact_match_pct_by_seed",
        "edit_similarity_pct_difference": "edit_similarity_pct_by_seed",
        "exact_prefix_pct_difference": "exact_prefix_pct_by_seed",
    }
    # Two-sided t critical values for 95% CIs. Seed is the independent unit;
    # examples within one trained model are not incorrectly treated as independent.
    t_critical = {1: None, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    summary = {"reference": reference}
    for output_name, field in fields.items():
        reference_values = tokenizers[reference][field]
        summary[output_name] = {}
        for name, metrics in tokenizers.items():
            differences = [
                value - baseline for value, baseline in zip(metrics[field], reference_values)
            ]
            stats = _mean_and_stdev(differences)
            critical = t_critical.get(len(differences), 1.96)
            if critical is None:
                ci = [None, None]
            else:
                half_width = critical * stats["stdev"] / math.sqrt(len(differences))
                ci = [stats["mean"] - half_width, stats["mean"] + half_width]
            summary[output_name][name] = {
                "by_seed": differences,
                **stats,
                "ci95": ci,
            }
    return summary


def run(
    root: Path,
    vocab_size: int,
    languages: tuple[str, ...],
    tokenizer_names: tuple[str, ...],
    seeds: tuple[int, ...],
    steps: int,
    batch_size: int,
    seq_len: int,
    dim: int,
    layers: int,
    heads: int,
    context_bytes: int,
    target_bytes: int,
    eval_per_language: int,
    eval_batch_size: int,
) -> dict:
    train_texts = _load_split(root, "dev", languages)
    train_text = "\n".join(train_texts.values())
    train_examples = _load_examples(root, "dev", languages, context_bytes, target_bytes)
    eval_examples = _load_examples(root, "devtest", languages, context_bytes, target_bytes)
    report = {
        "methodology": (
            "deterministic marked-span extraction with exact-match generation; "
            "matched vocabulary, model shape, parameter count, examples, padded "
            "sequence length, batch order, optimizer, updates, greedy decode budget, and seeds"
        ),
        "task": "multilingual_marked_span_extraction",
        "vocab_size": vocab_size,
        "languages": list(languages),
        "tokenizers_compared": list(tokenizer_names),
        "seeds": list(seeds),
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "model": {"dim": dim, "layers": layers, "heads": heads},
        "context_bytes": context_bytes,
        "target_bytes": target_bytes,
        "eval_per_language": eval_per_language,
        "tokenizers": {},
    }

    with tempfile.TemporaryDirectory() as directory:
        tokenizers, train_seconds = _train_tokenizers(
            train_text, vocab_size, Path(directory), tokenizer_names
        )
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
        eval_indices = _stratified_eval_indices(eval_examples, eval_fit, eval_per_language)
        if not train_fit or not eval_indices:
            raise ValueError("no shared examples fit; increase seq_len or reduce context_bytes")
        if len(eval_indices) != eval_per_language * len(languages):
            raise ValueError(
                f"only {len(eval_indices)} shared evaluation examples fit; expected "
                f"{eval_per_language * len(languages)}"
            )

        max_new_tokens = max(
            len(eval_encoded[name][index].target_ids)
            for name in tokenizers
            for index in eval_indices
        )
        report["shared_train_examples"] = len(train_fit)
        report["shared_eval_examples"] = len(eval_indices)
        report["max_new_tokens"] = max_new_tokens

        parameter_counts = set()
        for name, tokenizer in tokenizers.items():
            target_lengths = [len(train_encoded[name][index].target_ids) for index in train_fit]
            prompt_lengths = [len(eval_encoded[name][index].prompt_ids) for index in eval_indices]
            target_eval_lengths = [len(eval_encoded[name][index].target_ids) for index in eval_indices]
            per_seed = []
            example_rows = {}
            for seed in seeds:
                seed_everything(seed)
                corpus = _SpanTaskCorpus(tokenizer, train_encoded[name], train_fit, seq_len, seed)
                model = CachedTransformerLM(
                    TransformerConfig(
                        vocab_size=vocab_size,
                        dim=dim,
                        layers=layers,
                        heads=heads,
                        max_seq_len=seq_len,
                    )
                )
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                parameter_counts.add(parameter_count)
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
                )
                train_started = time.perf_counter()
                trainer.train_steps(steps, batch_size, seq_len, log_every=0)
                model_train_seconds = time.perf_counter() - train_started
                metrics, rows = _evaluate_generation(
                    trainer.raw_model,
                    tokenizer,
                    eval_encoded[name],
                    eval_examples,
                    eval_indices,
                    max_new_tokens,
                    eval_batch_size,
                )
                metrics["seed"] = seed
                metrics["model_train_seconds"] = model_train_seconds
                per_seed.append(metrics)
                example_rows[str(seed)] = rows
                print(name, seed, json.dumps(metrics), flush=True)

            result = {
                "vocab_size": tokenizer.vocab_size,
                "parameter_count": parameter_count,
                "tokenizer_train_seconds": train_seconds[name],
                "mean_train_target_tokens_including_eos": statistics.mean(target_lengths),
                "mean_eval_prompt_tokens": statistics.mean(prompt_lengths),
                "mean_eval_target_tokens_including_eos": statistics.mean(target_eval_lengths),
                "exact_match_pct_by_seed": [row["exact_match_pct"] for row in per_seed],
                "edit_similarity_pct_by_seed": [
                    row["mean_edit_similarity_pct"] for row in per_seed
                ],
                "exact_prefix_pct_by_seed": [row["mean_exact_prefix_pct"] for row in per_seed],
                "eos_rate_pct_by_seed": [row["eos_rate_pct"] for row in per_seed],
                "mean_exact_match_pct": statistics.mean(
                    row["exact_match_pct"] for row in per_seed
                ),
                "stdev_exact_match_pct": (
                    statistics.stdev(row["exact_match_pct"] for row in per_seed)
                    if len(per_seed) > 1
                    else 0.0
                ),
                "mean_edit_similarity_pct": statistics.mean(
                    row["mean_edit_similarity_pct"] for row in per_seed
                ),
                "mean_exact_prefix_pct": statistics.mean(
                    row["mean_exact_prefix_pct"] for row in per_seed
                ),
                "mean_eos_rate_pct": statistics.mean(row["eos_rate_pct"] for row in per_seed),
                "mean_model_train_seconds": statistics.mean(
                    row["model_train_seconds"] for row in per_seed
                ),
                "mean_generation_seconds": statistics.mean(
                    row["generation_seconds"] for row in per_seed
                ),
                "per_seed": per_seed,
                "examples_by_seed": example_rows,
            }
            report["tokenizers"][name] = result

        if len(parameter_counts) != 1:
            raise AssertionError(f"model parameter counts differ: {sorted(parameter_counts)}")
        report["parameter_count"] = parameter_counts.pop()
        report["paired_summary"] = _paired_seed_summary(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flores-root", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--languages", nargs="*", default=DEFAULT_LANGUAGES)
    parser.add_argument(
        "--tokenizers",
        nargs="*",
        default=("multigear", "spt", "byte_bpe", "sentencepiece_bpe", "sentencepiece_unigram"),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=80)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--context-bytes", type=int, default=48)
    parser.add_argument("--target-bytes", type=int, default=4)
    parser.add_argument("--eval-per-language", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/tokenizer/spt_bench/tokenizer_generation.json"),
    )
    args = parser.parse_args()
    report = run(
        args.flores_root,
        args.vocab_size,
        tuple(args.languages),
        tuple(args.tokenizers),
        tuple(args.seeds),
        args.steps,
        args.batch_size,
        args.seq_len,
        args.dim,
        args.layers,
        args.heads,
        args.context_bytes,
        args.target_bytes,
        args.eval_per_language,
        args.eval_batch_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        name: {
            "exact_match_pct": metrics["mean_exact_match_pct"],
            "edit_similarity_pct": metrics["mean_edit_similarity_pct"],
            "exact_prefix_pct": metrics["mean_exact_prefix_pct"],
        }
        for name, metrics in report["tokenizers"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
