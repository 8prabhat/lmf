#!/usr/bin/env python
"""Quantitative and three-rater blinded generation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lmf.data.corpora import EduCombinedCorpus
from lmf.models.pure_parallel_gear import PureParallelGearLM
from lmf.models.rhca.state import SamplingConfig
try:
    from scripts.pure_parallel_gear_common import load_model
except ModuleNotFoundError:
    from pure_parallel_gear_common import load_model


DOMAINS = (
    "cosmopedia",
    "fineweb_edu",
    "open_web_math",
    "pes2o",
    "pg19",
    "stack_exchange",
    "wikipedia",
)
CONTINUATION_LENGTHS = (64, 256, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blind-key-output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompts-per-domain", type=int, default=30)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument(
        "--continuation-lengths",
        nargs="+",
        type=int,
        default=CONTINUATION_LENGTHS,
    )
    parser.add_argument("--seed", type=int, default=20261100)
    parser.add_argument(
        "--training-4gram-hashes",
        type=Path,
        help="Optional newline-delimited hashes of training token 4-grams.",
    )
    return parser.parse_args()


def token_statistics(
    tokens: torch.Tensor,
    training_hashes: set[str] | None = None,
) -> dict[str, float]:
    values = [int(value) for value in tokens.flatten().tolist()]
    ngrams = {
        width: [tuple(values[index : index + width]) for index in range(len(values) - width + 1)]
        for width in (1, 2, 3, 4)
    }
    counts = Counter(values)
    probabilities = [count / max(len(values), 1) for count in counts.values()]
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
    third = max(1, len(values) // 3)
    first = set(ngrams[2][:third])
    last = set(ngrams[2][-third:])
    result = {
        **{
            f"distinct_{width}": len(set(rows)) / max(len(rows), 1)
            for width, rows in ngrams.items()
        },
        "adjacent_repetition": sum(
            left == right for left, right in zip(values, values[1:])
        )
        / max(len(values) - 1, 1),
        "repeated_4gram_fraction": 1.0
        - len(set(ngrams[4])) / max(len(ngrams[4]), 1),
        "empirical_token_entropy": entropy,
        "first_last_bigram_jaccard": len(first & last) / max(len(first | last), 1),
    }
    if training_hashes is not None:
        hashes = [
            hashlib.blake2b(
                ",".join(str(value) for value in row).encode(),
                digest_size=8,
            ).hexdigest()
            for row in ngrams[4]
        ]
        result["training_4gram_overlap"] = sum(
            value in training_hashes for value in hashes
        ) / max(len(hashes), 1)
    return result


@torch.no_grad()
def reference_nll(model, prompt, reference) -> float:
    joined = torch.cat((prompt, reference), dim=1)
    kwargs = {}
    if isinstance(model, PureParallelGearLM):
        if model._boundary_detector is None:
            raise RuntimeError(
                "Pure Gear reference NLL requires its boundary detector"
            )
        kwargs["sentence_end_mask"] = torch.stack(
            [
                model._boundary_detector.scan_tokens(row.tolist())[1]
                for row in joined.detach().cpu()
            ]
        )
    logits, _ = model(joined, **kwargs)
    prediction = logits[:, prompt.shape[1] - 1 : -1]
    return float(
        F.cross_entropy(
            prediction.reshape(-1, prediction.shape[-1]),
            reference.reshape(-1),
        )
    )


def main() -> None:
    args = parse_args()
    checkpoints = {}
    for value in args.checkpoint:
        name, path = value.split("=", 1)
        checkpoints[name] = Path(path)
    models = {
        name: load_model(path, args.device)
        for name, path in checkpoints.items()
    }
    corpus = EduCombinedCorpus(
        root=str(args.corpus_root),
        tokenizer_name=args.tokenizer_name,
        domains=list(DOMAINS),
        load_tokenizer=True,
        seed=args.seed,
    )
    tokenizer = corpus.tokenizer
    training_hashes = (
        None
        if args.training_4gram_hashes is None
        else {
            line.strip()
            for line in args.training_4gram_hashes.read_text().splitlines()
            if line.strip()
        }
    )
    for model in models.values():
        if isinstance(model, PureParallelGearLM):
            model.configure_boundary_detector(tokenizer)
    decoding = {
        "greedy": SamplingConfig(deterministic=True),
        "t07_p09": SamplingConfig(temperature=0.7, top_p=0.9),
        "t09_p095": SamplingConfig(temperature=0.9, top_p=0.95),
    }
    report: dict[str, Any] = {
        "protocol": {
            "prompts_per_domain": args.prompts_per_domain,
            "total_prompts": args.prompts_per_domain * len(DOMAINS),
            "prompt_tokens": args.prompt_tokens,
            "continuation_lengths": args.continuation_lengths,
            "decoding": list(decoding),
            "required_blind_raters": 3,
            "blind_dimensions": [
                "coherence",
                "relevance",
                "grammar",
                "factual_consistency",
                "non_repetition",
                "useful_novelty",
            ],
            "seed": args.seed,
            "training_4gram_overlap_available": training_hashes is not None,
        },
        "examples": [],
        "blind_review": [],
        "rating_file_schema": {
            "rater_id": "independent-rater-identifier",
            "ratings": [
                {
                    "id": "blind-item-id",
                    "preference": "A|B|tie",
                    "dimensions": {
                        "coherence": "A|B|tie",
                        "relevance": "A|B|tie",
                        "grammar": "A|B|tie",
                        "factual_consistency": "A|B|tie",
                        "non_repetition": "A|B|tie",
                        "useful_novelty": "A|B|tie",
                    },
                }
            ],
        },
    }
    blind_key = []
    rng = random.Random(args.seed)
    for domain in DOMAINS:
        tokens = torch.load(
            args.corpus_root / domain / f"test_{args.tokenizer_name}.pt",
            map_location="cpu",
        ).long().flatten()
        maximum_width = args.prompt_tokens + max(args.continuation_lengths)
        maximum = max(1, len(tokens) - maximum_width)
        starts = [
            round(index * maximum / max(args.prompts_per_domain - 1, 1))
            for index in range(args.prompts_per_domain)
        ]
        for prompt_index, start in enumerate(starts):
            prompt = tokens[start : start + args.prompt_tokens][None].to(args.device)
            example: dict[str, Any] = {
                "domain": domain,
                "prompt_index": prompt_index,
                "prompt": tokenizer.decode(prompt[0].cpu().tolist()),
                "models": {},
            }
            for model_name, model in models.items():
                model_result: dict[str, Any] = {"lengths": {}}
                for continuation_length in args.continuation_lengths:
                    reference = tokens[
                        start
                        + args.prompt_tokens : start
                        + args.prompt_tokens
                        + continuation_length
                    ][None].to(args.device)
                    length_result = {
                        "reference_nll": reference_nll(model, prompt, reference),
                        "reference": tokenizer.decode(reference[0].cpu().tolist()),
                        "decoding": {},
                    }
                    for mode, sampling in decoding.items():
                        samples = 1 if mode == "greedy" else 4
                        generated_samples = []
                        for sample in range(samples):
                            digest = hashlib.blake2b(
                                f"{domain}:{prompt_index}:{continuation_length}:{mode}:{sample}".encode(),
                                digest_size=8,
                            ).digest()
                            torch.manual_seed(
                                args.seed + int.from_bytes(digest, "little")
                            )
                            generated = model.generate(
                                prompt,
                                continuation_length,
                                sampling,
                            )[0].cpu()
                            generated_samples.append(
                                {
                                    "text": tokenizer.decode(generated.tolist()),
                                    **token_statistics(
                                        generated,
                                        training_hashes=training_hashes,
                                    ),
                                }
                            )
                        length_result["decoding"][mode] = generated_samples
                    model_result["lengths"][str(continuation_length)] = length_result
                example["models"][model_name] = model_result
            if "gear" in models and "transformer" in models:
                for continuation_length in args.continuation_lengths:
                    names = ["gear", "transformer"]
                    rng.shuffle(names)
                    item_id = (
                        f"{domain}-{prompt_index:02d}-{continuation_length}"
                    )
                    row = {
                        "id": item_id,
                        "prompt": example["prompt"],
                        "continuation_tokens": continuation_length,
                        "A": example["models"][names[0]]["lengths"][
                            str(continuation_length)
                        ]["decoding"]["t09_p095"][0]["text"],
                        "B": example["models"][names[1]]["lengths"][
                            str(continuation_length)
                        ]["decoding"]["t09_p095"][0]["text"],
                        "ratings": {
                            "rater_1": None,
                            "rater_2": None,
                            "rater_3": None,
                        },
                    }
                    report["blind_review"].append(row)
                    blind_key.append(
                        {"id": item_id, "A": names[0], "B": names[1]}
                    )
            report["examples"].append(example)

    aggregate: dict[str, Any] = {}
    metrics = tuple(
        next(
            sample
            for example in report["examples"]
            for row in example["models"].values()
            for length in row["lengths"].values()
            for samples in length["decoding"].values()
            for sample in samples
        ).keys()
    )
    metrics = tuple(name for name in metrics if name != "text")
    for model_name in models:
        aggregate[model_name] = {}
        for continuation_length in args.continuation_lengths:
            rows = [
                example["models"][model_name]["lengths"][
                    str(continuation_length)
                ]
                for example in report["examples"]
            ]
            aggregate[model_name][str(continuation_length)] = {
                "reference_nll": statistics.fmean(
                    row["reference_nll"] for row in rows
                ),
                "decoding": {},
            }
            for mode in decoding:
                samples = [
                    sample
                    for row in rows
                    for sample in row["decoding"][mode]
                ]
                aggregate[model_name][str(continuation_length)]["decoding"][
                    mode
                ] = {
                    metric: statistics.fmean(
                        float(sample[metric]) for sample in samples
                    )
                    for metric in metrics
                }
    report["aggregate"] = aggregate
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    args.blind_key_output.parent.mkdir(parents=True, exist_ok=True)
    args.blind_key_output.write_text(
        json.dumps({"blind_key": blind_key}, indent=2, sort_keys=True)
    )
    print(f"wrote {args.output}")
    print(f"wrote evaluator-only key {args.blind_key_output}")


if __name__ == "__main__":
    main()
