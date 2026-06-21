#!/usr/bin/env python
"""Prepare deduplicated indices and immutable Pure Parallel Gear manifests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lmf.data import (
    build_document_index,
    build_exhaustive_evaluation_manifest,
    build_paired_training_manifest,
)


DEFAULT_DOMAINS = (
    "cosmopedia",
    "fineweb_edu",
    "open_web_math",
    "pes2o",
    "pg19",
    "stack_exchange",
    "wikipedia",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus-root", type=Path, required=True)
    common.add_argument("--tokenizer-name", required=True)
    common.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    common.add_argument("--max-sentence-tokens", type=int, default=128)

    index = subparsers.add_parser("index", parents=[common])
    index.add_argument("--output-root", type=Path, required=True)
    index.add_argument("--bos-id", type=int, required=True)
    index.add_argument("--eos-id", type=int, required=True)
    index.add_argument("--sealed-per-mille", type=int, default=5)

    train = subparsers.add_parser("train-manifest", parents=[common])
    train.add_argument("--index-root", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--training-tokens", type=int, required=True)
    train.add_argument("--capacity-multiplier", type=float, default=2.5)
    train.add_argument("--rows", nargs="+")

    evaluation = subparsers.add_parser("eval-manifest", parents=[common])
    evaluation.add_argument("--index-root", type=Path, required=True)
    evaluation.add_argument("--output-root", type=Path, required=True)
    evaluation.add_argument("--seq-len", type=int, default=4096)
    evaluation.add_argument(
        "--split", choices=("valid", "test", "sealed"), required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "index":
        report = build_document_index(
            args.corpus_root,
            args.output_root,
            tokenizer_name=args.tokenizer_name,
            domains=args.domains,
            bos_id=args.bos_id,
            eos_id=args.eos_id,
            sealed_per_mille=args.sealed_per_mille,
        )
    elif args.command == "train-manifest":
        if args.rows:
            rows = {
                int(length): int(count)
                for length, count in (
                    value.split(":", 1) for value in args.rows
                )
            }
        else:
            if args.capacity_multiplier < 1.0:
                raise ValueError("capacity multiplier must be at least one")
            rows = {
                length: math.ceil(
                    args.training_tokens
                    * args.capacity_multiplier
                    * fraction
                    / length
                )
                for length, fraction in zip(
                    (128, 256, 512, 1024, 2048, 4096),
                    (0.10, 0.15, 0.20, 0.20, 0.20, 0.15),
                )
            }
        report = build_paired_training_manifest(
            args.corpus_root,
            args.index_root,
            args.output_root,
            tokenizer_name=args.tokenizer_name,
            rows_by_length=rows,
            seed=args.seed,
            domains=args.domains,
            max_sentence_tokens=args.max_sentence_tokens,
        )
    else:
        report = build_exhaustive_evaluation_manifest(
            args.corpus_root,
            args.index_root,
            args.output_root,
            tokenizer_name=args.tokenizer_name,
            seq_len=args.seq_len,
            domains=args.domains,
            split=args.split,
            max_sentence_tokens=args.max_sentence_tokens,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
