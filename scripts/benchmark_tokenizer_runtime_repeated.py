"""Repeated runtime measurement for the tokenizers used by generation tests."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

try:
    from benchmark_tokenizer_generation import (
        DEFAULT_LANGUAGES,
        _load_split,
        _train_tokenizers,
    )
except ModuleNotFoundError:
    from scripts.benchmark_tokenizer_generation import (
        DEFAULT_LANGUAGES,
        _load_split,
        _train_tokenizers,
    )


def _timed(callable_, repeats: int) -> list[float]:
    callable_()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        values.append(time.perf_counter() - started)
    return values


def run(
    root: Path,
    vocab_size: int,
    languages: tuple[str, ...],
    tokenizer_names: tuple[str, ...],
    repeats: int,
) -> dict:
    train_text = "\n".join(_load_split(root, "dev", languages).values())
    eval_texts = list(_load_split(root, "devtest", languages).values())
    total_bytes = sum(len(text.encode("utf-8")) for text in eval_texts)
    report = {
        "methodology": (
            "same tokenizer training text and vocabulary as exact generation; "
            "one warmup followed by repeated full-corpus encode/decode measurements"
        ),
        "vocab_size": vocab_size,
        "languages": list(languages),
        "repeats": repeats,
        "eval_bytes": total_bytes,
        "tokenizers": {},
    }
    with tempfile.TemporaryDirectory() as directory:
        tokenizers, train_seconds = _train_tokenizers(
            train_text, vocab_size, Path(directory), tokenizer_names
        )
        for name, tokenizer in tokenizers.items():
            encoded = [tokenizer.encode(text) for text in eval_texts]
            if any(tokenizer.decode(ids) != text for tokenizer, ids, text in (
                (tokenizer, ids, text) for ids, text in zip(encoded, eval_texts)
            )):
                raise AssertionError(f"{name} failed exact roundtrip")
            encode_seconds = _timed(
                lambda: [tokenizer.encode(text) for text in eval_texts], repeats
            )
            decode_seconds = _timed(
                lambda: [tokenizer.decode(ids) for ids in encoded], repeats
            )
            tokens = sum(len(ids) for ids in encoded)
            result = {
                "tokenizer_train_seconds_single_run": train_seconds[name],
                "eval_tokens": tokens,
                "bytes_per_token": total_bytes / tokens,
                "encode_seconds_by_repeat": encode_seconds,
                "encode_megabytes_per_second_by_repeat": [
                    total_bytes / seconds / 1e6 for seconds in encode_seconds
                ],
                "median_encode_megabytes_per_second": statistics.median(
                    total_bytes / seconds / 1e6 for seconds in encode_seconds
                ),
                "decode_seconds_by_repeat": decode_seconds,
                "median_decode_megabytes_per_second": statistics.median(
                    total_bytes / seconds / 1e6 for seconds in decode_seconds
                ),
                "exact_roundtrip": True,
            }
            report["tokenizers"][name] = result
            print(name, json.dumps(result), flush=True)
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
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.flores_root,
        args.vocab_size,
        tuple(args.languages),
        tuple(args.tokenizers),
        args.repeats,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
