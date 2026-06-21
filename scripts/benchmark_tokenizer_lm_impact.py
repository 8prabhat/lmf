"""Compare matched-vocabulary tokenizers by downstream language-model loss.

This benchmark holds transformer shape and token-update compute fixed. It
therefore measures the practical combination of token learnability and sequence
compression. Report bits/byte, not bits/token, across tokenizers.
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
from lmf.data.corpora import WikiTextCorpus
from lmf.data.tokenizers import FastBPETokenizer, MultiGearTokenizer, SurprisalPhaseTokenizer
from lmf.experiments.spt_bench.runner import _downstream_bpt

from benchmark_multilingual_tokenizers import DEFAULT_LANGUAGES, _load_split


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


def _train_tokenizers(
    train_text: str,
    vocab_size: int,
    directory: Path,
    tokenizer_names: tuple[str, ...] | None = None,
) -> dict:
    import sentencepiece as spm

    tokenizers = {}
    candidates = (
        ("multigear", MultiGearTokenizer(vocab_size)),
        ("multigear_viterbi", MultiGearTokenizer(vocab_size, inference="viterbi")),
        ("spt", SurprisalPhaseTokenizer(vocab_size)),
        (
            "spt_legacy_byte_boundaries",
            SurprisalPhaseTokenizer(
                vocab_size,
                threshold=0.8,
                boundary_unit="byte",
                grapheme_vocab_fraction=0.1,
                max_token_bytes=1_000_000,
            ),
        ),
        ("spt_with_lexical_boundaries", SurprisalPhaseTokenizer(vocab_size, pretokenize=True)),
        ("byte_bpe", FastBPETokenizer(vocab_size)),
    )
    selected = set(tokenizer_names) if tokenizer_names is not None else None
    for name, tokenizer in candidates:
        if selected is not None and name not in selected:
            continue
        started = time.perf_counter()
        tokenizer.train([train_text])
        tokenizers[name] = (tokenizer, time.perf_counter() - started)

    path = directory / "train.txt"
    path.write_text(train_text, encoding="utf-8")
    for model_type in ("bpe", "unigram"):
        name = f"sentencepiece_{model_type}"
        if selected is not None and name not in selected:
            continue
        prefix = str(directory / f"sentencepiece_{model_type}")
        started = time.perf_counter()
        spm.SentencePieceTrainer.train(
            input=str(path),
            model_prefix=prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            byte_fallback=True,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            hard_vocab_limit=False,
            minloglevel=2,
        )
        processor = spm.SentencePieceProcessor(model_file=prefix + ".model")
        tokenizers[name] = (
            _SentencePieceAdapter(processor),
            time.perf_counter() - started,
        )
    return tokenizers


def run(
    root: Path,
    vocab_size: int,
    languages: tuple[str, ...],
    seeds: tuple[int, ...],
    steps: int,
    batch_size: int,
    seq_len: int,
    matched_bytes: bool = True,
    tokenizer_names: tuple[str, ...] | None = None,
) -> dict:
    train = "\n".join(_load_split(root, "dev", languages).values())
    valid = "\n".join(_load_split(root, "devtest", languages).values())
    report = {
        "methodology": (
            "matched vocabulary/model shape with both fixed-token-compute and "
            "fixed-raw-byte-exposure evaluations"
        ),
        "vocab_size": vocab_size,
        "languages": list(languages),
        "seeds": list(seeds),
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "matched_bytes": matched_bytes,
        "selected_tokenizers": list(tokenizer_names) if tokenizer_names is not None else None,
        "tokenizers": {},
    }
    cfg = {
        "device": "auto",
        "precision": "fp32",
        "model": {
            "name": "transformer",
            "dim": 64,
            "layers": 2,
            "heads": 2,
            "max_seq_len": max(256, seq_len),
        },
        "trainer": {
            "name": "transformer",
            "lr": 3e-3,
            "warmup_steps": min(30, max(1, steps // 10)),
            "total_steps": steps,
        },
        "run": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "eval_batches": 20,
        },
    }

    with tempfile.TemporaryDirectory() as directory:
        tokenizers = _train_tokenizers(train, vocab_size, Path(directory), tokenizer_names)
        encoded = {}
        for name, (tokenizer, train_seconds) in tokenizers.items():
            train_ids = torch.tensor(tokenizer.encode(train), dtype=torch.long)
            valid_ids = torch.tensor(tokenizer.encode(valid), dtype=torch.long)
            train_bytes_per_token = len(train.encode("utf-8")) / len(train_ids)
            valid_bytes_per_token = len(valid.encode("utf-8")) / len(valid_ids)
            encoded[name] = (
                tokenizer,
                train_seconds,
                train_ids,
                valid_ids,
                train_bytes_per_token,
                valid_bytes_per_token,
            )

        if "byte_bpe" not in encoded:
            raise ValueError("byte_bpe must be selected as the fixed-byte reference")
        reference_bytes_per_token = encoded["byte_bpe"][4]
        target_train_bytes = steps * batch_size * seq_len * reference_bytes_per_token

        for name, values in encoded.items():
            (
                tokenizer,
                train_seconds,
                train_ids,
                valid_ids,
                train_bytes_per_token,
                valid_bytes_per_token,
            ) = values

            def evaluate(n_steps: int) -> list[float]:
                local_cfg = {
                    **cfg,
                    "trainer": {
                        **cfg["trainer"],
                        "warmup_steps": min(30, max(1, n_steps // 10)),
                        "total_steps": n_steps,
                    },
                }
                scores = []
                for seed in seeds:
                    seed_everything(seed)
                    corpus = WikiTextCorpus(
                        tokenizer,
                        train_ids,
                        valid_ids,
                        valid_ids,
                        seed=seed,
                    )
                    bpt = _downstream_bpt(corpus, local_cfg, n_steps)
                    scores.append(bpt / valid_bytes_per_token)
                return scores

            bits_per_byte = evaluate(steps)
            matched_byte_steps = round(
                target_train_bytes / (batch_size * seq_len * train_bytes_per_token)
            )
            matched_byte_scores = evaluate(matched_byte_steps) if matched_bytes else []
            report["tokenizers"][name] = {
                "vocab_size": tokenizer.vocab_size,
                "tokenizer_train_seconds": train_seconds,
                "train_bytes_per_token": train_bytes_per_token,
                "valid_bytes_per_token": valid_bytes_per_token,
                "estimated_train_megabytes_seen": (
                    steps * batch_size * seq_len * train_bytes_per_token / 1e6
                ),
                "bits_per_byte_by_seed": bits_per_byte,
                "mean_bits_per_byte": statistics.mean(bits_per_byte),
                "stdev_bits_per_byte": (
                    statistics.stdev(bits_per_byte) if len(bits_per_byte) > 1 else 0.0
                ),
                "matched_byte_steps": matched_byte_steps,
                "matched_byte_bits_per_byte_by_seed": matched_byte_scores,
                "mean_matched_byte_bits_per_byte": (
                    statistics.mean(matched_byte_scores) if matched_byte_scores else None
                ),
                "stdev_matched_byte_bits_per_byte": (
                    statistics.stdev(matched_byte_scores)
                    if len(matched_byte_scores) > 1
                    else 0.0 if matched_byte_scores else None
                ),
            }
            print(name, json.dumps(report["tokenizers"][name]), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flores-root", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--languages", nargs="*", default=DEFAULT_LANGUAGES)
    parser.add_argument("--seeds", nargs="*", type=int, default=(0, 1, 2))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--skip-matched-bytes", action="store_true")
    parser.add_argument(
        "--tokenizers",
        nargs="*",
        default=None,
        help="optional subset; byte_bpe is required for matched-byte evaluation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/tokenizer/spt_bench/tokenizer_lm_impact.json"),
    )
    args = parser.parse_args()
    report = run(
        args.flores_root,
        args.vocab_size,
        tuple(args.languages),
        tuple(args.seeds),
        args.steps,
        args.batch_size,
        args.seq_len,
        not args.skip_matched_bytes,
        tuple(args.tokenizers) if args.tokenizers else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
