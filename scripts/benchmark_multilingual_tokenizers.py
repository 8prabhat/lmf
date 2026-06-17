"""Benchmark SPT against matched-vocabulary and pretrained multilingual references.

Requires the extracted FLORES-200 dataset directory containing ``dev`` and
``devtest`` folders. Install optional dependencies with:

    pip install -e '.[multilingual]'
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from lmf.data.tokenizers import FastBPETokenizer, MultiGearTokenizer, SurprisalPhaseTokenizer


DEFAULT_LANGUAGES = (
    "eng_Latn", "spa_Latn", "fra_Latn", "deu_Latn",
    "rus_Cyrl", "ukr_Cyrl", "arb_Arab", "heb_Hebr",
    "hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu",
    "zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang",
    "tha_Thai", "khm_Khmr", "amh_Ethi", "tur_Latn",
    "vie_Latn", "swh_Latn",
)


def _load_split(root: Path, split: str, languages: tuple[str, ...]) -> dict[str, str]:
    return {
        language: (root / split / f"{language}.{split}").read_text(encoding="utf-8")
        for language in languages
    }


def _measure(
    tokenizer,
    train_texts: dict[str, str],
    test_texts: dict[str, str],
    vocab_size: int,
) -> dict:
    train_ids = {language: tokenizer.encode(text) for language, text in train_texts.items()}
    train_counts: Counter[int] = Counter(
        token_id for ids in train_ids.values() for token_id in ids
    )
    started = time.perf_counter()
    test_ids = {language: tokenizer.encode(text) for language, text in test_texts.items()}
    encode_seconds = time.perf_counter() - started

    by_language = {}
    for language, text in test_texts.items():
        ids = test_ids[language]
        by_language[language] = {
            "tokens": len(ids),
            "bytes_per_token": len(text.encode("utf-8")) / len(ids),
            "chars_per_token": len(text) / len(ids),
        }
    total_tokens = sum(row["tokens"] for row in by_language.values())
    total_bytes = sum(len(text.encode("utf-8")) for text in test_texts.values())
    total_chars = sum(len(text) for text in test_texts.values())
    test_counts: Counter[int] = Counter(
        token_id for ids in test_ids.values() for token_id in ids
    )
    rare_train_ids = {token_id for token_id, count in train_counts.items() if count <= 5}
    unseen_train_ids = set(test_counts).difference(train_counts)
    multiword_mass = 0
    for token_id, count in train_counts.items():
        try:
            piece = tokenizer.decode([token_id])
        except Exception:
            continue
        if piece.strip() and len(piece.strip().split()) > 1:
            multiword_mass += count
    train_token_total = sum(train_counts.values())
    return {
        "micro_bytes_per_token": total_bytes / total_tokens,
        "micro_chars_per_token": total_chars / total_tokens,
        "macro_chars_per_token": (
            sum(row["chars_per_token"] for row in by_language.values()) / len(by_language)
        ),
        "encode_megabytes_per_second": total_bytes / max(encode_seconds, 1e-9) / 1e6,
        "used_train_vocab_pct": 100.0 * len(train_counts) / vocab_size,
        "rare_train_types_pct_of_used": (
            100.0 * len(rare_train_ids) / max(len(train_counts), 1)
        ),
        "test_token_mass_from_rare_train_types_pct": (
            100.0 * sum(test_counts[token_id] for token_id in rare_train_ids)
            / max(total_tokens, 1)
        ),
        "test_token_mass_unseen_in_train_pct": (
            100.0 * sum(test_counts[token_id] for token_id in unseen_train_ids)
            / max(total_tokens, 1)
        ),
        "train_multiword_token_mass_pct": (
            100.0 * multiword_mass / max(train_token_total, 1)
        ),
        "by_language": by_language,
    }


def run(
    root: Path,
    vocab_size: int,
    languages: tuple[str, ...],
    tokenizer_names: tuple[str, ...] | None = None,
) -> dict:
    import sentencepiece as spm
    import tiktoken

    train = _load_split(root, "dev", languages)
    test = _load_split(root, "devtest", languages)
    report = {
        "vocab_size": vocab_size,
        "languages": list(languages),
        "selected_tokenizers": list(tokenizer_names) if tokenizer_names is not None else None,
        "tokenizers": {},
    }

    tokenizers = {}
    candidates = (
        ("multigear", MultiGearTokenizer(vocab_size)),
        ("multigear_viterbi", MultiGearTokenizer(vocab_size, inference="viterbi")),
        ("spt", SurprisalPhaseTokenizer(vocab_size)),
        ("byte_bpe", FastBPETokenizer(vocab_size)),
    )
    selected = set(tokenizer_names) if tokenizer_names is not None else None
    for name, tokenizer in candidates:
        if selected is not None and name not in selected:
            continue
        started = time.perf_counter()
        tokenizer.train(list(train.values()))
        tokenizers[name] = (tokenizer, time.perf_counter() - started)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.txt"
        path.write_text("\n".join(train.values()), encoding="utf-8")
        for model_type in ("bpe", "unigram"):
            name = f"sentencepiece_{model_type}"
            if selected is not None and name not in selected:
                continue
            prefix = str(Path(directory) / f"sentencepiece_{model_type}")
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
            sentencepiece = spm.SentencePieceProcessor(model_file=prefix + ".model")
            tokenizers[name] = (
                sentencepiece,
                time.perf_counter() - started,
            )

        for name, (tokenizer, train_seconds) in tokenizers.items():
            actual_vocab_size = (
                tokenizer.vocab_size()
                if name.startswith("sentencepiece_")
                else tokenizer.vocab_size
            )
            report["tokenizers"][name] = {
                "vocab_size": actual_vocab_size,
                "train_seconds": train_seconds,
                **_measure(tokenizer, train, test, actual_vocab_size),
            }

    for encoding_name in ("cl100k_base", "o200k_base"):
        if selected is not None and encoding_name not in selected:
            continue
        encoding = tiktoken.get_encoding(encoding_name)
        report["tokenizers"][encoding_name] = {
            "vocab_size": encoding.n_vocab,
            "train_seconds": None,
            **_measure(encoding, train, test, encoding.n_vocab),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flores-root", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--languages", nargs="*", default=DEFAULT_LANGUAGES)
    parser.add_argument("--tokenizers", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/spt_bench/multilingual.json"))
    args = parser.parse_args()
    report = run(
        args.flores_root,
        args.vocab_size,
        tuple(args.languages),
        tuple(args.tokenizers) if args.tokenizers else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {
        key: value for key, value in metrics.items() if key != "by_language"
    } for name, metrics in report["tokenizers"].items()}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
