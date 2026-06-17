"""Offline tokenizer training and token-id materialization utilities."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .corpora import tokenizer_fingerprint
from .tokenizers import (
    MultiGearPredictionAwareTokenizer,
    MultiGearTokenizer,
    SentencePieceTokenizer,
    SpecialTokenTokenizer,
)


_TEXT_SUFFIXES = {".txt", ".text", ".md", ".jsonl"}


def _safe_domain_name(path: Path, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem if path.is_file() else path.name)
    stem = stem.strip("._-") or "text"
    name = stem
    index = 2
    while name in used:
        name = f"{stem}_{index}"
        index += 1
    used.add(name)
    return name


def _read_jsonl(path: Path, text_key: str) -> str:
    parts: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                parts.append(line.rstrip("\n"))
                continue
            text = value.get(text_key) if isinstance(value, dict) else None
            parts.append(str(text) if text is not None else line.rstrip("\n"))
    return "\n".join(parts)


def _read_text_source(path: Path, jsonl_text_key: str) -> str:
    if path.is_dir():
        files = sorted(
            child
            for child in path.rglob("*")
            if child.is_file()
            and not child.name.startswith(".")
            and child.suffix.lower() in _TEXT_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"no text files found under {path}")
        return "\n".join(_read_text_source(child, jsonl_text_key) for child in files)
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path, jsonl_text_key)
    return path.read_text(encoding="utf-8", errors="replace")


def _split_text(text: str, train_frac: float, valid_frac: float) -> tuple[str, str, str]:
    train_cut = int(len(text) * train_frac)
    valid_cut = int(len(text) * (train_frac + valid_frac))
    return text[:train_cut], text[train_cut:valid_cut], text[valid_cut:]


def _token_dtype(vocab_size: int, dtype: str) -> np.dtype:
    if dtype == "auto":
        return np.dtype("uint16" if vocab_size <= np.iinfo(np.uint16).max else "uint32")
    resolved = np.dtype(dtype)
    if np.iinfo(resolved).max < vocab_size - 1:
        raise ValueError(f"dtype={resolved.name} cannot store vocab_size={vocab_size}")
    return resolved


def _write_train_tokens(path: Path, ids: list[int], dtype: np.dtype) -> None:
    array = np.asarray(ids, dtype=dtype)
    array.tofile(path)


def _write_eval_tokens(path: Path, ids: list[int]) -> None:
    torch.save(torch.tensor(ids, dtype=torch.int32), path)


def _sentencepiece_training_text(texts: list[str], max_line_chars: int = 2048) -> str:
    lines: list[str] = []
    for text in texts:
        for raw_line in text.splitlines() or [text]:
            line = raw_line.strip()
            if not line:
                continue
            for start in range(0, len(line), max_line_chars):
                chunk = line[start:start + max_line_chars].strip()
                if chunk:
                    lines.append(chunk)
    if not lines:
        raise ValueError("SentencePiece training text is empty after line splitting")
    return "\n".join(lines)


def _materialize_multigear_split_texts(
    split_texts: dict[str, tuple[str, str, str]],
    domains: list[dict[str, str]],
    output_root: str | Path,
    *,
    tokenizer_name: str,
    vocab_size: int,
    tokenizer_kwargs: dict[str, Any] | None,
    dtype: str,
    force: bool,
    tokenizer_cls: type[MultiGearTokenizer] = MultiGearTokenizer,
    tokenizer_format: str = "lmf_multigear_tokenizer_v1",
) -> dict[str, Any]:
    output = Path(output_root).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / f"shared_tokenizer_{tokenizer_name}.pt"
    tokenizer_manifest_path = tokenizer_path.with_suffix(tokenizer_path.suffix + ".manifest.json")
    if tokenizer_path.exists() and not force:
        raise FileExistsError(f"{tokenizer_path} already exists; pass force=True to overwrite")
    existing_domains = [output / domain["name"] for domain in domains if (output / domain["name"]).exists()]
    if existing_domains and not force:
        names = ", ".join(str(path) for path in existing_domains)
        raise FileExistsError(f"{names} already exist; pass force=True to overwrite")

    started = time.perf_counter()
    base = tokenizer_cls(max_vocab=int(vocab_size), **(tokenizer_kwargs or {}))
    base.train([parts[0] for parts in split_texts.values()])
    tokenizer = SpecialTokenTokenizer(base)
    train_seconds = time.perf_counter() - started
    fingerprint = tokenizer_fingerprint(tokenizer)
    token_dtype = _token_dtype(tokenizer.vocab_size, dtype)

    torch.save(tokenizer, tokenizer_path)
    tokenizer_manifest = {
        "format": tokenizer_format,
        "tokenizer_name": tokenizer_name,
        "tokenizer_fingerprint": fingerprint,
        "tokenizer_class": f"{base.__class__.__module__}.{base.__class__.__name__}",
        "base_vocab_size": int(base.vocab_size),
        "vocab_size": int(tokenizer.vocab_size),
        "train_seconds": train_seconds,
        "tokenizer_kwargs": tokenizer_kwargs or {},
    }
    tokenizer_manifest_path.write_text(json.dumps(tokenizer_manifest, indent=2, sort_keys=True))

    report: dict[str, Any] = {
        **tokenizer_manifest,
        "output_root": str(output),
        "domains": [],
    }
    for domain in domains:
        name = domain["name"]
        train_text, valid_text, test_text = split_texts[name]
        domain_dir = output / name
        domain_dir.mkdir(parents=True, exist_ok=True)

        train_ids = tokenizer.encode(train_text)
        valid_ids = tokenizer.encode(valid_text)
        test_ids = tokenizer.encode(test_text)
        train_path = domain_dir / f"train_{tokenizer_name}.bin"
        valid_path = domain_dir / f"valid_{tokenizer_name}.pt"
        test_path = domain_dir / f"test_{tokenizer_name}.pt"
        _write_train_tokens(train_path, train_ids, token_dtype)
        _write_eval_tokens(valid_path, valid_ids)
        _write_eval_tokens(test_path, test_ids)

        manifest = {
            "format": "lmf_pretokenized_text_v1",
            "tokenizer_name": tokenizer_name,
            "tokenizer_fingerprint": fingerprint,
            "vocab_size": int(tokenizer.vocab_size),
            "dtype": token_dtype.name,
            "domain": name,
            "source": domain["source"],
            "train_chars": len(train_text),
            "valid_chars": len(valid_text),
            "test_chars": len(test_text),
            "train_tokens": len(train_ids),
            "valid_tokens": len(valid_ids),
            "test_tokens": len(test_ids),
            **{k: v for k, v in domain.items() if k not in {"name", "source"}},
        }
        train_path.with_suffix(train_path.suffix + ".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        report["domains"].append(manifest)
    return report


def _materialize_sentencepiece_bpe_split_texts(
    split_texts: dict[str, tuple[str, str, str]],
    domains: list[dict[str, Any]],
    output_root: str | Path,
    *,
    tokenizer_name: str,
    vocab_size: int,
    dtype: str,
    force: bool,
) -> dict[str, Any]:
    output = Path(output_root).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / f"shared_tokenizer_{tokenizer_name}.pt"
    model_path = output / f"shared_tokenizer_{tokenizer_name}.model"
    tokenizer_manifest_path = tokenizer_path.with_suffix(tokenizer_path.suffix + ".manifest.json")
    for path in (tokenizer_path, model_path):
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists; pass force=True to overwrite")
    existing_domains = [output / domain["name"] for domain in domains if (output / domain["name"]).exists()]
    if existing_domains and not force:
        names = ", ".join(str(path) for path in existing_domains)
        raise FileExistsError(f"{names} already exist; pass force=True to overwrite")

    import sentencepiece as spm

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        train_path = tmp / "sentencepiece_train.txt"
        train_path.write_text(
            _sentencepiece_training_text([parts[0] for parts in split_texts.values()]),
            encoding="utf-8",
        )
        prefix = tmp / "sentencepiece_bpe"
        spm.SentencePieceTrainer.train(
            input=str(train_path),
            model_prefix=str(prefix),
            vocab_size=int(vocab_size),
            model_type="bpe",
            character_coverage=1.0,
            byte_fallback=True,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            hard_vocab_limit=False,
            minloglevel=2,
        )
        shutil.copyfile(prefix.with_suffix(".model"), model_path)

    base = SentencePieceTokenizer(model_file=str(model_path))
    tokenizer = SpecialTokenTokenizer(base)
    train_seconds = time.perf_counter() - started
    fingerprint = tokenizer_fingerprint(tokenizer)
    token_dtype = _token_dtype(tokenizer.vocab_size, dtype)

    torch.save(tokenizer, tokenizer_path)
    tokenizer_manifest = {
        "format": "lmf_sentencepiece_bpe_tokenizer_v1",
        "tokenizer_name": tokenizer_name,
        "tokenizer_fingerprint": fingerprint,
        "requested_base_vocab_size": int(vocab_size),
        "base_vocab_size": int(base.vocab_size),
        "vocab_size": int(tokenizer.vocab_size),
        "train_seconds": train_seconds,
        "sentencepiece_model": model_path.name,
        "sentencepiece_options": {
            "model_type": "bpe",
            "byte_fallback": True,
            "normalization_rule_name": "identity",
            "add_dummy_prefix": False,
            "remove_extra_whitespaces": False,
            "hard_vocab_limit": False,
        },
    }
    tokenizer_manifest_path.write_text(json.dumps(tokenizer_manifest, indent=2, sort_keys=True))

    report: dict[str, Any] = {
        **tokenizer_manifest,
        "output_root": str(output),
        "domains": [],
    }
    for domain in domains:
        name = domain["name"]
        train_text, valid_text, test_text = split_texts[name]
        domain_dir = output / name
        domain_dir.mkdir(parents=True, exist_ok=True)
        train_ids = tokenizer.encode(train_text)
        valid_ids = tokenizer.encode(valid_text)
        test_ids = tokenizer.encode(test_text)
        train_bin = domain_dir / f"train_{tokenizer_name}.bin"
        valid_path = domain_dir / f"valid_{tokenizer_name}.pt"
        test_path = domain_dir / f"test_{tokenizer_name}.pt"
        _write_train_tokens(train_bin, train_ids, token_dtype)
        _write_eval_tokens(valid_path, valid_ids)
        _write_eval_tokens(test_path, test_ids)
        manifest = {
            "format": "lmf_pretokenized_text_v1",
            "tokenizer_name": tokenizer_name,
            "tokenizer_fingerprint": fingerprint,
            "vocab_size": int(tokenizer.vocab_size),
            "dtype": token_dtype.name,
            "domain": name,
            "source": domain["source"],
            "train_chars": len(train_text),
            "valid_chars": len(valid_text),
            "test_chars": len(test_text),
            "train_tokens": len(train_ids),
            "valid_tokens": len(valid_ids),
            "test_tokens": len(test_ids),
            **{k: v for k, v in domain.items() if k not in {"name", "source"}},
        }
        train_bin.with_suffix(train_bin.suffix + ".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        report["domains"].append(manifest)
    return report


def materialize_multigear_dataset(
    sources: list[str | Path],
    output_root: str | Path,
    *,
    tokenizer_name: str = "multigear32768_v1",
    vocab_size: int = 32768,
    tokenizer_kwargs: dict[str, Any] | None = None,
    train_frac: float = 0.85,
    valid_frac: float = 0.075,
    dtype: str = "auto",
    force: bool = False,
    jsonl_text_key: str = "text",
    tokenizer_cls: type[MultiGearTokenizer] = MultiGearTokenizer,
    tokenizer_format: str = "lmf_multigear_tokenizer_v1",
) -> dict[str, Any]:
    """Train MultiGear once and persist tokenized train/valid/test splits.

    The on-disk layout intentionally matches ``EduCombinedCorpus``:

    ``shared_tokenizer_{tokenizer_name}.pt``
        ``torch.save`` of a ``SpecialTokenTokenizer(MultiGearTokenizer(...))``.
    ``<domain>/train_{tokenizer_name}.bin``
        Raw uint16/uint32 token ids, memory-mapped by training.
    ``<domain>/valid_{tokenizer_name}.pt`` and ``test_{tokenizer_name}.pt``
        Small eval tensors loaded by evaluation/generation flows.
    """

    if not sources:
        raise ValueError("at least one source path is required")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0.0 <= valid_frac < 1.0:
        raise ValueError("valid_frac must be in [0, 1)")
    if train_frac + valid_frac >= 1.0:
        raise ValueError("train_frac + valid_frac must leave a non-empty test split")

    used_names: set[str] = set()
    domains: list[dict[str, str]] = []
    split_texts: dict[str, tuple[str, str, str]] = {}
    for source in sources:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        name = _safe_domain_name(path, used_names)
        text = _read_text_source(path, jsonl_text_key)
        if len(text) < 3:
            raise ValueError(f"{path} is too small to split into train/valid/test text")
        split_texts[name] = _split_text(text, train_frac, valid_frac)
        domains.append({"name": name, "source": str(path)})
    return _materialize_multigear_split_texts(
        split_texts,
        domains,
        output_root,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        tokenizer_kwargs=tokenizer_kwargs,
        dtype=dtype,
        force=force,
        tokenizer_cls=tokenizer_cls,
        tokenizer_format=tokenizer_format,
    )


def materialize_prediction_aware_multigear_dataset(
    sources: list[str | Path],
    output_root: str | Path,
    *,
    tokenizer_name: str = "multigear_prediction_aware32768_v1",
    vocab_size: int = 32768,
    tokenizer_kwargs: dict[str, Any] | None = None,
    train_frac: float = 0.85,
    valid_frac: float = 0.075,
    dtype: str = "auto",
    force: bool = False,
    jsonl_text_key: str = "text",
) -> dict[str, Any]:
    """Train prediction-aware MultiGear once and persist tokenized splits."""

    kwargs = {"inference": "prediction_aware", **(tokenizer_kwargs or {})}
    return materialize_multigear_dataset(
        sources,
        output_root,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        tokenizer_kwargs=kwargs,
        train_frac=train_frac,
        valid_frac=valid_frac,
        dtype=dtype,
        force=force,
        jsonl_text_key=jsonl_text_key,
        tokenizer_cls=MultiGearPredictionAwareTokenizer,
        tokenizer_format="lmf_multigear_prediction_aware_tokenizer_v1",
    )


def materialize_sentencepiece_bpe_dataset(
    sources: list[str | Path],
    output_root: str | Path,
    *,
    tokenizer_name: str = "sentencepiece_bpe32768_v1",
    vocab_size: int = 32768,
    train_frac: float = 0.85,
    valid_frac: float = 0.075,
    dtype: str = "auto",
    force: bool = False,
    jsonl_text_key: str = "text",
) -> dict[str, Any]:
    """Train SentencePiece BPE once and persist tokenized train/valid/test splits."""

    if not sources:
        raise ValueError("at least one source path is required")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0.0 <= valid_frac < 1.0:
        raise ValueError("valid_frac must be in [0, 1)")
    if train_frac + valid_frac >= 1.0:
        raise ValueError("train_frac + valid_frac must leave a non-empty test split")

    used_names: set[str] = set()
    domains: list[dict[str, str]] = []
    split_texts: dict[str, tuple[str, str, str]] = {}
    for source in sources:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        name = _safe_domain_name(path, used_names)
        text = _read_text_source(path, jsonl_text_key)
        if len(text) < 3:
            raise ValueError(f"{path} is too small to split into train/valid/test text")
        split_texts[name] = _split_text(text, train_frac, valid_frac)
        domains.append({"name": name, "source": str(path)})
    return _materialize_sentencepiece_bpe_split_texts(
        split_texts,
        domains,
        output_root,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        dtype=dtype,
        force=force,
    )


def _decode_stratified_windows(
    tokenizer: Any,
    tokens: Any,
    length: int,
    *,
    target_tokens: int,
    window_tokens: int,
    seed: int,
) -> tuple[str, int, int]:
    if length <= 0 or target_tokens <= 0:
        return "", 0, 0
    window = max(1, min(int(window_tokens), int(length), int(target_tokens)))
    count = max(1, (int(target_tokens) + window - 1) // window)
    max_start = max(0, int(length) - window)
    rng = np.random.default_rng(seed)
    pieces: list[str] = []
    sampled = 0
    for index in range(count):
        if max_start == 0:
            start = 0
        else:
            lo = int(index * max_start / count)
            hi = int((index + 1) * max_start / count)
            start = int(rng.integers(lo, max(lo + 1, hi + 1)))
        size = min(window, int(target_tokens) - sampled)
        if size <= 0:
            break
        chunk = np.asarray(tokens[start:start + size], dtype=np.int64).copy()
        pieces.append(tokenizer.decode([int(value) for value in chunk.tolist()]))
        sampled += int(chunk.size)
    return "\n".join(pieces), sampled, count


def materialize_multigear_from_edu_combined(
    source_root: str | Path,
    output_root: str | Path,
    *,
    source_tokenizer_name: str = "bpe32768_v2",
    tokenizer_name: str = "multigear_edu10pct_v1",
    vocab_size: int = 32768,
    tokenizer_kwargs: dict[str, Any] | None = None,
    fraction: float = 0.10,
    max_bpe_tokens_per_domain: int | None = None,
    window_tokens: int = 65536,
    domains: list[str] | None = None,
    train_frac: float = 0.85,
    valid_frac: float = 0.075,
    dtype: str = "auto",
    force: bool = False,
    seed: int = 0,
    tokenizer_cls: type[MultiGearTokenizer] = MultiGearTokenizer,
    tokenizer_format: str = "lmf_multigear_tokenizer_v1",
) -> dict[str, Any]:
    """Sample existing ``edu_combined`` BPE shards, decode, and re-tokenize with MultiGear."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    from .corpora import EduCombinedCorpus, NumericFallbackTokenizer

    source = EduCombinedCorpus(
        str(source_root),
        tokenizer_name=source_tokenizer_name,
        domains=domains,
        seed=seed,
        load_tokenizer=True,
    )
    if isinstance(source.tokenizer, NumericFallbackTokenizer):
        raise RuntimeError(
            f"could not load source tokenizer {source_tokenizer_name!r}; "
            "decoding token ids to text would be lossy"
        )

    split_texts: dict[str, tuple[str, str, str]] = {}
    domain_specs: list[dict[str, Any]] = []
    for offset, shard in enumerate(source.train_shards):
        requested = max(1, int(shard.length * fraction))
        target = (
            requested
            if max_bpe_tokens_per_domain is None
            else min(requested, int(max_bpe_tokens_per_domain))
        )
        text, sampled, windows = _decode_stratified_windows(
            source.tokenizer,
            shard.tokens,
            shard.length,
            target_tokens=target,
            window_tokens=window_tokens,
            seed=seed + 104729 * (offset + 1),
        )
        if len(text) < 3:
            raise ValueError(f"sampled text for {shard.name} is too small")
        split_texts[shard.name] = _split_text(text, train_frac, valid_frac)
        domain_specs.append(
            {
                "name": shard.name,
                "source": str(shard.path),
                "source_tokenizer_name": source_tokenizer_name,
                "source_train_tokens": shard.length,
                "requested_fraction": fraction,
                "requested_bpe_tokens": requested,
                "sampled_bpe_tokens": sampled,
                "sample_windows": windows,
            }
        )

    report = _materialize_multigear_split_texts(
        split_texts,
        domain_specs,
        output_root,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        tokenizer_kwargs=tokenizer_kwargs,
        dtype=dtype,
        force=force,
        tokenizer_cls=tokenizer_cls,
        tokenizer_format=tokenizer_format,
    )
    report["source_root"] = str(Path(source_root).expanduser())
    report["source_tokenizer_name"] = source_tokenizer_name
    report["requested_fraction"] = fraction
    report["max_bpe_tokens_per_domain"] = max_bpe_tokens_per_domain
    report["sampled_bpe_tokens_total"] = sum(
        int(domain.get("sampled_bpe_tokens", 0)) for domain in domain_specs
    )
    report["requested_bpe_tokens_total"] = sum(
        int(domain.get("requested_bpe_tokens", 0)) for domain in domain_specs
    )
    return report


def materialize_prediction_aware_multigear_from_edu_combined(
    source_root: str | Path,
    output_root: str | Path,
    *,
    source_tokenizer_name: str = "bpe32768_v2",
    tokenizer_name: str = "multigear_prediction_aware_edu10pct_v1",
    vocab_size: int = 32768,
    tokenizer_kwargs: dict[str, Any] | None = None,
    fraction: float = 0.10,
    max_bpe_tokens_per_domain: int | None = None,
    window_tokens: int = 65536,
    domains: list[str] | None = None,
    train_frac: float = 0.85,
    valid_frac: float = 0.075,
    dtype: str = "auto",
    force: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Sample ``edu_combined`` and materialize prediction-aware MultiGear ids."""

    kwargs = {"inference": "prediction_aware", **(tokenizer_kwargs or {})}
    return materialize_multigear_from_edu_combined(
        source_root,
        output_root,
        source_tokenizer_name=source_tokenizer_name,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        tokenizer_kwargs=kwargs,
        fraction=fraction,
        max_bpe_tokens_per_domain=max_bpe_tokens_per_domain,
        window_tokens=window_tokens,
        domains=domains,
        train_frac=train_frac,
        valid_frac=valid_frac,
        dtype=dtype,
        force=force,
        seed=seed,
        tokenizer_cls=MultiGearPredictionAwareTokenizer,
        tokenizer_format="lmf_multigear_prediction_aware_tokenizer_v1",
    )


def materialize_sentencepiece_bpe_from_edu_combined(
    source_root: str | Path,
    output_root: str | Path,
    *,
    source_tokenizer_name: str = "bpe32768_v2",
    tokenizer_name: str = "sentencepiece_bpe_edu_subset_v1",
    vocab_size: int = 32768,
    fraction: float = 0.10,
    max_bpe_tokens_per_domain: int | None = None,
    window_tokens: int = 65536,
    domains: list[str] | None = None,
    train_frac: float = 0.85,
    valid_frac: float = 0.075,
    dtype: str = "auto",
    force: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Sample existing ``edu_combined`` BPE shards and re-tokenize with SentencePiece BPE."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    from .corpora import EduCombinedCorpus, NumericFallbackTokenizer

    source = EduCombinedCorpus(
        str(source_root),
        tokenizer_name=source_tokenizer_name,
        domains=domains,
        seed=seed,
        load_tokenizer=True,
    )
    if isinstance(source.tokenizer, NumericFallbackTokenizer):
        raise RuntimeError(
            f"could not load source tokenizer {source_tokenizer_name!r}; "
            "decoding token ids to text would be lossy"
        )

    split_texts: dict[str, tuple[str, str, str]] = {}
    domain_specs: list[dict[str, Any]] = []
    for offset, shard in enumerate(source.train_shards):
        requested = max(1, int(shard.length * fraction))
        target = (
            requested
            if max_bpe_tokens_per_domain is None
            else min(requested, int(max_bpe_tokens_per_domain))
        )
        text, sampled, windows = _decode_stratified_windows(
            source.tokenizer,
            shard.tokens,
            shard.length,
            target_tokens=target,
            window_tokens=window_tokens,
            seed=seed + 104729 * (offset + 1),
        )
        if len(text) < 3:
            raise ValueError(f"sampled text for {shard.name} is too small")
        split_texts[shard.name] = _split_text(text, train_frac, valid_frac)
        domain_specs.append(
            {
                "name": shard.name,
                "source": str(shard.path),
                "source_tokenizer_name": source_tokenizer_name,
                "source_train_tokens": shard.length,
                "requested_fraction": fraction,
                "requested_bpe_tokens": requested,
                "sampled_bpe_tokens": sampled,
                "sample_windows": windows,
            }
        )

    report = _materialize_sentencepiece_bpe_split_texts(
        split_texts,
        domain_specs,
        output_root,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        dtype=dtype,
        force=force,
    )
    report["source_root"] = str(Path(source_root).expanduser())
    report["source_tokenizer_name"] = source_tokenizer_name
    report["requested_fraction"] = fraction
    report["max_bpe_tokens_per_domain"] = max_bpe_tokens_per_domain
    report["sampled_bpe_tokens_total"] = sum(
        int(domain.get("sampled_bpe_tokens", 0)) for domain in domain_specs
    )
    report["requested_bpe_tokens_total"] = sum(
        int(domain.get("requested_bpe_tokens", 0)) for domain in domain_specs
    )
    return report
