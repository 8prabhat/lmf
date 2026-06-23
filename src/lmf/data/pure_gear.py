"""Document-aware immutable data utilities for canonical Pure Parallel Gear.

The decisive comparison uses immutable paired manifests.  A manifest describes
the exact document spans packed into every row; both architectures therefore
receive identical tokens and segment boundaries independent of sampler state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ..core.registry import CORPORA
from .batch import TrainingBatch
from .corpora import EduCombinedCorpus
from .sentence_boundaries import SentenceBoundaryDetector


INDEX_FORMAT = "lmf_document_index_v1"
MANIFEST_FORMAT = "lmf_paired_document_windows_v2"
SPLIT_TRAIN = 0
SPLIT_SEALED = 1
SPLIT_EXCLUDED = 2


_REFACTORED_PATH_PREFIXES = (
    (
        Path("outputs/sentencepiece_bpe_prepared"),
        Path("outputs/tokenizer/sentencepiece_bpe_prepared"),
    ),
    (
        Path("outputs/pure_parallel_gear_360_proxy"),
        Path("outputs/pure_parallel_gear/360_proxy"),
    ),
)


def _resolve_manifest_artifact_path(value: str | Path) -> Path:
    """Resolve artifact paths embedded before the 2026 directory refactor.

    Paired manifests are immutable experiment artifacts, so rewriting them
    would alter their hashes and break provenance. Resolve the two documented
    repository moves at load time while preserving the declared path in the
    manifest itself.
    """
    path = Path(value).expanduser()
    if path.exists() or path.is_absolute():
        return path
    for old_prefix, new_prefix in _REFACTORED_PATH_PREFIXES:
        try:
            suffix = path.relative_to(old_prefix)
        except ValueError:
            continue
        relocated = new_prefix / suffix
        if relocated.exists():
            return relocated
    return path


def _token_hash(tokens: np.ndarray) -> int:
    digest = hashlib.blake2b(
        np.asarray(tokens, dtype=np.uint16).tobytes(),
        digest_size=8,
        person=b"lmf-doc-v1",
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def _iter_documents(
    tokens,
    *,
    bos_id: int,
    eos_id: int,
    chunk_tokens: int = 8_000_000,
):
    """Yield inclusive/exclusive document boundaries without loading a shard."""
    length = len(tokens)
    open_start: int | None = None
    yielded = False
    for chunk_start in range(0, length, chunk_tokens):
        chunk_end = min(length, chunk_start + chunk_tokens)
        chunk = np.asarray(tokens[chunk_start:chunk_end])
        boundary = np.flatnonzero((chunk == bos_id) | (chunk == eos_id))
        for local in boundary.tolist():
            position = chunk_start + int(local)
            token = int(chunk[local])
            if token == bos_id:
                if open_start is not None and position > open_start:
                    yielded = True
                    yield open_start, position
                open_start = position
            elif open_start is not None:
                yielded = True
                yield open_start, position + 1
                open_start = None
    if open_start is not None and open_start < length:
        yielded = True
        yield open_start, length
    if not yielded and length >= 2:
        # Some prepared corpora expose a split as one contiguous token stream.
        # Treat it as one document rather than silently producing an empty set.
        yield 0, length


def _tensor_documents(
    tensor: torch.Tensor,
    *,
    bos_id: int,
    eos_id: int,
):
    array = tensor.detach().cpu().numpy()
    yield from _iter_documents(array, bos_id=bos_id, eos_id=eos_id)


def build_document_index(
    corpus_root: str | Path,
    output_root: str | Path,
    *,
    tokenizer_name: str,
    domains: Iterable[str] | None = None,
    bos_id: int = 32768,
    eos_id: int = 32769,
    sealed_per_mille: int = 5,
    chunk_tokens: int = 8_000_000,
) -> dict:
    """Index documents, remove exact duplicates, and reserve a sealed split.

    Official validation/test document hashes are inserted before train scanning,
    so duplicates cannot leak into training or the hash-selected confirmation
    split.  Re-running is deterministic.
    """
    source = Path(corpus_root).expanduser()
    output = Path(output_root).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    selected = (
        sorted(path.name for path in source.iterdir() if path.is_dir())
        if domains is None
        else sorted(str(value) for value in domains)
    )
    seen: set[int] = set()
    report = {
        "format": INDEX_FORMAT,
        "source_root": str(source),
        "tokenizer_name": tokenizer_name,
        "bos_id": int(bos_id),
        "eos_id": int(eos_id),
        "sealed_per_mille": int(sealed_per_mille),
        "domains": {},
        "official_evaluation": {},
    }
    # Evaluation hashes take precedence over all training documents.
    for split in ("valid", "test"):
        report["official_evaluation"][split] = {}
        for domain in selected:
            path = source / domain / f"{split}_{tokenizer_name}.pt"
            if not path.exists():
                continue
            tensor = torch.load(path, map_location="cpu").flatten()
            starts: list[int] = []
            ends: list[int] = []
            hashes: list[int] = []
            splits: list[int] = []
            for start, end in _tensor_documents(
                tensor,
                bos_id=bos_id,
                eos_id=eos_id,
            ):
                value = _token_hash(tensor[start:end].numpy())
                duplicate = value in seen
                seen.add(value)
                starts.append(start)
                ends.append(end)
                hashes.append(value)
                splits.append(
                    SPLIT_EXCLUDED if duplicate else SPLIT_SEALED
                )
            official_dir = output / f"official_{split}" / domain
            official_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                official_dir / "starts.npy",
                np.asarray(starts, dtype=np.uint64),
            )
            np.save(
                official_dir / "ends.npy",
                np.asarray(ends, dtype=np.uint64),
            )
            np.save(
                official_dir / "hashes.npy",
                np.asarray(hashes, dtype=np.uint64),
            )
            np.save(
                official_dir / "splits.npy",
                np.asarray(splits, dtype=np.uint8),
            )
            split_array = np.asarray(splits)
            report["official_evaluation"][split][domain] = {
                "documents": len(starts),
                "unique_documents": int(
                    (split_array == SPLIT_SEALED).sum()
                ),
                "excluded_duplicates": int(
                    (split_array == SPLIT_EXCLUDED).sum()
                ),
            }

    for domain in selected:
        train_path = source / domain / f"train_{tokenizer_name}.bin"
        manifest_path = train_path.with_suffix(train_path.suffix + ".manifest.json")
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {"dtype": "uint16"}
        )
        tokens = np.memmap(
            train_path,
            dtype=np.dtype(manifest.get("dtype", "uint16")),
            mode="r",
        )
        starts: list[int] = []
        ends: list[int] = []
        hashes: list[int] = []
        splits: list[int] = []
        for start, end in _iter_documents(
            tokens,
            bos_id=bos_id,
            eos_id=eos_id,
            chunk_tokens=chunk_tokens,
        ):
            if end - start < 2:
                continue
            value = _token_hash(tokens[start:end])
            if value in seen:
                split = SPLIT_EXCLUDED
            else:
                seen.add(value)
                split = (
                    SPLIT_SEALED
                    if value % 1000 < sealed_per_mille
                    else SPLIT_TRAIN
                )
            starts.append(start)
            ends.append(end)
            hashes.append(value)
            splits.append(split)
        domain_dir = output / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        np.save(domain_dir / "starts.npy", np.asarray(starts, dtype=np.uint64))
        np.save(domain_dir / "ends.npy", np.asarray(ends, dtype=np.uint64))
        np.save(domain_dir / "hashes.npy", np.asarray(hashes, dtype=np.uint64))
        np.save(domain_dir / "splits.npy", np.asarray(splits, dtype=np.uint8))
        split_array = np.asarray(splits)
        report["domains"][domain] = {
            "documents": len(starts),
            "train_documents": int((split_array == SPLIT_TRAIN).sum()),
            "sealed_documents": int((split_array == SPLIT_SEALED).sum()),
            "excluded_documents": int((split_array == SPLIT_EXCLUDED).sum()),
            "train_tokens": int(
                sum(
                    end - start
                    for start, end, split in zip(starts, ends, splits)
                    if split == SPLIT_TRAIN
                )
            ),
        }
    (output / "index.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


@dataclass
class _DomainDocuments:
    name: str
    tokens: object
    starts: np.ndarray
    ends: np.ndarray
    splits: np.ndarray
    train_documents: np.ndarray
    train_token_cdf: np.ndarray


def _load_documents(
    corpus_root: Path,
    index_root: Path,
    tokenizer_name: str,
    domains: Iterable[str],
    source_split: str = "train",
) -> list[_DomainDocuments]:
    output = []
    for domain in domains:
        if source_split == "train":
            train_path = (
                corpus_root / domain / f"train_{tokenizer_name}.bin"
            )
            manifest_path = train_path.with_suffix(
                train_path.suffix + ".manifest.json"
            )
            manifest = json.loads(manifest_path.read_text())
            tokens = np.memmap(
                train_path,
                dtype=np.dtype(manifest.get("dtype", "uint16")),
                mode="r",
            )
        elif source_split in {"valid", "test"}:
            tokens = (
                torch.load(
                    corpus_root
                    / domain
                    / f"{source_split}_{tokenizer_name}.pt",
                    map_location="cpu",
                )
                .flatten()
                .numpy()
            )
        else:
            raise ValueError(f"unsupported source split: {source_split}")
        starts = np.load(index_root / domain / "starts.npy", mmap_mode="r")
        ends = np.load(index_root / domain / "ends.npy", mmap_mode="r")
        splits = np.load(index_root / domain / "splits.npy", mmap_mode="r")
        train_documents = np.flatnonzero(splits == SPLIT_TRAIN).astype(
            np.uint32,
            copy=False,
        )
        train_lengths = (
            ends[train_documents] - starts[train_documents]
        ).astype(np.float64)
        output.append(
            _DomainDocuments(
                domain,
                tokens,
                starts,
                ends,
                splits,
                train_documents,
                np.cumsum(train_lengths),
            )
        )
    return output


def _domain_probabilities(documents: list[_DomainDocuments]) -> np.ndarray:
    totals = np.asarray(
        [
            max(
                1,
                int(
                    (
                        (domain.ends - domain.starts)
                        * (domain.splits == SPLIT_TRAIN)
                    ).sum()
                ),
            )
            for domain in documents
        ],
        dtype=np.float64,
    )
    root = np.sqrt(totals)
    # Exact 5% floor for seven domains; distribute the remaining mass by sqrt.
    floor = min(0.05, 0.99 / len(documents))
    return floor + (1.0 - floor * len(documents)) * root / root.sum()


def _save_manifest_arrays(
    output: Path,
    length: int,
    row_ptr: list[int],
    span_domain: list[int],
    span_document: list[int],
    span_offset: list[int],
    span_length: list[int],
) -> None:
    np.save(output / f"length_{length}_row_ptr.npy", np.asarray(row_ptr, dtype=np.uint64))
    np.save(output / f"length_{length}_span_domain.npy", np.asarray(span_domain, dtype=np.uint8))
    np.save(output / f"length_{length}_span_document.npy", np.asarray(span_document, dtype=np.uint32))
    np.save(output / f"length_{length}_span_offset.npy", np.asarray(span_offset, dtype=np.uint32))
    np.save(output / f"length_{length}_span_length.npy", np.asarray(span_length, dtype=np.uint16))


def _save_sentence_arrays(
    output: Path,
    length: int,
    row_ptr: list[int],
    span_domain: list[int],
    span_document: list[int],
    span_offset: list[int],
    span_length: list[int],
    documents: list[_DomainDocuments],
    detector: SentenceBoundaryDetector,
) -> None:
    rows = len(row_ptr) - 1
    sentence_ids = np.full((rows, length), -1, dtype=np.int32)
    sentence_end = np.zeros((rows, length), dtype=np.bool_)
    forced = np.zeros((rows, length), dtype=np.bool_)
    for row in range(rows):
        tokens: list[int] = []
        segments: list[int] = []
        begin, end = row_ptr[row], row_ptr[row + 1]
        for segment, span_index in enumerate(range(begin, end)):
            domain = documents[span_domain[span_index]]
            document = span_document[span_index]
            offset = span_offset[span_index]
            take = span_length[span_index]
            start = int(domain.starts[document]) + offset
            values = np.asarray(
                domain.tokens[start : start + take],
                dtype=np.int64,
            ).tolist()
            tokens.extend(int(value) for value in values)
            segments.extend([segment] * len(values))
        tokens = tokens[:length]
        segments = segments[:length]
        ids, ends, forced_ends = detector.scan_tokens(tokens, segments)
        real = len(tokens)
        sentence_ids[row, :real] = ids.numpy()
        sentence_end[row, :real] = ends.numpy()
        forced[row, :real] = forced_ends.numpy()
    np.save(output / f"length_{length}_sentence_ids.npy", sentence_ids)
    np.save(output / f"length_{length}_sentence_end_mask.npy", sentence_end)
    np.save(output / f"length_{length}_forced_boundary_mask.npy", forced)


def build_paired_training_manifest(
    corpus_root: str | Path,
    index_root: str | Path,
    output_root: str | Path,
    *,
    tokenizer_name: str,
    rows_by_length: dict[int, int],
    seed: int,
    domains: Iterable[str],
    max_sentence_tokens: int = 128,
) -> dict:
    """Create deterministic packed rows shared by all compared models."""
    corpus_path = Path(corpus_root).expanduser()
    index_path = Path(index_root).expanduser()
    output = Path(output_root).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    domain_names = tuple(domains)
    documents = _load_documents(
        corpus_path,
        index_path,
        tokenizer_name,
        domain_names,
    )
    probabilities = _domain_probabilities(documents)
    tokenizer = EduCombinedCorpus(
        root=str(corpus_path),
        tokenizer_name=tokenizer_name,
        domains=list(domain_names),
        load_tokenizer=True,
    ).tokenizer
    detector = SentenceBoundaryDetector(
        tokenizer,
        max_sentence_tokens=max_sentence_tokens,
    )
    rng = np.random.default_rng(int(seed))
    metadata = {
        "format": MANIFEST_FORMAT,
        "kind": "training",
        "seed": int(seed),
        "tokenizer_name": tokenizer_name,
        "corpus_root": str(corpus_path),
        "index_root": str(index_path),
        "domains": list(domain_names),
        "domain_probabilities": probabilities.tolist(),
        "boundary_detector_version": detector.version,
        "boundary_detector_hash": detector.fingerprint,
        "max_sentence_tokens": int(max_sentence_tokens),
        "rows_by_length": {
            str(int(length)): int(rows)
            for length, rows in rows_by_length.items()
        },
    }
    for length, rows in sorted(rows_by_length.items()):
        row_ptr = [0]
        span_domain: list[int] = []
        span_document: list[int] = []
        span_offset: list[int] = []
        span_length: list[int] = []
        for _ in range(int(rows)):
            remaining = int(length)
            while remaining > 0:
                domain_index = int(rng.choice(len(documents), p=probabilities))
                domain = documents[domain_index]
                if not len(domain.train_documents):
                    raise RuntimeError(f"domain {domain.name} has no train documents")
                # Length-proportional document sampling without constructing a
                # corpus-wide token table. The CDF is precomputed once because
                # rebuilding it for every packed span is prohibitive at scale.
                chosen = int(
                    domain.train_documents[
                        np.searchsorted(
                            domain.train_token_cdf,
                            rng.random() * domain.train_token_cdf[-1],
                            side="right",
                        )
                    ]
                )
                doc_length = int(domain.ends[chosen] - domain.starts[chosen])
                offset = int(rng.integers(0, max(1, doc_length - 1)))
                take = min(remaining, doc_length - offset)
                if take < 2 and remaining > 1:
                    continue
                span_domain.append(domain_index)
                span_document.append(chosen)
                span_offset.append(offset)
                span_length.append(take)
                remaining -= take
            row_ptr.append(len(span_domain))
        _save_manifest_arrays(
            output,
            int(length),
            row_ptr,
            span_domain,
            span_document,
            span_offset,
            span_length,
        )
        _save_sentence_arrays(
            output,
            int(length),
            row_ptr,
            span_domain,
            span_document,
            span_offset,
            span_length,
            documents,
            detector,
        )
    (output / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    return metadata


def build_exhaustive_evaluation_manifest(
    corpus_root: str | Path,
    index_root: str | Path,
    output_root: str | Path,
    *,
    tokenizer_name: str,
    seq_len: int,
    domains: Iterable[str],
    split: str = "sealed",
    max_sentence_tokens: int = 128,
) -> dict:
    """Create one-pass rows whose supervised targets cover each document once."""
    if split not in {"sealed", "valid", "test"}:
        raise ValueError("split must be sealed, valid, or test")
    corpus_path = Path(corpus_root).expanduser()
    index_path = Path(index_root).expanduser()
    output = Path(output_root).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    domain_names = tuple(domains)
    source_split = "train" if split == "sealed" else split
    source_index = (
        index_path
        if split == "sealed"
        else index_path / f"official_{split}"
    )
    documents = _load_documents(
        corpus_path,
        source_index,
        tokenizer_name,
        domain_names,
        source_split=source_split,
    )
    tokenizer = EduCombinedCorpus(
        root=str(corpus_path),
        tokenizer_name=tokenizer_name,
        domains=list(domain_names),
        load_tokenizer=True,
    ).tokenizer
    detector = SentenceBoundaryDetector(
        tokenizer,
        max_sentence_tokens=max_sentence_tokens,
    )
    row_ptr = [0]
    span_domain: list[int] = []
    span_document: list[int] = []
    span_offset: list[int] = []
    span_length: list[int] = []
    for domain_index, domain in enumerate(documents):
        for document_index in np.flatnonzero(
            domain.splits == SPLIT_SEALED
        ).tolist():
            length = int(
                domain.ends[document_index] - domain.starts[document_index]
            )
            offset = 0
            while offset < length - 1:
                take = min(int(seq_len), length - offset)
                span_domain.append(domain_index)
                span_document.append(document_index)
                span_offset.append(offset)
                span_length.append(take)
                row_ptr.append(len(span_domain))
                if offset + take >= length:
                    break
                offset += take - 1  # one-token context overlap, no target overlap
    _save_manifest_arrays(
        output,
        int(seq_len),
        row_ptr,
        span_domain,
        span_document,
        span_offset,
        span_length,
    )
    _save_sentence_arrays(
        output,
        int(seq_len),
        row_ptr,
        span_domain,
        span_document,
        span_offset,
        span_length,
        documents,
        detector,
    )
    metadata = {
        "format": MANIFEST_FORMAT,
        "kind": "evaluation",
        "split": split,
        "source_split": source_split,
        "tokenizer_name": tokenizer_name,
        "corpus_root": str(corpus_path),
        "index_root": str(source_index),
        "domains": list(domain_names),
        "boundary_detector_version": detector.version,
        "boundary_detector_hash": detector.fingerprint,
        "max_sentence_tokens": int(max_sentence_tokens),
        "rows_by_length": {str(int(seq_len)): len(row_ptr) - 1},
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    return metadata


@CORPORA.register("paired_document_manifest")
class PairedDocumentManifestCorpus:
    """Disk-backed corpus serving immutable packed rows from a manifest."""

    def __init__(
        self,
        manifest_root: str,
        seed: int = 0,
        wrap: bool = False,
    ) -> None:
        self.root = Path(manifest_root).expanduser()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != MANIFEST_FORMAT:
            raise ValueError("unsupported paired manifest format")
        self.seed = int(seed)
        self.wrap = bool(wrap)
        self.declared_corpus_root = Path(self.manifest["corpus_root"])
        self.declared_index_root = Path(self.manifest["index_root"])
        self.corpus_root = _resolve_manifest_artifact_path(
            self.declared_corpus_root
        )
        self.index_root = _resolve_manifest_artifact_path(
            self.declared_index_root
        )
        self.source_split = str(
            self.manifest.get("source_split", "train")
        )
        self.tokenizer_name = str(self.manifest["tokenizer_name"])
        self.domains = tuple(self.manifest["domains"])
        self._documents = _load_documents(
            self.corpus_root,
            self.index_root,
            self.tokenizer_name,
            self.domains,
            source_split=self.source_split,
        )
        loader = EduCombinedCorpus(
            root=str(self.corpus_root),
            tokenizer_name=self.tokenizer_name,
            domains=list(self.domains),
            seed=seed,
            load_tokenizer=True,
        )
        self.tokenizer = loader.tokenizer
        self.vocab_size = loader.vocab_size
        special = getattr(self.tokenizer, "special_to_id", {})
        self.pad_id = int(special.get("<|pad|>", 0))
        self._arrays: dict[int, dict[str, np.ndarray]] = {}
        self._cursor = {
            int(length): 0
            for length in self.manifest["rows_by_length"]
        }

    def _load_arrays(self, seq_len: int) -> dict[str, np.ndarray]:
        if seq_len not in self._arrays:
            prefix = self.root / f"length_{seq_len}"
            self._arrays[seq_len] = {
                name: np.load(
                    f"{prefix}_{name}.npy",
                    mmap_mode="r",
                )
                for name in (
                    "row_ptr",
                    "span_domain",
                    "span_document",
                    "span_offset",
                    "span_length",
                    "sentence_ids",
                    "sentence_end_mask",
                    "forced_boundary_mask",
                )
            }
        return self._arrays[seq_len]

    def _row(
        self,
        arrays: dict[str, np.ndarray],
        row_index: int,
        seq_len: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        begin = int(arrays["row_ptr"][row_index])
        end = int(arrays["row_ptr"][row_index + 1])
        tokens: list[torch.Tensor] = []
        segments: list[torch.Tensor] = []
        domain_ids: list[torch.Tensor] = []
        for segment, span_index in enumerate(range(begin, end)):
            domain_index = int(arrays["span_domain"][span_index])
            domain = self._documents[domain_index]
            document = int(arrays["span_document"][span_index])
            offset = int(arrays["span_offset"][span_index])
            length = int(arrays["span_length"][span_index])
            start = int(domain.starts[document]) + offset
            array = np.asarray(
                domain.tokens[start : start + length],
                dtype=np.int64,
            ).copy()
            tokens.append(torch.from_numpy(array).long())
            segments.append(torch.full((length,), segment, dtype=torch.long))
            domain_ids.append(
                torch.full((length,), domain_index, dtype=torch.long)
            )
        row = torch.cat(tokens)[:seq_len]
        segment_ids = torch.cat(segments)[:seq_len]
        token_domains = torch.cat(domain_ids)[:seq_len]
        real = row.numel()
        if real < seq_len:
            row = F_pad(row, seq_len - real, self.pad_id)
            segment_ids = F_pad(segment_ids, seq_len - real, -1)
            token_domains = F_pad(token_domains, seq_len - real, -1)
        attention = torch.arange(seq_len) < real
        loss = attention.clone()
        loss[0] = False
        loss[1:] &= segment_ids[1:] == segment_ids[:-1]
        sentence_ids = torch.from_numpy(
            np.asarray(arrays["sentence_ids"][row_index], dtype=np.int64).copy()
        )
        sentence_end = torch.from_numpy(
            np.asarray(
                arrays["sentence_end_mask"][row_index],
                dtype=np.bool_,
            ).copy()
        )
        forced = torch.from_numpy(
            np.asarray(
                arrays["forced_boundary_mask"][row_index],
                dtype=np.bool_,
            ).copy()
        )
        sentence_end &= attention
        forced &= attention
        return (
            row,
            attention,
            loss,
            segment_ids,
            token_domains,
            sentence_ids,
            sentence_end,
            forced,
        )

    def sample_batch(
        self,
        batch: int,
        seq_len: int,
        split: str = "train",
    ) -> TrainingBatch:
        del split
        arrays = self._load_arrays(int(seq_len))
        rows = len(arrays["row_ptr"]) - 1
        cursor = self._cursor[int(seq_len)]
        indices = []
        for _ in range(int(batch)):
            if cursor >= rows:
                if not self.wrap:
                    raise StopIteration(
                        f"paired manifest exhausted at length {seq_len}"
                    )
                cursor = 0
            indices.append(cursor)
            cursor += 1
        self._cursor[int(seq_len)] = cursor
        return self.batch_from_indices(indices, seq_len)

    def batch_from_indices(
        self,
        indices: list[int] | tuple[int, ...],
        seq_len: int,
    ) -> TrainingBatch:
        """Materialize deterministic manifest rows without moving the sampler."""
        seq_len = int(seq_len)
        arrays = self._load_arrays(seq_len)
        rows = len(arrays["row_ptr"]) - 1
        indices = [int(index) for index in indices]
        if not indices:
            raise ValueError("indices cannot be empty")
        if any(index < 0 or index >= rows for index in indices):
            raise IndexError("manifest row index is out of range")
        packed = [
            self._row(arrays, index, seq_len)
            for index in indices
        ]
        (
            tokens,
            attention,
            loss,
            segment_ids,
            token_domains,
            sentence_ids,
            sentence_end,
            forced,
        ) = (
            torch.stack(values) for values in zip(*packed)
        )
        return TrainingBatch(
            tokens,
            attention,
            loss,
            metadata={
                "segment_ids": segment_ids,
                "token_domain_ids": token_domains,
                "sentence_ids": sentence_ids,
                "sentence_end_mask": sentence_end,
                "forced_boundary_mask": forced,
                "manifest_row_ids": torch.tensor(indices, dtype=torch.long),
            },
        )

    def sample_tokenized(
        self,
        batch: int,
        seq_len: int,
        split: str = "train",
    ) -> torch.Tensor:
        return self.sample_batch(batch, seq_len, split).tokens

    def sampler_state(self) -> dict:
        return {"cursor": dict(self._cursor)}

    def load_sampler_state(self, state: dict) -> None:
        self._cursor = {
            int(length): int(cursor)
            for length, cursor in state["cursor"].items()
        }

    def diagnostics(self) -> dict:
        return {
            "type": "paired_document_manifest",
            "manifest_root": str(self.root),
            "manifest_format": self.manifest["format"],
            "rows_by_length": self.manifest["rows_by_length"],
            "domains": list(self.domains),
            "source_split": self.source_split,
            "boundary_detector_version": self.manifest[
                "boundary_detector_version"
            ],
            "boundary_detector_hash": self.manifest[
                "boundary_detector_hash"
            ],
            "seed": self.seed,
        }


def F_pad(tensor: torch.Tensor, amount: int, value: int) -> torch.Tensor:
    if amount <= 0:
        return tensor
    return torch.cat(
        [
            tensor,
            torch.full(
                (amount,),
                value,
                dtype=tensor.dtype,
            ),
        ]
    )
