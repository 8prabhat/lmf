"""Corpora: synthetic, in-memory text, WikiText-103, and memory-mapped.

Each corpus exposes ``sample_tokenized(batch, seq_len, split) -> LongTensor`` and
(optionally) ``sample_batch(...) -> TrainingBatch``. They self-register in the
``CORPORA`` registry so the CLI/trainer can build them by name from config.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..core.registry import CORPORA
from .batch import TrainingBatch, lm_batch
from .tokenizers import MultiGearTokenizer, SpecialTokenTokenizer, build_bpe_tokenizer


def tokenizer_fingerprint(tokenizer) -> str:
    """Stable content hash of a tokenizer (vocab/merges/special tokens)."""
    parts: list[bytes] = [type(tokenizer).__name__.encode()]
    base = getattr(tokenizer, "base", None)
    if base is not None:
        parts.append(tokenizer_fingerprint(base).encode())
        parts.append(repr(getattr(tokenizer, "special_tokens", None)).encode())
    else:
        hf = getattr(tokenizer, "_tok", None)
        if hf is not None and hasattr(hf, "to_str"):
            parts.append(hf.to_str(pretty=False).encode())
        else:
            sp_model = getattr(tokenizer, "_model_proto", None)
            if sp_model is not None:
                parts.append(bytes(sp_model))
            merges = getattr(tokenizer, "_merges", None)
            vocab = getattr(tokenizer, "_vocab", None)
            if merges is not None and vocab is not None:
                parts.append(repr(merges).encode())
                parts.append(repr(sorted(vocab.items())).encode())
                # SPT encoding also depends on its boundary model and settings.
                # Hashing only vocab/merges can accept a checkpoint whose
                # tokenizer produces a different token stream.
                for attr in (
                    "threshold",
                    "phase_merges",
                    "unicode_safe",
                    "pretokenize",
                    "_effective_pretokenize",
                    "boundary_unit",
                    "grapheme_vocab_fraction",
                    "balance_texts",
                    "max_token_bytes",
                    "_bigram_surprisal",
                    "_unigram_surprisal",
                    "_grapheme_bigram_surprisal",
                    "_grapheme_unigram_surprisal",
                    "_unknown_grapheme_surprisal",
                    "_surprisal_mean",
                    "_surprisal_std",
                    "inference",
                    "gear_spans",
                    "gear_fractions",
                    "transition_weight",
                    "unigram_iterations",
                    "chunk_bytes",
                    "_merge_gears",
                    "_token_gears",
                    "_token_costs",
                    "_transition_costs",
                ):
                    if hasattr(tokenizer, attr):
                        value = getattr(tokenizer, attr)
                        if isinstance(value, dict):
                            value = sorted(value.items())
                        parts.append(f"{attr}={value!r}".encode())
            else:
                parts.append(str(getattr(tokenizer, "vocab_size", "")).encode())
    return hashlib.sha256(b"|".join(parts)).hexdigest()[:16]


def corpus_fingerprint(corpus) -> str:
    payload = corpus.diagnostics() if hasattr(corpus, "diagnostics") else {
        "type": type(corpus).__name__, "vocab_size": getattr(corpus, "vocab_size", None)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


@CORPORA.register("procedural")
class ProceduralCorpus:
    """Deterministic synthetic language with local + long-range structure.

    Tokens follow a first-order Markov chain over content tokens with periodic
    "echo" positions that copy a token from a fixed distance back — giving the
    model both local n-gram structure and a genuine long-range dependency for the
    exact-recall tail to learn. Cheap, dependency-free, reproducible: ideal for
    smoke training and falsification kernels.
    """

    def __init__(self, vocab_size: int = 512, seed: int = 0, echo_distance: int = 24,
                 echo_every: int = 8) -> None:
        self.vocab_size = int(vocab_size)
        self.echo_distance = int(echo_distance)
        self.echo_every = int(echo_every)
        self.seed = int(seed)
        # A fixed transition matrix defines the "language"; derived from a separate
        # generator so it does not consume the per-split sampling streams.
        trans_g = torch.Generator().manual_seed(self.seed)
        logits = torch.randn(self.vocab_size, self.vocab_size, generator=trans_g) * 2.0
        self._trans = torch.softmax(logits, dim=-1)
        # Persistent per-split generators. Each call ADVANCES the generator so every
        # batch is fresh, while a fixed seed per split keeps runs reproducible and
        # the splits disjoint (review finding 2 — the old code rebuilt the generator
        # each call and returned an identical batch every time).
        self._gens = {
            split: torch.Generator().manual_seed(self.seed + 1000 * off)
            for split, off in {"train": 1, "valid": 2, "test": 3}.items()
        }

    def _gen(self, split: str) -> torch.Generator:
        return self._gens["valid" if split == "eval" else split]

    def sample_tokenized(self, batch: int, seq_len: int, split: str = "train") -> torch.Tensor:
        g = self._gen(split)
        out = torch.zeros(batch, seq_len, dtype=torch.long)
        out[:, 0] = torch.randint(0, self.vocab_size, (batch,), generator=g)
        for t in range(1, seq_len):
            if t >= self.echo_distance and t % self.echo_every == 0:
                out[:, t] = out[:, t - self.echo_distance]  # long-range echo
            else:
                probs = self._trans[out[:, t - 1]]
                out[:, t] = torch.multinomial(probs, 1, generator=g).squeeze(-1)
        return out

    def sample_batch(self, batch: int, seq_len: int, split: str = "train") -> TrainingBatch:
        return lm_batch(self.sample_tokenized(batch, seq_len, split))

    def sampler_state(self) -> dict:
        return {split: g.get_state() for split, g in self._gens.items()}

    def load_sampler_state(self, state: dict) -> None:
        for split, s in state.items():
            if split in self._gens:
                self._gens[split].set_state(s.cpu() if hasattr(s, "cpu") else s)

    def diagnostics(self) -> dict:
        return {"type": "procedural", "vocab_size": self.vocab_size,
                "echo_distance": self.echo_distance, "echo_every": self.echo_every,
                "seed": self.seed}


@CORPORA.register("text")
class InMemoryTextCorpus:
    """BPE-tokenize an in-memory string; serve random windows by slicing.

    The text is split into disjoint train/valid/test **character** regions FIRST,
    the tokenizer is trained on the train region ONLY, and each region is encoded
    separately (review finding 5). This removes both the BPE-on-eval-text leak and
    the previous bug where ``valid`` and ``test`` aliased the same slice.
    """

    def __init__(self, text: str, tokenizer=None, max_vocab: int = 8192,
                 train_frac: float = 0.85, valid_frac: float = 0.075, seed: int = 0,
                 wrap_special: bool = True) -> None:
        n = len(text)
        train_cut = int(n * train_frac)
        valid_cut = int(n * (train_frac + valid_frac))
        train_text, valid_text, test_text = text[:train_cut], text[train_cut:valid_cut], text[valid_cut:]
        if tokenizer is None:
            tokenizer = build_bpe_tokenizer(max_vocab)
            tokenizer.train([train_text])          # train split only — no eval leakage
        if wrap_special and not isinstance(tokenizer, SpecialTokenTokenizer):
            tokenizer = SpecialTokenTokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.seed = int(seed)
        enc = lambda s: torch.tensor(tokenizer.encode(s), dtype=torch.long)  # noqa: E731
        train_ids, valid_ids, test_ids = enc(train_text), enc(valid_text), enc(test_text)
        self._splits = {"train": train_ids, "valid": valid_ids, "test": test_ids, "eval": valid_ids}
        # Persistent per-split generators (matching ProceduralCorpus): sampling
        # from "valid"/"test" during training must not perturb the "train"
        # random stream, or training batch order becomes dependent on the eval
        # schedule.
        self._gens = {
            split: torch.Generator().manual_seed(int(seed) + 1000 * off)
            for split, off in {"train": 1, "valid": 2, "test": 3}.items()
        }

    def _gen(self, split: str) -> torch.Generator:
        return self._gens["valid" if split == "eval" else split]

    def sample_tokenized(self, batch: int, seq_len: int, split: str = "train") -> torch.Tensor:
        ids = self._splits[split]
        if seq_len >= len(ids):
            raise ValueError(f"seq_len={seq_len} >= {split} length={len(ids)}")
        g = self._gen(split)
        starts = torch.randint(0, len(ids) - seq_len, (batch,), generator=g)
        return torch.stack([ids[s:s + seq_len] for s in starts])

    def sample_batch(self, batch: int, seq_len: int, split: str = "train") -> TrainingBatch:
        return lm_batch(self.sample_tokenized(batch, seq_len, split))

    def decode_text(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids.tolist())

    def sampler_state(self) -> dict:
        return {split: g.get_state() for split, g in self._gens.items()}

    def load_sampler_state(self, state: dict) -> None:
        for split, s in state.items():
            if split in self._gens:
                self._gens[split].set_state(s.cpu() if hasattr(s, "cpu") else s)

    def diagnostics(self) -> dict:
        return {"type": "text", "tokenizer_fingerprint": tokenizer_fingerprint(self.tokenizer),
                "train_tokens": int(len(self._splits["train"])),
                "valid_tokens": int(len(self._splits["valid"])),
                "test_tokens": int(len(self._splits["test"])),
                "seed": self.seed}


@CORPORA.register("multigear_text")
class MultiGearTextCorpus(InMemoryTextCorpus):
    """In-memory/file text corpus trained with MultiGear on the train split only."""

    def __init__(
        self,
        text: str | None = None,
        text_file: str | None = None,
        max_vocab: int = 8192,
        tokenizer_kwargs: dict | None = None,
        train_frac: float = 0.85,
        valid_frac: float = 0.075,
        seed: int = 0,
    ) -> None:
        if (text is None) == (text_file is None):
            raise ValueError("provide exactly one of text or text_file")
        if text_file is not None:
            text = Path(text_file).read_text(encoding="utf-8", errors="replace")
        assert text is not None
        train_cut = int(len(text) * train_frac)
        base = MultiGearTokenizer(max_vocab=max_vocab, **(tokenizer_kwargs or {}))
        base.train([text[:train_cut]])
        tokenizer = SpecialTokenTokenizer(base)
        super().__init__(
            text,
            tokenizer=tokenizer,
            train_frac=train_frac,
            valid_frac=valid_frac,
            seed=seed,
            wrap_special=False,
        )


class WikiTextCorpus:
    """WikiText-103 with official train/valid/test splits (BPE on train only)."""

    def __init__(self, tokenizer, train_ids, valid_ids, test_ids, seed: int = 0) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.seed = int(seed)
        self._splits = {"train": train_ids, "valid": valid_ids, "test": test_ids, "eval": valid_ids}
        # Persistent per-split generators (matching ProceduralCorpus): sampling
        # from "valid"/"test" during training must not perturb the "train"
        # random stream, or training batch order becomes dependent on the eval
        # schedule.
        self._gens = {
            split: torch.Generator().manual_seed(int(seed) + 1000 * off)
            for split, off in {"train": 1, "valid": 2, "test": 3}.items()
        }

    def _gen(self, split: str) -> torch.Generator:
        return self._gens["valid" if split == "eval" else split]

    def sample_tokenized(self, batch: int, seq_len: int, split: str = "train") -> torch.Tensor:
        ids = self._splits[split]
        if seq_len >= len(ids):
            raise ValueError(f"seq_len={seq_len} >= {split} length={len(ids)}")
        g = self._gen(split)
        starts = torch.randint(0, len(ids) - seq_len, (batch,), generator=g)
        return torch.stack([ids[s:s + seq_len] for s in starts])

    def sample_batch(self, batch: int, seq_len: int, split: str = "train") -> TrainingBatch:
        return lm_batch(self.sample_tokenized(batch, seq_len, split))

    def sampler_state(self) -> dict:
        return {split: g.get_state() for split, g in self._gens.items()}

    def load_sampler_state(self, state: dict) -> None:
        for split, s in state.items():
            if split in self._gens:
                self._gens[split].set_state(s.cpu() if hasattr(s, "cpu") else s)

    def diagnostics(self) -> dict:
        return {"type": "wikitext103",
                "tokenizer_fingerprint": tokenizer_fingerprint(self.tokenizer),
                "train_tokens": int(len(self._splits["train"])),
                "seed": self.seed}


def build_wikitext103(cache_dir: str = "data/wikitext103", bpe_vocab: int = 32768,
                      max_train_chars: int | None = None, seed: int = 0) -> WikiTextCorpus:
    """Download WikiText-103, train BPE on the train split, return a corpus."""
    from datasets import load_dataset

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)

    def _join(split):
        return "\n".join(t for t in split["text"] if t.strip())

    train_text = _join(ds["train"])
    if max_train_chars is not None:
        train_text = train_text[:max_train_chars]
    valid_text, test_text = _join(ds["validation"]), _join(ds["test"])
    tok = build_bpe_tokenizer(bpe_vocab)
    tok.train([train_text])
    tok = SpecialTokenTokenizer(tok)
    enc = lambda s: torch.tensor(tok.encode(s), dtype=torch.long)  # noqa: E731
    return WikiTextCorpus(tok, enc(train_text), enc(valid_text), enc(test_text), seed=seed)


class NumericFallbackTokenizer:
    """Best-effort tokenizer for pre-tokenized corpora without source tokenizer code.

    Text prompts are encoded as UTF-8 byte ids so generation remains usable even
    when the external pickled tokenizer cannot be imported. Generated token ids
    above 255 decode as explicit ``<id:N>`` markers.
    """

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = int(vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        stripped = text.strip()
        if stripped.startswith("ids:"):
            ids = [int(part) for part in stripped[4:].replace(",", " ").split()]
            if any(token_id < 0 or token_id >= self._vocab_size for token_id in ids):
                raise ValueError("ids: prompt contains token outside tokenizer vocabulary")
            return ids
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: list[int]) -> str:
        chunks: list[str] = []
        byte_run = bytearray()
        for token_id in ids:
            value = int(token_id)
            if 0 <= value <= 255:
                byte_run.append(value)
                continue
            if byte_run:
                chunks.append(byte_run.decode("utf-8", errors="replace"))
                byte_run.clear()
            chunks.append(f"<id:{value}>")
        if byte_run:
            chunks.append(byte_run.decode("utf-8", errors="replace"))
        return "".join(chunks)


def _maybe_add_quanthelion_to_path(root: Path) -> None:
    """Make the sibling Quanthelion package importable when it is present."""

    candidates = []
    env = None
    try:
        import os
        env = os.environ.get("RHCA_QUANTHELION_ROOT")
    except Exception:  # pragma: no cover - environment access is effectively infallible
        env = None
    if env:
        candidates.append(Path(env).expanduser())
    for parent in root.resolve().parents:
        candidates.append(parent / "Quanthelion")
    for candidate in candidates:
        if (candidate / "quanthelion" / "__init__.py").exists():
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
            return


def _load_optional_tokenizer(root: Path, name: str, vocab_size: int):
    path = root / f"shared_tokenizer_{name}.pt"
    if not path.exists():
        return NumericFallbackTokenizer(vocab_size)
    _maybe_add_quanthelion_to_path(root)
    try:
        tokenizer = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
            return tokenizer
    except Exception:
        return NumericFallbackTokenizer(vocab_size)
    return NumericFallbackTokenizer(vocab_size)


@dataclass
class _TokenShard:
    name: str
    tokens: object
    length: int
    path: Path
    manifest: dict


@CORPORA.register("edu_combined")
class EduCombinedCorpus:
    """Disk-backed corpus for the pre-tokenized edu_combined dataset.

    Train splits are sampled directly from uint16 ``.bin`` files via numpy
    memmap, so the 100GB+ corpus is not loaded into RAM. Valid/test ``.pt``
    tensors are loaded lazily per selected domain.
    """

    def __init__(
        self,
        root: str,
        tokenizer_name: str = "bpe32768_v2",
        domains: list[str] | None = None,
        seed: int = 0,
        sample_weights: str = "tokens",
        eval_max_tokens_per_domain: int | None = None,
        max_train_tokens_per_domain: int | None = None,
        load_tokenizer: bool = True,
        vocab_size: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.tokenizer_name = tokenizer_name
        self.seed = int(seed)
        self.sample_weights = sample_weights
        self.eval_max_tokens_per_domain = (
            None if eval_max_tokens_per_domain is None else int(eval_max_tokens_per_domain)
        )
        self.max_train_tokens_per_domain = (
            None if max_train_tokens_per_domain is None else int(max_train_tokens_per_domain)
        )
        if sample_weights not in {"tokens", "uniform"}:
            raise ValueError("sample_weights must be 'tokens' or 'uniform'")
        if not self.root.exists():
            raise FileNotFoundError(f"edu_combined root does not exist: {self.root}")
        selected = set(domains) if domains is not None else None
        self.train_shards = self._discover_train_shards(selected)
        if not self.train_shards:
            raise FileNotFoundError(
                f"no train_*{tokenizer_name}*.bin shards found under {self.root}"
            )
        self.vocab_size = max(int(shard.manifest.get("vocab_size", 0)) for shard in self.train_shards)
        if self.vocab_size <= 0:
            self.vocab_size = 32781
        self.tokenizer = (
            _load_optional_tokenizer(self.root, tokenizer_name, self.vocab_size)
            if load_tokenizer
            else NumericFallbackTokenizer(self.vocab_size)
        )
        self._splits: dict[str, list[_TokenShard]] = {
            "train": self.train_shards,
            "valid": self._load_eval_shards("valid", selected),
            "test": self._load_eval_shards("test", selected),
        }
        self._splits["eval"] = self._splits["valid"]
        if not self._splits["valid"]:
            self._splits["valid"] = self.train_shards
            self._splits["eval"] = self.train_shards
        if not self._splits["test"]:
            self._splits["test"] = self._splits["valid"]
        self._weights = {
            split: self._weights_for(shards)
            for split, shards in self._splits.items()
            if split != "eval"
        }
        self._weights["eval"] = self._weights["valid"]
        self._gens = {
            split: torch.Generator().manual_seed(self.seed + 1000 * offset)
            for split, offset in {"train": 1, "valid": 2, "test": 3}.items()
        }

    def _manifest_for(self, train_path: Path) -> dict:
        manifest_path = train_path.with_suffix(train_path.suffix + ".manifest.json")
        if not manifest_path.exists():
            return {"dtype": "uint16", "vocab_size": 32781}
        return json.loads(manifest_path.read_text())

    def _discover_train_shards(self, selected: set[str] | None) -> list[_TokenShard]:
        shards = []
        for domain_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            if selected is not None and domain_dir.name not in selected:
                continue
            train_path = domain_dir / f"train_{self.tokenizer_name}.bin"
            if not train_path.exists():
                continue
            manifest = self._manifest_for(train_path)
            dtype_name = manifest.get("dtype", "uint16")
            dtype = np.dtype(dtype_name)
            mmap = np.memmap(train_path, dtype=dtype, mode="r")
            length = int(len(mmap))
            if self.max_train_tokens_per_domain is not None:
                length = min(length, self.max_train_tokens_per_domain)
            shards.append(_TokenShard(domain_dir.name, mmap, length, train_path, manifest))
        return shards

    def _load_eval_shards(self, split: str, selected: set[str] | None) -> list[_TokenShard]:
        shards = []
        for domain_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            if selected is not None and domain_dir.name not in selected:
                continue
            path = domain_dir / f"{split}_{self.tokenizer_name}.pt"
            if not path.exists():
                continue
            tensor = torch.load(path, map_location="cpu")
            if not torch.is_tensor(tensor):
                raise TypeError(f"{path} must contain a tensor, got {type(tensor).__name__}")
            tensor = tensor.to(torch.long).flatten()
            if self.eval_max_tokens_per_domain is not None:
                tensor = tensor[:self.eval_max_tokens_per_domain]
            shards.append(_TokenShard(domain_dir.name, tensor, int(tensor.numel()), path, {}))
        return shards

    def _weights_for(self, shards: list[_TokenShard]) -> torch.Tensor:
        if not shards:
            return torch.empty(0)
        if self.sample_weights == "uniform":
            weights = torch.ones(len(shards), dtype=torch.float)
        else:
            weights = torch.tensor([max(1, shard.length) for shard in shards], dtype=torch.float)
        return weights / weights.sum()

    def _gen(self, split: str) -> torch.Generator:
        return self._gens["valid" if split == "eval" else split]

    @staticmethod
    def _slice(shard: _TokenShard, start: int, seq_len: int) -> torch.Tensor:
        data = shard.tokens
        if torch.is_tensor(data):
            return data[start:start + seq_len].to(torch.long)
        array = np.asarray(data[start:start + seq_len], dtype=np.int64).copy()
        return torch.from_numpy(array).to(torch.long)

    def sample_tokenized(self, batch: int, seq_len: int, split: str = "train") -> torch.Tensor:
        if batch < 1:
            raise ValueError("batch must be positive")
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        split = "valid" if split == "eval" else split
        shards = self._splits[split]
        weights = self._weights[split]
        if any(shard.length <= seq_len for shard in shards):
            bad = [shard.name for shard in shards if shard.length <= seq_len]
            raise ValueError(f"seq_len={seq_len} too large for shards: {bad}")
        g = self._gen(split)
        choices = torch.multinomial(weights, batch, replacement=True, generator=g)
        rows = []
        for choice in choices.tolist():
            shard = shards[int(choice)]
            start = int(torch.randint(0, shard.length - seq_len, (1,), generator=g).item())
            rows.append(self._slice(shard, start, seq_len))
        return torch.stack(rows)

    def sample_batch(self, batch: int, seq_len: int, split: str = "train") -> TrainingBatch:
        return lm_batch(self.sample_tokenized(batch, seq_len, split))

    def decode_text(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode([int(value) for value in ids.flatten().tolist()])

    def sampler_state(self) -> dict:
        return {split: generator.get_state() for split, generator in self._gens.items()}

    def load_sampler_state(self, state: dict) -> None:
        for split, value in state.items():
            if split in self._gens:
                self._gens[split].set_state(value.cpu() if hasattr(value, "cpu") else value)

    def diagnostics(self) -> dict:
        fingerprints = sorted(
            {
                str(shard.manifest.get("tokenizer_fingerprint", ""))
                for shard in self.train_shards
                if shard.manifest.get("tokenizer_fingerprint")
            }
        )
        return {
            "type": "edu_combined",
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_fingerprints": fingerprints,
            "vocab_size": self.vocab_size,
            "domains": [shard.name for shard in self.train_shards],
            "train_tokens": sum(shard.length for shard in self.train_shards),
            "valid_tokens": sum(shard.length for shard in self._splits["valid"]),
            "test_tokens": sum(shard.length for shard in self._splits["test"]),
            "sample_weights": self.sample_weights,
            "seed": self.seed,
        }


def build_corpus(spec: dict) -> object:
    """Build a corpus from a ``data`` config block.

    ``{"name": "procedural", "vocab_size": 512}`` or
    ``{"name": "text", "text": "..."}`` or ``{"name": "wikitext103", ...}``.
    """
    spec = dict(spec)
    name = spec.pop("name", "procedural")
    if name == "wikitext103":
        return build_wikitext103(**spec)
    return CORPORA.create(name, **spec)
