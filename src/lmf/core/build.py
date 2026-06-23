"""Shared corpus/model/trainer construction.

The single place the CLI and the ablation runner both go through, so neither
can drift from registry-based construction.
"""

from __future__ import annotations

from typing import Any

from .config import ExperimentConfig
from .registry import MODELS, TRAINERS
from .seeding import seed_everything
from ..data.corpora import build_corpus


def special_token_ids(corpus) -> dict:
    """Map the corpus tokenizer's reserved tokens to RHCA's expected role names."""
    tok = getattr(corpus, "tokenizer", None)
    table = getattr(tok, "special_to_id", None)
    if not table:
        return {}
    role = {"eos": "<|eos|>", "pad": "<|pad|>", "bos": "<|bos|>"}
    return {name: table[t] for name, t in role.items() if t in table}


def initialize_token_embeddings(model, tokenizer, strategy: str | None) -> bool:
    """Apply an opt-in tokenizer-aware initialization after model construction."""
    if strategy in (None, "independent"):
        return False
    if strategy != "merge_compositional":
        raise ValueError(f"unknown token_embedding_init strategy: {strategy}")
    embedding = getattr(model, "token", None)
    weight = getattr(embedding, "weight", None)
    initializer = getattr(tokenizer, "initialize_embeddings_from_merges", None)
    if weight is None:
        raise TypeError(f"{type(model).__name__} has no token embedding weight")
    if initializer is None:
        raise TypeError(f"{type(tokenizer).__name__} has no merge-tree initializer")
    initializer(weight)
    return True


def configure_token_hierarchy(model, tokenizer) -> bool:
    """Provide tokenizer hierarchy metadata to models that can consume it."""
    configure = getattr(model, "configure_token_hierarchy", None)
    hierarchy = getattr(tokenizer, "token_hierarchy", None)
    if configure is None or hierarchy is None:
        return False
    if not hasattr(model, "_token_gears"):
        return False
    configure(**hierarchy())
    return True


def configure_boundary_detector(model, tokenizer) -> bool:
    """Install the corpus tokenizer for models with incremental boundaries."""
    configure = getattr(model, "configure_boundary_detector", None)
    if configure is None or tokenizer is None:
        return False
    configure(tokenizer)
    return True


def build(cfg: ExperimentConfig, *, run_overrides: dict[str, Any] | None = None):
    """Build (corpus, model, trainer, run) purely from registry names in cfg."""
    seed_everything(int(cfg.get("seed", 0)))
    corpus = build_corpus(cfg.data or {"name": "procedural", "vocab_size": 512})
    model_cfg = dict(cfg.model)
    model_name = model_cfg.pop("name", "rhca")
    declared_vocab_size = model_cfg.get("vocab_size")
    if (
        declared_vocab_size is not None
        and int(declared_vocab_size) != int(corpus.vocab_size)
    ):
        raise ValueError(
            f"model vocab_size={declared_vocab_size} does not match "
            f"corpus vocab_size={corpus.vocab_size}; silent vocabulary "
            "replacement is disabled"
        )
    token_embedding_init = model_cfg.pop("token_embedding_init", None)
    # Wire the corpus's reserved control-token IDs into the model so EOS-stopping
    # and pad handling actually work at generation time (review finding 7).
    if model_name == "rhca" and "special_token_ids" not in model_cfg:
        model_cfg["special_token_ids"] = special_token_ids(corpus)
    model = MODELS.create(model_name, model_cfg, corpus.vocab_size)
    tokenizer = getattr(corpus, "tokenizer", None)
    configure_token_hierarchy(model, tokenizer)
    configure_boundary_detector(model, tokenizer)
    if token_embedding_init is not None:
        initialize_token_embeddings(model, tokenizer, token_embedding_init)
    trainer_cfg = dict(cfg.trainer)
    trainer_name = trainer_cfg.pop("name", model_name)
    run = {**(cfg.get("run", {}) or {}), **(run_overrides or {})}
    batch_size = int(run.get("batch_size", 8))
    seq_len = int(run.get("seq_len", 256))
    trainer = TRAINERS.create(
        trainer_name, model, corpus,
        device=cfg.get("device", "auto"), precision=cfg.get("precision", "bf16"),
        batch_size=batch_size, seq_len=seq_len, **trainer_cfg)
    return corpus, model, trainer, run
