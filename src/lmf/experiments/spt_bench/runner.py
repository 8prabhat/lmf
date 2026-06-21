"""Config-driven benchmark: SPT vs BPE at matched vocabulary size.

Compares the standalone, architecture-agnostic ``SurprisalPhaseTokenizer``
(``lmf.data.tokenizers``) against byte-level BPE on three axes, holding
vocabulary size, corpus, and downstream model/training budget fixed:

* ``compression_ratio_bytes_per_token`` — corpus bytes / token count.
* ``valid_bytes_per_token`` — held-out validation bytes / token count.
* ``train_seconds`` — tokenizer training wall-clock.
* ``downstream_bits_per_token`` — bits/token of a small transformer trained
  from scratch on each tokenizer's token stream.
* ``downstream_bits_per_byte`` — the comparable downstream metric across
  tokenizers with different segmentation lengths.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ...core.config import load_config
from ...core.registry import MODELS, TRAINERS
from ...core.seeding import seed_everything
from ...data import (
    InMemoryTextCorpus,
    MultiGearTokenizer,
    SurprisalPhaseTokenizer,
    build_bpe_tokenizer,
)

_SAMPLE_CORPUS = Path(__file__).parent / "sample_corpus.txt"


def _load_text(cfg: dict) -> str:
    data_cfg = cfg.get("data", {})
    if "text" in data_cfg:
        return str(data_cfg["text"])
    text_file = data_cfg.get("text_file")
    path = Path(text_file) if text_file else _SAMPLE_CORPUS
    return path.read_text(encoding="utf-8", errors="replace")


def _downstream_bpt(corpus, cfg: dict, steps: int) -> float:
    run = cfg.get("run", {})
    model_cfg = {k: v for k, v in cfg.get("model", {}).items() if k != "name"}
    model = MODELS.create("transformer", model_cfg, corpus.vocab_size)
    trainer_cfg = {k: v for k, v in cfg.get("trainer", {}).items() if k != "name"}
    batch_size = int(run.get("batch_size", 8))
    seq_len = int(run.get("seq_len", 64))
    trainer = TRAINERS.create(
        "transformer", model, corpus,
        device=cfg.get("device", "auto"), precision=cfg.get("precision", "bf16"),
        batch_size=batch_size, seq_len=seq_len, **trainer_cfg)
    trainer.train_steps(steps, batch_size, seq_len, log_every=0)
    return trainer.evaluate_bpt(batch_size, seq_len, n_batches=int(run.get("eval_batches", 10)),
                                split="valid")


def _bytes_per_token(tokenizer, text: str) -> float:
    return len(text.encode("utf-8")) / max(len(tokenizer.encode(text)), 1)


def run(config_path: str, block: str = "smoke", out_dir: str = "outputs/tokenizer/spt_bench") -> dict:
    cfg = load_config(config_path, block=block).raw
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    text = _load_text(cfg)
    vocab_size = int(cfg.get("vocab_size", 512))
    steps = int(cfg.get("run", {}).get("steps", 100))

    train_cut = int(len(text) * 0.85)
    valid_cut = int(len(text) * 0.925)
    train_text = text[:train_cut]
    valid_text = text[train_cut:valid_cut]
    test_text = text[valid_cut:]

    report: dict = {"config": config_path, "block": block, "vocab_size": vocab_size,
                    "corpus_bytes": len(text.encode("utf-8")), "tokenizers": {}}

    for name, tokenizer in (
        ("bpe", build_bpe_tokenizer(vocab_size)),
        ("multigear", MultiGearTokenizer(max_vocab=vocab_size)),
        ("multigear_viterbi", MultiGearTokenizer(max_vocab=vocab_size, inference="viterbi")),
        (
            "multigear_viterbi_pruned",
            MultiGearTokenizer(
                max_vocab=vocab_size,
                inference="viterbi",
                prune_fraction=0.25,
                transition_weight=0.05,
            ),
        ),
        ("spt", SurprisalPhaseTokenizer(max_vocab=vocab_size)),
    ):
        started = time.perf_counter()
        tokenizer.train([train_text])
        train_seconds = time.perf_counter() - started

        compression_ratio = _bytes_per_token(tokenizer, text)
        train_bytes_per_token = _bytes_per_token(tokenizer, train_text)
        valid_bytes_per_token = _bytes_per_token(tokenizer, valid_text)
        test_bytes_per_token = _bytes_per_token(tokenizer, test_text)

        seed_everything(seed)
        corpus = InMemoryTextCorpus(text, tokenizer=tokenizer, seed=seed)
        bpt = _downstream_bpt(corpus, cfg, steps)
        bits_per_byte = bpt / valid_bytes_per_token

        report["tokenizers"][name] = {
            "vocab_size": tokenizer.vocab_size,
            "train_seconds": round(train_seconds, 3),
            "compression_ratio_bytes_per_token": round(compression_ratio, 4),
            "train_bytes_per_token": round(train_bytes_per_token, 4),
            "valid_bytes_per_token": round(valid_bytes_per_token, 4),
            "test_bytes_per_token": round(test_bytes_per_token, 4),
            "estimated_train_bytes_seen": round(
                steps
                * int(cfg.get("run", {}).get("batch_size", 8))
                * int(cfg.get("run", {}).get("seq_len", 64))
                * train_bytes_per_token
            ),
            "downstream_bits_per_token": round(bpt, 4),
            "downstream_bits_per_byte": round(bits_per_byte, 4),
        }
        print(f"[spt_bench] {name}: vocab={tokenizer.vocab_size} "
              f"train={train_seconds:.3f}s valid_bytes/token={valid_bytes_per_token:.3f} "
              f"bpt={bpt:.4f} bits/byte={bits_per_byte:.4f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"spt_bench_{block}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2))
    report["out_file"] = str(path)
    print(f"[spt_bench] wrote {path}")
    return report
