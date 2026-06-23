#!/usr/bin/env python3
"""Three-seed 200K-token screening for block-rate Bounded Hybrid Gear variants."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lmf.ablation.stats import analytic_confidence_interval
from lmf.core.hashing import file_sha256, git_tree_sha256, json_sha256
from lmf.core.seeding import seed_everything
from lmf.data import (
    ContiguousDocumentLaneCorpus,
    PairedDocumentManifestCorpus,
    tokenizer_fingerprint,
)
from lmf.models.bounded_hybrid_gear import (
    BlockHybridGearV4Config,
    BlockHybridGearV4LM,
    BoundedTransformerConfig,
    BoundedTransformerLM,
    PureParallelGearV3Trainer,
)
from lmf.models.transformer import CachedTransformerLM, TransformerConfig
from lmf.training.base_trainer import BaseTrainer
from lmf.training.checkpoints import architecture_fingerprint


DOMAINS = (
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
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("outputs/tokenizer/sentencepiece_bpe_prepared"),
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("outputs/pure_parallel_gear/360_proxy/index"),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("outputs/pure_parallel_gear/360_proxy/validation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bounded_hybrid_gear/screen_200k"),
    )
    parser.add_argument(
        "--qualification",
        type=Path,
        required=True,
        help="candidate-matched passed engineering qualification JSON",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="sentencepiece_bpe_edu_subset_v1",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--tokens", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--seeds", default="20262160,20262161,20262162")
    parser.add_argument("--max-validation-rows", type=int, default=176)
    parser.add_argument(
        "--candidate-name",
        choices=(
            "bounded_hybrid_gear_block_additive",
            "bounded_hybrid_gear_block_selective_film",
            "bounded_hybrid_gear_block_bank_router",
        ),
        default="bounded_hybrid_gear_block_additive",
    )
    parser.add_argument("--v4-ffn-dim", type=int, default=349)
    parser.add_argument("--fusion-rank", type=int, default=32)
    parser.add_argument("--attention-window", type=int, default=128)
    parser.add_argument("--block-tokens", type=int, default=128)
    return parser.parse_args()


def model_configs(vocab_size: int, args: argparse.Namespace):
    selective = args.candidate_name == "bounded_hybrid_gear_block_selective_film"
    bank_router = args.candidate_name == "bounded_hybrid_gear_block_bank_router"
    return {
        args.candidate_name: BlockHybridGearV4Config(
            vocab_size=vocab_size,
            dim=112,
            layers=2,
            ffn_dim=args.v4_ffn_dim,
            heads=7,
            kv_heads=1,
            attention_window=args.attention_window,
            block_tokens=args.block_tokens,
            cell_dim=12,
            bank_rank=12,
            fusion_mode=(
                "bank_router"
                if bank_router
                else ("selective_film" if selective else "additive")
            ),
            fusion_rank=args.fusion_rank,
        ),
        "bounded_transformer": BoundedTransformerConfig(
            vocab_size=vocab_size,
            dim=112,
            layers=2,
            ffn_dim=381,
            heads=7,
            kv_heads=1,
            attention_window=args.attention_window,
        ),
        "full_transformer": TransformerConfig(
            vocab_size=vocab_size,
            dim=112,
            layers=2,
            heads=7,
            max_seq_len=4096,
        ),
    }


def build_model(name: str, config):
    if name in {
        "bounded_hybrid_gear_block_additive",
        "bounded_hybrid_gear_block_selective_film",
        "bounded_hybrid_gear_block_bank_router",
    }:
        return BlockHybridGearV4LM(config)
    if name == "bounded_transformer":
        return BoundedTransformerLM(config)
    return CachedTransformerLM(config)


def build_trainer(name, model, corpus, args):
    common = {
        "device": args.device,
        "precision": "fp32",
        "lr": 1e-3,
        "warmup_tokens": max(1, args.tokens // 10),
        "total_training_tokens": args.tokens,
        "schedule_mode": "tokens",
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        "total_steps": 10_000,
    }
    if name in {
        "bounded_hybrid_gear_block_additive",
        "bounded_hybrid_gear_block_selective_film",
        "bounded_hybrid_gear_block_bank_router",
        "bounded_transformer",
    }:
        return PureParallelGearV3Trainer(
            model,
            corpus,
            stateful=True,
            tbptt_chunks=2,
            **common,
        )
    return BaseTrainer(
        model,
        corpus,
        grad_accum_steps=2,
        **common,
    )


@torch.no_grad()
def evaluate(model, manifest_root: Path, batch_size: int, max_rows: int):
    corpus = PairedDocumentManifestCorpus(str(manifest_root), wrap=False)
    lengths = sorted(int(value) for value in corpus.manifest["rows_by_length"])
    if len(lengths) != 1:
        raise ValueError("screening validation manifest must have one length")
    seq_len = lengths[0]
    available = int(corpus.manifest["rows_by_length"][str(seq_len)])
    rows = min(available, max_rows)
    selected_rows = [
        index * available // rows for index in range(rows)
    ]
    parameters = inspect.signature(model.forward).parameters
    domain_loss = {domain: [0.0, 0] for domain in corpus.domains}
    total_loss = 0.0
    total_targets = 0
    model.eval()
    for start in range(0, rows, batch_size):
        indices = selected_rows[start : start + batch_size]
        batch = corpus.batch_from_indices(indices, seq_len).to(
            next(model.parameters()).device
        )
        kwargs = {"attention_mask": batch.attention_mask}
        if "segment_ids" in parameters:
            kwargs["segment_ids"] = batch.metadata["segment_ids"]
        if "sentence_end_mask" in parameters:
            kwargs["sentence_end_mask"] = batch.metadata["sentence_end_mask"]
        logits, _ = model(batch.tokens, **kwargs)
        targets = batch.tokens[:, 1:]
        losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        valid = batch.loss_mask[:, 1:] & batch.attention_mask[:, 1:]
        total_loss += float(losses[valid].sum())
        total_targets += int(valid.sum())
        domains = batch.metadata["token_domain_ids"][:, 1:]
        for index, domain in enumerate(corpus.domains):
            selected = valid & (domains == index)
            domain_loss[domain][0] += float(losses[selected].sum())
            domain_loss[domain][1] += int(selected.sum())
    per_domain = {
        domain: loss / count
        for domain, (loss, count) in domain_loss.items()
        if count
    }
    return {
        "nll": total_loss / max(1, total_targets),
        "macro_domain_nll": statistics.fmean(per_domain.values()),
        "worst_domain_nll": max(per_domain.values()),
        "per_domain_nll": per_domain,
        "targets": total_targets,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    qualification = json.loads(args.qualification.read_text())
    current_code_hash = git_tree_sha256()
    qualified_code_hash = qualification.get("environment", {}).get(
        "code_hash"
    )
    if qualified_code_hash != current_code_hash:
        raise RuntimeError(
            "qualification was produced by different code "
            f"({qualified_code_hash} != {current_code_hash}); rerun it"
        )
    if not qualification.get("block_hybrid_gear_qualified", False):
        raise RuntimeError(
            "the supplied block-hybrid engineering qualification did not pass"
        )
    qualified_config = qualification["models"]["block_hybrid_gear"][
        "instantiated_config"
    ]
    requested_fusion = {
        "bounded_hybrid_gear_block_additive": "additive",
        "bounded_hybrid_gear_block_selective_film": "selective_film",
        "bounded_hybrid_gear_block_bank_router": "bank_router",
    }[args.candidate_name]
    expected = {
        "fusion_mode": requested_fusion,
        "fusion_rank": args.fusion_rank,
        "ffn_dim": args.v4_ffn_dim,
        "attention_window": args.attention_window,
        "block_tokens": args.block_tokens,
    }
    mismatches = {
        key: {
            "qualified": qualified_config.get(key),
            "requested": value,
        }
        for key, value in expected.items()
        if qualified_config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "qualification does not match the requested candidate: "
            f"{mismatches}"
        )
    seeds = tuple(int(value) for value in args.seeds.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe_corpus = ContiguousDocumentLaneCorpus(
        str(args.corpus_root),
        str(args.index_root),
        args.tokenizer_name,
        DOMAINS,
        seed=seeds[0],
        whole_windows=True,
    )
    configs = model_configs(probe_corpus.vocab_size, args)
    report = {
        "stage": "bounded_hybrid_gear_200k_screen",
        "environment": {
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": args.device,
            "precision": "fp32",
            "code_hash": current_code_hash,
        },
        "qualification": {
            "path": str(args.qualification),
            "sha256": file_sha256(args.qualification),
        },
        "data": {
            "corpus_root": str(args.corpus_root),
            "index_root": str(args.index_root),
            "index_hash": file_sha256(args.index_root / "index.json"),
            "validation_manifest": str(args.validation_manifest),
            "validation_manifest_hash": file_sha256(
                args.validation_manifest / "manifest.json"
            ),
            "tokenizer_name": args.tokenizer_name,
            "tokenizer_fingerprint": tokenizer_fingerprint(
                probe_corpus.tokenizer
            ),
            "domains": DOMAINS,
        },
        "tokens_requested": args.tokens,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "seeds": seeds,
        "configs": {
            name: config.to_dict() for name, config in configs.items()
        },
        "runs": {},
    }
    del probe_corpus
    for name, config in configs.items():
        report["runs"][name] = []
        for seed in seeds:
            seed_everything(seed)
            corpus = ContiguousDocumentLaneCorpus(
                str(args.corpus_root),
                str(args.index_root),
                args.tokenizer_name,
                DOMAINS,
                seed=seed,
                whole_windows=True,
            )
            model = build_model(name, config)
            trainer = build_trainer(name, model, corpus, args)
            tokens_per_step = (
                args.batch_size
                * (args.seq_len - 1)
                * 2
            )
            steps = math.ceil(args.tokens / tokens_per_step)
            started = time.perf_counter()
            trainer.train_steps(
                steps,
                args.batch_size,
                args.seq_len,
                log_every=0,
            )
            wall_seconds = time.perf_counter() - started
            validation = evaluate(
                trainer.raw_model,
                args.validation_manifest,
                args.batch_size,
                args.max_validation_rows,
            )
            checkpoint = (
                args.output_dir / "checkpoints" / f"{name}_seed_{seed}.pt"
            )
            trainer.save_checkpoint(checkpoint)
            manifest = trainer.raw_model.architecture_manifest()
            run = {
                "seed": seed,
                "parameters": sum(
                    parameter.numel()
                    for parameter in trainer.raw_model.parameters()
                ),
                "supervised_tokens": trainer.supervised_tokens_seen,
                "optimization_seconds": trainer.optimization_seconds,
                "wall_seconds": wall_seconds,
                "tokens_per_second": (
                    trainer.supervised_tokens_seen
                    / max(trainer.optimization_seconds, 1e-9)
                ),
                "validation": validation,
                "manifest": manifest,
                "manifest_hash": json_sha256(manifest),
                "architecture_fingerprint": architecture_fingerprint(
                    trainer.raw_model
                ),
                "checkpoint": str(checkpoint),
                "checkpoint_hash": file_sha256(checkpoint),
            }
            report["runs"][name].append(run)
            (args.output_dir / "results.partial.json").write_text(
                json.dumps(report, indent=2, sort_keys=True)
            )
            del trainer, model, corpus
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    summaries = {}
    for name, runs in report["runs"].items():
        summaries[name] = {
            "macro_nll": analytic_confidence_interval(
                [run["validation"]["macro_domain_nll"] for run in runs]
            ),
            "throughput": analytic_confidence_interval(
                [run["tokens_per_second"] for run in runs]
            ),
        }
    report["summaries"] = summaries
    baseline_speed = summaries["full_transformer"]["throughput"]["mean"]
    v4_nll = summaries[args.candidate_name]["macro_nll"]["mean"]
    report["checks"] = {
        "v4_throughput_at_least_half_transformer": (
            summaries[args.candidate_name]["throughput"]["mean"]
            / baseline_speed
            >= 0.5
        ),
        "v4_within_3pct_full_transformer": (
            v4_nll
            <= 1.03 * summaries["full_transformer"]["macro_nll"]["mean"]
        ),
        "v4_within_3pct_bounded_transformer": (
            v4_nll
            <= 1.03 * summaries["bounded_transformer"]["macro_nll"]["mean"]
        ),
        "parameter_match_within_half_percent": all(
            abs(
                run["parameters"]
                / report["runs"]["full_transformer"][0]["parameters"]
                - 1.0
            )
            <= 0.005
            for runs in report["runs"].values()
            for run in runs
        ),
    }
    report["passed"] = all(report["checks"].values())
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": report["checks"],
                "summaries": report["summaries"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
