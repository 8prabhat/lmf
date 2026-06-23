"""Single CLI entrypoint: ``lmf {train,eval,generate,rfk}``.

The CLI is thin and family-agnostic. It reads a config block, builds the corpus,
model, and trainer purely by registry name, and dispatches. Adding a model family
requires no change here.
"""

from __future__ import annotations

import argparse
import json

import torch

from ..core.build import build as _build
from ..core.config import load_config

# Import families for their registration side effects.
from .. import models  # noqa: F401


def cmd_train(args) -> None:
    cfg = load_config(args.config, args.block, args.env, args.set)
    corpus, model, trainer, run = _build(cfg)
    steps = int(
        run.get("steps", 200) if args.steps is None else args.steps
    )
    print(f"[train] family={cfg.model.get('name', 'rhca')} block={cfg.block} "
          f"params={sum(p.numel() for p in model.parameters()):,} steps={steps}")
    trainer.train_steps(steps, int(run.get("batch_size", 8)), int(run.get("seq_len", 256)),
                        log_every=int(args.log_every))
    if args.checkpoint:
        trainer.save_checkpoint(args.checkpoint)
        print(f"[train] checkpoint -> {args.checkpoint}")


def cmd_eval(args) -> None:
    cfg = load_config(args.config, args.block, args.env, args.set)
    corpus, model, trainer, run = _build(cfg)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint, strict=not args.allow_mismatch)
    batch_size = int(run.get("batch_size", 8))
    seq_len = int(run.get("seq_len", 256))
    from ..evaluation import lm_metrics
    with trainer.frozen_sampling():
        metrics = lm_metrics(
            model, corpus, batch_size, seq_len,
            n_batches=int(args.n_batches), split=args.split)
    out = {"bits_per_token": round(metrics["bits_per_token"], 4), "block": cfg.block}
    if "bits_per_byte" in metrics:
        out["bits_per_byte"] = round(metrics["bits_per_byte"], 4)
        out["bytes_per_token"] = round(metrics["bytes_per_token"], 4)
        out["eval_tokens"] = int(metrics["eval_tokens"])
        out["eval_bytes"] = int(metrics["eval_bytes"])
    if hasattr(model, "prefill"):
        from ..evaluation.benchmarks import long_context_throughput, tokens_per_settle
        out["tokens_per_settle"] = tokens_per_settle(model)
        out["long_context"] = long_context_throughput(model, contexts=(256, 1024))
    print(json.dumps(out, indent=2))


def cmd_generate(args) -> None:
    cfg = load_config(args.config, args.block, args.env, args.set)
    corpus, model, trainer, run = _build(cfg)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint, strict=not args.allow_mismatch)
    device = next(model.parameters()).device
    if args.prompt and hasattr(corpus, "tokenizer"):
        ids = torch.tensor([corpus.tokenizer.encode(args.prompt)], device=device)
    else:
        ids = torch.randint(3, corpus.vocab_size, (1, 8), device=device)
    result = model.generate(ids, args.max_new_tokens)
    # RHCA returns a GenerationResult; the transformer baseline returns a tensor.
    token_ids = result.token_ids[0] if hasattr(result, "token_ids") else result[0]
    if hasattr(corpus, "decode_text"):
        text = corpus.decode_text(token_ids)
    elif hasattr(corpus, "tokenizer") and hasattr(corpus.tokenizer, "decode"):
        text = corpus.tokenizer.decode(token_ids.detach().cpu().tolist())
    else:
        text = None
    out = {"text": text, "tokens": token_ids.tolist()}
    if hasattr(result, "tokens_per_settle"):
        out["tokens_per_settle"] = round(result.tokens_per_settle, 3)
    print(json.dumps(out, indent=2))


def cmd_rfk(args) -> None:
    from ..experiments.rfk import run as run_rfk
    run_rfk(args.config, block=args.block, only=args.only)


def cmd_spt_bench(args) -> None:
    from ..experiments.spt_bench import run as run_spt_bench
    run_spt_bench(args.config, block=args.block)


def cmd_ablate(args) -> None:
    from ..ablation import load_ablation_spec, run_ablation
    spec = load_ablation_spec(args.config)
    result = run_ablation(spec, resume=not args.force, force=args.force,
                          workers=args.workers, only=args.only,
                          max_cells=args.max_cells, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
        return
    print(json.dumps(result, indent=2))


def cmd_ablate_status(args) -> None:
    from ..ablation import load_results
    results = load_results(args.results_dir)
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    print(json.dumps({"results_dir": args.results_dir, "n_results": len(results),
                       "by_status": by_status}, indent=2))


def cmd_ablate_report(args) -> None:
    from ..ablation import build_report, load_ablation_spec, maybe_plot, write_report
    spec = load_ablation_spec(args.spec) if args.spec else None
    report = build_report(args.results_dir, spec=spec)
    out = write_report(args.results_dir, report, fmt=args.format, out=args.out)
    print(f"[ablate-report] wrote {out}")
    if args.plot:
        plots = maybe_plot(args.results_dir, report, spec=spec)
        for p in plots:
            print(f"[ablate-report] wrote {p}")


def cmd_diagnose(args) -> None:
    cfg = load_config(args.config, args.block, args.env, args.set)
    corpus, model, trainer, run = _build(cfg)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint, strict=not args.allow_mismatch)
    from ..diagnostics import diagnose
    report = diagnose(model, corpus, batch_size=args.batch_size, seq_len=args.seq_len)
    print(json.dumps(report, indent=2, default=str))


def cmd_pretokenize_multigear(args) -> None:
    from ..data import materialize_multigear_dataset

    tokenizer_kwargs = {
        "inference": args.inference,
        "max_token_bytes": args.max_token_bytes,
        "transition_weight": args.transition_weight,
        "unigram_iterations": args.unigram_iterations,
        "chunk_bytes": args.chunk_bytes,
        "prune_fraction": args.prune_fraction,
    }
    report = materialize_multigear_dataset(
        args.source,
        args.output_root,
        tokenizer_name=args.tokenizer_name,
        vocab_size=args.vocab_size,
        tokenizer_kwargs=tokenizer_kwargs,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        dtype=args.dtype,
        force=args.force,
        jsonl_text_key=args.jsonl_text_key,
    )
    print(json.dumps(report, indent=2, default=str))


def cmd_pretokenize_sentencepiece_bpe(args) -> None:
    from ..data import materialize_sentencepiece_bpe_dataset

    report = materialize_sentencepiece_bpe_dataset(
        args.source,
        args.output_root,
        tokenizer_name=args.tokenizer_name,
        vocab_size=args.vocab_size,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        dtype=args.dtype,
        force=args.force,
        jsonl_text_key=args.jsonl_text_key,
    )
    print(json.dumps(report, indent=2, default=str))


def cmd_pretokenize_edu_multigear(args) -> None:
    from ..data import materialize_multigear_from_edu_combined

    tokenizer_kwargs = {
        "inference": args.inference,
        "max_token_bytes": args.max_token_bytes,
        "transition_weight": args.transition_weight,
        "unigram_iterations": args.unigram_iterations,
        "chunk_bytes": args.chunk_bytes,
        "prune_fraction": args.prune_fraction,
    }
    report = materialize_multigear_from_edu_combined(
        args.source_root,
        args.output_root,
        source_tokenizer_name=args.source_tokenizer_name,
        tokenizer_name=args.tokenizer_name,
        vocab_size=args.vocab_size,
        tokenizer_kwargs=tokenizer_kwargs,
        fraction=args.fraction,
        max_bpe_tokens_per_domain=args.max_bpe_tokens_per_domain,
        window_tokens=args.window_tokens,
        domains=args.domain,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        dtype=args.dtype,
        force=args.force,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, default=str))


def cmd_pretokenize_edu_multigear_prediction_aware(args) -> None:
    from ..data import materialize_prediction_aware_multigear_from_edu_combined

    tokenizer_kwargs = {
        "inference": "prediction_aware",
        "max_token_bytes": args.max_token_bytes,
        "transition_weight": args.transition_weight,
        "unigram_iterations": args.unigram_iterations,
        "chunk_bytes": args.chunk_bytes,
        "prune_fraction": args.prune_fraction,
        "prediction_alpha": args.prediction_alpha,
        "byte_reward": args.byte_reward,
        "gear0_penalty": args.gear0_penalty,
        "rare_threshold": args.rare_threshold,
        "rare_penalty": args.rare_penalty,
        "long_rare_penalty": args.long_rare_penalty,
        "unseen_penalty": args.unseen_penalty,
        "prediction_transition_weight": args.prediction_transition_weight,
    }
    report = materialize_prediction_aware_multigear_from_edu_combined(
        args.source_root,
        args.output_root,
        source_tokenizer_name=args.source_tokenizer_name,
        tokenizer_name=args.tokenizer_name,
        vocab_size=args.vocab_size,
        tokenizer_kwargs=tokenizer_kwargs,
        fraction=args.fraction,
        max_bpe_tokens_per_domain=args.max_bpe_tokens_per_domain,
        window_tokens=args.window_tokens,
        domains=args.domain,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        dtype=args.dtype,
        force=args.force,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, default=str))


def cmd_pretokenize_edu_sentencepiece_bpe(args) -> None:
    from ..data import materialize_sentencepiece_bpe_from_edu_combined

    report = materialize_sentencepiece_bpe_from_edu_combined(
        args.source_root,
        args.output_root,
        source_tokenizer_name=args.source_tokenizer_name,
        tokenizer_name=args.tokenizer_name,
        vocab_size=args.vocab_size,
        fraction=args.fraction,
        max_bpe_tokens_per_domain=args.max_bpe_tokens_per_domain,
        window_tokens=args.window_tokens,
        domains=args.domain,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        dtype=args.dtype,
        force=args.force,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lmf", description="Language Model Foundry")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", required=True)
        p.add_argument("--block", default=None)
        p.add_argument("--env", default=None)
        p.add_argument("--set", action="append", default=[], help="dotted.key=value override")

    pt = sub.add_parser("train"); common(pt)
    pt.add_argument("--steps", type=int, default=None)
    pt.add_argument("--log-every", type=int, default=25)
    pt.add_argument("--checkpoint", default=None)
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval"); common(pe)
    pe.add_argument("--checkpoint", default=None)
    pe.add_argument("--n-batches", type=int, default=10)
    pe.add_argument("--split", default="valid")
    pe.add_argument("--allow-mismatch", action="store_true",
                    help="load the checkpoint even if its architecture/corpus/tokenizer "
                         "fingerprints don't match the current config")
    pe.set_defaults(func=cmd_eval)

    pg = sub.add_parser("generate"); common(pg)
    pg.add_argument("--checkpoint", default=None)
    pg.add_argument("--prompt", default=None)
    pg.add_argument("--max-new-tokens", type=int, default=64)
    pg.add_argument("--allow-mismatch", action="store_true",
                    help="load the checkpoint even if its architecture/corpus/tokenizer "
                         "fingerprints don't match the current config")
    pg.set_defaults(func=cmd_generate)

    pr = sub.add_parser("rfk"); common(pr)
    pr.add_argument("--only", nargs="*", default=None)
    pr.set_defaults(func=cmd_rfk)

    psb = sub.add_parser("spt-bench"); common(psb)
    psb.set_defaults(func=cmd_spt_bench)

    pa = sub.add_parser("ablate")
    pa.add_argument("--config", required=True, help="path to an ablation spec YAML (not an experiment config)")
    pa.add_argument("--dry-run", action="store_true")
    pa.add_argument("--force", action="store_true")
    pa.add_argument("--workers", type=int, default=1)
    pa.add_argument("--only", nargs="*", default=None)
    pa.add_argument("--max-cells", type=int, default=None)
    pa.set_defaults(func=cmd_ablate)

    pas = sub.add_parser("ablate-status")
    pas.add_argument("--results-dir", required=True)
    pas.set_defaults(func=cmd_ablate_status)

    par = sub.add_parser("ablate-report")
    par.add_argument("--results-dir", required=True)
    par.add_argument("--spec", default=None, help="ablation spec YAML, for metric/mode-aware reports")
    par.add_argument("--format", choices=["md", "csv", "json"], default="md")
    par.add_argument("--out", default=None)
    par.add_argument("--plot", action="store_true")
    par.set_defaults(func=cmd_ablate_report)

    pd = sub.add_parser("diagnose"); common(pd)
    pd.add_argument("--checkpoint", default=None)
    pd.add_argument("--batch-size", type=int, default=4)
    pd.add_argument("--seq-len", type=int, default=64)
    pd.add_argument("--allow-mismatch", action="store_true",
                    help="load the checkpoint even if its architecture/corpus/tokenizer "
                         "fingerprints don't match the current config")
    pd.set_defaults(func=cmd_diagnose)

    ppm = sub.add_parser(
        "pretokenize-multigear",
        help="train MultiGear once and materialize tokenized splits for fast training",
    )
    ppm.add_argument(
        "--source",
        action="append",
        required=True,
        help="text file or directory; repeat to create multiple domains",
    )
    ppm.add_argument("--output-root", required=True)
    ppm.add_argument("--tokenizer-name", default="multigear32768_v1")
    ppm.add_argument("--vocab-size", type=int, default=32768)
    ppm.add_argument("--inference", choices=["bpe", "viterbi"], default="bpe")
    ppm.add_argument("--max-token-bytes", type=int, default=16)
    ppm.add_argument("--transition-weight", type=float, default=0.15)
    ppm.add_argument("--unigram-iterations", type=int, default=2)
    ppm.add_argument("--chunk-bytes", type=int, default=65536)
    ppm.add_argument("--prune-fraction", type=float, default=0.0)
    ppm.add_argument("--train-frac", type=float, default=0.85)
    ppm.add_argument("--valid-frac", type=float, default=0.075)
    ppm.add_argument("--dtype", default="auto", choices=["auto", "uint16", "uint32", "int32", "int64"])
    ppm.add_argument("--jsonl-text-key", default="text")
    ppm.add_argument("--force", action="store_true")
    ppm.set_defaults(func=cmd_pretokenize_multigear)

    pspt = sub.add_parser(
        "pretokenize-sentencepiece-bpe",
        help="train SentencePiece BPE once and materialize tokenized splits",
    )
    pspt.add_argument(
        "--source",
        action="append",
        required=True,
        help="text file or directory; repeat to create multiple domains",
    )
    pspt.add_argument("--output-root", required=True)
    pspt.add_argument("--tokenizer-name", default="sentencepiece_bpe32768_v1")
    pspt.add_argument("--vocab-size", type=int, default=32768)
    pspt.add_argument("--train-frac", type=float, default=0.85)
    pspt.add_argument("--valid-frac", type=float, default=0.075)
    pspt.add_argument("--dtype", default="auto", choices=["auto", "uint16", "uint32", "int32", "int64"])
    pspt.add_argument("--jsonl-text-key", default="text")
    pspt.add_argument("--force", action="store_true")
    pspt.set_defaults(func=cmd_pretokenize_sentencepiece_bpe)

    pem = sub.add_parser(
        "pretokenize-edu-multigear",
        help="sample edu_combined BPE shards, decode them, and materialize MultiGear ids",
    )
    pem.add_argument("--source-root", required=True)
    pem.add_argument("--output-root", required=True)
    pem.add_argument("--source-tokenizer-name", default="bpe32768_v2")
    pem.add_argument("--tokenizer-name", default="multigear_edu10pct_v1")
    pem.add_argument("--vocab-size", type=int, default=32768)
    pem.add_argument("--fraction", type=float, default=0.10)
    pem.add_argument(
        "--max-bpe-tokens-per-domain",
        type=int,
        default=None,
        help="safety cap for interactive runs; omit for the literal requested fraction",
    )
    pem.add_argument("--window-tokens", type=int, default=65536)
    pem.add_argument("--domain", action="append", default=None)
    pem.add_argument("--inference", choices=["bpe", "viterbi"], default="bpe")
    pem.add_argument("--max-token-bytes", type=int, default=16)
    pem.add_argument("--transition-weight", type=float, default=0.15)
    pem.add_argument("--unigram-iterations", type=int, default=2)
    pem.add_argument("--chunk-bytes", type=int, default=65536)
    pem.add_argument("--prune-fraction", type=float, default=0.0)
    pem.add_argument("--train-frac", type=float, default=0.85)
    pem.add_argument("--valid-frac", type=float, default=0.075)
    pem.add_argument("--dtype", default="auto", choices=["auto", "uint16", "uint32", "int32", "int64"])
    pem.add_argument("--seed", type=int, default=0)
    pem.add_argument("--force", action="store_true")
    pem.set_defaults(func=cmd_pretokenize_edu_multigear)

    pemp = sub.add_parser(
        "pretokenize-edu-multigear-prediction-aware",
        help="sample edu_combined BPE shards and materialize prediction-aware MultiGear ids",
    )
    pemp.add_argument("--source-root", required=True)
    pemp.add_argument("--output-root", required=True)
    pemp.add_argument("--source-tokenizer-name", default="bpe32768_v2")
    pemp.add_argument("--tokenizer-name", default="multigear_prediction_aware_edu10pct_v1")
    pemp.add_argument("--vocab-size", type=int, default=32768)
    pemp.add_argument("--fraction", type=float, default=0.10)
    pemp.add_argument(
        "--max-bpe-tokens-per-domain",
        type=int,
        default=None,
        help="safety cap for interactive runs; omit for the literal requested fraction",
    )
    pemp.add_argument("--window-tokens", type=int, default=65536)
    pemp.add_argument("--domain", action="append", default=None)
    pemp.add_argument("--max-token-bytes", type=int, default=16)
    pemp.add_argument("--transition-weight", type=float, default=0.15)
    pemp.add_argument("--unigram-iterations", type=int, default=2)
    pemp.add_argument("--chunk-bytes", type=int, default=65536)
    pemp.add_argument("--prune-fraction", type=float, default=0.0)
    pemp.add_argument("--prediction-alpha", type=float, default=0.25)
    pemp.add_argument("--byte-reward", type=float, default=0.28)
    pemp.add_argument("--gear0-penalty", type=float, default=0.70)
    pemp.add_argument("--rare-threshold", type=int, default=3)
    pemp.add_argument("--rare-penalty", type=float, default=0.45)
    pemp.add_argument("--long-rare-penalty", type=float, default=0.25)
    pemp.add_argument("--unseen-penalty", type=float, default=0.60)
    pemp.add_argument("--prediction-transition-weight", type=float, default=None)
    pemp.add_argument("--train-frac", type=float, default=0.85)
    pemp.add_argument("--valid-frac", type=float, default=0.075)
    pemp.add_argument("--dtype", default="auto", choices=["auto", "uint16", "uint32", "int32", "int64"])
    pemp.add_argument("--seed", type=int, default=0)
    pemp.add_argument("--force", action="store_true")
    pemp.set_defaults(func=cmd_pretokenize_edu_multigear_prediction_aware)

    pes = sub.add_parser(
        "pretokenize-edu-sentencepiece-bpe",
        help="sample edu_combined BPE shards, decode them, and materialize SentencePiece BPE ids",
    )
    pes.add_argument("--source-root", required=True)
    pes.add_argument("--output-root", required=True)
    pes.add_argument("--source-tokenizer-name", default="bpe32768_v2")
    pes.add_argument("--tokenizer-name", default="sentencepiece_bpe_edu_subset_v1")
    pes.add_argument("--vocab-size", type=int, default=32768)
    pes.add_argument("--fraction", type=float, default=0.10)
    pes.add_argument(
        "--max-bpe-tokens-per-domain",
        type=int,
        default=None,
        help="safety cap for interactive runs; omit for the literal requested fraction",
    )
    pes.add_argument("--window-tokens", type=int, default=65536)
    pes.add_argument("--domain", action="append", default=None)
    pes.add_argument("--train-frac", type=float, default=0.85)
    pes.add_argument("--valid-frac", type=float, default=0.075)
    pes.add_argument("--dtype", default="auto", choices=["auto", "uint16", "uint32", "int32", "int64"])
    pes.add_argument("--seed", type=int, default=0)
    pes.add_argument("--force", action="store_true")
    pes.set_defaults(func=cmd_pretokenize_edu_sentencepiece_bpe)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
