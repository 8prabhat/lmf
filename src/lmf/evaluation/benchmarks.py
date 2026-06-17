"""Structural benchmarks — the axes where the rolling-frontier design should win.

These operationalise the review's §7 win condition. They are the headline,
currently-unmeasured comparisons: RHCA's state is O(1) in context length while a
transformer's KV cache grows linearly, and RHCA commits several tokens per settle
where AR decoding emits one token per forward.

* ``long_context_throughput`` — prefill tokens/s and peak memory at growing context.
* ``tokens_per_settle`` — mean committed tokens per settle during generation.
* ``needle_in_tail`` — exact-recall accuracy for a token planted within tail_size.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from ..core.device import sync


def _peak_memory(device: torch.device, fn) -> int:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        fn()
        sync(device)
        return max(0, torch.cuda.max_memory_allocated(device) - before)
    if device.type == "mps":
        torch.mps.empty_cache()
        before = torch.mps.current_allocated_memory()
        fn()
        sync(device)
        return max(0, torch.mps.current_allocated_memory() - before)
    fn()
    return 0


@torch.no_grad()
def long_context_throughput(model, contexts=(512, 2048, 8192), batch_size=1,
                            vocab_size: int | None = None) -> dict[str, Any]:
    """Measure prefill/ingest tokens/s and peak memory at growing context length.

    ``prefill``/forward compute and memory are context-length-dependent for
    *every* architecture, RHCA included — that's exactly what ``results``
    measures. ``generation_state_size_constant`` is a separate, static
    architectural property: RHCA's carried ``GenerationState`` (memory/frontier/
    tail) has a fixed size set by config, independent of how much context was
    prefilled, whereas a transformer's KV cache grows with context. It is not a
    per-context measurement and must not be confused with the prefill numbers
    above.
    """
    device = next(model.parameters()).device
    vocab_size = vocab_size or model.config.vocab_size
    is_rhca = hasattr(model, "prefill")
    was_training = model.training
    model.eval()
    results = []
    try:
        for ctx in contexts:
            tokens = torch.randint(3, vocab_size, (batch_size, ctx), device=device)

            def _run():
                if is_rhca:
                    model.prefill(tokens)
                else:
                    model(tokens)

            sync(device)
            started = time.perf_counter()
            peak = _peak_memory(device, _run)
            sync(device)
            elapsed = time.perf_counter() - started
            results.append({
                "context": ctx,
                "tokens_per_second": round(batch_size * ctx / max(elapsed, 1e-9), 1),
                "peak_memory_mb": round(peak / 1e6, 2),
            })
    finally:
        model.train(was_training)
    return {"results": results, "generation_state_size_constant": is_rhca}


@torch.no_grad()
def tokens_per_settle(model, prompt_len=16, gen_len=64, batch_size=2,
                      vocab_size: int | None = None) -> dict[str, float]:
    """RHCA's answer to speculative decoding: mean committed tokens per settle."""
    if not hasattr(model, "generate") or not hasattr(model, "prefill"):
        return {"tokens_per_settle": 1.0, "note": "autoregressive baseline emits 1 token/forward"}
    device = next(model.parameters()).device
    vocab_size = vocab_size or model.config.vocab_size
    prompt = torch.randint(3, vocab_size, (batch_size, prompt_len), device=device)
    was_training = model.training
    model.eval()
    try:
        result = model.generate(prompt, gen_len)
    finally:
        model.train(was_training)
    return {"tokens_per_settle": round(result.tokens_per_settle, 3),
            "tokens_per_second": round(result.tokens_per_second, 1),
            "settles": result.cycles}


@torch.no_grad()
def needle_in_tail(model, distance: int, batch_size=8, vocab_size: int | None = None,
                   trials: int = 4) -> dict[str, float]:
    """Induction/copy probe with an explicit cue (review finding 8).

    Each sequence is ``[A, B, <random filler>, A]`` where the cue token ``A`` first
    appears at position 0 followed by ``B``, then appears again as the final token.
    A model with working in-context recall (through the exact-recall tail, or
    through memory once ``distance`` exceeds ``tail_size``) should predict ``B`` as
    the continuation. The first/second ``A`` are ``distance`` apart, so this
    measures recall at a controllable range. Chance level is ~1/vocab — the result
    is only meaningful for a trained model; on a fresh model expect ~chance.
    """
    if not hasattr(model, "prefill"):
        return {"note": "needle test applies to rolling-frontier models only"}
    device = next(model.parameters()).device
    vocab_size = vocab_size or model.config.vocab_size
    hits, total = 0, 0
    was_training = model.training
    model.eval()
    try:
        for t in range(trials):
            g = torch.Generator(device="cpu").manual_seed(1234 + t)
            a = torch.randint(3, vocab_size, (batch_size,), generator=g)
            b = torch.randint(3, vocab_size, (batch_size,), generator=g)
            filler = torch.randint(3, vocab_size, (batch_size, max(0, distance - 1)), generator=g)
            seq = torch.cat([a[:, None], b[:, None], filler, a[:, None]], dim=1).to(device)
            state = model.prefill(seq)                       # final token is the cue A
            frontier, _ = model.settle(state, active_only=True)
            pred = model.codebook.logits(frontier[:, 0, 0]).argmax(dim=-1)
            hits += int((pred == b.to(device)).sum())
            total += batch_size
    finally:
        model.train(was_training)
    return {"distance": distance, "recall_accuracy": round(hits / max(1, total), 4),
            "chance_level": round(1.0 / vocab_size, 6)}
