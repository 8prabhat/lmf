"""Comparable quality metrics for RHCA and the transformer baseline.

The two architectures must be scored on *identical* tasks for their bits-per-token
numbers to be comparable. The transformer measures causal teacher-forced
next-token prediction via one parallel forward. RHCA measures the same
chain-conditioned, ``max_commit``-token-window prediction it is trained on:
settle once per window, score the ``frontier_size``-wide chain-conditioned
window, commit ``max_commit`` tokens, and re-settle — the same state dynamics
used by ``carried_state_training_step`` and ``advance`` (generation).
"""

from __future__ import annotations

import inspect
import math
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from ..data.batch import TrainingBatch, lm_batch


def _sample_batch(corpus, batch_size, seq_len, split) -> TrainingBatch:
    if hasattr(corpus, "sample_batch"):
        return corpus.sample_batch(batch_size, seq_len, split)
    return lm_batch(corpus.sample_tokenized(batch_size, seq_len, split))


def _forward_language_model(model, batch: TrainingBatch):
    """Forward a causal LM while passing optional batch metadata it supports."""
    kwargs = {"attention_mask": batch.attention_mask}
    parameters = inspect.signature(model.forward).parameters
    for name in ("segment_ids", "sentence_end_mask"):
        if name in parameters and name in batch.metadata:
            kwargs[name] = batch.metadata[name]
    return model(batch.tokens, **kwargs)


def _decoded_byte_count(corpus, targets: torch.Tensor, valid: torch.Tensor) -> int | None:
    """Count UTF-8 bytes represented by valid target tokens.

    Token-level loss is not comparable across tokenizers with different average
    token lengths. For byte-normalized loss, decode contiguous valid target runs
    with the active tokenizer and count their UTF-8 bytes.
    """
    tokenizer = getattr(corpus, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return None
    total = 0
    target_rows = targets.detach().cpu().tolist()
    valid_rows = valid.detach().cpu().tolist()
    for row, mask in zip(target_rows, valid_rows):
        run: list[int] = []
        for token_id, is_valid in zip(row, mask):
            if is_valid:
                run.append(int(token_id))
            elif run:
                total += len(tokenizer.decode(run).encode("utf-8"))
                run = []
        if run:
            total += len(tokenizer.decode(run).encode("utf-8"))
    return total


@contextmanager
def _eval_mode(model):
    """Set eval mode for the duration, then restore the prior mode (finding 6)."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


@torch.no_grad()
def rhca_lm_metrics(model, corpus, batch_size=8, seq_len=256, n_batches=10,
                    split="valid") -> dict[str, float]:
    device = next(model.parameters()).device
    nll = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)
    byte_count = 0
    saw_byte_count = False
    h = model.config.frontier_size
    stride = model.config.max_commit
    with _eval_mode(model):
        for _ in range(n_batches):
            batch = _sample_batch(corpus, batch_size, seq_len, split).to(device)
            tokens = batch.tokens
            valid = batch.loss_mask & batch.attention_mask
            b, n = tokens.shape
            if n < h + stride:
                continue
            state = model.prefill(tokens[:, :h], batch.attention_mask[:, :h])
            pos = h
            while pos + stride <= n:
                frontier, _ = model.settle(state, active_only=True)
                # Score only the max_commit-wide block advance() actually
                # commits this cycle — matching calibrate_commit_threshold and
                # carried_state_training_step, not the full frontier_size draft.
                targets = tokens[:, pos:pos + stride]
                _, logits = model._chain_conditioned_fields_and_logits(frontier, targets)
                logits1 = logits[:, 0]
                per_tok = F.cross_entropy(
                    logits1.reshape(-1, model.config.vocab_size),
                    targets.reshape(-1), reduction="none").reshape(b, stride)
                m = valid[:, pos:pos + stride]
                nll += (per_tok * m).sum()
                count += m.sum()
                decoded_bytes = _decoded_byte_count(corpus, targets, m)
                if decoded_bytes is not None:
                    byte_count += decoded_bytes
                    saw_byte_count = True
                # Carry the SETTLED frontier forward and commit the same
                # max_commit-wide block, exactly as carried_state_training_step
                # / advance do (finding 4) — not a single-token re-settle loop.
                cmask = batch.attention_mask[:, pos:pos + stride]
                state = model._advance_state(state, frontier, targets, cmask)
                pos += stride
    bits = float(nll.item() / math.log(2.0))
    tokens = float(count.item())
    metrics = {
        "bits_per_token": bits / max(tokens, 1.0),
        "eval_tokens": tokens,
    }
    if saw_byte_count and byte_count > 0:
        metrics["bits_per_byte"] = bits / float(byte_count)
        metrics["bytes_per_token"] = float(byte_count) / max(tokens, 1.0)
        metrics["eval_bytes"] = float(byte_count)
    return metrics


@torch.no_grad()
def rhca_bits_per_token(model, corpus, batch_size=8, seq_len=256, n_batches=10,
                        split="valid") -> float:
    return rhca_lm_metrics(model, corpus, batch_size, seq_len, n_batches, split)[
        "bits_per_token"
    ]


@torch.no_grad()
def transformer_lm_metrics(model, corpus, batch_size=8, seq_len=256, n_batches=10,
                           split="valid") -> dict[str, float]:
    device = next(model.parameters()).device
    nll = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)
    byte_count = 0
    saw_byte_count = False
    with _eval_mode(model):
        for _ in range(n_batches):
            batch = _sample_batch(corpus, batch_size, seq_len, split).to(device)
            logits, _ = _forward_language_model(model, batch)
            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), batch.tokens[:, 1:].reshape(-1),
                reduction="none").reshape(batch.tokens.shape[0], -1)
            valid = batch.loss_mask[:, 1:] & batch.attention_mask[:, 1:]
            nll += (losses * valid).sum()
            count += valid.sum()
            decoded_bytes = _decoded_byte_count(corpus, batch.tokens[:, 1:], valid)
            if decoded_bytes is not None:
                byte_count += decoded_bytes
                saw_byte_count = True
    bits = float(nll.item() / math.log(2.0))
    tokens = float(count.item())
    metrics = {
        "bits_per_token": bits / max(tokens, 1.0),
        "eval_tokens": tokens,
    }
    if saw_byte_count and byte_count > 0:
        metrics["bits_per_byte"] = bits / float(byte_count)
        metrics["bytes_per_token"] = float(byte_count) / max(tokens, 1.0)
        metrics["eval_bytes"] = float(byte_count)
    return metrics


@torch.no_grad()
def transformer_bits_per_token(model, corpus, batch_size=8, seq_len=256, n_batches=10,
                               split="valid") -> float:
    return transformer_lm_metrics(model, corpus, batch_size, seq_len, n_batches, split)[
        "bits_per_token"
    ]


def bits_per_token(model, corpus, batch_size=8, seq_len=256, n_batches=10, split="valid") -> float:
    """Dispatch to the architecture-appropriate scorer."""
    if hasattr(model, "prefill"):
        return rhca_bits_per_token(model, corpus, batch_size, seq_len, n_batches, split)
    return transformer_bits_per_token(model, corpus, batch_size, seq_len, n_batches, split)


def lm_metrics(model, corpus, batch_size=8, seq_len=256, n_batches=10,
               split="valid") -> dict[str, float]:
    """Dispatch to architecture-appropriate token and byte-normalized metrics."""
    if hasattr(model, "prefill"):
        return rhca_lm_metrics(model, corpus, batch_size, seq_len, n_batches, split)
    return transformer_lm_metrics(model, corpus, batch_size, seq_len, n_batches, split)


def bits_per_byte(model, corpus, batch_size=8, seq_len=256, n_batches=10,
                  split="valid") -> float:
    """Return byte-normalized autoregressive loss.

    Raises when the corpus has no lossless decoder, because falling back to
    one byte per token would silently make tokenizer comparisons unreliable.
    """
    metrics = lm_metrics(model, corpus, batch_size, seq_len, n_batches, split)
    if "bits_per_byte" not in metrics:
        raise ValueError("bits_per_byte requires a corpus tokenizer with decode()")
    return metrics["bits_per_byte"]


@torch.no_grad()
def calibrate_commit_threshold(model, corpus, batch_size=8, seq_len=256, n_batches=5,
                               split="valid", precision_target: float = 0.85) -> dict[str, Any]:
    """Choose the largest entropy threshold whose commit precision >= target.

    Collects (normalized token entropy, correct) over teacher-forced draft blocks, then sweeps
    candidate thresholds and returns the largest one (most permissive, highest
    commit rate) that still keeps committed-token precision above the target.
    """
    device = next(model.parameters()).device
    h, c = model.config.frontier_size, model.config.max_commit
    entropies, corrects = [], []
    with _eval_mode(model):
        for _ in range(n_batches):
            batch = _sample_batch(corpus, batch_size, seq_len, split).to(device)
            tokens = batch.tokens
            n = tokens.shape[1]
            prefix = min(h, n - h)
            if prefix < 1:
                continue
            state = model.prefill(tokens[:, :prefix])
            pos = prefix
            while pos + h <= n:
                targets = tokens[:, pos:pos + h]
                frontier, _ = model.settle(state, active_only=True)
                _, logits = model._chain_conditioned_fields_and_logits(frontier, targets[:, :c])
                probs = logits[:, 0].softmax(dim=-1)
                entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1) / math.log(
                    model.config.vocab_size)
                pred = logits[:, 0].argmax(dim=-1)
                entropies.append(entropy.flatten())
                corrects.append((pred == targets[:, :c]).flatten())
                # Advance by max_commit, carrying the settled frontier (finding 1).
                state = model._advance_state(state, frontier, targets[:, :c])
                pos += c
    if not entropies:
        return {"error": "no data"}
    e = torch.cat(entropies).float()
    ok = torch.cat(corrects).float()
    order = e.argsort()
    e_sorted, ok_sorted = e[order], ok[order]
    # Prefix-cumulative precision: commit everything below each entropy candidate.
    cum_correct = ok_sorted.cumsum(0)
    cum_count = torch.arange(1, len(e) + 1, device=e.device).float()
    precision = cum_correct / cum_count
    feasible = precision >= precision_target
    chosen = None
    if bool(feasible.any()):
        idx = int(torch.nonzero(feasible).max())
        chosen = float(e_sorted[idx])
    return {
        "n_samples": len(e),
        "model_accuracy": round(float(ok.mean()), 4),
        "threshold_at_precision": chosen,
        "precision_target": precision_target,
        "commit_rate_at_threshold": (round(float((e < chosen).float().mean()), 4)
                                     if chosen is not None else 0.0),
    }


@torch.no_grad()
def repetition_rate(model, corpus, batch_size=4, prompt_len=16, gen_len=64,
                    n_batches=2, split="valid", ngram=4) -> float:
    from ..models.rhca.state import SamplingConfig
    device = next(model.parameters()).device
    cfg = SamplingConfig(deterministic=False, temperature=0.9, top_k=50, top_p=0.95,
                         repetition_penalty=1.1)
    total, seqs = 0.0, 0
    with _eval_mode(model):
        for _ in range(n_batches):
            batch = _sample_batch(corpus, batch_size, prompt_len + gen_len, split).to(device)
            result = model.generate(batch.tokens[:, :prompt_len], gen_len, cfg)
            if torch.is_tensor(result):
                generated = result
                lengths = torch.full(
                    (result.shape[0],),
                    result.shape[1],
                    dtype=torch.long,
                    device=result.device,
                )
            else:
                generated = result.token_ids
                lengths = result.generated_lengths
            for b in range(batch_size):
                gen = generated[b, :int(lengths[b])].tolist()
                if len(gen) < ngram:
                    continue
                grams = [tuple(gen[i:i + ngram]) for i in range(len(gen) - ngram + 1)]
                total += 1.0 - len(set(grams)) / max(1, len(grams))
                seqs += 1
    return round(total / max(1, seqs), 4)
