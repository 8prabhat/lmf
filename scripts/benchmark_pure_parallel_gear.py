#!/usr/bin/env python
"""Paired, parameter-matched Pure Parallel Gear training studies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lmf.ablation.stats import percentile
from lmf.core.device import sync
from lmf.core.hashing import file_sha256, git_tree_sha256
from lmf.core.io import atomic_write_json as write_json
from lmf.core.seeding import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
from lmf.data import PairedDocumentManifestCorpus, tokenizer_fingerprint
from lmf.diagnostics import cache_bytes, parameter_count as count_parameters
from lmf.models.gru import GRULM, GRULMConfig
from lmf.models.pure_parallel_gear import (
    PureParallelGearConfig,
    PureParallelGearLM,
    PureParallelGearTrainer,
)
from lmf.models.transformer import CachedTransformerLM, TransformerConfig
from lmf.models.transformer.trainer import TransformerTrainer
from lmf.training.base_trainer import BaseTrainer


MODEL_NAMES = ("transformer", "gru", "gear")
STAGES = {
    "proxy": {
        "seeds": 5,
        "tokens": 1_000_000,
        "effective": 8_192,
        "lrs": (3e-4, 1e-3, 3e-3),
        "transformer": (112, 3, 4),
    },
    "3m": {
        "seeds": 5,
        "tokens": 60_000_000,
        "effective": 32_768,
        "lrs": (3e-4, 1e-3, 3e-3),
        "transformer": (80, 3, 4),
    },
    "15m": {
        "seeds": 3,
        "tokens": 300_000_000,
        "effective": 65_536,
        "lrs": (1e-4, 3e-4, 1e-3),
        "transformer": (288, 6, 8),
    },
    "50m": {
        "seeds": 2,
        "tokens": 1_000_000_000,
        "effective": 131_072,
        "lrs": (6e-5, 2e-4, 6e-4),
        "transformer": (576, 8, 8),
    },
}


def _configured_parameter_count(model_type, config) -> int:
    with torch.device("meta"):
        return count_parameters(model_type(config))


def _gear_layer_parameter_count(
    config: PureParallelGearConfig,
    *,
    banks: int,
    gears: int,
    use_ffn: bool,
) -> int:
    dim = config.dim
    channels = config.rotor_channels
    state = banks * gears * channels
    feature_dim = (
        2 * state
        + state
        + 3 * state
        + 2 * banks * (gears - 1) * channels
        + (3 if config.use_load_state else 2) * state
        + 2 * max(banks - 1, 0) * gears * channels
    )
    count = (
        dim
        + 5 * dim * state
        + 3 * state
        + (state if config.learned_angular_velocity else 0)
        + (
            banks * (gears - 1) * channels * 8
            if config.boundary_settling
            else 0
        )
        + (
            banks * gears * channels * 8
            if config.boundary_settling
            and config.cross_bank_coupling
            and banks > 1
            else 0
        )
        + (
            max(config.settling_rounds, 1)
            * banks
            * (gears - 1)
            * channels
            if config.boundary_settling
            else 0
        )
        + (
            max(config.settling_rounds, 1) * banks * gears * channels
            if config.boundary_settling
            and config.cross_bank_coupling
            and banks > 1
            else 0
        )
        + (
            2 * state
            if config.boundary_settling and config.use_load_state
            else 0
        )
        + (
            2 * state
            if config.boundary_settling
            and config.learned_angular_velocity
            else 0
        )
        + (
            2 * banks
            if config.boundary_settling and config.adaptive_settling_depth
            else 0
        )
        + feature_dim
        + feature_dim * dim
        + 1
    )
    if use_ffn:
        ffn = int(config.ffn_dim)
        count += dim + 2 * ffn * dim + 2 * ffn + ffn * dim + 1
    return count


def _fast_weight_memory_parameter_count(config: PureParallelGearConfig) -> int:
    if not config.use_fast_weight_memory:
        return 0
    dim = config.dim
    bank_key = config.fast_weight_banks * config.fast_weight_key_dim
    bank_value = config.fast_weight_banks * config.fast_weight_value_dim
    return (
        bank_key * dim  # key_proj
        + bank_key * dim  # query_proj
        + config.fast_weight_value_dim * dim  # value_down_proj
        + dim * bank_value  # value_up_proj
        + (dim + bank_value) + 1  # gate_proj (weight + bias)
        + dim * bank_value  # memory_out_proj
        + 1  # memory_residual
        + (
            config.fast_weight_banks if config.unify_memory_consolidation else 0
        )  # consolidation_gate (Phase 3.5)
    )


def _future_heads_parameter_count(config: PureParallelGearConfig) -> int:
    """One bias-free Linear(gears_per_bank*rotor_channels*2 -> dim) per
    bank, backing the training-only future-rotor auxiliary loss (Phase
    3.1) -- never read at inference, see _inference_parameter_count."""
    return (
        config.num_banks
        * (config.gears_per_bank * config.rotor_channels * 2)
        * config.dim
    )


def gear_parameter_count(config: PureParallelGearConfig) -> int:
    return (
        config.vocab_size * config.dim
        + config.layers
        * _gear_layer_parameter_count(
            config,
            banks=config.num_banks,
            gears=config.gears_per_bank,
            use_ffn=config.use_local_swiglu,
        )
        + (
            _gear_layer_parameter_count(
                config,
                banks=1,
                gears=config.predictor_gears,
                use_ffn=False,
            )
            if config.use_predictor_gear
            else 0
        )
        + config.dim
        + _fast_weight_memory_parameter_count(config)
        + _future_heads_parameter_count(config)
    )


def _matched_gear(
    target: int,
    vocab_size: int,
    *,
    baseline_dim: int,
    layers: int,
    extra_config: dict | None = None,
) -> PureParallelGearConfig:
    extra_config = extra_config or {}
    # Matches on the *inference-relevant* parameter count (total minus the
    # training-only future_heads scaffold -- see _inference_parameter_count)
    # since `target` is the Transformer baseline's total, and the
    # Transformer has no equivalent training-only head to net against.
    # Matching on the raw total here would silently shrink dim/ffn_dim to
    # "pay for" future_heads, leaving the gear model under-resourced
    # relative to the Transformer at the inference time that actually
    # matters for a fair comparison.
    def inference_count(cfg: PureParallelGearConfig) -> int:
        return gear_parameter_count(cfg) - _future_heads_parameter_count(cfg)

    best: tuple[float, PureParallelGearConfig] | None = None
    for dim in range(max(16, int(0.55 * baseline_dim)), int(1.35 * baseline_dim) + 1):
        low = PureParallelGearConfig(
            vocab_size=vocab_size,
            dim=dim,
            layers=layers,
            ffn_dim=dim,
            **extra_config,
        )
        low_count = inference_count(low)
        high = replace(low, ffn_dim=dim + 1)
        slope = inference_count(high) - low_count
        if slope <= 0:
            continue
        estimate = max(dim, round(dim + (target - low_count) / slope))
        for ffn in range(max(dim, estimate - 3), estimate + 4):
            if ffn > 8 * dim:
                continue
            candidate = replace(low, ffn_dim=ffn)
            count = inference_count(candidate)
            error = abs(count / target - 1.0)
            score = (
                error,
                abs(dim / baseline_dim - 1.0),
                ffn / dim,
            )
            if best is None or score < best[0]:
                best = (score, candidate)
    if best is None or best[0][0] > 0.005:
        raise RuntimeError(
            f"unable to parameter-match Pure Gear within 0.5%; best={best}"
        )
    return best[1]


def _matched_gru(
    target: int,
    vocab_size: int,
    *,
    baseline_dim: int,
    layers: int,
) -> GRULMConfig:
    best: tuple[float, GRULMConfig] | None = None
    for dim in range(max(8, int(0.75 * baseline_dim)), int(1.25 * baseline_dim) + 1):
        for hidden in range(max(8, baseline_dim // 2), 2 * baseline_dim + 1):
            candidate = GRULMConfig(
                vocab_size=vocab_size,
                dim=dim,
                hidden_dim=hidden,
                layers=max(1, layers),
            )
            count = (
                vocab_size * dim
                + 4 * hidden * dim
                + (6 * layers - 3) * hidden * hidden
                + 6 * layers * hidden
            )
            error = abs(count / target - 1.0)
            if best is None or error < best[0]:
                best = (error, candidate)
    if best is None or best[0] > 0.005:
        raise RuntimeError(f"unable to parameter-match GRU within 0.5%; best={best}")
    return best[1]


def configs(stage: str, vocab_size: int) -> dict[str, Any]:
    dim, layers, heads = STAGES[stage]["transformer"]
    transformer = TransformerConfig(
        vocab_size=vocab_size,
        dim=dim,
        layers=layers,
        heads=heads,
        max_seq_len=4096,
    )
    target = _configured_parameter_count(CachedTransformerLM, transformer)
    return {
        "transformer": transformer,
        "gru": _matched_gru(
            target,
            vocab_size,
            baseline_dim=dim,
            layers=layers,
        ),
        "gear": _matched_gear(
            target,
            vocab_size,
            baseline_dim=dim,
            layers=layers,
        ),
    }


def build_model(name: str, config: Any, seed: int) -> torch.nn.Module:
    seed_everything(seed)
    if name == "transformer":
        return CachedTransformerLM(config)
    if name == "gru":
        return GRULM(config)
    return PureParallelGearLM(config)


def trainer_class(name: str):
    if name == "transformer":
        return TransformerTrainer
    if name == "gear":
        return PureParallelGearTrainer
    return BaseTrainer


def _inference_parameter_count(model: torch.nn.Module) -> int:
    """Parameter count restricted to what actually runs at inference.

    `PureParallelGearLM.future_heads` backs a training-only auxiliary loss
    (the future-rotor objective, Phase 3.1) that `_future_loss` only ever
    reads from `training_step` -- never from `forward()` or generation. The
    Transformer/GRU baselines being compared against here have no analogous
    training-only head, so counting future_heads against the gear model
    would penalize it for capacity it never uses to answer a query, making
    this fairness check stricter than what it is meant to guarantee
    (matched *inference-time* capacity).
    """
    total = count_parameters(model)
    future_heads = getattr(model, "future_heads", None)
    if future_heads is not None:
        total -= count_parameters(future_heads)
    return total


def assert_fair_configs(configuration: dict[str, Any]) -> dict[str, Any]:
    models = {
        name: build_model(name, config, seed=1)
        for name, config in configuration.items()
    }
    parameters = {
        name: _inference_parameter_count(model) for name, model in models.items()
    }
    baseline = parameters["transformer"]
    relative = {
        name: value / baseline - 1.0 for name, value in parameters.items()
    }
    if any(abs(value) > 0.005 for value in relative.values()):
        raise RuntimeError(f"parameter mismatch exceeds 0.5%: {relative}")
    manifest = models["gear"].architecture_manifest()
    forbidden = (
        "self_attention",
        "qkv_projections",
        "token_similarity",
        "history_retrieval",
        "history_tensor",
        "kv_cache",
        "token_routing",
        "transformer_blocks",
    )
    if any(manifest["invariants"][name] for name in forbidden):
        raise RuntimeError("Pure Gear architecture contract was violated")
    return {
        "parameters": parameters,
        "relative_to_transformer": relative,
        "manifests": {
            name: model.architecture_manifest()
            for name, model in models.items()
        },
        # Compatibility for existing report readers.
        "gear_manifest": manifest,
    }


def _model_kwargs(model, batch) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {
        "attention_mask": batch.attention_mask
    }
    if isinstance(model, (PureParallelGearLM, CachedTransformerLM, GRULM)):
        kwargs["segment_ids"] = batch.metadata["segment_ids"]
    if isinstance(model, PureParallelGearLM):
        kwargs.update(
            {
                "sentence_end_mask": batch.metadata["sentence_end_mask"],
            }
        )
    return kwargs


@torch.no_grad()
def evaluate_manifest(
    model: torch.nn.Module,
    manifest_root: Path,
    *,
    batch_size: int,
    device: str,
    max_rows: int | None = None,
) -> dict[str, Any]:
    corpus = PairedDocumentManifestCorpus(str(manifest_root), wrap=False)
    lengths = sorted(int(value) for value in corpus.manifest["rows_by_length"])
    if len(lengths) != 1:
        raise ValueError("evaluation manifest must contain one sequence length")
    seq_len = lengths[0]
    total_rows_available = int(
        corpus.manifest["rows_by_length"][str(seq_len)]
    )
    model.eval()
    total_loss = total_targets = top1 = top5 = byte_count = 0
    brier_sum = 0.0
    domain_loss = {domain: [0.0, 0] for domain in corpus.domains}
    bin_count = torch.zeros(15, dtype=torch.float64)
    bin_confidence = torch.zeros(15, dtype=torch.float64)
    bin_correct = torch.zeros(15, dtype=torch.float64)
    selected_rows = (
        list(range(total_rows_available))
        if max_rows is None or int(max_rows) >= total_rows_available
        else [
            index * total_rows_available // int(max_rows)
            for index in range(int(max_rows))
        ]
    )
    seen = 0
    while seen < len(selected_rows):
        row_indices = selected_rows[seen : seen + batch_size]
        batch = corpus.batch_from_indices(row_indices, seq_len).to(device)
        logits, _ = model(batch.tokens, **_model_kwargs(model, batch))
        logits, targets = logits[:, :-1], batch.tokens[:, 1:]
        valid = batch.loss_mask[:, 1:] & batch.attention_mask[:, 1:]
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        probabilities = logits.softmax(dim=-1)
        target_probability = probabilities.gather(
            -1, targets[..., None]
        ).squeeze(-1)
        brier = probabilities.square().sum(dim=-1) - 2.0 * target_probability + 1.0
        predictions = logits.argmax(dim=-1)
        total_loss += float(losses[valid].sum())
        brier_sum += float(brier[valid].sum())
        total_targets += int(valid.sum())
        top1 += int(((predictions == targets) & valid).sum())
        top5 += int(
            (
                (
                    logits.topk(min(5, logits.shape[-1]), dim=-1).indices
                    == targets[..., None]
                ).any(-1)
                & valid
            ).sum()
        )
        confidence = probabilities.max(dim=-1).values
        bins = (confidence * 15).long().clamp_max(14)
        for index in range(15):
            selected = valid & (bins == index)
            if bool(selected.any()):
                bin_count[index] += int(selected.sum())
                bin_confidence[index] += float(confidence[selected].sum())
                bin_correct[index] += int(
                    (predictions[selected] == targets[selected]).sum()
                )
        domains = batch.metadata["token_domain_ids"][:, 1:]
        for index, domain in enumerate(corpus.domains):
            selected = valid & (domains == index)
            domain_loss[domain][0] += float(losses[selected].sum())
            domain_loss[domain][1] += int(selected.sum())
        for target_row, valid_row in zip(targets.cpu(), valid.cpu()):
            run = []
            for token, selected in zip(target_row.tolist(), valid_row.tolist()):
                if selected:
                    run.append(int(token))
                elif run:
                    byte_count += len(corpus.tokenizer.decode(run).encode("utf-8"))
                    run = []
            if run:
                byte_count += len(corpus.tokenizer.decode(run).encode("utf-8"))
        seen += len(row_indices)
    per_domain = {
        domain: loss / count
        for domain, (loss, count) in domain_loss.items()
        if count
    }
    ece = 0.0
    for index in range(15):
        count = int(bin_count[index])
        if count:
            ece += count / total_targets * abs(
                float(bin_correct[index] / count)
                - float(bin_confidence[index] / count)
            )
    nll = total_loss / max(total_targets, 1)
    return {
        "nll": nll,
        "perplexity": math.exp(min(30.0, nll)),
        "bits_per_token": nll / math.log(2.0),
        "bits_per_byte": (
            total_loss / math.log(2.0) / byte_count if byte_count else None
        ),
        "top1_accuracy": top1 / max(total_targets, 1),
        "top5_accuracy": top5 / max(total_targets, 1),
        "ece_15_bin": ece,
        "brier_score": brier_sum / max(total_targets, 1),
        "macro_domain_nll": statistics.fmean(per_domain.values()),
        "worst_domain_nll": max(per_domain.values()),
        "per_domain_nll": per_domain,
        "per_domain_targets": {
            domain: count for domain, (_, count) in domain_loss.items()
        },
        "targets": total_targets,
        "rows": seen,
    }


def _cached_forward(model, tokens, cache=None, *, use_cache=True, boundary=None):
    if isinstance(model, CachedTransformerLM):
        return model(tokens, caches=cache, use_cache=use_cache)
    kwargs = {"cache": cache, "use_cache": use_cache}
    if isinstance(model, PureParallelGearLM):
        kwargs["sentence_end_mask"] = boundary
    return model(tokens, **kwargs)


@torch.no_grad()
def efficiency_samples(
    model: torch.nn.Module,
    *,
    vocab_size: int,
    seq_len: int,
    device: str,
    repeats: int = 5,
    incremental_steps: int = 64,
) -> dict[str, Any]:
    """Measure synchronized full-context prefill and cached token transitions."""
    model = model.to(device).eval()
    tokens = torch.randint(0, vocab_size, (1, seq_len), device=device)
    prompt_boundary = None
    trailing_sentence = None
    if isinstance(model, PureParallelGearLM):
        if model._boundary_detector is None:
            raise RuntimeError(
                "Pure Gear efficiency measurement requires a configured "
                "sentence-boundary detector"
            )
        prompt_values = tokens[0].detach().cpu().tolist()
        prompt_boundary = model._boundary_detector.scan_tokens(
            prompt_values, close_final=False
        )[1][None]
        trailing_sentence = model._boundary_detector.trailing_sentence(
            prompt_values
        )

    # Warm up every incremental cache shape without recording the sample.
    logits, cache = _cached_forward(
        model, tokens, boundary=prompt_boundary
    )
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    warm_sentence = (
        None if trailing_sentence is None else list(trailing_sentence)
    )
    for _step in range(incremental_steps):
        boundary = None
        if isinstance(model, PureParallelGearLM):
            assert warm_sentence is not None
            value = int(token[0, 0].item())
            warm_sentence.append(value)
            is_boundary = (
                model._boundary_detector.is_boundary_incremental(
                    warm_sentence
                )
            )
            boundary = torch.tensor([[is_boundary]], dtype=torch.bool)
            if is_boundary:
                warm_sentence = []
        logits, cache = _cached_forward(
            model, token, cache, boundary=boundary
        )
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
    sync(device)

    prefill_times: list[float] = []
    incremental_times: list[float] = []
    measured_cache_bytes = 0
    for _ in range(repeats):
        sync(device)
        started = time.perf_counter()
        logits, cache = _cached_forward(
            model, tokens, boundary=prompt_boundary
        )
        sync(device)
        prefill_times.append(time.perf_counter() - started)
        measured_cache_bytes = cache_bytes(cache)

        sentence = None if trailing_sentence is None else list(trailing_sentence)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        sync(device)
        started = time.perf_counter()
        for _step in range(incremental_steps):
            boundary = None
            if isinstance(model, PureParallelGearLM):
                assert sentence is not None
                value = int(token[0, 0].item())
                sentence.append(value)
                is_boundary = (
                    model._boundary_detector.is_boundary_incremental(sentence)
                )
                boundary = torch.tensor([[is_boundary]], dtype=torch.bool)
                if is_boundary:
                    sentence = []
            logits, cache = _cached_forward(
                model, token, cache, boundary=boundary
            )
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
        sync(device)
        incremental_times.append(time.perf_counter() - started)
    return {
        "prefill_times": prefill_times,
        "incremental_times": incremental_times,
        "cache_bytes": measured_cache_bytes,
        "incremental_steps": incremental_steps,
    }


def throughput(
    model: torch.nn.Module,
    *,
    vocab_size: int,
    seq_len: int,
    device: str,
    repeats: int = 5,
) -> dict[str, float]:
    samples = efficiency_samples(
        model,
        vocab_size=vocab_size,
        seq_len=seq_len,
        device=device,
        repeats=repeats,
    )
    prefill_seconds = statistics.median(samples["prefill_times"])
    generation_seconds = statistics.median(samples["incremental_times"])
    return {
        "prefill_tokens_per_second": seq_len / max(prefill_seconds, 1e-9),
        "incremental_tokens_per_second": samples["incremental_steps"]
        / max(generation_seconds, 1e-9),
        "prefill_p50_seconds": prefill_seconds,
        "prefill_p95_seconds": percentile(samples["prefill_times"], 0.95),
        "incremental_p50_seconds": generation_seconds,
        "incremental_p95_seconds": percentile(
            samples["incremental_times"], 0.95
        ),
        "cache_bytes": samples["cache_bytes"],
    }


def _context_length(progress: float) -> int:
    cumulative = 0.0
    for length, fraction in zip(
        (128, 256, 512, 1024, 2048, 4096),
        (0.10, 0.15, 0.20, 0.20, 0.20, 0.15),
    ):
        cumulative += fraction
        if progress < cumulative:
            return length
    return 4096


def _trainer(
    name: str,
    model: torch.nn.Module,
    corpus,
    *,
    lr: float,
    device: str,
    precision: str,
    total_tokens: int | None = None,
    total_seconds: float | None = None,
):
    if isinstance(model, PureParallelGearLM):
        tokenizer = getattr(corpus, "tokenizer", None)
        if tokenizer is None:
            raise TypeError(
                "Pure Gear training requires the corpus tokenizer so generation "
                "uses the same frozen sentence-boundary policy"
            )
        model.configure_boundary_detector(tokenizer)
    kwargs = {
        "device": device,
        "precision": precision,
        "lr": lr,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        "total_steps": 1_000_000_000,
        "warmup_steps": 1,
    }
    if total_seconds is not None:
        kwargs.update(
            schedule_mode="time",
            total_seconds=total_seconds,
            warmup_seconds=0.1 * total_seconds,
        )
    else:
        kwargs.update(
            schedule_mode="tokens",
            total_training_tokens=total_tokens,
            warmup_tokens=max(1, int(total_tokens or 1) // 10),
        )
    return trainer_class(name)(model, corpus, **kwargs)


def train_run(
    name: str,
    config: Any,
    manifest: Path,
    validation: Path,
    output: Path,
    *,
    seed: int,
    lr: float,
    device: str,
    precision: str,
    effective_tokens: int,
    micro_batch: int,
    total_tokens: int | None = None,
    total_seconds: float | None = None,
    max_validation_rows: int | None = None,
    eval_every_fraction: float = 0.1,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    corpus = PairedDocumentManifestCorpus(str(manifest), wrap=False)
    model = build_model(name, config, seed)
    trainer = _trainer(
        name,
        model,
        corpus,
        lr=lr,
        device=device,
        precision=precision,
        total_tokens=total_tokens,
        total_seconds=total_seconds,
    )
    started = time.perf_counter()
    target_seconds = float("inf") if total_seconds is None else total_seconds
    target_tokens = 2**63 - 1 if total_tokens is None else total_tokens
    next_eval = float(eval_every_fraction)
    validation_history = []
    active_micro_batch = int(micro_batch)
    oom_recoveries = []
    while (
        trainer.supervised_tokens_seen < target_tokens
        and trainer.optimization_seconds < target_seconds
    ):
        progress = (
            trainer.supervised_tokens_seen / target_tokens
            if total_tokens is not None
            else trainer.optimization_seconds / target_seconds
        )
        length = _context_length(progress)
        trainer.grad_accum_steps = max(
            1, math.ceil(effective_tokens / (active_micro_batch * length))
        )
        sampler_state = corpus.sampler_state()
        rng_state = capture_rng_state(trainer.device)
        counters = (
            trainer.step,
            trainer.tokens_seen,
            trainer.supervised_tokens_seen,
        )
        try:
            trainer.train_steps(
                1,
                active_micro_batch,
                length,
                log_every=0,
                max_seconds=(
                    None
                    if total_seconds is None
                    else max(
                        0.0,
                        target_seconds - trainer.optimization_seconds,
                    )
                ),
            )
        except RuntimeError as error:
            if (
                trainer.device.type != "mps"
                or "out of memory" not in str(error).lower()
                or active_micro_batch <= 1
            ):
                raise
            restore_rng_state(rng_state)
            corpus.load_sampler_state(sampler_state)
            (
                trainer.step,
                trainer.tokens_seen,
                trainer.supervised_tokens_seen,
            ) = counters
            trainer.optimizer.zero_grad(set_to_none=True)
            torch.mps.synchronize()
            torch.mps.empty_cache()
            previous = active_micro_batch
            active_micro_batch = max(1, active_micro_batch // 2)
            oom_recoveries.append(
                {
                    "step": trainer.step,
                    "sequence_length": length,
                    "old_micro_batch": previous,
                    "new_micro_batch": active_micro_batch,
                    "error": str(error),
                }
            )
            continue
        current_fraction = (
            trainer.supervised_tokens_seen / target_tokens
            if total_tokens is not None
            else trainer.optimization_seconds / target_seconds
        )
        if current_fraction >= next_eval:
            validation_history.append(
                {
                    "fraction": min(1.0, current_fraction),
                    "elapsed_seconds": trainer.optimization_seconds,
                    "wall_elapsed_seconds": time.perf_counter() - started,
                    "supervised_tokens": trainer.supervised_tokens_seen,
                    "metrics": evaluate_manifest(
                        trainer.raw_model,
                        validation,
                        batch_size=active_micro_batch,
                        device=device,
                        max_rows=max_validation_rows,
                    ),
                }
            )
            next_eval += float(eval_every_fraction)
    wall_elapsed = time.perf_counter() - started
    elapsed = trainer.optimization_seconds
    result = evaluate_manifest(
        trainer.raw_model,
        validation,
        batch_size=active_micro_batch,
        device=device,
        max_rows=max_validation_rows,
    )
    checkpoint = output / f"{name}_seed_{seed}.pt"
    trainer.save_checkpoint(checkpoint)
    return trainer.raw_model, {
        "seconds": elapsed,
        "wall_seconds": wall_elapsed,
        "supervised_tokens": trainer.supervised_tokens_seen,
        "tokens_per_second": trainer.supervised_tokens_seen / max(elapsed, 1e-9),
        "parameter_token_proxy": count_parameters(trainer.raw_model)
        * trainer.supervised_tokens_seen,
        "validation": result,
        "validation_history": validation_history,
        "oom_recoveries": oom_recoveries,
        "final_micro_batch_size": active_micro_batch,
        "checkpoint": str(checkpoint),
    }


def environment_fingerprint(device: str, precision: str) -> dict[str, Any]:
    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "precision": precision,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "dependency_fingerprint": hashlib.sha256(
            "\n".join(packages).encode()
        ).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--train-manifest-template", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-gate", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--seed-start", type=int, default=20261000)
    parser.add_argument("--tune-token-fraction", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "mps" and args.precision != "fp32":
        raise RuntimeError(
            "decisive Pure Gear comparisons on MPS require precision=fp32: "
            "Pure Gear forces FP32 execution, so a BF16 setting would train "
            "only the Transformer/GRU controls in lower precision"
        )
    if args.device == "mps":
        if args.qualification is None:
            raise RuntimeError("MPS runs require --qualification")
        qualification = json.loads(args.qualification.read_text())
        if not qualification.get("qualified", False):
            raise RuntimeError("Pure Gear engineering qualification did not pass")
        current_code_hash = git_tree_sha256()
        qualified_code_hash = qualification.get("environment", {}).get(
            "code_hash"
        )
        if qualified_code_hash != current_code_hash:
            raise RuntimeError(
                "Pure Gear qualification was produced by different code "
                f"({qualified_code_hash} != {current_code_hash}); rerun it"
            )
    if args.stage in {"15m", "50m"}:
        if args.prior_gate is None:
            raise RuntimeError(
                f"{args.stage} requires the preceding passed final gate"
            )
        prior = json.loads(args.prior_gate.read_text())
        if not prior.get("gate", {}).get("passed", False):
            raise RuntimeError("preceding final gate did not pass")
    stage = STAGES[args.stage]
    first_manifest = Path(
        args.train_manifest_template.format(seed=args.seed_start)
    )
    corpus = PairedDocumentManifestCorpus(str(first_manifest), wrap=False)
    configuration = configs(args.stage, corpus.vocab_size)
    report: dict[str, Any] = {
        "stage": args.stage,
        "environment": environment_fingerprint(args.device, args.precision),
        "data": {
            "first_manifest": str(first_manifest),
            "first_manifest_hash": file_sha256(first_manifest / "manifest.json"),
            "validation_manifest_hash": file_sha256(
                args.validation_manifest / "manifest.json"
            ),
            "confirmation_manifest_hash": file_sha256(
                args.confirmation_manifest / "manifest.json"
            ),
            "tokenizer_fingerprint": tokenizer_fingerprint(corpus.tokenizer),
            "boundary_detector_hash": corpus.manifest.get(
                "boundary_detector_hash"
            ),
        },
        "fairness": assert_fair_configs(configuration),
        "runs": [],
        "protocol": {
            "models": MODEL_NAMES,
            "same_manifest_rows": True,
            "equal_lr_search_budget": True,
            "equal_compute_proxy": (
                "trainable_parameter_count_x_supervised_tokens; reported "
                "separately because exact hardware FLOPs are architecture-dependent"
            ),
            "tokens": stage["tokens"],
            "seeds": stage["seeds"],
        },
    }
    tuning_tokens = max(
        100_000, int(stage["tokens"] * args.tune_token_fraction)
    )
    selected_lrs: dict[str, float] = {}
    for name in MODEL_NAMES:
        trials = []
        for lr in stage["lrs"]:
            _, result = train_run(
                name,
                configuration[name],
                first_manifest,
                args.validation_manifest,
                args.output_dir / "lr_trials" / name / f"{lr:g}",
                seed=args.seed_start,
                lr=lr,
                device=args.device,
                precision=args.precision,
                effective_tokens=stage["effective"],
                micro_batch=args.micro_batch_size,
                total_tokens=tuning_tokens,
                max_validation_rows=args.max_validation_rows,
                eval_every_fraction=1.0,
            )
            trials.append({"lr": lr, **result})
        best = min(
            trials,
            key=lambda item: (
                item["validation"]["macro_domain_nll"],
                item["validation"]["worst_domain_nll"],
            ),
        )
        selected_lrs[name] = best["lr"]
        report.setdefault("lr_tuning", {})[name] = {
            "selected": best["lr"],
            "trials": trials,
        }
        write_json(args.output_dir / "results.partial.json", report)

    for offset in range(stage["seeds"]):
        seed = args.seed_start + offset
        manifest = Path(args.train_manifest_template.format(seed=seed))
        run: dict[str, Any] = {
            "seed": seed,
            "equal_token": {},
            "equal_time": {},
            "equal_compute_proxy": {},
        }
        for name in MODEL_NAMES:
            model, training = train_run(
                name,
                configuration[name],
                manifest,
                args.confirmation_manifest,
                args.output_dir / "equal_token",
                seed=seed,
                lr=selected_lrs[name],
                device=args.device,
                precision=args.precision,
                effective_tokens=stage["effective"],
                micro_batch=args.micro_batch_size,
                total_tokens=stage["tokens"],
                max_validation_rows=args.max_validation_rows,
            )
            training["efficiency"] = throughput(
                model,
                vocab_size=corpus.vocab_size,
                seq_len=min(1024, model.config.max_seq_len),
                device=args.device,
            )
            run["equal_token"][name] = training
            run["equal_compute_proxy"][name] = {
                "source": "equal_token",
                "parameter_token_proxy": training["parameter_token_proxy"],
                "validation": training["validation"],
            }
        wall_budget = run["equal_token"]["transformer"]["seconds"]
        for name in MODEL_NAMES:
            _, training = train_run(
                name,
                configuration[name],
                manifest,
                args.confirmation_manifest,
                args.output_dir / "equal_time",
                seed=seed,
                lr=selected_lrs[name],
                device=args.device,
                precision=args.precision,
                effective_tokens=stage["effective"],
                micro_batch=args.micro_batch_size,
                total_seconds=wall_budget,
                max_validation_rows=args.max_validation_rows,
            )
            run["equal_time"][name] = training
        report["runs"].append(run)
        write_json(args.output_dir / "results.partial.json", report)
    report["training_complete"] = True
    write_json(args.output_dir / "results.json", report)


if __name__ == "__main__":
    main()
