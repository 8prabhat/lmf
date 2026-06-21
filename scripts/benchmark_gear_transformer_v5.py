#!/usr/bin/env python
"""Train and evaluate the stacked-parallel V5 gear architecture.

The benchmark intentionally uses one tokenizer and one corpus for both models.
The headline baseline is selected to match total trainable parameter count, not
just the smaller Transformer trunk inside the gear model. Both models train from
scratch on identical sampled windows for identical update/token budgets. The
report includes held-out NLL, component ablations, and forward/backward
throughput so predictive gains cannot hide a runtime bottleneck.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lmf.data import InMemoryTextCorpus
from lmf.models.gear_transformer import (
    GearTransformerConfig,
    MHGTransformerLM,
    SimplifiedGearTransformerLM,
)
from lmf.models.gear_transformer.trainer import GearTransformerTrainer
from lmf.models.transformer import CachedTransformerLM, TransformerConfig
from lmf.models.transformer.trainer import TransformerTrainer
from lmf.training.checkpoints import save_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "outputs/gear_transformer/comparison/sample_corpus_repeated.txt"
DEFAULT_TOKENIZER = (
    ROOT
    / "outputs/gear_transformer/sentencepiece_repeated"
    / "shared_tokenizer_sentencepiece_bpe_repeated_v1.pt"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--eval-sequences", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--variant",
        choices=("full", "simplified"),
        default="full",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/gear_transformer_v5_results.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/checkpoints/gear_transformer_v5.pt",
    )
    return parser.parse_args()


def _fixed_windows(
    tokens: torch.Tensor,
    seq_len: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    available = max(1, (tokens.numel() - 1) // seq_len)
    count = min(count, available)
    return torch.stack(
        [
            tokens[index * seq_len : (index + 1) * seq_len + 1]
            for index in range(count)
        ]
    ).to(device)


@torch.no_grad()
def _nll(model: torch.nn.Module, windows: torch.Tensor) -> float:
    model.eval()
    inputs, targets = windows[:, :-1], windows[:, 1:]
    logits, _ = model(inputs)
    return float(
        F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _throughput(
    model: torch.nn.Module,
    windows: torch.Tensor,
    repeats: int = 8,
) -> dict[str, float]:
    device = windows.device
    inputs = windows[:, :-1]
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(inputs)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(inputs)
        _synchronize(device)
    forward_seconds = time.perf_counter() - started

    model.train()
    for parameter in model.parameters():
        parameter.grad = None
    _synchronize(device)
    started = time.perf_counter()
    for repeat in range(max(2, repeats // 2)):
        if hasattr(model, "training_step"):
            loss = model.training_step(
                windows,
                {"training_step": 100_000 + repeat},
            )["total"]
        else:
            logits, _ = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                windows[:, 1:].reshape(-1),
            )
        loss.backward()
        for parameter in model.parameters():
            parameter.grad = None
    _synchronize(device)
    backward_repeats = max(2, repeats // 2)
    training_seconds = time.perf_counter() - started
    tokens_per_repeat = inputs.numel()
    return {
        "forward_tokens_per_second": (
            repeats * tokens_per_repeat / max(forward_seconds, 1e-9)
        ),
        "training_tokens_per_second": (
            backward_repeats
            * tokens_per_repeat
            / max(training_seconds, 1e-9)
        ),
    }


@torch.no_grad()
def _continuation(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    length: int = 64,
) -> str:
    device = next(model.parameters()).device
    prompt_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=device,
    )[None]
    generated = model.generate(prompt_ids, length)
    return tokenizer.decode(generated[0].detach().cpu().tolist())


def _gear_config(vocab_size: int, steps: int) -> GearTransformerConfig:
    return GearTransformerConfig(
        vocab_size=vocab_size,
        dim=72,
        layers=3,
        heads=4,
        max_seq_len=256,
        dropout=0.0,
        num_gears=5,
        gear_dim=8,
        gear_system="parallel_v5",
        gear_lane_sizes=(2, 1, 1, 1),
        gear_speeds=(1.4, 0.7, 0.3, 0.11, 0.035),
        gear_slots=(8, 8, 6, 4, 4),
        gear_receptive_fields=(4, 16, 64, 256, 1024),
        gear_rotation_dims=8,
        gear_layer_strategy="explicit",
        gear_layers=(0, 2),
        gear_bank_temporal_strides=(1, 4),
        phase_coupling_topology="adjacent_anchor",
        phase_coupling_init=0.12,
        phase_coupling_max=0.35,
        phase_lock_loss_weight=0.002,
        temporal_context_retention=0.85,
        interbank_coupling_init=0.15,
        bank_specialization_strength=0.35,
        lane_mixing_init=0.12,
        lane_dropout=0.05,
        routing_temperature=1.0,
        future_horizons=(2, 4),
        future_dim=36,
        future_loss_weight=0.02,
        future_token_loss_weight=0.01,
        future_logit_weight=0.15,
        lane_prediction_horizons=(1, 2, 4, 8),
        lane_prediction_loss_weight=0.005,
        lane_token_loss_weight=0.002,
        prediction_loss_stride=8,
        diversity_loss_weight=0.001,
        slot_usage_loss_weight=0.003,
        alignment_loss_weight=0.005,
        consistency_loss_weight=0.005,
        gear_warmup_steps=0,
        gear_ramp_steps=max(10, steps // 8),
        phase_warmup_steps=max(5, steps // 12),
        phase_ramp_steps=max(10, steps // 6),
        auxiliary_warmup_steps=max(10, steps // 8),
        auxiliary_ramp_steps=max(10, steps // 8),
        auxiliary_loss_interval=2,
        future_loss_interval=2,
        future_warmup_steps=max(20, steps // 4),
        future_ramp_steps=max(10, steps // 6),
        gear_lr_multiplier=1.0,
    )


def _simplified_gear_config(
    vocab_size: int,
    steps: int,
) -> GearTransformerConfig:
    """Single widened fast bank plus rotation, context, lanes, and future logits."""
    return GearTransformerConfig(
        vocab_size=vocab_size,
        dim=72,
        layers=3,
        heads=4,
        max_seq_len=256,
        dropout=0.0,
        num_gears=5,
        gear_dim=16,
        gear_system="parallel_v5",
        gear_lane_sizes=(2, 1, 1, 1),
        gear_speeds=(1.4, 0.7, 0.3, 0.11, 0.035),
        gear_slots=(8, 8, 6, 4, 4),
        gear_receptive_fields=(4, 16, 64, 256, 1024),
        gear_rotation_dims=16,
        gear_layer_strategy="explicit",
        gear_layers=(0,),
        gear_bank_temporal_strides=(1,),
        phase_coupling_enabled=False,
        phase_coupling_init=0.0,
        phase_lock_loss_weight=0.0,
        temporal_context_retention=0.85,
        interbank_coupling_init=0.0,
        bank_specialization_strength=0.0,
        lane_mixing_init=0.12,
        lane_dropout=0.05,
        routing_temperature=1.0,
        future_horizons=(2, 4),
        future_dim=36,
        future_loss_weight=0.02,
        future_token_loss_weight=0.01,
        future_logit_weight=0.15,
        lane_prediction_horizons=(1, 2, 4, 8),
        lane_prediction_loss_weight=0.005,
        lane_token_loss_weight=0.002,
        prediction_loss_stride=8,
        diversity_loss_weight=0.001,
        slot_usage_loss_weight=0.003,
        alignment_loss_weight=0.005,
        consistency_loss_weight=0.005,
        gear_warmup_steps=0,
        gear_ramp_steps=max(10, steps // 8),
        phase_warmup_steps=0,
        phase_ramp_steps=max(10, steps // 8),
        auxiliary_warmup_steps=max(10, steps // 8),
        auxiliary_ramp_steps=max(10, steps // 8),
        auxiliary_loss_interval=2,
        future_loss_interval=2,
        future_warmup_steps=max(20, steps // 4),
        future_ramp_steps=max(10, steps // 6),
        gear_lr_multiplier=1.0,
    )


def _parameter_matched_baseline_config(
    vocab_size: int,
    target_parameters: int,
) -> TransformerConfig:
    """Choose the closest small modern Transformer by total parameter count."""
    candidates: list[tuple[int, int, int, TransformerConfig]] = []
    for heads in (2, 3, 4, 5, 6, 8):
        for layers in range(2, 7):
            for head_dim in range(8, 50, 2):
                dim = heads * head_dim
                config = TransformerConfig(
                    vocab_size=vocab_size,
                    dim=dim,
                    layers=layers,
                    heads=heads,
                    max_seq_len=256,
                )
                parameters = sum(
                    parameter.numel()
                    for parameter in CachedTransformerLM(config).parameters()
                )
                candidates.append(
                    (
                        abs(parameters - target_parameters),
                        abs(layers - 3),
                        dim,
                        config,
                    )
                )
    close = [
        item
        for item in candidates
        if item[0] / max(target_parameters, 1) <= 0.02
    ]
    if close:
        # Once capacity is matched within two percent, hold depth constant
        # before chasing tiny parameter-count differences. This avoids selecting
        # a pathologically narrow/deep baseline simply because it is 0.1%
        # closer in size.
        return min(close, key=lambda item: (item[1], item[0], -item[2]))[3]
    return min(candidates, key=lambda item: (item[0], item[1], -item[2]))[3]


def main() -> None:
    args = _parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = torch.load(
        DEFAULT_TOKENIZER,
        map_location="cpu",
        weights_only=False,
    )
    corpus = InMemoryTextCorpus(
        DEFAULT_CORPUS.read_text(),
        tokenizer=tokenizer,
        seed=args.seed,
        wrap_special=False,
    )
    valid_tokens = corpus._splits["valid"].long()
    windows = _fixed_windows(
        valid_tokens,
        args.seq_len,
        args.eval_sequences,
        device,
    )
    initial_sampler_state = corpus.sampler_state()
    if args.variant == "simplified":
        gear_config = _simplified_gear_config(corpus.vocab_size, args.steps)
        gear_type = SimplifiedGearTransformerLM
        ablation_components = (
            "gears",
            "phase",
            "rotation",
            "temporal_context",
            "lane_mixing",
            "future",
        )
    else:
        gear_config = _gear_config(corpus.vocab_size, args.steps)
        gear_type = MHGTransformerLM
        ablation_components = (
            "gears",
            "phase",
            "phase_coupling",
            "rotation",
            "temporal_context",
            "interbank_coupling",
            "lane_mixing",
            "future",
        )
    torch.manual_seed(args.seed)
    gear = gear_type(gear_config)
    gear_parameters = sum(parameter.numel() for parameter in gear.parameters())
    baseline_config = _parameter_matched_baseline_config(
        corpus.vocab_size,
        gear_parameters,
    )
    torch.manual_seed(args.seed)
    baseline = CachedTransformerLM(baseline_config)
    baseline_initial_nll = _nll(baseline, windows)
    baseline_trainer = TransformerTrainer(
        baseline,
        corpus,
        device=args.device,
        precision="fp32",
        lr=1e-3,
        warmup_steps=max(10, args.steps // 10),
        total_steps=args.steps,
    )
    started = time.perf_counter()
    baseline_trainer.train_steps(
        args.steps,
        args.batch_size,
        args.seq_len,
        log_every=0,
    )
    baseline_training_seconds = time.perf_counter() - started
    corpus.load_sampler_state(initial_sampler_state)
    torch.manual_seed(args.seed)
    gear = gear_type(gear_config)
    gear_initial_nll = _nll(gear, windows)
    gear_trainer = GearTransformerTrainer(
        gear,
        corpus,
        device=args.device,
        precision="fp32",
        lr=1e-3,
        warmup_steps=max(10, args.steps // 10),
        total_steps=args.steps,
        trunk_freeze_steps=0,
    )
    started = time.perf_counter()
    gear_trainer.train_steps(
        args.steps,
        args.batch_size,
        args.seq_len,
        log_every=0,
    )
    gear_training_seconds = time.perf_counter() - started

    baseline = baseline_trainer.raw_model
    gear = gear_trainer.raw_model
    baseline_nll = _nll(baseline, windows)
    gear_nll = _nll(gear, windows)
    component_metrics = gear.component_ablation_metrics(
        windows,
        ablation_components,
    )
    baseline_speed = _throughput(
        baseline,
        windows[: min(8, len(windows))],
    )
    gear_speed = _throughput(gear, windows[: min(8, len(windows))])

    prompts = (
        "The river wound slowly through the valley,",
        "Engineers building the new railway line debated",
        "The professor who guided her research had spent decades studying",
        "At dawn, the research team opened the sealed container and found",
        "The council postponed its final decision because the evidence",
    )
    predictions = [
        {
            "prompt": prompt,
            "baseline": _continuation(baseline, tokenizer, prompt),
            "gear": _continuation(gear, tokenizer, prompt),
        }
        for prompt in prompts
    ]
    diagnostics = gear.gear_diagnostics(windows[:4, :-1])
    component_positive = {
        name: metrics["delta_nll"] > 0.0
        for name, metrics in component_metrics.items()
        if name != "full"
    }
    forward_ratio = (
        gear_speed["forward_tokens_per_second"]
        / baseline_speed["forward_tokens_per_second"]
    )
    training_ratio = (
        gear_speed["training_tokens_per_second"]
        / baseline_speed["training_tokens_per_second"]
    )
    report = {
        "seed": args.seed,
        "variant": args.variant,
        "comparison_contract": {
            "parameter_matched": True,
            "same_tokenizer_and_corpus": True,
            "same_sampled_training_windows": True,
            "same_optimizer_updates": True,
            "same_training_tokens": True,
            "same_base_learning_rate_and_warmup": True,
            "trained_from_scratch": True,
        },
        "steps_per_model": args.steps,
        "training_tokens_per_model": (
            args.steps * args.batch_size * args.seq_len
        ),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "parameters": {
            "baseline": sum(p.numel() for p in baseline.parameters()),
            "gear": sum(p.numel() for p in gear.parameters()),
            "relative_gap": (
                sum(p.numel() for p in gear.parameters())
                / sum(p.numel() for p in baseline.parameters())
                - 1.0
            ),
        },
        "baseline_config": baseline_config.to_dict(),
        "validation": {
            "baseline_initial_nll": baseline_initial_nll,
            "gear_initial_nll": gear_initial_nll,
            "baseline_nll": baseline_nll,
            "baseline_bits_per_token": baseline_nll / math.log(2.0),
            "gear_nll": gear_nll,
            "gear_bits_per_token": gear_nll / math.log(2.0),
        },
        "training_seconds": {
            "baseline": baseline_training_seconds,
            "gear": gear_training_seconds,
        },
        "throughput": {
            "baseline": baseline_speed,
            "gear": gear_speed,
            "gear_to_baseline_forward_ratio": forward_ratio,
            "gear_to_baseline_training_ratio": training_ratio,
        },
        "component_ablations": component_metrics,
        "component_positive_impact": component_positive,
        "all_components_positive": all(component_positive.values()),
        "acceptance": {
            "all_components_positive": all(component_positive.values()),
            "parameter_gap_at_most_2_percent": abs(
                sum(p.numel() for p in gear.parameters())
                / sum(p.numel() for p in baseline.parameters())
                - 1.0
            ) <= 0.02,
            "beats_parameter_matched_baseline": gear_nll < baseline_nll,
            "forward_ratio_at_least_0_35": forward_ratio >= 0.35,
            "training_ratio_at_least_0_35": training_ratio >= 0.35,
        },
        "diagnostics": diagnostics,
        "predictions": predictions,
        "gear_config": gear.config.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    save_checkpoint(
        args.checkpoint,
        gear,
        gear_trainer.optimizer,
        gear_trainer.step,
        extra={"benchmark": f"{args.variant}_gear_transformer"},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
