#!/usr/bin/env python
"""Engineering qualification before any decisive Pure Gear training run."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from lmf.core.hashing import git_tree_sha256, json_sha256
from lmf.core.device import sync
from lmf.data import (
    NumericFallbackTokenizer,
    ProceduralCorpus,
    boundary_detector_hash,
)
from lmf.diagnostics import cache_bytes, parameter_count
from lmf.models.pure_parallel_gear import (
    PureGearLayer,
    PureParallelGearConfig,
    PureParallelGearLM,
    PureParallelGearTrainer,
)
from lmf.models.pure_parallel_gear.model import _rotate
from lmf.models.transformer import CachedTransformerLM

try:
    from scripts.benchmark_pure_parallel_gear import (
        _inference_parameter_count,
        configs as benchmark_configs,
        throughput as generation_throughput,
    )
except ModuleNotFoundError:
    from benchmark_pure_parallel_gear import (
        _inference_parameter_count,
        configs as benchmark_configs,
        throughput as generation_throughput,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20261050)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--generation-context", type=int, default=512)
    return parser.parse_args()


def config() -> PureParallelGearConfig:
    return PureParallelGearConfig(
        vocab_size=257,
        dim=32,
        layers=2,
        ffn_dim=64,
        num_banks=2,
        gears_per_bank=4,
        rotor_channels=2,
        predictor_gears=3,
        max_sentence_tokens=16,
        max_seq_len=512,
    )


def closed_form_parity(model: PureParallelGearLM) -> dict:
    layer = model.layers[0]
    hidden_a = torch.randn(2, 11, model.config.dim, requires_grad=True)
    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    # Compare one row because _token_dynamics treats its leading dimension as
    # a sentence, not a batch.
    closed, _, _ = layer._token_dynamics(
        hidden_a[0],
        layer.initial_state(1, hidden_a.device),
        fixed_omega=False,
    )
    source = layer.input_norm(hidden_b[0]).float()
    shape = (hidden_b.shape[1], layer.banks, layer.gears, layer.channels)
    delta = model.config.theta_limit * torch.tanh(
        layer.angle_projection(source).reshape(shape)
    )
    clutch = torch.sigmoid(layer.clutch_projection(source).reshape(shape))
    torque = model.config.torque_limit * clutch[..., None] * torch.tanh(
        layer.torque_projection(source).reshape(*shape, 2)
    )
    retention = layer.retention_low + (
        layer.retention_high - layer.retention_low
    ) * torch.sigmoid(
        layer.retention_projection(source).reshape(shape)
    )
    sequential = []
    loop_state = layer.initial_state(1, hidden_b.device)
    rotor = loop_state.rotor
    omega = loop_state.omega
    for index in range(hidden_b.shape[1]):
        rotor = (
            retention[index][None, ..., None]
            * _rotate(rotor, delta[index] + omega)
            + torque[index]
        )
        sequential.append(rotor)
    sequential_tensor = torch.cat(sequential, dim=0)
    closed.sum().backward()
    sequential_tensor.sum().backward()
    return {
        "max_output_error": float(
            (closed.detach() - sequential_tensor.detach()).abs().max()
        ),
        "max_input_gradient_error": float(
            (hidden_a.grad[0] - hidden_b.grad[0]).abs().max()
        ),
    }


def streaming_parity(model, device):
    tokens = torch.randint(0, model.config.vocab_size, (2, 47), device=device)
    boundaries = torch.zeros_like(tokens, dtype=torch.bool)
    boundaries[:, (9, 23, 39)] = True
    full, _ = model(tokens, sentence_end_mask=boundaries)
    pieces = []
    cache = None
    for position in range(tokens.shape[1]):
        logits, cache = model(
            tokens[:, position : position + 1],
            cache=cache,
            use_cache=True,
            sentence_end_mask=boundaries[:, position : position + 1],
        )
        pieces.append(logits)
    streamed = torch.cat(pieces, dim=1)
    return {
        "max_logit_error": float((full - streamed).abs().max().detach()),
        "cache_bytes": cache_bytes(cache),
    }


def optimizer_precision(device):
    corpus = ProceduralCorpus(vocab_size=257, seed=11)
    model = PureParallelGearLM(config())
    trainer = PureParallelGearTrainer(
        model,
        corpus,
        device=device,
        precision="bf16",
        lr=1e-3,
        total_training_tokens=1024,
        warmup_tokens=1,
        context_lengths=(32,),
        context_fractions=(1.0,),
    )
    trainer.train_steps(1, 2, 32, log_every=0)
    parameter_dtypes = sorted(
        {str(parameter.dtype) for parameter in trainer.raw_model.parameters()}
    )
    moment_dtypes = sorted(
        {
            str(value.dtype)
            for state in trainer.optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value) and value.is_floating_point()
        }
    )
    return {
        "parameter_dtypes": parameter_dtypes,
        "optimizer_moment_dtypes": moment_dtypes,
        "gradient_skips": trainer.total_gradient_skips,
    }


def training_throughput(model, tokens, device, repeats: int) -> dict:
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    samples = []
    for iteration in range(repeats + 1):
        optimizer.zero_grad(set_to_none=True)
        sync(device)
        started = time.perf_counter()
        loss = model.training_step(tokens)["total"]
        loss.backward()
        optimizer.step()
        sync(device)
        if iteration:
            samples.append(time.perf_counter() - started)
    median = statistics.median(samples)
    return {
        "tokens_per_second": tokens.numel() / max(median, 1e-9),
        "median_step_seconds": median,
        "samples": samples,
        "includes_optimizer_step": True,
    }


def performance_qualification(args, device: torch.device) -> dict:
    configurations = benchmark_configs("proxy", 257)
    transformer = CachedTransformerLM(configurations["transformer"])
    gear = PureParallelGearLM(configurations["gear"])
    gear.configure_boundary_detector(NumericFallbackTokenizer(257))
    tokens = torch.randint(
        0,
        257,
        (args.batch_size, args.seq_len),
        device=device,
    )
    transformer_training = training_throughput(
        transformer, tokens, device, args.repeats
    )
    gear_training = training_throughput(
        gear, tokens, device, args.repeats
    )
    transformer_generation = generation_throughput(
        transformer,
        vocab_size=257,
        seq_len=args.generation_context,
        device=str(device),
        repeats=args.repeats,
    )
    gear_generation = generation_throughput(
        gear,
        vocab_size=257,
        seq_len=args.generation_context,
        device=str(device),
        repeats=args.repeats,
    )
    parameters = {
        "transformer": parameter_count(transformer),
        # _inference_parameter_count excludes gear's future_heads -- a
        # training-only auxiliary-loss scaffold (Phase 3.1) never read at
        # inference -- so this gate measures inference-time capacity
        # parity, not raw checkpoint size.
        "gear": _inference_parameter_count(gear),
    }
    return {
        "parameters": parameters,
        "parameter_gap": parameters["gear"] / parameters["transformer"] - 1.0,
        "training": {
            "transformer": transformer_training,
            "gear": gear_training,
            "gear_to_transformer_ratio": (
                gear_training["tokens_per_second"]
                / transformer_training["tokens_per_second"]
            ),
        },
        "generation": {
            "transformer": transformer_generation,
            "gear": gear_generation,
            "gear_to_transformer_ratio": (
                gear_generation["incremental_tokens_per_second"]
                / transformer_generation["incremental_tokens_per_second"]
            ),
            "cache_ratio": (
                gear_generation["cache_bytes"]
                / max(1, transformer_generation["cache_bytes"])
            ),
        },
        "configs": {
            "transformer": configurations["transformer"].to_dict(),
            "gear": configurations["gear"].to_dict(),
        },
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cpu_model = PureParallelGearLM(config()).eval()
    closed = closed_form_parity(cpu_model)
    device = torch.device(args.device)
    device_model = PureParallelGearLM(config()).to(device).eval()
    device_model.load_state_dict(cpu_model.state_dict())
    streaming = streaming_parity(device_model, device)
    cache_sizes = []
    for length in (16, 64, 256):
        tokens = torch.randint(
            0, device_model.config.vocab_size, (1, length), device=device
        )
        _, cache = device_model(tokens, use_cache=True)
        cache_sizes.append(cache_bytes(cache))
    precision = optimizer_precision(args.device)
    performance = performance_qualification(args, device)
    manifest = device_model.architecture_manifest()
    source = (
        inspect.getsource(PureParallelGearLM)
        + inspect.getsource(PureGearLayer)
    ).lower()
    forbidden_terms = (
        "scaled_dot_product_attention",
        "retrieval_window",
        "pointer_weights",
    )
    source_violations = [term for term in forbidden_terms if term in source]
    checks = {
        "closed_form_output": closed["max_output_error"] <= 1e-5,
        "closed_form_gradient": closed["max_input_gradient_error"] <= 1e-5,
        "streaming_equivalence": streaming["max_logit_error"] <= 2e-5,
        "constant_cache": len(set(cache_sizes)) == 1,
        "fp32_parameters": precision["parameter_dtypes"] == ["torch.float32"],
        "fp32_optimizer_moments": precision["optimizer_moment_dtypes"]
        == ["torch.float32"],
        "no_gradient_skips": precision["gradient_skips"] == 0,
        "no_forbidden_source_terms": not source_violations,
        "manifest_reports_host_control": (
            manifest["invariants"]["host_scalar_control_flow"] is True
        ),
        "manifest_reports_boundary_settling": (
            manifest["invariants"]["sentence_execution"]
            == "parallel_affine_scan_with_sequential_boundary_settling"
        ),
        "performance_parameter_match": (
            abs(performance["parameter_gap"]) <= 0.005
        ),
        "training_throughput_at_least_half_transformer": (
            performance["training"]["gear_to_transformer_ratio"] >= 0.5
        ),
        "incremental_generation_at_least_1_5x_transformer": (
            performance["generation"]["gear_to_transformer_ratio"] >= 1.5
        ),
        "cache_at_most_quarter_transformer": (
            performance["generation"]["cache_ratio"] <= 0.25
        ),
    }
    report = {
        "qualified": all(checks.values()),
        "checks": checks,
        "closed_form": closed,
        "streaming": streaming,
        "cache_sizes": cache_sizes,
        "precision": precision,
        "performance": performance,
        "source_violations": source_violations,
        "manifest": manifest,
        "manifest_hash": json_sha256(manifest),
        "instantiated_config": device_model.config.to_dict(),
        "boundary_detector_hash": boundary_detector_hash(
            device_model.config.max_sentence_tokens
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": args.device,
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "code_hash": git_tree_sha256(),
            "seed": args.seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
