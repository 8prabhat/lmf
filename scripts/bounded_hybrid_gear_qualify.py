#!/usr/bin/env python3
"""Engineering and performance qualification for the Bounded Hybrid Gear family."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from lmf.models.bounded_hybrid_gear import (
    BoundedTransformerConfig,
    BoundedTransformerLM,
    BlockHybridGearV4Config,
    BlockHybridGearV4LM,
    HybridParallelGearConfig,
    HybridParallelGearLM,
    PureParallelGearV3Config,
    PureParallelGearV3LM,
    chunked_affine_scan,
    complex_mul,
)
from lmf.models.transformer import CachedTransformerLM, TransformerConfig
try:
    from scripts.pure_parallel_gear_common import cache_bytes
except ModuleNotFoundError:
    from pure_parallel_gear_common import cache_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--generation-steps", type=int, default=32)
    parser.add_argument(
        "--block-fusion-mode",
        choices=("additive", "selective_film", "bank_router"),
        default="additive",
    )
    parser.add_argument("--block-fusion-rank", type=int, default=32)
    parser.add_argument("--block-ffn-dim", type=int, default=349)
    parser.add_argument("--block-attention-window", type=int, default=128)
    parser.add_argument("--block-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20262150)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def sequential_scan(multiplier, bias, initial):
    state = initial
    rows = []
    for position in range(multiplier.shape[1]):
        state = complex_mul(multiplier[:, position], state) + bias[:, position]
        rows.append(state)
    return torch.stack(rows, dim=1)


def scan_proof(device: torch.device) -> dict:
    torch.manual_seed(3)
    phase = torch.randn(2, 37, 3, 2, 1, device=device)
    magnitude = 0.90 + 0.09 * torch.rand(
        2, 37, 3, 2, 1, 1, device=device
    )
    multiplier = (
        magnitude
        * torch.stack((phase.cos(), phase.sin()), dim=-1)
    ).requires_grad_(True)
    bias = (0.03 * torch.randn_like(multiplier)).requires_grad_(True)
    initial = torch.randn(2, 3, 2, 1, 2, device=device, requires_grad=True)
    multiplier_ref = multiplier.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    initial_ref = initial.detach().clone().requires_grad_(True)
    scan, _, _ = chunked_affine_scan(
        multiplier, bias, initial, chunk_size=8
    )
    reference = sequential_scan(multiplier_ref, bias_ref, initial_ref)
    weight = torch.randn_like(scan)
    (scan * weight).sum().backward()
    (reference * weight).sum().backward()
    return {
        "output_max_error": float(
            (scan - reference).abs().max().detach().cpu()
        ),
        "multiplier_gradient_max_error": float(
            (multiplier.grad - multiplier_ref.grad).abs().max().detach().cpu()
        ),
        "bias_gradient_max_error": float(
            (bias.grad - bias_ref.grad).abs().max().detach().cpu()
        ),
        "initial_gradient_max_error": float(
            (initial.grad - initial_ref.grad).abs().max().detach().cpu()
        ),
    }


def model_set(vocab_size: int, args: argparse.Namespace):
    strict = PureParallelGearV3LM(
        PureParallelGearV3Config(
            vocab_size=vocab_size,
            dim=96,
            layers=2,
            ffn_dim=601,
            cell_dim=12,
            bank_rank=12,
        )
    )
    hybrid = HybridParallelGearLM(
        HybridParallelGearConfig(
            vocab_size=vocab_size,
            dim=96,
            layers=2,
            ffn_dim=557,
            cell_dim=12,
            bank_rank=12,
            attention_heads=6,
            attention_kv_heads=2,
        )
    )
    bounded = BoundedTransformerLM(
        BoundedTransformerConfig(
            vocab_size=vocab_size,
            dim=112,
            layers=2,
            ffn_dim=381,
            heads=7,
            kv_heads=1,
            attention_window=args.block_attention_window,
        )
    )
    transformer = CachedTransformerLM(
        TransformerConfig(
            vocab_size=vocab_size,
            dim=112,
            layers=2,
            heads=7,
        )
    )
    v4 = BlockHybridGearV4LM(
        BlockHybridGearV4Config(
            vocab_size=vocab_size,
            dim=112,
            layers=2,
            ffn_dim=args.block_ffn_dim,
            heads=7,
            kv_heads=1,
            attention_window=args.block_attention_window,
            cell_dim=12,
            bank_rank=12,
            block_tokens=args.block_tokens,
            fusion_mode=args.block_fusion_mode,
            fusion_rank=args.block_fusion_rank,
        )
    )
    return {
        "strict_v3": strict,
        "hybrid": hybrid,
        "bounded_transformer": bounded,
        "full_transformer": transformer,
        "block_hybrid_gear": v4,
    }


def precision_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def streaming_error(model, tokens, device, dtype):
    model.eval()
    with precision_context(device, dtype):
        full, _ = model(tokens)
    fp32, _ = model(tokens)
    cache = None
    pieces = []
    sizes = []
    for position in range(tokens.shape[1]):
        with precision_context(device, dtype):
            if model.architecture_manifest()["name"] == "CachedTransformerLM":
                logits, cache = model(
                    tokens[:, position : position + 1],
                    caches=cache,
                    use_cache=True,
                )
            else:
                logits, cache = model(
                    tokens[:, position : position + 1],
                    cache=cache,
                    use_cache=True,
                )
        pieces.append(logits)
        sizes.append(cache_bytes(cache))
    return {
        "max_logit_error": float(
            (full - torch.cat(pieces, dim=1)).abs().max().detach().cpu()
        ),
        "mixed_precision_max_logit_error": float(
            (full.float() - fp32.float()).abs().max().detach().cpu()
        ),
        "cache_bytes": sizes[-1],
        "cache_trace_bytes": sizes,
        "constant_cache": len(set(sizes)) == 1,
    }


def code_hash() -> str:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "src",
            "scripts",
            "configs",
            "tests",
            "pyproject.toml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    digest = hashlib.sha256()
    for name in sorted(result.stdout.splitlines()):
        path = Path(name)
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def manifest_hash(manifest: dict) -> str:
    encoded = json.dumps(manifest, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def throughput(model, tokens, device, dtype, repeats):
    model.train()
    with precision_context(device, dtype):
        loss = model.training_step(tokens)["total"]
    loss.backward()
    model.zero_grad(set_to_none=True)
    sync(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        with precision_context(device, dtype):
            loss = model.training_step(tokens)["total"]
        loss.backward()
        sync(device)
        samples.append(time.perf_counter() - started)
        model.zero_grad(set_to_none=True)
    seconds = statistics.median(samples)
    return {
        "median_step_seconds": seconds,
        "tokens_per_second": tokens.numel() / max(seconds, 1e-9),
        "samples": samples,
    }


@torch.no_grad()
def incremental_profile(
    model,
    *,
    vocab_size: int,
    context_len: int,
    generation_steps: int,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict:
    model.eval()
    prompt = torch.randint(
        0,
        vocab_size,
        (1, context_len),
        device=device,
    )
    with precision_context(device, dtype):
        if model.architecture_manifest()["name"] == "CachedTransformerLM":
            _, cache = model(prompt, use_cache=True)
        else:
            _, cache = model(prompt, use_cache=True)
    sync(device)
    token = prompt[:, -1:]
    samples = []
    for _ in range(generation_steps):
        started = time.perf_counter()
        with precision_context(device, dtype):
            if model.architecture_manifest()["name"] == "CachedTransformerLM":
                logits, cache = model(token, caches=cache, use_cache=True)
            else:
                logits, cache = model(token, cache=cache, use_cache=True)
        sync(device)
        samples.append(time.perf_counter() - started)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
    median = statistics.median(samples)
    return {
        "context_tokens": context_len,
        "steps": generation_steps,
        "cache_bytes": cache_bytes(cache),
        "median_token_seconds": median,
        "tokens_per_second": 1.0 / max(median, 1e-9),
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.precision == "bf16" else None
    tokens = torch.randint(
        0,
        4109,
        (args.batch_size, args.seq_len),
        device=device,
    )
    models = model_set(4109, args)
    report = {
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "precision": args.precision,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "code_hash": code_hash(),
            "seed": args.seed,
            "checkpoint_hash": None,
            "checkpoint_hash_reason": "engineering qualification uses fresh weights",
        },
        "scan_proof": scan_proof(device),
        "models": {},
    }
    for name, model in models.items():
        model = model.to(device)
        short_tokens = tokens[:, : min(64, args.seq_len)]
        manifest = model.architecture_manifest()
        report["models"][name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "manifest": manifest,
            "manifest_hash": manifest_hash(manifest),
            "instantiated_config": model.config.to_dict(),
            "streaming": streaming_error(
                model, short_tokens, device, dtype
            ),
            "training": throughput(
                model, tokens, device, dtype, args.repeats
            ),
            "incremental": incremental_profile(
                model,
                vocab_size=4109,
                context_len=args.context_len,
                generation_steps=args.generation_steps,
                device=device,
                dtype=dtype,
            ),
        }
    baseline = report["models"]["full_transformer"]["training"][
        "tokens_per_second"
    ]
    baseline_parameters = report["models"]["full_transformer"]["parameters"]
    baseline_incremental = report["models"]["full_transformer"][
        "incremental"
    ]["tokens_per_second"]
    baseline_cache = report["models"]["full_transformer"]["incremental"][
        "cache_bytes"
    ]
    report["checks"] = {
        "scan_output_proof": report["scan_proof"]["output_max_error"] <= 1e-5,
        "scan_gradient_proof": max(
            value
            for key, value in report["scan_proof"].items()
            if "gradient" in key
        )
        <= 2e-5,
        "all_streaming_equivalent": all(
            value["streaming"]["max_logit_error"]
            <= (2e-3 if args.precision == "bf16" else 2e-5)
            for value in report["models"].values()
        ),
        "mixed_precision_logit_error": all(
            value["streaming"]["mixed_precision_max_logit_error"] <= 2e-3
            for value in report["models"].values()
        ),
        "parameter_match_within_half_percent": all(
            abs(value["parameters"] / baseline_parameters - 1.0) <= 0.005
            for value in report["models"].values()
        ),
        "strict_cache_constant": report["models"]["strict_v3"]["streaming"][
            "constant_cache"
        ],
        "hybrid_cache_bounded": report["models"]["hybrid"]["streaming"][
            "constant_cache"
        ],
        "bounded_transformer_cache_bounded": report["models"][
            "bounded_transformer"
        ]["streaming"]["constant_cache"],
        "strict_throughput_at_least_half_transformer": (
            report["models"]["strict_v3"]["training"]["tokens_per_second"]
            / baseline
            >= 0.5
        ),
        "hybrid_throughput_at_least_half_transformer": (
            report["models"]["hybrid"]["training"]["tokens_per_second"]
            / baseline
            >= 0.5
        ),
        "block_hybrid_gear_throughput_at_least_half_transformer": (
            report["models"]["block_hybrid_gear"]["training"][
                "tokens_per_second"
            ]
            / baseline
            >= 0.5
        ),
        "strict_generation_at_least_1_5x_transformer": (
            report["models"]["strict_v3"]["incremental"]["tokens_per_second"]
            / baseline_incremental
            >= 1.5
        ),
        "hybrid_generation_at_least_1_5x_transformer": (
            report["models"]["hybrid"]["incremental"]["tokens_per_second"]
            / baseline_incremental
            >= 1.5
        ),
        "block_hybrid_gear_generation_at_least_1_5x_transformer": (
            report["models"]["block_hybrid_gear"]["incremental"][
                "tokens_per_second"
            ]
            / baseline_incremental
            >= 1.5
        ),
        "hybrid_cache_at_most_quarter_transformer": (
            report["models"]["hybrid"]["incremental"]["cache_bytes"]
            / max(1, baseline_cache)
            <= 0.25
        ),
        "block_hybrid_gear_cache_at_most_quarter_transformer": (
            report["models"]["block_hybrid_gear"]["incremental"][
                "cache_bytes"
            ]
            / max(1, baseline_cache)
            <= 0.25
        ),
    }
    report["unresolved_bottlenecks"] = []
    for check, passed in report["checks"].items():
        if not passed:
            report["unresolved_bottlenecks"].append(
                {
                    "priority": "P1",
                    "check": check,
                    "status": "unresolved",
                    "scaling_blocked": True,
                }
            )
    report["qualified"] = all(report["checks"].values())
    common_checks = (
        "scan_output_proof",
        "scan_gradient_proof",
        "all_streaming_equivalent",
        "mixed_precision_logit_error",
        "parameter_match_within_half_percent",
    )
    block_hybrid_gear_checks = (
        "block_hybrid_gear_throughput_at_least_half_transformer",
        "block_hybrid_gear_generation_at_least_1_5x_transformer",
        "block_hybrid_gear_cache_at_most_quarter_transformer",
    )
    report["block_hybrid_gear_qualified"] = all(
        report["checks"][name] for name in (*common_checks, *block_hybrid_gear_checks)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "qualified": report["qualified"],
                "block_hybrid_gear_qualified": report["block_hybrid_gear_qualified"],
                **report["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
