#!/usr/bin/env python
"""360-degree post-training evaluation for Gear, Transformer, and GRU."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lmf.ablation.stats import percentile
from lmf.data import EduCombinedCorpus, SentenceBoundaryDetector
from lmf.diagnostics import cache_bytes
from lmf.evaluation import PredictiveTaskCorpus
from lmf.models.pure_parallel_gear import PureParallelGearLM
from lmf.models.rhca.state import SamplingConfig
from lmf.research_utils import load_model
try:
    from scripts.benchmark_pure_parallel_gear import (
        efficiency_samples,
        evaluate_manifest,
    )
except ModuleNotFoundError:
    from benchmark_pure_parallel_gear import efficiency_samples, evaluate_manifest


DOMAINS = (
    "cosmopedia",
    "fineweb_edu",
    "open_web_math",
    "pes2o",
    "pg19",
    "stack_exchange",
    "wikipedia",
)
TASKS = (
    "associative_recall",
    "induction",
    "selective_copy",
    "variable_echo",
    "noisy_recall",
    "repeated_pattern",
    "order_reversal",
    "nested_structure",
    "sentence_transition",
)
ABLATIONS = (
    "one_bank_only",
    "no_boundary_settling",
    "no_cross_bank_coupling",
    "commuting_coupling_only",
    "fixed_angular_velocities",
    "no_load_state",
    "no_predictor_gear",
    "no_local_swiglu",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--evaluation-manifest", action="append", default=[])
    parser.add_argument("--external-token-file", action="append", default=[])
    parser.add_argument("--minimal-pair-file", action="append", default=[])
    parser.add_argument("--paraphrase-file", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--windows-per-domain", type=int, default=256)
    parser.add_argument("--robustness-windows", type=int, default=64)
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=(128, 256, 512, 1024, 2048, 4096, 8192),
    )
    parser.add_argument(
        "--task-distances",
        nargs="+",
        type=int,
        default=(16, 64, 256, 1024, 2048),
    )
    parser.add_argument("--task-batches", type=int, default=8)
    parser.add_argument("--task-batch-size", type=int, default=8)
    parser.add_argument(
        "--efficiency-lengths",
        nargs="+",
        type=int,
        default=(128, 256, 512, 1024, 2048, 4096, 8192),
    )
    parser.add_argument("--efficiency-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261150)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def fixed_windows(tokens: torch.Tensor, length: int, count: int) -> torch.Tensor:
    maximum = max(0, len(tokens) - length)
    starts = [
        round(index * maximum / max(count - 1, 1))
        for index in range(count)
    ]
    return torch.stack([tokens[start : start + length] for start in starts])


def boundary_masks(
    windows: torch.Tensor,
    detector: SentenceBoundaryDetector,
) -> torch.Tensor:
    return torch.stack(
        [detector.scan_tokens(row.tolist())[1] for row in windows]
    )


def model_kwargs(model, tokens, detector=None):
    if isinstance(model, PureParallelGearLM):
        boundaries = (
            boundary_masks(tokens.cpu(), detector).to(tokens.device)
            if detector is not None
            else None
        )
        return {"sentence_end_mask": boundaries}
    return {}


@torch.no_grad()
def window_metrics(
    model,
    windows,
    *,
    device,
    detector=None,
    target_mask=None,
) -> dict[str, float]:
    loss_sum = brier_sum = 0.0
    count = top1 = top5 = 0
    confidences = []
    correctness = []
    for start in range(0, len(windows), 8):
        tokens = windows[start : start + 8].to(device)
        logits, _ = model(tokens, **model_kwargs(model, tokens, detector))
        logits, targets = logits[:, :-1], tokens[:, 1:]
        valid = torch.ones_like(targets, dtype=torch.bool)
        if target_mask is not None:
            valid &= target_mask[start : start + len(tokens), 1:].to(device)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        probabilities = logits.softmax(-1)
        target_p = probabilities.gather(-1, targets[..., None]).squeeze(-1)
        brier = probabilities.square().sum(-1) - 2 * target_p + 1
        prediction = logits.argmax(-1)
        loss_sum += float(losses[valid].sum())
        brier_sum += float(brier[valid].sum())
        count += int(valid.sum())
        top1 += int((prediction[valid] == targets[valid]).sum())
        top5 += int(
            (
                logits.topk(min(5, logits.shape[-1]), -1).indices[valid]
                == targets[valid][..., None]
            ).any(-1).sum()
        )
        confidences.extend(probabilities.max(-1).values[valid].cpu().tolist())
        correctness.extend((prediction[valid] == targets[valid]).cpu().tolist())
    ece = 0.0
    for index in range(15):
        selected = [
            row
            for row, value in enumerate(confidences)
            if index / 15 <= value < (index + 1) / 15
            or (index == 14 and value == 1.0)
        ]
        if selected:
            accuracy = statistics.fmean(float(correctness[row]) for row in selected)
            confidence = statistics.fmean(confidences[row] for row in selected)
            ece += len(selected) / count * abs(accuracy - confidence)
    nll = loss_sum / max(count, 1)
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 30.0)),
        "bits_per_token": nll / math.log(2.0),
        "top1_accuracy": top1 / max(count, 1),
        "top5_accuracy": top5 / max(count, 1),
        "ece_15_bin": ece,
        "brier_score": brier_sum / max(count, 1),
        "targets": count,
    }


@torch.no_grad()
def task_metrics(
    model,
    task,
    distance,
    device,
    seed,
    *,
    batches: int,
    batch_size: int,
):
    corpus = PredictiveTaskCorpus(
        task,
        seed=seed,
        distance=distance,
        pairs=8,
    )
    correct = count = 0
    loss_sum = 0.0
    seq_len = max(128, distance + 64)
    for _ in range(batches):
        batch = corpus.sample_batch(batch_size, seq_len, "test").to(device)
        logits, _ = model(batch.tokens)
        prediction = logits[:, :-1].argmax(-1)
        targets = batch.tokens[:, 1:]
        valid = batch.loss_mask[:, 1:]
        losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        correct += int(((prediction == targets) & valid).sum())
        count += int(valid.sum())
        loss_sum += float(losses[valid].sum())
    return {
        "accuracy": correct / max(count, 1),
        "nll": loss_sum / max(count, 1),
        "targets": count,
    }


@torch.no_grad()
@torch.no_grad()
def typo_prompt_metrics(model, windows, tokenizer, detector, device):
    clean_losses, typo_losses = [], []
    for row in windows[:16]:
        split = max(2, row.numel() // 2)
        prompt = row[:split].tolist()
        reference = row[split:].to(device)
        text = tokenizer.decode(prompt)
        characters = list(text)
        for index in range(9, len(characters), 19):
            if characters[index].isalnum():
                characters[index] = "?"
        typo_ids = tokenizer.encode("".join(characters))
        for prompt_ids, output in (
            (prompt, clean_losses),
            (typo_ids, typo_losses),
        ):
            if not prompt_ids:
                continue
            joined = torch.tensor(
                [prompt_ids + reference.cpu().tolist()],
                dtype=torch.long,
                device=device,
            )
            logits, _ = model(
                joined,
                **model_kwargs(model, joined, detector),
            )
            start = len(prompt_ids) - 1
            prediction = logits[:, start : start + len(reference)]
            output.append(
                float(
                    F.cross_entropy(
                        prediction.reshape(-1, prediction.shape[-1]),
                        reference.reshape(-1),
                    )
                )
            )
    clean = statistics.fmean(clean_losses)
    typo = statistics.fmean(typo_losses)
    return {"clean_nll": clean, "typo_nll": typo, "delta_nll": typo - clean}


def robustness_metrics(model, windows, tokenizer, detector, device, seed):
    generator = torch.Generator().manual_seed(seed)
    output = {}
    for probability in (0.05, 0.10, 0.20):
        corrupted = windows.clone()
        mask = torch.rand(corrupted.shape, generator=generator) < probability
        replacements = torch.randint(
            0, model.config.vocab_size, corrupted.shape, generator=generator
        )
        corrupted[mask] = replacements[mask]
        output[f"token_corruption_{probability}"] = window_metrics(
            model, corrupted, device=device, detector=detector
        )
    for prefix_length in (16, 64, 128):
        prefix = torch.randint(
            0,
            model.config.vocab_size,
            (len(windows), prefix_length),
            generator=generator,
        )
        joined = torch.cat((prefix, windows), dim=1)
        target_mask = torch.zeros_like(joined, dtype=torch.bool)
        target_mask[:, prefix_length:] = True
        output[f"irrelevant_prefix_{prefix_length}"] = window_metrics(
            model,
            joined,
            device=device,
            detector=detector,
            target_mask=target_mask,
        )
    for keep in (64, 128, 256):
        if keep < windows.shape[1]:
            output[f"truncated_{keep}"] = window_metrics(
                model,
                windows[:, -keep:],
                device=device,
                detector=detector,
            )
    output["text_typo_prompt"] = typo_prompt_metrics(
        model, windows, tokenizer, detector, device
    )
    return output


def state_metrics(model: PureParallelGearLM, tokens, detector, device):
    records = model.diagnostics(
        tokens.to(device),
        sentence_end_mask=boundary_masks(tokens, detector).to(device),
    )
    output = []
    for record in records:
        rotor = record["rotor"].float().flatten(1)
        centered = rotor - rotor.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered.cpu()) if len(rotor) > 1 else torch.zeros(1)
        energy = singular.square()
        effective_rank = (
            float(energy.sum().square() / energy.square().sum().clamp_min(1e-12))
            if bool(energy.any())
            else 0.0
        )
        omega = record["omega"].float()
        clutch = record["clutch"].float()
        output.append(
            {
                "rotor_energy_mean": float(record["rotor_energy"].float().mean()),
                "rotor_energy_std": float(record["rotor_energy"].float().std()),
                "phase_spread": float(
                    torch.atan2(
                        record["rotor"][..., 1],
                        record["rotor"][..., 0],
                    ).float().std()
                ),
                "omega_saturation_fraction": float(
                    (
                        omega.abs() >= 0.9 * model.config.omega_limit
                    ).float().mean()
                ),
                "dead_gear_fraction": float(
                    (
                        (clutch.mean(dim=0) < 0.01)
                        | (clutch.mean(dim=0) > 0.99)
                    )
                    .float()
                    .mean()
                ),
                "clutch_mean": float(clutch.mean()),
                "coupling_activity": float(record["coupling_activity"]),
                "effective_state_rank": effective_rank,
            }
        )
    return output


def gradient_health(model, tokens, detector, device):
    model.train()
    model.zero_grad(set_to_none=True)
    metadata = {}
    if isinstance(model, PureParallelGearLM):
        metadata["sentence_end_mask"] = boundary_masks(tokens, detector).to(device)
    loss = model.training_step(tokens.to(device), metadata)["total"]
    loss.backward()
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            rows.append(
                {
                    "name": name,
                    "missing": parameter.grad is None,
                    "norm": (
                        None
                        if parameter.grad is None
                        else float(parameter.grad.float().norm())
                    ),
                    "finite": (
                        False
                        if parameter.grad is None
                        else bool(torch.isfinite(parameter.grad).all())
                    ),
                }
            )
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "loss": float(loss.detach()),
        "missing": [row["name"] for row in rows if row["missing"]],
        "nonfinite": [row["name"] for row in rows if not row["finite"]],
        "zero": [row["name"] for row in rows if row["norm"] == 0.0],
        "smallest": sorted(
            (row for row in rows if row["norm"] is not None),
            key=lambda row: row["norm"],
        )[:10],
    }


@torch.no_grad()
def efficiency_profile(model, vocab_size, device, lengths, repeats):
    contexts = []
    for length in lengths:
        try:
            samples = efficiency_samples(
                model,
                vocab_size=vocab_size,
                seq_len=length,
                device=device,
                repeats=repeats,
            )
            prefill = samples["prefill_times"]
            generation = samples["incremental_times"]
            cache_size = samples["cache_bytes"]
            contexts.append(
                {
                    "length": length,
                    "prefill_p50_seconds": percentile(prefill, 0.5),
                    "prefill_p95_seconds": percentile(prefill, 0.95),
                    "incremental_p50_seconds": percentile(generation, 0.5),
                    "incremental_p95_seconds": percentile(generation, 0.95),
                    "incremental_tokens_per_second": samples["incremental_steps"]
                    / percentile(generation, 0.5),
                    "cache_bytes": cache_size,
                    "cache_bytes_per_context_token": cache_size / length,
                }
            )
        except RuntimeError as error:
            contexts.append(
                {"length": length, "failed": True, "error": str(error)}
            )
    return {"contexts": contexts}


def training_memory(model, vocab_size, device):
    if torch.device(device).type != "mps":
        return {"available": False, "reason": "MPS allocator required"}
    torch.mps.synchronize()
    torch.mps.empty_cache()
    before = int(torch.mps.current_allocated_memory())
    tokens = torch.randint(0, vocab_size, (2, 512), device=device)
    model.zero_grad(set_to_none=True)
    model.training_step(tokens)["total"].backward()
    torch.mps.synchronize()
    after = int(torch.mps.current_allocated_memory())
    model.zero_grad(set_to_none=True)
    return {
        "available": True,
        "peak_available": False,
        "measurement_kind": "allocator_snapshot_after_backward_not_peak",
        "peak_proxy_bytes": after,
        "snapshot_after_backward_bytes": after,
        "delta_bytes": after - before,
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }


def profiler_contract(model, vocab_size, device):
    profile_model = (
        model
        if torch.device(device).type == "cpu"
        else copy.deepcopy(model).to("cpu").eval()
    )
    tokens = torch.randint(0, vocab_size, (1, 256))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as profile:
        profile_model(tokens)
    rows = sorted(
        (
            {
                "op": event.key,
                "self_cpu_us": event.self_cpu_time_total,
                "cpu_memory_bytes": event.cpu_memory_usage,
            }
            for event in profile.key_averages()
        ),
        key=lambda row: row["self_cpu_us"],
        reverse=True,
    )
    forbidden_patterns = ("scaled_dot_product", "attention")
    violations = [
        row["op"]
        for row in rows
        if isinstance(profile_model, PureParallelGearLM)
        and any(pattern in row["op"].lower() for pattern in forbidden_patterns)
    ]
    sequence_square = []
    if isinstance(profile_model, PureParallelGearLM):
        for event in profile.events():
            for shape in event.input_shapes:
                if (
                    len(shape) >= 2
                    and shape[-1] == 256
                    and shape[-2] == 256
                ):
                    sequence_square.append(
                        {"op": event.name, "shape": list(shape)}
                    )
    return {
        "available": True,
        "top_operations": rows[:30],
        "forbidden_operation_violations": violations,
        "sequence_square_violations": sequence_square,
        "passed": not violations and not sequence_square,
    }


@torch.no_grad()
def ablation_metrics(model, windows, detector, device):
    if not isinstance(model, PureParallelGearLM):
        return None
    tokens = windows.to(device)
    targets = tokens[:, 1:]
    boundary = boundary_masks(windows, detector).to(device)
    full = model.component_logits(tokens, sentence_end_mask=boundary)[:, :-1]
    full_nll = float(
        F.cross_entropy(full.reshape(-1, full.shape[-1]), targets.reshape(-1))
    )
    output = {"full": {"nll": full_nll}}
    for name in ABLATIONS:
        disabled = (name,)
        if name == "one_bank_only":
            # Bank masking is represented by disabling cross-bank composition;
            # a separately configured one-bank model is required for decisive runs.
            output[name] = {
                "requires_retrained_model": True,
                "reason": "shape-changing ablation cannot be validly post-hoc",
            }
            continue
        logits = model.component_logits(
            tokens,
            disabled,
            sentence_end_mask=boundary,
        )[:, :-1]
        nll = float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
        )
        output[name] = {"nll": nll, "delta_nll": nll - full_nll}
    output["punctuation_vs_forced_fixed"] = {
        "requires_retrained_model": True,
        "reason": "boundary-policy ablation must be trained, not only evaluated",
    }
    return output


def exhaustive_windows(tokens, length):
    rows, masks = [], []
    offset = 0
    while offset < len(tokens) - 1:
        row = tokens[offset : offset + length]
        real = len(row)
        if real < length:
            row = F.pad(row, (0, length - real))
        mask = torch.zeros(length, dtype=torch.bool)
        mask[1:real] = True
        rows.append(row)
        masks.append(mask)
        if offset + real >= len(tokens):
            break
        offset += real - 1
    return torch.stack(rows), torch.stack(masks)


@torch.no_grad()
def minimal_pairs(model, tokenizer, path, device, detector):
    correct = 0
    margins = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            good = row.get("good", row.get("sentence_good"))
            bad = row.get("bad", row.get("sentence_bad"))
            scores = []
            for text in (good, bad):
                ids = tokenizer.encode(text)
                tokens = torch.tensor([ids], device=device)
                logits, _ = model(
                    tokens,
                    **model_kwargs(model, tokens, detector),
                )
                scores.append(
                    float(
                        F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            tokens[:, 1:].reshape(-1),
                            reduction="sum",
                        )
                    )
                )
            margin = scores[1] - scores[0]
            margins.append(margin)
            correct += int(margin > 0)
    return {
        "accuracy": correct / max(len(margins), 1),
        "mean_bad_minus_good_nll": statistics.fmean(margins),
        "pairs": len(margins),
    }


@torch.no_grad()
def paraphrase_metrics(model, tokenizer, path, device, detector):
    differences = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            texts = (row["original"], row["paraphrase"])
            scores = []
            for text in texts:
                ids = tokenizer.encode(text)
                tokens = torch.tensor([ids], dtype=torch.long, device=device)
                logits, _ = model(
                    tokens,
                    **model_kwargs(model, tokens, detector),
                )
                scores.append(
                    float(
                        F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            tokens[:, 1:].reshape(-1),
                        )
                    )
                )
            differences.append(abs(scores[0] - scores[1]))
    return {
        "mean_absolute_nll_difference": statistics.fmean(differences),
        "p95_absolute_nll_difference": percentile(differences, 0.95),
        "pairs": len(differences),
    }


def main() -> None:
    args = parse_args()
    named_paths = dict(value.split("=", 1) for value in args.checkpoint)
    models = {
        name: load_model(Path(path), args.device)
        for name, path in named_paths.items()
    }
    manifests = dict(value.split("=", 1) for value in args.evaluation_manifest)
    external = dict(value.split("=", 1) for value in args.external_token_file)
    pair_files = dict(value.split("=", 1) for value in args.minimal_pair_file)
    paraphrase_files = dict(
        value.split("=", 1) for value in args.paraphrase_file
    )
    corpus = EduCombinedCorpus(
        root=str(args.corpus_root),
        tokenizer_name=args.tokenizer_name,
        domains=list(DOMAINS),
        load_tokenizer=True,
        seed=args.seed,
    )
    detector = SentenceBoundaryDetector(corpus.tokenizer)
    for model in models.values():
        if isinstance(model, PureParallelGearLM):
            model.configure_boundary_detector(corpus.tokenizer)
    report: dict[str, Any] = {
        "protocol": {
            **vars(args),
            "corpus_root": str(args.corpus_root),
            "output": str(args.output),
        },
        "models": {},
    }
    for model_name, model in models.items():
        result: dict[str, Any] = {
            "complete_manifests": {
                name: evaluate_manifest(
                    model,
                    Path(path),
                    batch_size=8,
                    device=args.device,
                )
                for name, path in manifests.items()
            },
            "natural": {},
            "predictive_tasks": {},
            "external_frozen_tokenizer": {},
            "minimal_pairs": {},
            "paraphrase_robustness": {},
            "efficiency": efficiency_profile(
                model,
                corpus.vocab_size,
                args.device,
                args.efficiency_lengths,
                args.efficiency_repeats,
            ),
            "training_memory": training_memory(
                model, corpus.vocab_size, args.device
            ),
            "profiler_contract": profiler_contract(
                model, corpus.vocab_size, args.device
            )
            if args.profile
            else {"available": False, "reason": "--profile not requested"},
        }
        diagnostic_windows = None
        for domain in DOMAINS:
            tokens = torch.load(
                args.corpus_root
                / domain
                / f"test_{args.tokenizer_name}.pt",
                map_location="cpu",
            ).long().flatten()
            domain_result = {}
            for length in args.context_lengths:
                if length >= len(tokens):
                    continue
                windows = fixed_windows(
                    tokens,
                    length,
                    min(args.windows_per_domain, max(1, len(tokens) // length)),
                )
                domain_result[str(length)] = window_metrics(
                    model,
                    windows,
                    device=args.device,
                    detector=detector,
                )
            base = fixed_windows(
                tokens,
                min(512, len(tokens) - 1),
                min(
                    args.robustness_windows,
                    max(1, len(tokens) // 512),
                ),
            )
            diagnostic_windows = base[: min(8, len(base))]
            domain_result["robustness"] = robustness_metrics(
                model,
                base,
                corpus.tokenizer,
                detector,
                args.device,
                args.seed,
            )
            result["natural"][domain] = domain_result
        for task in TASKS:
            result["predictive_tasks"][task] = {
                str(distance): task_metrics(
                    model,
                    task,
                    distance,
                    args.device,
                    args.seed + distance,
                    batches=args.task_batches,
                    batch_size=args.task_batch_size,
                )
                for distance in args.task_distances
            }
        for name, path in external.items():
            tokens = torch.load(path, map_location="cpu").long().flatten()
            windows, target_mask = exhaustive_windows(
                tokens, min(4096, len(tokens))
            )
            result["external_frozen_tokenizer"][name] = window_metrics(
                model,
                windows,
                device=args.device,
                detector=detector,
                target_mask=target_mask,
            )
        for name, path in pair_files.items():
            result["minimal_pairs"][name] = minimal_pairs(
                model,
                corpus.tokenizer,
                Path(path),
                args.device,
                detector,
            )
        for name, path in paraphrase_files.items():
            result["paraphrase_robustness"][name] = paraphrase_metrics(
                model,
                corpus.tokenizer,
                Path(path),
                args.device,
                detector,
            )
        assert diagnostic_windows is not None
        result["gradient_health"] = gradient_health(
            model, diagnostic_windows, detector, args.device
        )
        result["mechanism"] = (
            state_metrics(
                model,
                diagnostic_windows,
                detector,
                args.device,
            )
            if isinstance(model, PureParallelGearLM)
            else None
        )
        result["ablations"] = ablation_metrics(
            model, diagnostic_windows, detector, args.device
        )
        report["models"][model_name] = result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
