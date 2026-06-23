"""Equal-budget bake-off: Spectral Memory vs baselines, + SM-LM ablations.

Trains every model under an identical budget (same data stream, optimizer, lr,
seed, steps) and reports recall-focused metrics. Two tasks:

* ``mqar`` (default) — Multi-Query Associative Recall: a sequence of distinct
  ``key value`` pairs followed by queries (a key reappears; the model must emit
  its value). This is *content-addressable* recall — exactly what the delta rule
  targets and what fixed-filter conv / SSM memories struggle with. Accuracy is
  measured only at answer positions.
* ``echo`` — ProceduralCorpus fixed-distance positional copy. Included as a
  contrast: it needs *positional* addressing (full-attention RoPE), not content
  memory, so bounded-state models are expected to be near chance.

Reported per model: parameters, final train loss, validation bits/token, overall
next-token accuracy, recall accuracy at answer/echo positions, and train tok/s.

Run:
    ./.venv/bin/python scripts/benchmark_spectral_memory.py --task mqar --device mps
    ./.venv/bin/python scripts/benchmark_spectral_memory.py --task echo --device mps
"""

from __future__ import annotations

import argparse
import time

import torch

from lmf.core.registry import MODELS
from lmf.data import ProceduralCorpus

VOCAB = 256
ECHO_DISTANCE = 40
ECHO_EVERY = 8
MQAR_PAIRS = 16
MQAR_QUERIES = 24


def mqar_batch(batch: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct key/value pairs, then queries; answer = value after a repeated key."""
    half = VOCAB // 2
    rows, masks = [], []
    for _ in range(batch):
        keys = torch.randperm(half, generator=gen)[:MQAR_PAIRS]
        values = half + torch.randint(0, VOCAB - half, (MQAR_PAIRS,), generator=gen)
        seq, amask = [], []
        for i in range(MQAR_PAIRS):                       # context: k v k v ...
            seq += [int(keys[i]), int(values[i])]
            amask += [False, False]
        for qi in torch.randint(0, MQAR_PAIRS, (MQAR_QUERIES,), generator=gen).tolist():
            seq += [int(keys[qi]), int(values[qi])]       # query key, then its value
            amask += [False, True]                        # the value is the answer
        rows.append(seq)
        masks.append(amask)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_models() -> dict[str, torch.nn.Module]:
    """Param-matched headline contenders + SM-LM ablations."""
    sm = dict(
        dim=192, layers=4, banks=4, head_dim=48,
        attn_heads=4, attention_layers=[2], window=24, chunk=32, mlp_ratio=2,
    )
    return {
        # --- headline baselines (param-matched) ---
        "transformer": MODELS.create("transformer", dict(dim=192, layers=4, heads=4), VOCAB),
        "bounded_transformer": MODELS.create(
            "bounded_transformer", dict(dim=192, layers=4, heads=4, kv_heads=2, attention_window=24), VOCAB
        ),
        "hybrid_parallel_gear": MODELS.create(
            "hybrid_parallel_gear",
            dict(dim=192, layers=4, attention_heads=4, attention_kv_heads=2, attention_window=24),
            VOCAB,
        ),
        "spectral_memory": MODELS.create("spectral_memory", dict(sm), VOCAB),
        # --- SM-LM ablations ---
        "  sm/pure_linear": MODELS.create("spectral_memory", {**sm, "attention_layers": []}, VOCAB),
        "  sm/free_decay": MODELS.create("spectral_memory", {**sm, "decay_mode": "free"}, VOCAB),
        "  sm/router_none": MODELS.create("spectral_memory", {**sm, "router": "none"}, VOCAB),
        "  sm/single_bank": MODELS.create("spectral_memory", {**sm, "banks": 1}, VOCAB),
    }


def make_sampler(task: str, seq_len: int, seed: int, split_offset: int):
    """Return ``sampler(batch) -> (tokens, answer_mask)`` with a fixed stream.

    Identical (task, seed, split_offset) reproduces the same batches across
    models, so every contender sees the same data.
    """
    if task == "mqar":
        gen = torch.Generator().manual_seed(seed + split_offset)

        def sampler(batch):
            return mqar_batch(batch, gen)

        return sampler

    corpus = ProceduralCorpus(VOCAB, seed=seed, echo_distance=ECHO_DISTANCE, echo_every=ECHO_EVERY)
    split = "train" if split_offset == 0 else "valid"
    pos = torch.arange(1, seq_len)
    echo = (pos % ECHO_EVERY == 0) & (pos >= ECHO_DISTANCE)
    echo_full = torch.cat([torch.zeros(1, dtype=torch.bool), echo])

    def sampler(batch):
        tokens = corpus.sample_tokenized(batch, seq_len, split)
        return tokens, echo_full[None, :].expand(batch, -1)

    return sampler


@torch.no_grad()
def evaluate(model, sampler, device, batches=8) -> dict:
    model.eval()
    nll = tot = 0.0
    correct = seen = rec_correct = rec_seen = 0
    for _ in range(batches):
        tokens, ans = sampler(64)
        tokens, ans = tokens.to(device), ans.to(device)
        logits, _ = model(tokens)
        pred_logits = logits[:, :-1]
        targets = tokens[:, 1:]
        ans_t = ans[:, 1:]
        nll += torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, VOCAB), targets.reshape(-1), reduction="sum"
        ).item()
        tot += targets.numel()
        preds = pred_logits.argmax(-1)
        correct += (preds == targets).sum().item()
        seen += targets.numel()
        rec_correct += ((preds == targets) & ans_t).sum().item()
        rec_seen += ans_t.sum().item()
    model.train()
    return {
        "bpt": nll / tot / 0.6931471805599453,
        "acc": correct / seen,
        "recall": rec_correct / max(rec_seen, 1),
    }


def train_one(model, task, device, steps, lr, batch, seq_len, seed) -> tuple:
    train_sampler = make_sampler(task, seq_len, seed, split_offset=0)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    torch.manual_seed(seed)
    last = 0.0
    t0 = time.time()
    n_tok = 0
    for _ in range(steps):
        tokens, _ = train_sampler(batch)
        tokens = tokens.to(device)
        n_tok += tokens.numel()
        opt.zero_grad()
        loss = model.training_step(tokens)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = loss.item()
    tok_per_s = n_tok / (time.time() - t0)
    metrics = evaluate(model, make_sampler(task, seq_len, seed, split_offset=99), device)
    return last, tok_per_s, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mqar", "echo"], default="mqar")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    seq_len = 2 * MQAR_PAIRS + 2 * MQAR_QUERIES if args.task == "mqar" else args.seq_len

    note = (
        "content-addressable recall (delta-rule home turf); SM attention window=24"
        if args.task == "mqar"
        else f"fixed-distance positional copy (echo={ECHO_DISTANCE}); SM window=24"
    )
    print(f"task={args.task}  device={device}  steps={args.steps}  batch={args.batch}  "
          f"seq_len={seq_len}\n  {note}\n")
    header = (f"{'model':22s} {'params':>9s} {'train_loss':>10s} {'val_bpt':>8s} "
              f"{'acc':>6s} {'recall':>7s} {'tok/s':>8s}")
    print(header)
    print("-" * len(header))
    for name, model in build_models().items():
        params = sum(p.numel() for p in model.parameters())
        loss, tok_s, m = train_one(
            model, args.task, device, args.steps, args.lr, args.batch, seq_len, args.seed
        )
        print(f"{name:22s} {params:9,d} {loss:10.4f} {m['bpt']:8.3f} {m['acc']:6.3f} "
              f"{m['recall']:7.3f} {tok_s:8.0f}")


if __name__ == "__main__":
    main()
