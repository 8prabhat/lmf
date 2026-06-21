# Stacked Parallel Gear Transformer V5

V5 implements gears as functional multi-scale state controllers rather than
phase-conditioned cosmetic adapters.

## Topology

The efficient configuration has two independent parallel banks:

```text
Transformer block 0
  -> fast gear bank (stride 1)
  -> gated carrier
Transformer blocks
  -> slow gear bank (stride 4)
  -> LM trunk and future head
```

Each bank contains parallel local, phrase, semantic, and discourse lanes.
Three-bank/nine-gear configurations are supported through
`gear_layer_strategy: stacked_parallel`.

## Per-bank update

1. A vectorized affine scan builds the causal context available before the
   current token.
2. A drive-only predictor produces positive phase advances.
3. Sparse adjacent and lane-anchor gear edges correct those advances with a
   sinusoidal tooth-ratio error.
4. Phase-addressed slots select the current latent mode.
5. Paired memory dimensions undergo real 2-D geometric rotation.
6. A chunk-stable closed-form affine rotation scan updates persistent memory.
7. Gears fuse within lanes; lane attention and routing fuse the bank output.
8. The output is returned to the Transformer and emitted as a gated carrier for
   the next bank.

Slow banks update only on their active temporal stride. Their phase and causal
retention account for the number of elapsed tokens, and the most recent output
is held between updates. Cached decoding and full-sequence execution are
numerically equivalent.

## Training

- A Transformer trunk can be warm-started with
  `initialize_trunk_from_transformer`.
- Gear parameters use their own learning-rate multiplier.
- Gear, phase, auxiliary, and future paths have independent warmup/ramp
  schedules.
- Sequence-length curricula are provided through the trainer.
- Lane/context latent prediction, sampled lane/future token prediction,
  phase-lock, diversity, usage balance, consistency, and calibration losses
  are independently weighted.
- Expensive auxiliary and future objectives can run at intervals while the LM
  and gear state paths remain active every scheduled bank step.

## Evaluation contract

`component_ablation_metrics` disables one mechanism at a time without mutating
the checkpoint. It reports NLL change and top-1 prediction change for:

- complete gears;
- phase and geometric rotation;
- sparse phase coupling;
- causal temporal context;
- explicit inter-bank coupling;
- lane mixing;
- future prediction;
- every individual bank.

The benchmark also reports fixed-corpus predictions, parameter counts,
parameter-matched baseline loss, training time, training-token count, and
forward/backward throughput. Both models are initialized from scratch and see
the same sampled windows:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_gear_transformer_v5.py
```

The June 19, 2026 audit also removed trainable capacity that could never affect
the output: dense masked coupling entries, padded slot rows, the first bank's
unused inter-bank projection, and the final bank's unused outgoing carrier.
Representative full-loss backward passes now give every remaining trainable
parameter a finite, non-zero gradient.

In a 100-step falsification run on the repeated narrative corpus, the gear model
had 302,716 parameters and the matched Transformer had 296,820 (+1.99% for
gear). Validation NLL was 5.5383 for gear versus 5.3288 for the Transformer.
Gear throughput was 31% of baseline forward throughput and 34% of training
throughput. Every Transformer block had positive acute ablation impact, but
temporal context, lane mixing, and the slow bank were slightly harmful at this
short checkpoint. Disabling temporal context plus lane mixing improved NLL only
from 5.53830 to 5.53807, which is too small and too under-replicated to justify
changing the default.

V5 therefore remains experimental. A credible positive claim requires at least
three seeds on non-repeated data, matched parameters and training tokens, and
separate matched-wall-clock results.
