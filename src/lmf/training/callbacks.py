"""Training callbacks — the Open/Closed seam for periodic side-tasks.

The base Trainer's loop never names a concrete side-task (eval, calibration,
checkpointing). Instead it invokes callbacks at step boundaries, so new periodic
behaviour is added by writing a Callback, not by editing the loop.
"""

from __future__ import annotations

from typing import Any, Protocol


class Callback(Protocol):
    def on_step_end(self, trainer: Any, step: int, record: dict[str, float]) -> None: ...


class PeriodicCheckpoint:
    def __init__(self, path: str, every: int) -> None:
        self.path = path
        self.every = max(1, int(every))

    def on_step_end(self, trainer, step, record) -> None:
        if step % self.every == 0:
            trainer.save_checkpoint(self.path)


class PeriodicEval:
    """Run a held-out eval and stash the result on the record."""

    def __init__(self, every: int, batch_size: int, seq_len: int, n_batches: int = 10,
                 split: str = "valid") -> None:
        self.every = max(1, int(every))
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_batches = n_batches
        self.split = split

    def on_step_end(self, trainer, step, record) -> None:
        if step % self.every == 0:
            bpt = trainer.evaluate_bpt(self.batch_size, self.seq_len, self.n_batches, self.split)
            record["eval_bpt"] = bpt
            print(f"  [eval] step={step}  bpt={bpt:.4f}")
