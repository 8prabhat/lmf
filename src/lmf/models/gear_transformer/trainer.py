"""Trainer registration for the Gear Transformer family."""

from __future__ import annotations

import torch

from ...core.registry import TRAINERS
from ..transformer.trainer import TransformerTrainer


class GearTransformerTrainer(TransformerTrainer):
    """Transformer trainer with staged gear activation and faster gear updates."""

    def __init__(self, model, corpus, **kwargs) -> None:
        curriculum_lengths = kwargs.pop("seq_len_curriculum", ())
        curriculum_steps = kwargs.pop("seq_len_curriculum_steps", ())
        self.trunk_freeze_steps = int(kwargs.pop("trunk_freeze_steps", 0))
        if self.trunk_freeze_steps < 0:
            raise ValueError("trunk_freeze_steps must be non-negative")
        self.seq_len_curriculum = tuple(int(v) for v in curriculum_lengths)
        self.seq_len_curriculum_steps = tuple(int(v) for v in curriculum_steps)
        if bool(self.seq_len_curriculum) != bool(self.seq_len_curriculum_steps):
            raise ValueError(
                "seq_len_curriculum and seq_len_curriculum_steps must be set together"
            )
        if len(self.seq_len_curriculum) != len(self.seq_len_curriculum_steps):
            raise ValueError("sequence curriculum lengths and steps must match")
        if self.seq_len_curriculum_steps and (
            self.seq_len_curriculum_steps[0] != 0
            or any(
                left >= right
                for left, right in zip(
                    self.seq_len_curriculum_steps,
                    self.seq_len_curriculum_steps[1:],
                )
            )
        ):
            raise ValueError(
                "sequence curriculum steps must start at zero and increase"
            )
        if self.seq_len_curriculum and kwargs.get("prefetch", False):
            raise ValueError("sequence curriculum is incompatible with fixed prefetch")
        weight_decay = float(kwargs.get("weight_decay", 0.01))
        super().__init__(model, corpus, **kwargs)
        multiplier = float(getattr(model.config, "gear_lr_multiplier", 1.0))
        gear_parameters = []
        trunk_parameters = []
        for name, parameter in self.raw_model.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                name.startswith("shared_gears.")
                or ".gears." in name
                or ".gear_norm." in name
                or name.startswith("future")
                or name.startswith("consistency")
            ):
                gear_parameters.append(parameter)
            else:
                trunk_parameters.append(parameter)
        groups = [
            {
                "params": trunk_parameters,
                "lr": self.lr,
                "group_role": "trunk",
            }
        ]
        if gear_parameters:
            groups.append(
                {
                    "params": gear_parameters,
                    "lr": self.lr * multiplier,
                    "lr_multiplier": multiplier,
                    "group_role": "gear",
                }
            )
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=self.lr,
            weight_decay=weight_decay,
            fused=self.device.type == "cuda",
        )

    def batch_metadata(self, step: int) -> dict:
        return {
            **super().batch_metadata(step),
            "training_step": int(step),
            "sequence_length": self.effective_seq_len(
                self.seq_len_curriculum[-1]
                if self.seq_len_curriculum
                else 0,
                step,
            ),
        }

    def effective_seq_len(self, requested_seq_len: int, step: int) -> int:
        if not self.seq_len_curriculum:
            return super().effective_seq_len(requested_seq_len, step)
        selected = self.seq_len_curriculum[0]
        for boundary, length in zip(
            self.seq_len_curriculum_steps,
            self.seq_len_curriculum,
        ):
            if step < boundary:
                break
            selected = length
        return min(int(requested_seq_len), selected)

    def param_group_lr_multiplier(self, group: dict, step: int) -> float:
        if (
            group.get("group_role") == "trunk"
            and step < self.trunk_freeze_steps
        ):
            return 0.0
        return super().param_group_lr_multiplier(group, step)


@TRAINERS.register("gear_transformer")
def build_gear_transformer_trainer(model, corpus, **kwargs) -> GearTransformerTrainer:
    return GearTransformerTrainer(model, corpus, **kwargs)


@TRAINERS.register("mlgt")
def build_multi_rate_latent_gear_transformer_trainer(model, corpus, **kwargs) -> GearTransformerTrainer:
    return build_gear_transformer_trainer(model, corpus, **kwargs)


@TRAINERS.register("gear_only")
def build_gear_only_trainer(model, corpus, **kwargs) -> GearTransformerTrainer:
    return build_gear_transformer_trainer(model, corpus, **kwargs)


@TRAINERS.register("simplified_gear_transformer")
def build_simplified_gear_transformer_trainer(
    model,
    corpus,
    **kwargs,
) -> GearTransformerTrainer:
    return build_gear_transformer_trainer(model, corpus, **kwargs)
