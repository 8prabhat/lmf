"""Controlled compositional and long-range tasks for Pure Parallel Gear."""

from __future__ import annotations

import torch

from ..data.batch import TrainingBatch


class PredictiveTaskCorpus:
    """Deterministic synthetic tasks with explicit supervised target positions.

    The task vocabulary is partitioned into content, value, marker, and filler
    ranges so chance accuracy and leakage are straightforward to audit.
    """

    vocab_size = 256
    supported_tasks = {
        "associative_recall",
        "induction",
        "selective_copy",
        "variable_echo",
        "noisy_recall",
        "repeated_pattern",
        "order_reversal",
        "nested_structure",
        "sentence_transition",
    }

    def __init__(
        self,
        task: str,
        *,
        seed: int = 0,
        distance: int = 64,
        pairs: int = 8,
        noise_probability: float = 0.15,
    ) -> None:
        if task not in self.supported_tasks:
            raise ValueError(f"unknown predictive task: {task}")
        self.task = task
        self.seed = int(seed)
        self.distance = int(distance)
        self.pairs = int(pairs)
        self.noise_probability = float(noise_probability)
        self._gens = {
            split: torch.Generator().manual_seed(
                self.seed + 1000 * offset
            )
            for split, offset in {"train": 1, "valid": 2, "test": 3}.items()
        }

    def _gen(self, split: str) -> torch.Generator:
        return self._gens["valid" if split == "eval" else split]

    @staticmethod
    def _empty(
        batch: int,
        seq_len: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.randint(
            160,
            256,
            (batch, seq_len),
            generator=generator,
        )
        loss = torch.zeros_like(tokens, dtype=torch.bool)
        return tokens, loss

    def _associative(
        self,
        batch: int,
        seq_len: int,
        generator: torch.Generator,
        *,
        noisy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, loss = self._empty(batch, seq_len, generator)
        distances = torch.full_like(tokens, -1)
        pair_count = min(self.pairs, 24)
        for row in range(batch):
            keys = torch.randperm(48, generator=generator)[:pair_count] + 1
            values = torch.randperm(48, generator=generator)[:pair_count] + 65
            cursor = 1
            value_positions = {}
            for key, value in zip(keys, values):
                if cursor + 1 >= seq_len:
                    break
                tokens[row, cursor] = key
                tokens[row, cursor + 1] = value
                value_positions[int(key)] = cursor + 1
                cursor += 2
            query_start = min(
                seq_len - 2 * pair_count - 1,
                max(cursor + 1, self.distance),
            )
            query_start = max(cursor + 1, query_start)
            order = torch.randperm(pair_count, generator=generator)
            for index in order.tolist():
                if query_start + 1 >= seq_len:
                    break
                key, value = keys[index], values[index]
                tokens[row, query_start] = key
                tokens[row, query_start + 1] = value
                loss[row, query_start + 1] = True
                distances[row, query_start + 1] = (
                    query_start + 1 - value_positions[int(key)]
                )
                query_start += 2
            if noisy:
                corrupt = (
                    torch.rand(seq_len, generator=generator)
                    < self.noise_probability
                )
                protected = loss[row].clone()
                protected[:-1] |= loss[row, 1:]
                corrupt &= ~protected
                replacements = torch.randint(
                    160,
                    256,
                    (seq_len,),
                    generator=generator,
                )
                tokens[row] = torch.where(
                    corrupt,
                    replacements,
                    tokens[row],
                )
        return tokens, loss, distances

    def _induction(
        self,
        batch: int,
        seq_len: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, loss = self._empty(batch, seq_len, generator)
        distances = torch.full_like(tokens, -1)
        distance = min(max(4, self.distance), seq_len - 3)
        for row in range(batch):
            cue = int(torch.randint(1, 49, (1,), generator=generator))
            answer = int(torch.randint(65, 113, (1,), generator=generator))
            tokens[row, 0] = cue
            tokens[row, 1] = answer
            query = distance
            tokens[row, query] = cue
            tokens[row, query + 1] = answer
            loss[row, query + 1] = True
            distances[row, query + 1] = query
        return tokens, loss, distances

    def _selective_copy(
        self,
        batch: int,
        seq_len: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, loss = self._empty(batch, seq_len, generator)
        distances = torch.full_like(tokens, -1)
        for row in range(batch):
            keys = torch.randperm(32, generator=generator)[: self.pairs] + 1
            values = torch.randperm(32, generator=generator)[: self.pairs] + 65
            cursor = 1
            locations = {}
            for key, value in zip(keys, values):
                tokens[row, cursor : cursor + 3] = torch.tensor(
                    [129, int(key), int(value)]
                )
                locations[int(key)] = cursor + 2
                cursor += 3
            query_key = keys[
                int(torch.randint(0, len(keys), (1,), generator=generator))
            ]
            query_position = min(seq_len - 2, max(cursor + 2, self.distance))
            tokens[row, query_position] = 130
            tokens[row, query_position + 1] = query_key
            answer_position = min(seq_len - 1, query_position + 2)
            if answer_position == query_position + 2:
                tokens[row, answer_position] = values[
                    torch.nonzero(keys == query_key)[0]
                ]
                loss[row, answer_position] = True
                distances[row, answer_position] = (
                    answer_position - locations[int(query_key)]
                )
        return tokens, loss, distances

    def _variable_echo(
        self,
        batch: int,
        seq_len: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, loss = self._empty(batch, seq_len, generator)
        distances = torch.full_like(tokens, -1)
        choices = torch.tensor([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        choices = choices[choices < seq_len - 1]
        for row in range(batch):
            for position in range(8, seq_len, 11):
                valid = choices[choices <= position]
                distance = int(
                    valid[
                        int(
                            torch.randint(
                                0,
                                len(valid),
                                (1,),
                                generator=generator,
                            )
                        )
                    ]
                )
                tokens[row, position] = tokens[row, position - distance]
                loss[row, position] = True
                distances[row, position] = distance
        return tokens, loss, distances

    def _repeated_pattern(
        self,
        batch: int,
        seq_len: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, loss = self._empty(batch, seq_len, generator)
        distances = torch.full_like(tokens, -1)
        for row in range(batch):
            width = int(torch.randint(3, 13, (1,), generator=generator))
            pattern = torch.randint(1, 113, (width,), generator=generator)
            repeated = pattern.repeat((seq_len + width - 1) // width)[:seq_len]
            tokens[row] = repeated
            loss[row, seq_len // 2 :] = True
            distances[row, seq_len // 2 :] = width
        return tokens, loss, distances

    def _order_reversal(
        self,
        batch: int,
        seq_len: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, loss = self._empty(batch, seq_len, generator)
        distances = torch.full_like(tokens, -1)
        width = min(16, max(4, (seq_len - 4) // 3))
        for row in range(batch):
            sequence = torch.randperm(48, generator=generator)[:width] + 1
            tokens[row, 1 : width + 1] = sequence
            marker = min(seq_len - width - 1, max(width + 2, self.distance))
            tokens[row, marker] = 131
            answer = sequence.flip(0)
            end = min(seq_len, marker + 1 + width)
            count = end - marker - 1
            tokens[row, marker + 1 : end] = answer[:count]
            loss[row, marker + 1 : end] = True
            distances[row, marker + 1 : end] = marker
        return tokens, loss, distances

    def _nested(
        self,
        batch: int,
        seq_len: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, loss = self._empty(batch, seq_len, generator)
        distances = torch.full_like(tokens, -1)
        open_tokens = torch.arange(133, 141)
        close_tokens = torch.arange(141, 149)
        for row in range(batch):
            stack: list[tuple[int, int]] = []
            for position in range(seq_len):
                should_close = stack and (
                    len(stack) >= 8
                    or bool(torch.rand((), generator=generator) < 0.45)
                )
                if should_close:
                    kind, opened = stack.pop()
                    tokens[row, position] = close_tokens[kind]
                    loss[row, position] = True
                    distances[row, position] = position - opened
                else:
                    kind = int(torch.randint(0, 8, (1,), generator=generator))
                    tokens[row, position] = open_tokens[kind]
                    stack.append((kind, position))
        return tokens, loss, distances

    def _sentence_transition(
        self,
        batch: int,
        seq_len: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, loss = self._empty(batch, seq_len, generator)
        distances = torch.full_like(tokens, -1)
        sentence_width = max(6, min(24, seq_len // 4))
        for row in range(batch):
            topic = int(torch.randint(1, 49, (1,), generator=generator))
            answer = int(torch.randint(65, 113, (1,), generator=generator))
            for start in range(0, seq_len - 2, sentence_width):
                end = min(seq_len - 1, start + sentence_width - 1)
                tokens[row, start] = topic
                tokens[row, end] = 132
                if end + 1 < seq_len:
                    tokens[row, end + 1] = answer
                    loss[row, end + 1] = True
                    distances[row, end + 1] = end + 1 - start
                topic, answer = answer - 64, 65 + (topic % 48)
        return tokens, loss, distances

    def sample_batch(
        self,
        batch: int,
        seq_len: int,
        split: str = "train",
    ) -> TrainingBatch:
        generator = self._gen(split)
        if self.task == "associative_recall":
            tokens, loss, distances = self._associative(
                batch, seq_len, generator, noisy=False
            )
        elif self.task == "noisy_recall":
            tokens, loss, distances = self._associative(
                batch, seq_len, generator, noisy=True
            )
        elif self.task == "induction":
            tokens, loss, distances = self._induction(
                batch, seq_len, generator
            )
        elif self.task == "selective_copy":
            tokens, loss, distances = self._selective_copy(
                batch, seq_len, generator
            )
        elif self.task == "variable_echo":
            tokens, loss, distances = self._variable_echo(
                batch, seq_len, generator
            )
        elif self.task == "repeated_pattern":
            tokens, loss, distances = self._repeated_pattern(
                batch, seq_len, generator
            )
        elif self.task == "order_reversal":
            tokens, loss, distances = self._order_reversal(
                batch, seq_len, generator
            )
        elif self.task == "nested_structure":
            tokens, loss, distances = self._nested(
                batch, seq_len, generator
            )
        else:
            tokens, loss, distances = self._sentence_transition(
                batch, seq_len, generator
            )
        attention = torch.ones_like(tokens, dtype=torch.bool)
        return TrainingBatch(
            tokens,
            attention,
            loss,
            task=self.task,
            metadata={
                "target_distance": distances,
                "task_name": self.task,
            },
        )

    def sample_tokenized(
        self,
        batch: int,
        seq_len: int,
        split: str = "train",
    ) -> torch.Tensor:
        return self.sample_batch(batch, seq_len, split).tokens

    def sampler_state(self) -> dict:
        return {
            split: generator.get_state()
            for split, generator in self._gens.items()
        }

    def load_sampler_state(self, state: dict) -> None:
        for split, value in state.items():
            self._gens[split].set_state(value)

    def diagnostics(self) -> dict:
        return {
            "type": "predictive_task",
            "task": self.task,
            "distance": self.distance,
            "pairs": self.pairs,
            "seed": self.seed,
            "vocab_size": self.vocab_size,
        }
