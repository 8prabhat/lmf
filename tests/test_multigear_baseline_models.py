"""Cross-family checks for the mecm/mcpm/mgcf/mrwt MultiGear baseline models.

Per-architecture behavior is covered in test_mecm.py, test_mcpm.py,
test_mgcf.py, and test_mrwt.py; this file only checks registry wiring and the
generic build-from-config path shared by all four.
"""

from __future__ import annotations

import torch

from lmf.core.build import build
from lmf.core.config import ExperimentConfig
from lmf.core.registry import MODELS, TRAINERS


def test_multigear_baseline_registries_are_populated():
    for name in ("mecm", "mcpm", "mgcf", "mrwt"):
        assert name in MODELS
        assert name in TRAINERS


def test_build_from_config_for_all_multigear_baseline_models():
    for name in ("mecm", "mcpm", "mrwt"):
        cfg = ExperimentConfig(
            {
                "seed": 0,
                "device": "cpu",
                "precision": "fp32",
                "data": {"name": "procedural", "vocab_size": 64},
                "model": {
                    "name": name,
                    "vocab_size": 64,
                    "dim": 32,
                    "layers": 2,
                    **({"heads": 4} if name == "mrwt" else {"kernel_size": 5}),
                    "max_seq_len": 64,
                },
                "trainer": {"name": name, "lr": 1e-3, "total_steps": 2, "warmup_steps": 1},
                "run": {"batch_size": 2, "seq_len": 24, "steps": 2},
            },
            "test",
        )
        corpus, model, trainer, run = build(cfg)
        assert corpus.vocab_size == 64
        assert torch.isfinite(model.training_step(corpus.sample_tokenized(2, 24))["total"])
        assert run["steps"] == 2
