#!/usr/bin/env python
"""Train the Multi-Rate Latent Gear Transformer config.

Examples:
    python scripts/train_gear_transformer.py --block smoke --steps 10
    python scripts/train_gear_transformer.py --block sentencepiece_smoke --checkpoint outputs/mlgt.pt
"""

from __future__ import annotations

import sys

from lmf.cli.main import main


if __name__ == "__main__":
    main(["train", "--config", "configs/gear_transformer.yaml", *sys.argv[1:]])
