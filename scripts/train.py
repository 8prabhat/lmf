#!/usr/bin/env python
"""Thin wrapper: train a model from a config block.

    python scripts/train.py --config configs/rhca_v4.yaml --block smoke
"""

from __future__ import annotations

import sys

from lmf.cli.main import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
