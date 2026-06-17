#!/usr/bin/env python
"""Thin wrapper: evaluate a model (BPT + structural benchmarks)."""

from __future__ import annotations

import sys

from lmf.cli.main import main

if __name__ == "__main__":
    main(["eval", *sys.argv[1:]])
