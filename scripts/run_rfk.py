#!/usr/bin/env python
"""Thin wrapper: run the falsification kernels."""

from __future__ import annotations

import sys

from lmf.cli.main import main

if __name__ == "__main__":
    main(["rfk", *sys.argv[1:]])
