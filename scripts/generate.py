#!/usr/bin/env python
"""Thin wrapper: generate from a model."""

from __future__ import annotations

import sys

from lmf.cli.main import main

if __name__ == "__main__":
    main(["generate", *sys.argv[1:]])
