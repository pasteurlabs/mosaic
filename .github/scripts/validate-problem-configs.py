#!/usr/bin/env python3
"""Validate all registered Mosaic problem configs.

Loads every problem from the registry and calls ``.validate()`` on it.
Exits non-zero on the first validation failure.

Usage (in CI):
    python .github/scripts/validate-problem-configs.py
"""

from __future__ import annotations

from mosaic.benchmarks.problems import PROBLEMS, get_config

for p in PROBLEMS:
    get_config(p).validate()

print(f"Validated {len(PROBLEMS)} problem configs")
