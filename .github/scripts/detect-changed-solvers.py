#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect changed Tesseract solvers from a git diff.

Reads changed file paths from stdin (one per line) and prints a JSON
matrix of ``[{"domain": "...", "solver": "..."}]`` entries for solvers
whose directories contain changed files.

Prints ``[]`` if nothing changed.

Usage (in CI):
    git diff --name-only BASE HEAD | python .github/scripts/detect-changed-solvers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

TESSERACTS_DIR = Path("mosaic/tesseracts")


def main() -> None:
    diff_files = [line.strip() for line in sys.stdin if line.strip()]
    if not diff_files:
        print("[]")
        return

    # Match changed files to solver directories (any dir with
    # tesseract_config.yaml).
    changed: list[dict[str, str]] = []
    for cfg_path in sorted(TESSERACTS_DIR.rglob("tesseract_config.yaml")):
        solver_rel = str(cfg_path.parent.relative_to(TESSERACTS_DIR))
        with open(cfg_path) as f:
            mosaic_meta = ((yaml.safe_load(f) or {}).get("metadata") or {}).get(
                "mosaic"
            ) or {}
        prefix = f"mosaic/tesseracts/{solver_rel}/"
        if not any(f.startswith(prefix) for f in diff_files):
            continue
        parts = solver_rel.split("/", 1)
        entry: dict[str, str]
        if len(parts) == 2:
            entry = {"domain": parts[0], "solver": parts[1]}
        else:
            # Top-level solver (no domain subdirectory)
            entry = {"domain": parts[0], "solver": parts[0]}
        if mosaic_meta.get("name"):
            entry["display_name"] = mosaic_meta["name"]
        changed.append(entry)

    json.dump(changed, sys.stdout)
    print()


if __name__ == "__main__":
    main()
