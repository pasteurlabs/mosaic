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

    # Discover all solver dirs (any dir with tesseract_config.yaml)
    solver_dirs: list[str] = []
    for cfg in sorted(TESSERACTS_DIR.rglob("tesseract_config.yaml")):
        rel = str(cfg.parent.relative_to(TESSERACTS_DIR))
        solver_dirs.append(rel)

    # Match changed files to solver directories
    changed: list[dict[str, str]] = []
    seen: set[str] = set()
    for solver_rel in solver_dirs:
        prefix = f"mosaic/tesseracts/{solver_rel}/"
        if any(f.startswith(prefix) for f in diff_files) and solver_rel not in seen:
            seen.add(solver_rel)
            parts = solver_rel.split("/", 1)
            entry: dict[str, str]
            if len(parts) == 2:
                entry = {"domain": parts[0], "solver": parts[1]}
            else:
                # Top-level solver (no domain subdirectory)
                entry = {"domain": parts[0], "solver": parts[0]}
            # Read the display name from tesseract_config.yaml if available.
            cfg_path = TESSERACTS_DIR / solver_rel / "tesseract_config.yaml"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    tcfg = yaml.safe_load(f) or {}
                display = (tcfg.get("metadata") or {}).get("mosaic", {}).get("name", "")
                if display:
                    entry["display_name"] = display
            changed.append(entry)

    json.dump(changed, sys.stdout)
    print()


if __name__ == "__main__":
    main()
