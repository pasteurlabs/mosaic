#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map git-diff file paths to Mosaic problem names.

Reads changed file paths from stdin (one per line) and prints a
comma-separated list of affected problem names (or "all" if core
harness files changed, or "" if nothing benchmark-relevant changed).

Uses the live problem registry so new problems/tesseract dirs are
picked up automatically — no hardcoded domain lists.

Usage (in CI):
    git diff --name-only HEAD~1 HEAD | python .github/scripts/detect-changed-problems.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.problems import PROBLEMS as ALL
from mosaic.benchmarks.problems import get_config

_PROBLEMS_DIR = Path("mosaic/benchmarks/problems")


def _pkg_dir_to_cfg_name() -> dict[str, str]:
    """Map ``<problem_pkg_dirname>/`` -> ``cfg.name``.

    The CLI-facing slug ``cfg.name`` (e.g. ``"ns-grid"``) is not derivable
    from the package directory name (``"navier_stokes_grid"``); we have to
    import each candidate to read it.
    """
    out: dict[str, str] = {}
    for entry in sorted(_PROBLEMS_DIR.iterdir()):
        if entry.name.startswith("_") or entry.name == "__pycache__":
            continue
        # Resolve the same module path the registry would import.
        if entry.is_file() and entry.suffix == ".py":
            mod_name = f"mosaic.benchmarks.problems.{entry.stem}"
            key = f"mosaic/benchmarks/problems/{entry.name}"
        elif entry.is_dir() and (entry / "config.py").exists():
            mod_name = f"mosaic.benchmarks.problems.{entry.name}.config"
            key = f"mosaic/benchmarks/problems/{entry.name}/"
        elif entry.is_dir() and (entry / "__init__.py").exists():
            mod_name = f"mosaic.benchmarks.problems.{entry.name}"
            key = f"mosaic/benchmarks/problems/{entry.name}/"
        else:
            continue
        try:
            cfg = getattr(importlib.import_module(mod_name), "problem", None)
        except Exception:
            continue
        if isinstance(cfg, Problem):
            out[key] = cfg.name
    return out


diff_files = [line.strip() for line in sys.stdin if line.strip()]
if not diff_files:
    print("")
    sys.exit(0)

# Core harness changes trigger everything. ``problems/shared/`` is the
# new home of the per-suite runners (forward, gradient, cost, optimization)
# that used to live under ``suites/``; ``mosaic_shared/``
# is the renamed/relocated ``mosaic_shared/``.
CORE_PREFIXES = (
    "mosaic/benchmarks/core/",
    "mosaic/benchmarks/problems/shared/",
    "mosaic/mosaic_shared/",
)
if any(f.startswith(CORE_PREFIXES) for f in diff_files):
    print("all")
    sys.exit(0)

# Build reverse maps:
#   tesseract_dir_suffix -> [problem_names]
#   problem_pkg_prefix   -> problem_name  (any file under the dir triggers it)
tdir_to_problems: dict[str, list[str]] = {}
pkg_to_problem: dict[str, str] = _pkg_dir_to_cfg_name()

for name in ALL:
    cfg = get_config(name)
    # tesseract_dir is absolute; extract relative path under repo root.
    # Multiple problems may share one tesseract dir (ns-grid + ns-3d-grid).
    tdir = str(cfg.tesseract_dir)
    idx = tdir.find("mosaic/tesseracts/")
    if idx >= 0:
        suffix = tdir[idx:]
        tdir_to_problems.setdefault(suffix, []).append(name)

changed: set[str] = set()
for f in diff_files:
    matched = False
    for prefix, name in pkg_to_problem.items():
        if f.startswith(prefix):
            changed.add(name)
            matched = True
            break
    if matched:
        continue
    for prefix, problems in tdir_to_problems.items():
        if f.startswith(prefix + "/"):
            changed.update(problems)
            break

print(",".join(sorted(changed)) if changed else "")
