#!/usr/bin/env python3
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

import sys

from mosaic.benchmarks.problems import PROBLEMS as ALL
from mosaic.benchmarks.problems import get_config

diff_files = [line.strip() for line in sys.stdin if line.strip()]
if not diff_files:
    print("")
    sys.exit(0)

# Core harness changes trigger everything
CORE_PREFIXES = (
    "mosaic/benchmarks/core/",
    "mosaic/benchmarks/suites/",
    "mosaic/mosaic_shared/",
)
if any(f.startswith(CORE_PREFIXES) for f in diff_files):
    print("all")
    sys.exit(0)

# Build reverse maps: tesseract_dir_suffix -> [problem_names],
#                      config_file_path    -> problem_name
tdir_to_problems: dict[str, list[str]] = {}
config_to_problem: dict[str, str] = {}

for name in ALL:
    cfg = get_config(name)

    # tesseract_dir is absolute; extract relative path under repo root
    tdir = str(cfg.tesseract_dir)
    idx = tdir.find("mosaic/tesseracts/")
    if idx >= 0:
        suffix = tdir[idx:]  # e.g. mosaic/tesseracts/navier-stokes-grid
        tdir_to_problems.setdefault(suffix, []).append(name)

    # problem config file: derive from problem name
    config_to_problem[f"mosaic/benchmarks/problems/{name.replace('-', '_')}.py"] = name

changed: set[str] = set()
for f in diff_files:
    if f in config_to_problem:
        changed.add(config_to_problem[f])
        continue
    for prefix, problems in tdir_to_problems.items():
        if f.startswith(prefix + "/"):
            changed.update(problems)
            break

print(",".join(sorted(changed)) if changed else "")
