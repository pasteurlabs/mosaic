#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlay PR benchmark results on top of a baseline results directory.

Unlike a naive ``cp -r`` overlay, this script does **solver-level merging**
of ``result.json`` files.  A PR that only ran Exponax will update the
Exponax entries in each experiment's result.json while preserving all other
solvers' data from the baseline.

All other files (``*.png``, ``*.gif``, ``params.json``) are overlaid with
simple file-level copy (PR wins).

Usage:
    python .github/scripts/overlay-results.py <baseline-dir> <pr-results-dir> <output-dir>

The output directory receives the merged results.  It may be the same path
as *pr-results-dir* for in-place overlay (baseline is read first).
"""

from __future__ import annotations

# Reuse the solver-level merge logic from merge-results.py.
import importlib.util
import json
import shutil
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "merge_results", Path(__file__).parent / "merge-results.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_merge_result_pair = _mod._merge_result_pair


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <baseline-dir> <pr-results-dir> <output-dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    baseline = Path(sys.argv[1])
    pr_results = Path(sys.argv[2])
    output = Path(sys.argv[3])

    if not pr_results.is_dir():
        print(f"PR results directory not found: {pr_results}", file=sys.stderr)
        sys.exit(1)

    if not baseline.is_dir():
        # No baseline — just copy PR results as-is.
        print(f"No baseline directory ({baseline}), copying PR results only.")
        shutil.copytree(pr_results, output, dirs_exist_ok=True)
        return

    # Collect all relative paths from both directories.
    baseline_files: dict[str, Path] = {}
    for fpath in baseline.rglob("*"):
        if fpath.is_file():
            relpath = str(fpath.relative_to(baseline))
            baseline_files[relpath] = fpath

    pr_files: dict[str, Path] = {}
    for fpath in pr_results.rglob("*"):
        if fpath.is_file():
            relpath = str(fpath.relative_to(pr_results))
            pr_files[relpath] = fpath

    all_relpaths = sorted(set(baseline_files) | set(pr_files))

    n_merged = 0
    n_copied = 0
    n_baseline_only = 0

    # Skip meta files from baseline — they'll be regenerated.
    skip_basenames = {"snapshot.json", "status-report.md"}

    for relpath in all_relpaths:
        basename = Path(relpath).name
        if basename in skip_basenames:
            continue

        in_baseline = relpath in baseline_files
        in_pr = relpath in pr_files
        out_path = output / relpath

        if in_baseline and in_pr and relpath.endswith("result.json"):
            # Solver-level merge: baseline + PR overlay.
            try:
                base_data = _load_json(baseline_files[relpath])
                pr_data = _load_json(pr_files[relpath])
                merged = _merge_result_pair(base_data, pr_data)
                _save_json(merged, out_path)
                n_merged += 1
            except Exception as e:
                print(f"  warning: merge failed for {relpath}: {e}", file=sys.stderr)
                # Fall back to PR version.
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pr_files[relpath], out_path)
        elif in_pr:
            # PR-only file — copy as-is.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pr_files[relpath], out_path)
            n_copied += 1
        elif in_baseline:
            # Baseline-only file — carry forward.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_files[relpath], out_path)
            n_baseline_only += 1

    print(
        f"Overlay complete: {n_merged} result.json merged, "
        f"{n_copied} PR-only copied, {n_baseline_only} baseline-only carried forward."
    )


if __name__ == "__main__":
    main()
