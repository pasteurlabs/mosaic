#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge benchmark results from multiple CI artifact directories.

In CI, the GPU and CPU jobs for the same (suite, problem) write to
separate artifacts.  ``actions/download-artifact`` with
``merge-multiple: true`` does a naive file-level merge where
last-write-wins — so when both jobs produce the same ``result.json``,
one silently overwrites the other.

This script replaces that naive merge:

1. Walk every ``results-*/`` artifact directory.
2. Group files by their relative path (e.g.
   ``structural-mesh/forward/baseline/result.json``).
3. Deep-merge ``result.json`` files using the flat-list merge
   (schema_version=1: deduplicate by (solver, sweep_value), last wins).
4. Merge ``fields.npz`` files (combine per-solver arrays).
5. Write merged results to the final output directory.

Usage (in CI):
    python .github/scripts/merge-results.py staging-dir/ mosaic-results/
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=_NumpyEncoder, indent=2)


# ---------------------------------------------------------------------------
# result.json merge — schema_version=1 flat-list merge
# ---------------------------------------------------------------------------


def _merge_result_pair(existing: dict, new: dict) -> dict:
    """Merge two schema_version=1 result files. *new* wins on conflict.

    Deduplicates by (solver, sweep_value); later entries override earlier ones.
    """
    seen: dict[tuple, dict] = {}
    for entry in existing.get("results", []) + new.get("results", []):
        seen[(entry["solver"], entry.get("sweep_value"))] = entry
    merged = {**existing, **new, "results": list(seen.values())}
    # Merge provenance sub-dicts
    for key in ("tesseract_hashes", "wall_time_s"):
        a_val = existing.get("provenance", {}).get(key, {})
        b_val = new.get("provenance", {}).get(key, {})
        if a_val or b_val:
            merged.setdefault("provenance", {})[key] = {**a_val, **b_val}
    # Merge extras
    a_extras = existing.get("extras", {})
    b_extras = new.get("extras", {})
    if a_extras or b_extras:
        merged["extras"] = {**a_extras, **b_extras}
    return merged


def _merge_results(results: list[dict]) -> dict:
    """Merge a list of result dicts (from multiple artifact copies)."""
    merged = results[0]
    for r in results[1:]:
        merged = _merge_result_pair(merged, r)
    return merged


# ---------------------------------------------------------------------------
# fields.npz merge
# ---------------------------------------------------------------------------


def _merge_npz(paths: list[Path], out_path: Path) -> None:
    """Merge multiple .npz files, combining per-solver arrays."""
    merged: dict[str, np.ndarray] = {}
    for p in paths:
        try:
            with np.load(str(p), allow_pickle=False) as data:
                for key in data.files:
                    # Later artifacts override earlier ones for the same key.
                    # Metadata keys (sweep_values, solver_names, etc.) are
                    # overwritten; per-solver keys accumulate.
                    merged[key] = np.asarray(data[key])
        except Exception:
            continue
    if merged:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **merged)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <staging-dir> <output-dir>", file=sys.stderr)
        sys.exit(1)

    staging = Path(sys.argv[1])
    output = Path(sys.argv[2])

    if not staging.is_dir():
        print(f"Staging directory not found: {staging}", file=sys.stderr)
        sys.exit(1)

    # Collect all files grouped by relative path.
    # Staging layout: staging/<artifact-name>/<relative-path>
    # Each artifact-name dir like "results-forward-structural-mesh-gpu"
    # contains the same tree structure as mosaic-results/.
    artifact_dirs = sorted(
        d for d in staging.iterdir() if d.is_dir() and d.name.startswith("results-")
    )
    if not artifact_dirs:
        print(
            "No results-* artifact directories found in staging dir.", file=sys.stderr
        )
        sys.exit(0)

    # Group all files by their relative path (relative to the artifact dir).
    files_by_relpath: dict[str, list[Path]] = defaultdict(list)
    for adir in artifact_dirs:
        for fpath in adir.rglob("*"):
            if fpath.is_file():
                relpath = str(fpath.relative_to(adir))
                files_by_relpath[relpath].append(fpath)

    n_merged = 0

    for relpath, sources in sorted(files_by_relpath.items()):
        out_path = output / relpath

        if relpath.endswith("result.json") and len(sources) > 1:
            # Deep-merge result.json files.
            results = []
            for src in sources:
                try:
                    results.append(_load_json(src))
                except Exception as e:
                    print(f"  warning: failed to load {src}: {e}", file=sys.stderr)
            if not results:
                continue
            if len(results) == 1:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sources[0], out_path)
            else:
                merged = _merge_results(results)
                _save_json(merged, out_path)
                n_merged += 1
                print(f"  merged {len(results)} copies: {relpath}")

        elif relpath.endswith(".npz") and len(sources) > 1:
            # Merge npz files.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _merge_npz(sources, out_path)
            print(f"  merged {len(sources)} copies: {relpath}")

        else:
            # Single source or non-mergeable file — copy (last source wins
            # for any remaining conflicts, matching prior merge-multiple
            # behaviour).
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sources[-1], out_path)

    print(f"Merge complete: {n_merged} result.json file(s) deep-merged.")


if __name__ == "__main__":
    main()
