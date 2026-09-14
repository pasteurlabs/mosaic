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
4. Merge ``fields.npz`` files at the solver level (union per-solver arrays
   *and* the ``solver_names`` metadata array).
5. Write merged results to the final output directory.

Usage (in CI):
    python .github/scripts/merge-results.py staging-dir/ mosaic-results/
"""

from __future__ import annotations

import json
import re
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
# fields.npz merge — solver-level union
# ---------------------------------------------------------------------------
#
# Field-snapshot NPZs come in two on-disk layouts (see
# ``save_field_snapshots_npz`` in mosaic/benchmarks/core/io.py):
#
#   positional: ``{prefix}_{j}`` / ``{prefix}_{j}_{suffix}`` where ``j`` indexes
#               into the file's ``solver_names`` array (gradient/optimization)
#   flat:       ``{solver_name}_{suffix}`` (forward-agreement, e.g. ``exponax_0``
#               where the trailing index is a *sweep* index, not a solver index)
#
# Each file is decoded against *its own* ``solver_names`` into a
# ``{solver: {template: array}}`` map, where ``template`` is the original key
# with the solver's slot replaced by a ``{S}`` placeholder. Re-encoding then
# reproduces the exact key for whatever slot the solver lands in. Keys that
# don't resolve to a known solver are kept as shared (later files win).
#
# Order matters: a flat key like ``exponax_0`` would also match the positional
# pattern (prefix=``exponax``, idx=0). So each key is classified shared → flat →
# positional, in that order, and only falls back to positional once it's neither
# a known shared prefix nor prefixed by a known solver name.

_POSITIONAL_RE = re.compile(r"^(?P<prefix>.+?)_(?P<idx>-?\d+)(?:_(?P<suffix>.*))?$")

# Shared (non-per-solver) array prefixes used across mosaic's field NPZs. These
# must be matched before positional decoding because some (e.g. ``consensus_0``)
# carry a trailing integer that would otherwise be misread as a solver index.
# Mirrors the ``shared_prefixes`` / shared-key sets in
# mosaic/benchmarks/problems/**: forward (consensus, x_axis, ic, sweep_values),
# gradient/jacobian (singular_values, singular_vectors, ic), recovery (rep_val,
# rep_horizon, ic_true, ic_init), cost (sweep_values).
_SHARED_PREFIXES = (
    "sweep_values",
    "solver_names",
    "consensus",
    "x_axis",
    "singular_values",
    "singular_vectors",
    "rep_val",
    "rep_horizon",
    "ic_true",
    "ic_init",
    "ic",
)


def _is_shared_key(key: str) -> bool:
    """True if *key* is a shared array (matched by prefix), not per-solver."""
    return any(key == p or key.startswith(p + "_") for p in _SHARED_PREFIXES)


def _decode_npz(
    arrays: dict[str, np.ndarray], names: list[str]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Split a loaded NPZ into ({solver: {template: array}}, shared).

    ``template`` carries a ``{S}`` placeholder where the solver identity sat,
    so the slot is re-derivable at encode time regardless of layout.
    """
    # Longest-first so "ins_jl_diff" matches before "ins_jl" in the flat layout.
    names_by_len = sorted(names, key=len, reverse=True)
    per_solver: dict[str, dict[str, np.ndarray]] = {}
    shared: dict[str, np.ndarray] = {}

    for key, arr in arrays.items():
        if key == "solver_names":
            continue

        # Shared arrays (consensus, ic, singular_values, …) are not per-solver.
        # Checked first so e.g. "consensus_0" isn't decoded as solver-index 0.
        if _is_shared_key(key):
            shared[key] = arr
            continue

        # Flat: {solver_name}[_suffix]. Tried next — a flat key such as
        # "exponax_0" must not be misread as positional prefix="exponax", idx=0.
        matched = next(
            (s for s in names_by_len if key == s or key.startswith(s + "_")),
            None,
        )
        if matched is not None:
            suffix = key[len(matched) :]  # "" or "_suffix"
            per_solver.setdefault(matched, {})["{S}" + suffix] = arr
            continue

        # Positional: prefix_{j}[_suffix] where j indexes solver_names.
        m = _POSITIONAL_RE.match(key)
        if m:
            idx = int(m.group("idx"))
            if 0 <= idx < len(names):
                suffix = m.group("suffix")
                template = (
                    f"{m.group('prefix')}_{{S}}"
                    if suffix is None
                    else f"{m.group('prefix')}_{{S}}_{suffix}"
                )
                per_solver.setdefault(names[idx], {})[template] = arr
                continue

        shared[key] = arr

    return per_solver, shared


def _encode_template(template: str, idx: int, name: str) -> str:
    """Reproduce an on-disk key from a ``{S}`` template.

    Positional templates carry a ``_{S}`` index slot → fill with ``idx``;
    flat templates start with ``{S}`` → fill with the solver ``name``.
    """
    if template.startswith("{S}"):
        return name + template[len("{S}") :]
    return template.replace("{S}", str(idx))


def _merge_npz(paths: list[Path], out_path: Path) -> None:
    """Solver-level union of field-snapshot NPZs (later files win per solver).

    Each CI job's NPZ carries only the solvers that job ran — including the
    ``solver_names`` metadata array. A naive key-level merge would keep every
    per-solver array but let the last artifact's ``solver_names`` overwrite
    the others'; the field plots read their solver set from that array, so
    the merged file would silently hide the other jobs' solvers. Decode each
    file against its own ``solver_names``, union per solver, and re-encode.
    """
    merged_solver: dict[str, dict[str, np.ndarray]] = {}
    merged_shared: dict[str, np.ndarray] = {}
    ordered: list[str] = []
    for p in paths:
        try:
            with np.load(str(p), allow_pickle=False) as data:
                arrays = {k: np.asarray(data[k]) for k in data.files}
        except Exception as e:
            print(f"  warning: failed to merge {p}: {e}", file=sys.stderr)
            continue
        names = [str(n) for n in arrays.get("solver_names", np.array([]))]
        per_solver, shared = _decode_npz(arrays, names)
        merged_solver.update(per_solver)
        merged_shared.update(shared)
        ordered += [n for n in names if n not in ordered]

    ordered += [s for s in merged_solver if s not in ordered]
    if not ordered and not merged_shared:
        return

    payload: dict[str, np.ndarray] = {}
    if ordered:
        payload["solver_names"] = np.array(ordered)
    payload.update(merged_shared)
    for idx, name in enumerate(ordered):
        for template, arr in merged_solver.get(name, {}).items():
            payload[_encode_template(template, idx, name)] = arr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)


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
