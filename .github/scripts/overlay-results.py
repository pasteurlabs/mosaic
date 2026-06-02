#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlay PR benchmark results on top of a baseline results directory.

Unlike a naive ``cp -r`` overlay, this script does **solver-level merging**
of ``result.json`` files.  A PR that only ran Exponax will update the
Exponax entries in each experiment's result.json while preserving all other
solvers' data from the baseline.

``*.npz`` field snapshots get the same solver-level treatment: a PR that
only ran Exponax writes a ``gradient_fields.npz`` containing just Exponax's
arrays (and a one-entry ``solver_names``). A naive file-level copy would
clobber every baseline solver's fields, so the field plots — which read
their solver set straight from the NPZ's ``solver_names`` — would then show
only Exponax. :func:`_merge_npz_pair` instead unions the per-solver arrays,
PR winning per solver, so baseline solvers survive.

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
import re
import shutil
import sys
from pathlib import Path

import numpy as np

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


# ── per-solver NPZ overlay ──────────────────────────────────────────────────
#
# Field-snapshot NPZs come in two on-disk layouts (see
# ``save_field_snapshots_npz`` in mosaic/benchmarks/core/io.py):
#
#   positional: ``{prefix}_{j}`` / ``{prefix}_{j}_{suffix}`` where ``j`` indexes
#               into the file's ``solver_names`` array (gradient/optimization)
#   flat:       ``{solver_name}_{suffix}`` (forward-agreement, e.g. ``exponax_0``
#               where the trailing index is a *sweep* index, not a solver index)
#
# We decode each file against *its own* ``solver_names`` into a
# ``{solver: {template: array}}`` map, where ``template`` is the original key
# with the solver's slot replaced by a ``{S}`` placeholder. Re-encoding then
# reproduces the exact key for whatever slot the solver lands in. Keys that
# don't resolve to a known solver are kept as shared (PR wins).
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


def _merge_npz_pair(baseline_path: Path, pr_path: Path, out_path: Path) -> None:
    """Solver-level union of two field-snapshot NPZs (PR wins per solver).

    Baseline solvers absent from the PR are carried forward; shared
    (non-per-solver) arrays take the PR copy when present.
    """
    with np.load(str(baseline_path), allow_pickle=False) as b:
        base_arrays = {k: np.asarray(b[k]) for k in b.files}
    with np.load(str(pr_path), allow_pickle=False) as p:
        pr_arrays = {k: np.asarray(p[k]) for k in p.files}

    base_names = [str(n) for n in base_arrays.get("solver_names", np.array([]))]
    pr_names = [str(n) for n in pr_arrays.get("solver_names", np.array([]))]

    base_solver, base_shared = _decode_npz(base_arrays, base_names)
    pr_solver, pr_shared = _decode_npz(pr_arrays, pr_names)

    # PR wins per solver; baseline-only solvers carried forward.
    merged_solver: dict[str, dict[str, np.ndarray]] = {**base_solver, **pr_solver}
    merged_shared: dict[str, np.ndarray] = {**base_shared, **pr_shared}

    # Canonical ordering: PR solvers first, then baseline-only.
    ordered = list(pr_names) + [s for s in base_names if s not in pr_names]
    ordered += [s for s in merged_solver if s not in ordered]

    payload: dict[str, np.ndarray] = {"solver_names": np.array(ordered)}
    payload.update(merged_shared)
    for idx, name in enumerate(ordered):
        for template, arr in merged_solver.get(name, {}).items():
            payload[_encode_template(template, idx, name)] = arr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)


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
    n_npz_merged = 0
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
        elif in_baseline and in_pr and relpath.endswith(".npz"):
            # Solver-level NPZ union so a single-solver PR doesn't clobber the
            # baseline solvers' field arrays (which drive the field plots).
            try:
                _merge_npz_pair(baseline_files[relpath], pr_files[relpath], out_path)
                n_npz_merged += 1
            except Exception as e:
                print(
                    f"  warning: npz merge failed for {relpath}: {e}", file=sys.stderr
                )
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
        f"{n_npz_merged} npz merged, "
        f"{n_copied} PR-only copied, {n_baseline_only} baseline-only carried forward."
    )


if __name__ == "__main__":
    main()
