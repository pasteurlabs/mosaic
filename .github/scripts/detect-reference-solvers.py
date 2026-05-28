#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a per-problem solver map including changed + reference solvers.

Prints a per-problem solver map in the format understood by the
``mosaic run --solvers`` flag::

    ns-grid=Test Spectral,jax-cfd,OpenFOAM;ns-3d-grid=Test Spectral,Exponax

Each problem entry lists the changed solver(s) plus any reference solvers
that exist in that problem.  Reference solvers that were removed from a
problem (e.g. jax-cfd popped from ns-3d-grid) are omitted for that
problem, avoiding "Unknown solver" errors.

Reference identifiers in experiment configs use the solver *key* format
(underscored directory name, e.g. ``"jax_cfd"``), while the ``--solvers``
CLI filter and the ``--changed`` input use *display names*
(e.g. ``"jax-cfd"``).  This script resolves keys to display names so
the output can be used directly.

Usage (in CI):
    python .github/scripts/detect-reference-solvers.py \
        --problems ns-grid,ns-3d-grid \
        --changed "Test Spectral"
"""

from __future__ import annotations

import argparse

from mosaic.benchmarks.problems import PROBLEMS, get_config


def _collect_reference_solvers_per_problem(
    problem_names: list[str],
) -> dict[str, set[str]]:
    """Return {problem_name: {display_names}} of reference solvers per problem.

    Only returns solvers that actually exist in each problem's solver list.
    """
    per_problem: dict[str, set[str]] = {}
    for pname in problem_names:
        cfg = get_config(pname)
        display_names: set[str] = set()

        # Build key→display_name map for this problem's solvers.
        key_to_display = {s.key: s.name for s in cfg.solvers}
        # Also allow matching by display name directly.
        valid_display = {s.name for s in cfg.solvers}

        for exp in cfg.experiments.values():
            params = exp.params
            # Fine-grid reference: reference={"solvers": {"jax_cfd"}, ...}
            ref = params.get("reference")
            if isinstance(ref, dict) and "solvers" in ref:
                for solver_id in ref["solvers"]:
                    if solver_id in key_to_display:
                        display_names.add(key_to_display[solver_id])
                    elif solver_id in valid_display:
                        display_names.add(solver_id)
                    # else: solver was removed from this problem (e.g. jax_cfd
                    # popped from ns-3d-grid) — skip silently.
            # Named reference solver: reference_solver="openfoam"
            ref_solver = params.get("reference_solver")
            if ref_solver:
                if ref_solver in key_to_display:
                    display_names.add(key_to_display[ref_solver])
                elif ref_solver in valid_display:
                    display_names.add(ref_solver)

        if display_names:
            per_problem[pname] = display_names
    return per_problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems",
        default="all",
        help="Comma-separated problem names or 'all'",
    )
    parser.add_argument(
        "--changed",
        default="",
        help="Comma-separated solver display names to include in every problem",
    )
    args = parser.parse_args()

    problem_list = (
        list(PROBLEMS)
        if args.problems == "all"
        else [p.strip() for p in args.problems.split(",") if p.strip()]
    )
    changed = {s.strip() for s in args.changed.split(",") if s.strip()}

    per_problem_refs = _collect_reference_solvers_per_problem(problem_list)

    # Build per-problem solver map: changed solvers + their references,
    # but only including solvers that actually exist in each problem.
    parts = []
    for pname in sorted(problem_list):
        cfg = get_config(pname)
        valid = {s.name for s in cfg.solvers}
        # Start with changed solvers that exist in this problem.
        solvers = changed & valid
        # Add reference solvers (already validated to exist).
        solvers |= per_problem_refs.get(pname, set())
        if solvers:
            parts.append(f"{pname}={','.join(sorted(solvers))}")
    print(";".join(parts))


if __name__ == "__main__":
    main()
