#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect reference solvers needed by the given problems.

Prints a comma-separated list of solver display names that are used as
references (either via ``reference_solver`` or fine-grid
``reference={"solvers": {...}}`` experiment params) for the requested
problems.  Returns only solvers that are *not* already in the
``--changed`` set, so the caller can augment its solver list.

Reference identifiers in experiment configs use the solver *key* format
(underscored directory name, e.g. ``"jax_cfd"``), while the ``--solvers``
CLI filter and the ``--changed`` input use *display names*
(e.g. ``"jax-cfd"``).  This script resolves keys to display names so
the output can be appended directly to ``SOLVERS``.

Usage (in CI):
    python .github/scripts/detect-reference-solvers.py \
        --problems ns-grid,ns-3d-grid \
        --changed "XLB"
"""

from __future__ import annotations

import argparse

from mosaic.benchmarks.problems import PROBLEMS, get_config


def _collect_reference_solvers(problem_names: list[str]) -> set[str]:
    """Return display names of all solvers used as references."""
    display_names: set[str] = set()
    for pname in problem_names:
        cfg = get_config(pname)

        # Build key→display_name map for this problem's solvers.
        key_to_display = {s.key: s.name for s in cfg.solvers}

        for exp in cfg.experiments.values():
            params = exp.params
            # Fine-grid reference: reference={"solvers": {"jax_cfd"}, ...}
            ref = params.get("reference")
            if isinstance(ref, dict) and "solvers" in ref:
                for solver_id in ref["solvers"]:
                    if solver_id in key_to_display:
                        display_names.add(key_to_display[solver_id])
                    else:
                        # Might already be a display name.
                        display_names.add(solver_id)
            # Named reference solver: reference_solver="openfoam"
            ref_solver = params.get("reference_solver")
            if ref_solver:
                if ref_solver in key_to_display:
                    display_names.add(key_to_display[ref_solver])
                else:
                    display_names.add(ref_solver)
    return display_names


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
        help="Comma-separated solver display names already in the changed set",
    )
    args = parser.parse_args()

    problem_list = (
        list(PROBLEMS)
        if args.problems == "all"
        else [p.strip() for p in args.problems.split(",") if p.strip()]
    )
    changed = {s.strip() for s in args.changed.split(",") if s.strip()}

    refs = _collect_reference_solvers(problem_list)
    # Only report solvers not already being built
    extra = sorted(refs - changed)
    print(",".join(extra))


if __name__ == "__main__":
    main()
