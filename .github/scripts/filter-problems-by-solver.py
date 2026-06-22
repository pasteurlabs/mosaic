#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter a problem list to those containing specific solvers.

Prints a comma-separated list of problem names where at least one of
the ``--solvers`` display names exists.  Used by the ``benchmark:solver``
CI path to scope the matrix to the changed solvers' problems, derived
from the solver set rather than the diff — so harness/core changes never
widen the run (that's what ``benchmark:all`` is for).

Usage (in CI):
    python .github/scripts/filter-problems-by-solver.py \
        --problems all --solvers "Exponax"
"""

from __future__ import annotations

import argparse

from mosaic.benchmarks.problems import PROBLEMS, get_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems",
        default="all",
        help="Comma-separated problem names or 'all'",
    )
    parser.add_argument(
        "--solvers",
        required=True,
        help="Comma-separated solver display names",
    )
    args = parser.parse_args()

    problem_list = (
        list(PROBLEMS)
        if args.problems == "all"
        else [p.strip() for p in args.problems.split(",") if p.strip()]
    )
    wanted = {s.strip() for s in args.solvers.split(",") if s.strip()}

    hits = []
    for p in problem_list:
        try:
            names = {s.name for s in get_config(p).solvers}
        except Exception:
            continue
        if wanted & names:
            hits.append(p)

    print(",".join(hits))


if __name__ == "__main__":
    main()
