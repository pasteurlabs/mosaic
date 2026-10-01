#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter a problem list to those belonging to specific domains.

Prints a comma-separated list of problem names whose tesseract domain
directory (e.g. ``navier-stokes-grid``) is among ``--domains``. Used by
the ``benchmark:domain`` CI path to scope the matrix to every problem in
the changed solvers' domains, derived from the domain set rather than the
diff — so harness/core changes never widen the run (that's what
``benchmark:all`` is for).

A single domain can back multiple problems (e.g. ``navier-stokes-grid``
maps to both ``ns-grid`` and ``ns-3d-grid``), so the whole domain runs.

Usage (in CI):
    python .github/scripts/filter-problems-by-domain.py \
        --problems all --domains "navier-stokes-grid,thermal-mesh"
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
        "--domains",
        required=True,
        help="Comma-separated tesseract domain directory names",
    )
    args = parser.parse_args()

    problem_list = (
        list(PROBLEMS)
        if args.problems == "all"
        else [p.strip() for p in args.problems.split(",") if p.strip()]
    )
    wanted = {d.strip() for d in args.domains.split(",") if d.strip()}

    hits = []
    for p in problem_list:
        try:
            domain = get_config(p).tesseract_dir.name
        except Exception:  # noqa: S112 — unimportable problems can't match the filter
            continue
        if domain in wanted:
            hits.append(p)

    print(",".join(hits))


if __name__ == "__main__":
    main()
