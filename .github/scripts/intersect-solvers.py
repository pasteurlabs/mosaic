#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intersect a flat solver list with a single problem's solver set.

Prints the comma-separated canonical solver names that the requested
``--solvers`` list and the given ``--problem`` have in common (matched
case-insensitively, canonical casing preserved in the output). Empty output
means the problem has none of the requested solvers.

A flat ``--solvers`` list handed to a multi-problem benchmark matrix may name
solvers that exist only in some problems (e.g. ``torch-fem`` is thermal-only,
``TopOpt.jl`` is structural-only). ``mosaic run`` rejects a name that is not in
the problem it is run against, so applying one flat list to every matrix leg
breaks the legs whose problem lacks some of those solvers. Intersecting the
request with each problem's own solver set first keeps every leg valid.

Usage (in CI):
    python .github/scripts/intersect-solvers.py \
        --problem thermal-mesh --solvers "deal.II,FEniCS,torch-fem"
"""

from __future__ import annotations

import argparse

from mosaic.benchmarks.problems import get_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, help="Single problem name")
    parser.add_argument(
        "--solvers", required=True, help="Comma-separated solver display names"
    )
    args = parser.parse_args()

    wanted = {s.strip().lower() for s in args.solvers.split(",") if s.strip()}
    try:
        names = [s.name for s in get_config(args.problem).solvers]
    except Exception:
        names = []
    matched = [n for n in names if n.lower() in wanted]
    print(",".join(matched))


if __name__ == "__main__":
    main()
