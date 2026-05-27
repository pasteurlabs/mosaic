#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a JSON matrix of (suite, problem, hardware) combos for CI.

Outputs a JSON object with a single key ``include`` whose value is a list of
``{suite, problem, hardware}`` dicts — ready for ``fromJSON`` in a GitHub
Actions matrix strategy.

Usage (in CI):
    python .github/scripts/generate-benchmark-matrix.py \
        --problems all --suites all
"""

from __future__ import annotations

import argparse
import json
import sys

from mosaic.benchmarks.problems import PROBLEMS, get_config

SOLVER_SUITES = ["forward", "cost", "gradient", "optimization"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems", default="all", help="Comma-separated problems or 'all'"
    )
    parser.add_argument(
        "--suites", default="all", help="Comma-separated suites or 'all'"
    )
    args = parser.parse_args()

    problem_list = (
        PROBLEMS
        if args.problems == "all"
        else [p.strip() for p in args.problems.split(",") if p.strip()]
    )
    suite_list = (
        SOLVER_SUITES
        if args.suites == "all"
        else [
            s.strip()
            for s in args.suites.split(",")
            if s.strip() and s.strip() in SOLVER_SUITES
        ]
    )

    include = []
    for problem in problem_list:
        cfg = get_config(problem)
        has_gpu = any(getattr(s, "uses_gpu", True) for s in cfg.solvers)
        has_cpu = any(not getattr(s, "uses_gpu", True) for s in cfg.solvers)
        for suite in suite_list:
            if has_gpu:
                include.append({"suite": suite, "problem": problem, "hardware": "gpu"})
            if has_cpu:
                include.append({"suite": suite, "problem": problem, "hardware": "cpu"})

    json.dump({"include": include}, sys.stdout)
    print()  # trailing newline


if __name__ == "__main__":
    main()
