#!/usr/bin/env python3

# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a JSON matrix of (suite, problem, hardware) combos for CI.

Outputs a JSON object with a single key ``include`` whose value is a list of
``{suite, problem, hardware}`` dicts — ready for ``fromJSON`` in a GitHub
Actions matrix strategy.

When ``--solvers`` is provided, only problems that contain at least one
matching solver are included, and the hardware column is scoped to the
matching solvers' capabilities.

Usage (in CI):
    python .github/scripts/generate-benchmark-matrix.py \
        --problems all --suites all [--solvers "Solver Name"]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from typing import Any

from mosaic.benchmarks.core.utils import active_solvers
from mosaic.benchmarks.problems import PROBLEMS, get_config

SOLVER_SUITES = ["forward", "cost", "gradient", "optimization"]


def _experiment_keys_for_suite(
    cfg: Any, suite: str, selected: str = "all"
) -> list[str]:
    prefix = f"{suite}/"
    return [
        key[len(prefix) :]
        for key in cfg.experiments
        if key.startswith(prefix)
        and key[len(prefix) :]
        and (selected == "all" or key[len(prefix) :] == selected)
    ]


def _has_runnable_solvers(
    cfg: Any,
    suite: str,
    hardware: str,
    solver_filter: set[str] | None,
    selected_experiment: str = "all",
) -> bool:
    """True when at least one solver would run for *suite* on *hardware*.

    Mirrors the CLI's ``--hardware`` filter and per-experiment exclusion
    gating so CI does not dispatch runners for empty (suite, problem,
    hardware) combos — e.g. ns-grid optimization on CPU, where every
    runnable solver is GPU-only.
    """
    experiments = _experiment_keys_for_suite(cfg, suite, selected_experiment)
    if not experiments:
        return False

    want_gpu = hardware == "gpu"
    for experiment in experiments:
        with contextlib.redirect_stdout(io.StringIO()):
            names = active_solvers(cfg, suite, experiment)
        for name in names:
            if solver_filter and name not in solver_filter:
                continue
            spec = cfg.solver(name)
            is_gpu = getattr(spec, "uses_gpu", True)
            if want_gpu == is_gpu:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems", default="all", help="Comma-separated problems or 'all'"
    )
    parser.add_argument(
        "--suites", default="all", help="Comma-separated suites or 'all'"
    )
    parser.add_argument(
        "--solvers",
        default="",
        help="Comma-separated solver display names. When set, only problems "
        "that contain at least one matching solver are included.",
    )
    parser.add_argument(
        "--experiments",
        default="all",
        help="Single bare experiment name within requested suites, or all",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiments):
        parser.error("--experiments must be one bare name or all, not a path or list")

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
    solver_filter: set[str] | None = None
    if args.solvers:
        solver_filter = {s.strip() for s in args.solvers.split(",") if s.strip()}

    include = []
    registered_match = False
    for problem in problem_list:
        cfg = get_config(problem)
        registered_match |= any(
            _experiment_keys_for_suite(cfg, suite, args.experiments)
            for suite in suite_list
        )
        if solver_filter and not any(s.name in solver_filter for s in cfg.solvers):
            continue
        for suite in suite_list:
            if _has_runnable_solvers(
                cfg, suite, "gpu", solver_filter, args.experiments
            ):
                include.append({"suite": suite, "problem": problem, "hardware": "gpu"})
            if _has_runnable_solvers(
                cfg, suite, "cpu", solver_filter, args.experiments
            ):
                include.append({"suite": suite, "problem": problem, "hardware": "cpu"})

    if args.experiments != "all" and not registered_match:
        parser.error(
            "No registered experiment matches the requested problems, suites and experiment"
        )
    if args.experiments != "all" and not include:
        parser.error(
            "No active solvers match the requested experiment and solver selection"
        )
    json.dump({"include": include}, sys.stdout)
    print()  # trailing newline


if __name__ == "__main__":
    main()
