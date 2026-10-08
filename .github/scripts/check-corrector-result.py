#!/usr/bin/env python3
"""Fail the corrector CI cell when its canonical status is not fresh and OK."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from typer import Exit

from mosaic.benchmarks.cli._helpers import (
    _apply_solver_filter,
    _parse_per_problem_solver_map,
    _validate_solver_csv,
)
from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import RESULTS_DIR_ENV
from mosaic.benchmarks.core.status import OK, collect_status
from mosaic.benchmarks.core.utils import active_solvers
from mosaic.benchmarks.problems import get_config


def check_results(cfg: Problem, hardware: str, solvers: str = "") -> list[str]:
    """Check only active corrector solvers selected for this hardware/job."""
    try:
        _validate_solver_csv(solvers, [cfg.name])
        if "=" in solvers:
            mapping = _parse_per_problem_solver_map(solvers)
            requested = mapping.get(cfg.name)
            known = {name.lower() for name in cfg.solver_names}
            if requested is not None and (
                not requested or any(name.lower() not in known for name in requested)
            ):
                return [
                    f"Invalid corrector solver selection for {cfg.name}: {requested}"
                ]
        selected = _apply_solver_filter(cfg, solvers)
    except (ValueError, Exit) as exc:
        return [f"Invalid corrector solver selection: {solvers!r}: {exc}"]
    if selected is None:
        # A valid cross-problem union may contain only solvers from other domains.
        print("No selected solvers belong to ns-grid.")
        return []
    expected = [
        name
        for name in active_solvers(selected, "optimization", "solver_in_loop")
        if bool(selected.solver(name).uses_gpu) == (hardware == "gpu")
    ]
    if not expected:
        print("No active corrector cells in this hardware/solver scope.")
        return []
    rows = [
        row
        for row in collect_status(cfg, suites=["optimization"]).rows
        if row.experiment.split("/")[0] == "solver_in_loop"
    ]
    if not rows:
        return ["Missing registered optimization/solver_in_loop status row"]
    failures = []
    for row in rows:
        for name in expected:
            cell = row.cells.get(name)
            if cell is None:
                failures.append(f"{row.label}/{name}: missing status cell")
            elif cell.status != OK or cell.stale:
                failures.append(
                    f"{row.label}/{name}: {cell.status}"
                    f"{' (stale)' if cell.stale else ''}: {cell.reason}"
                )
    return failures


def main() -> int:
    """Return a failing process status for missing/failed/anomalous results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", required=True, choices=["cpu", "gpu"])
    parser.add_argument("--solvers", default="")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        os.environ[RESULTS_DIR_ENV] = str(args.output_dir.resolve())
    failures = check_results(get_config("ns-grid"), args.hardware, args.solvers)
    for failure in failures:
        print(f"Corrector CI failure: {failure}")
    if not failures:
        print("Corrector CI execution checks passed for all requested active cells.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
