# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-demand experiments run only when named."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.experiment import kernel

_ROOT = Path(__file__).resolve().parents[1]


@kernel(sweep_mode="none")
def _noop(t, ctx):  # pragma: no cover - never invoked
    return {"metrics": {}}


def _problem() -> Problem:
    problem = Problem.__new__(Problem)
    for attr, value in {
        "name": "toy",
        "experiments": {},
        "plot_fns": {},
        "make_ic": {},
        "solvers": [],
        "status_checks": {},
        "exclusions": {},
        "indicative_metrics": {},
        "output_key": "result",
        "ic_key": "v0",
        "domain_extent": 1.0,
        "reference": None,
    }.items():
        setattr(problem, attr, value)
    problem.add_experiment("optimization/regular", _noop, physics={"N": 8})
    problem.add_experiment(
        "optimization/campaign", _noop, physics={"N": 8}, on_demand=True
    )
    return problem


def _experiments(problem: Problem, calls: list) -> dict:
    def record(name):
        def fn(cfg, tags, **_):
            calls.append(name)
            return {}

        return fn

    return {k.split("/", 1)[1]: record(k) for k in problem.experiments}


def test_suite_runs_skip_on_demand_experiments(tmp_path, monkeypatch):
    from mosaic.benchmarks.core.runner import run_suite

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    problem = _problem()
    assert problem.experiments["optimization/campaign"].on_demand
    assert not problem.experiments["optimization/regular"].on_demand

    calls: list = []
    run_suite(
        problem,
        {},
        _experiments(problem, calls),
        plots=False,
        suite_name="optimization",
    )
    assert calls == ["optimization/regular"]

    calls.clear()
    run_suite(
        problem,
        {},
        _experiments(problem, calls),
        to_run=["campaign"],
        plots=False,
        suite_name="optimization",
    )
    assert calls == ["optimization/campaign"]


def test_benchmark_matrix_leaves_out_on_demand_experiments():
    spec = importlib.util.spec_from_file_location(
        "matrix", _ROOT / ".github" / "scripts" / "generate-benchmark-matrix.py"
    )
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    assert matrix._experiment_keys_for_suite(_problem(), "optimization") == ["regular"]


def test_status_lists_on_demand_experiments_only_once_they_have_run(
    tmp_path, monkeypatch
):
    from mosaic.benchmarks.core.status import collect_status

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    rows = collect_status(_problem(), ["optimization"]).rows
    assert [r.experiment for r in rows] == ["regular"]
