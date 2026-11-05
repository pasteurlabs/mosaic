# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A solver container that dies during a sweep.

The sweep records how the container stopped next to the failure, and in a
``default`` sweep it serves a new container so the later values run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mosaic.benchmarks.core import experiment, runner
from mosaic.benchmarks.core.config import Problem, SolverSpec

_SWEEP = [10, 500, 1000, 2000]
_DIED = {
    "status": "failed",
    "failure_type": "container_died",
    "exc_type": "ConnectionError",
    "exc_msg": "('Connection aborted.', RemoteDisconnected())",
}
_REPORT = {
    "container_exit": "container exited with code 134 (SIGABRT)",
    "container_log": "CUDA error: out of memory",
}


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> Problem:
    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path / "results"))
    spec = SolverSpec(
        dir="dummy", color="#000000", name="dummy", scheme="test", backend="python"
    )
    return Problem(
        name="test_problem",
        tesseract_dir=Path(tmp_path),
        output_key="result",
        solvers=[spec],
        make_ic={"zero": lambda **_: np.zeros(4)},
        make_inputs=lambda *a, **k: {},
        error_fn=lambda *a, **k: 0.0,
    )


@pytest.fixture
def restarts(monkeypatch) -> list:
    """Serve one fake Tesseract to every solver and record its restarts."""
    t = object()
    monkeypatch.setattr(
        runner,
        "run_with_gpu_pool",
        lambda names, tags, fn, **_: [fn(name, t) for name in names],
    )
    monkeypatch.setattr(experiment, "container_death_report", lambda _t: dict(_REPORT))
    calls: list = []
    monkeypatch.setattr(experiment, "restart_container", calls.append)
    return calls


def _run(cfg: Problem, kernel_fn, sweep_mode: str) -> dict:
    result = experiment.run_experiment(
        cfg,
        {"dummy": "dummy:latest"},
        kernel_fn,
        suite="cost",
        exp_key="vjp_cost/by_steps",
        runs=[
            {
                "ic": {"name": "zero"},
                "physics": {"steps": _SWEEP[0]},
                "sweep": {"key": "steps", "values": _SWEEP},
            }
        ],
        make_ic=cfg.make_ic,
        make_inputs=cfg.make_inputs,
        output_key="result",
        ic_key="v0",
        domain_extent=1.0,
        sweep_mode=sweep_mode,
        selector_fn=lambda *_: ["dummy"],
    )
    return {int(r["sweep_value"]): r["metrics"] for r in result["results"]}


def _dies_at(steps: int):
    def kernel_fn(t, ctx):
        if ctx.sweep_value == steps:
            return {"metrics": dict(_DIED)}
        return {"metrics": {"mean": 1.0}}

    return kernel_fn


def test_default_sweep_restarts_the_container_and_runs_the_rest(cfg, restarts):
    by_steps = _run(cfg, _dies_at(500), "default")
    assert by_steps[500] == {**_DIED, **_REPORT}
    assert len(restarts) == 1
    assert by_steps[1000]["mean"] == 1.0
    assert by_steps[2000]["mean"] == 1.0


def test_no_restart_after_the_last_value(cfg, restarts):
    by_steps = _run(cfg, _dies_at(2000), "default")
    assert by_steps[2000]["container_exit"] == _REPORT["container_exit"]
    assert restarts == []


def test_failed_restart_skips_the_rest(cfg, restarts, monkeypatch):
    def fail(_t):
        raise RuntimeError("image gone")

    monkeypatch.setattr(experiment, "restart_container", fail)
    by_steps = _run(cfg, _dies_at(500), "default")
    for steps in (1000, 2000):
        assert by_steps[steps]["status"] == "skipped"
        assert "could not be restarted" in by_steps[steps]["reason"]
        assert "image gone" in by_steps[steps]["reason"]


def test_limits_sweep_records_the_exit_without_restarting(cfg, restarts):
    def kernel_fn(t, ctx):
        if ctx.sweep_value == 500:
            raise ConnectionError("('Connection aborted.', RemoteDisconnected())")
        return {"metrics": {}}

    by_steps = _run(cfg, kernel_fn, "limits")
    assert by_steps[500]["failure_type"] == "container_died"
    assert by_steps[500]["container_exit"] == _REPORT["container_exit"]
    assert by_steps[1000]["status"] == "skipped"
    assert restarts == []
