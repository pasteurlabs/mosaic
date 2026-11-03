# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-setup check tables on the Results pages (docs/scoreboard.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docs import generate_results  # noqa: E402

scoreboard = generate_results.scoreboard
iv = scoreboard.views.iv

SOLVERS = ["PhiFlow", "XLB", "PICT"]
META = {"ns-grid": {s: {"display": s, "anchor": "", "gpu": True} for s in SOLVERS}}


def _cell(status: str = "ok", **kw) -> dict:
    return {"status": status, "reason": "", "category": "", "metrics": {}, **kw}


def _row(label: str, cells: dict[str, dict]) -> dict:
    suite, _, experiment = label.partition("/")
    return {"suite": suite, "experiment": experiment, "label": label, "cells": cells}


def _snapshot(root: Path, rows: list[dict]) -> None:
    snap = {"problems": {"ns-grid": {"solvers": SOLVERS, "rows": rows}}}
    (root / "snapshot.json").write_text(json.dumps(snap))


def _cost(root: Path, experiment: str, rows: list[dict]) -> None:
    d = root / "ns-grid" / "cost" / experiment
    d.mkdir(parents=True)
    data = {
        "results": rows,
        "extras": {"hardware": {"gpus": ["Tesla T4, 16384 MiB"]}},
    }
    (d / "result.json").write_text(json.dumps(data))


def _timing(solver: str, mean: float, std: float) -> dict:
    return {
        "solver": solver,
        "sweep_value": "64",
        "metrics": {
            "mean": mean,
            "std": std,
            "trials_s": [mean] * 3,
            "first_call_s": 10 * mean,
        },
    }


def _setup(data: dict, key: str) -> dict:
    return next(s for s in data["setups"] if s["key"] == key)


def _solver(setup: dict, name: str) -> dict:
    return next(r for r in setup["rows"] if r["name"] == name)


@pytest.fixture
def periodic(tmp_path: Path) -> dict:
    """A periodic-box snapshot covering each kind of cell."""
    _snapshot(
        tmp_path,
        [
            _row(
                "forward/agreement/tgv",
                {
                    "PhiFlow": _cell(),
                    "XLB": _cell(
                        "anomaly",
                        reason="error above the peer median",
                        explanation="LBM compressibility error",
                    ),
                    "PICT": _cell(),
                },
            ),
            _row(
                "gradient/fd_check",
                {
                    "PhiFlow": _cell(metrics={"rel_err": 2e-5, "cosine": 0.99999}),
                    "XLB": _cell(),
                    "PICT": _cell("excluded", category="unstable", reason="diverges"),
                },
            ),
            _row(
                "gradient/analytic_check",
                {
                    "PhiFlow": _cell(metrics={"rel_err": 3e-3, "cosine": 0.998}),
                    "XLB": _cell(),
                    "PICT": _cell("excluded", category="unstable", reason="diverges"),
                },
            ),
            _row(
                "gradient/horizon_sweep",
                {
                    "PhiFlow": _cell(resource_frontier="5120"),
                    "XLB": _cell(),
                    "PICT": _cell("excluded", category="wip"),
                },
            ),
            _row(
                "cost/spatial_cost",
                {s: _cell() for s in SOLVERS},
            ),
        ],
    )
    _cost(
        tmp_path,
        "spatial_cost",
        [
            _timing("PhiFlow", 1.00, 0.05),
            _timing("XLB", 1.05, 0.05),
            _timing("PICT", 3.00, 0.05),
        ],
    )
    data = scoreboard.compute(tmp_path, META)
    assert data is not None
    return _setup(data, "ns-2d-periodic")


def test_recorded_exclusions_block_a_check(periodic: dict) -> None:
    pict = _solver(periodic, "PICT")
    assert pict["checks"]["gradient"]["state"] == "blocked"
    assert pict["checks"]["gradient"]["blocked_as"] == "Unstable"
    # A work-in-progress exclusion is not a finding, so it reads as not run.
    assert pict["reliability"]["blocked"] == {"Unstable": 2}
    assert pict["reliability"]["notrun"] == 1
    html = scoreboard.render_setup(periodic)
    assert "Unstable</span>" in html
    assert "2 unstable" in html


def test_explained_anomalies_stay_borderline(periodic: dict) -> None:
    check = _solver(periodic, "XLB")["checks"]["forward"]
    assert check["state"] == "borderline"
    assert check["explained"]
    assert 'class="ck-explained"' in scoreboard.render_setup(periodic)


def test_resource_limits_are_listed_but_not_failures(periodic: dict) -> None:
    phiflow = _solver(periodic, "PhiFlow")
    assert phiflow["reliability"]["limits"] == 1
    assert phiflow["reliability"]["failed"] == 0
    # Without a result file there is no sweep key to name.
    assert phiflow["limits"] == [("gradient/horizon_sweep", "5120")]
    html = scoreboard.render_setup(periodic)
    assert "Resource limits (1)" in html
    assert "from 5120" in html


def test_gradient_check_shows_fd_and_analytic_side_by_side(periodic: dict) -> None:
    check = _solver(periodic, "PhiFlow")["checks"]["gradient"]
    assert [lab for lab, _ in check["metrics"]] == [
        "gradient/fd_check",
        "gradient/analytic_check",
    ]
    html = scoreboard.render_setup(periodic)
    assert "fd_check: error 2e-05, cos 0.99999" in html
    assert "analytic_check: error 0.003, cos 0.99800" in html


def test_cost_marks_gaps_within_the_trial_spread(periodic: dict) -> None:
    xlb = _solver(periodic, "XLB")["cost"]["Forward cost"]
    pict = _solver(periodic, "PICT")["cost"]["Forward cost"]
    assert xlb["within_noise"]
    assert not pict["within_noise"]
    assert periodic["gpu_model"] == "Tesla T4"
    html = scoreboard.render_setup(periodic)
    assert "≈ 1.1×" in html
    assert "First call" in html
    assert "measured on Tesla T4" in html


def test_check_tables_carry_effort_and_deviations(periodic: dict) -> None:
    assert periodic["has_effort"]
    html = scoreboard.render_setup(periodic)
    assert ">Effort</th>" in html
    assert ">Deviations</th>" in html
    xlb = _solver(periodic, "XLB")
    assert xlb["effort"][0] in iv.LEVELS
    # Effort and deviations link to the solver's record.
    href = f"solvers.qmd#{xlb['anchor']}" if xlb["anchor"] else "solvers.qmd"
    assert f'<td class="ck-effort"><a href="{href}"' in html
    # Reliability points at the notes under the table.
    assert '<a href="#notes-ns-2d-periodic">*</a>' in html
    assert 'id="notes-ns-2d-periodic"' in html


def test_partial_passes_read_as_partial() -> None:
    chk = {
        "state": "pass",
        "n_run": 1,
        "n_gates": 5,
        "reasons": [],
        "explained": False,
        "blocked_as": "Blocked",
        "metrics": [],
    }
    cell = scoreboard._check_td(chk, "notes-x")
    assert "Partial</span>" in cell
    assert "1 of 5 run" in cell
    assert "Pass</span>" in scoreboard._check_td({**chk, "n_run": 5}, "notes-x")


def test_setups_follow_the_intervention_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A setup defined in interventions.SETUPS takes its experiments along."""
    _snapshot(
        tmp_path,
        [
            _row("forward/agreement/tgv", {s: _cell() for s in SOLVERS}),
            _row("forward/kolmogorov", {s: _cell("failed") for s in SOLVERS}),
        ],
    )
    data = scoreboard.compute(tmp_path, META)
    tgv = _solver(_setup(data, "ns-2d-periodic"), "XLB")
    kolmogorov = _solver(_setup(data, "ns-2d-kolmogorov"), "XLB")
    assert tgv["checks"]["forward"]["state"] == "pass"
    assert kolmogorov["checks"]["forward"]["state"] == "fail"

    setups = {k: v for k, v in iv.SETUPS.items() if k != "ns-2d-kolmogorov"}
    monkeypatch.setattr(iv, "SETUPS", setups)
    without = scoreboard.compute(tmp_path, META)
    assert "ns-2d-kolmogorov" not in {s["key"] for s in without["setups"]}


def test_cost_follows_the_device_results_were_measured_on(tmp_path: Path) -> None:
    """Results from before a move to the GPU are compared with the CPU solvers."""
    _snapshot(tmp_path, [_row("cost/spatial_cost", {s: _cell() for s in SOLVERS})])
    d = tmp_path / "ns-grid" / "cost" / "spatial_cost"
    d.mkdir(parents=True)
    t4 = {"gpus": ["Tesla T4, 16384 MiB"], "cpu": "EPYC"}
    data = {
        "results": [
            _timing(s, t, 0.01) for s, t in zip(SOLVERS, (1.0, 2.0, 4.0), strict=True)
        ],
        "extras": {"hardware_by_solver": {"PhiFlow": t4, "XLB": t4, "PICT": {}}},
    }
    (d / "result.json").write_text(json.dumps(data))
    setup = _setup(scoreboard.compute(tmp_path, META), "ns-2d-periodic")
    pict = _solver(setup, "PICT")
    assert not pict["gpu"]
    assert pict["device_mismatch"]
    assert pict["cost"]["Forward cost"]["ratio"] == 1.0
    assert _solver(setup, "XLB")["cost"]["Forward cost"]["ratio"] == 2.0
    html = scoreboard.render_setup(setup)
    assert "measured on Tesla T4" in html
    assert "CPU run</span>" in html
