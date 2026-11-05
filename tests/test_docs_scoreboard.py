# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-setup check tables on the Results pages (docs/scripts/scoreboard.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docs.scripts import generate_results  # noqa: E402

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
    assert [kind for _, kind, _ in pict["flags"]] == ["Unstable", "Unstable", "notrun"]
    html = scoreboard.render_setup(periodic)
    assert "Unstable</span>" in html
    assert "Flags and exclusions (4)" in html


def test_explained_anomalies_stay_flagged(periodic: dict) -> None:
    check = _solver(periodic, "XLB")["checks"]["forward"]
    assert check["state"] == "flagged"
    # The reason and its explanation are listed once, in the notes below the
    # table, not on hover.
    html = scoreboard.render_setup(periodic)
    assert html.count("(explained: LBM compressibility error)") == 1


def test_resource_limits_are_listed_but_not_failures(periodic: dict) -> None:
    phiflow = _solver(periodic, "PhiFlow")
    assert phiflow["flags"] == []
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
    assert "1.1×" in html
    assert "≈" not in html
    assert "Within the spread of the fastest solver" in html
    assert "First call" in html
    assert (
        "Forward and VJP at N=64, multiples of the fastest GPU solver, measured on Tesla T4"
        in html
    )


def test_check_tables_carry_effort_and_deviations(periodic: dict) -> None:
    assert periodic["has_effort"]
    html = scoreboard.render_setup(periodic)
    assert ">Effort</th>" in html
    assert ">Deviations</th>" not in html
    xlb = _solver(periodic, "XLB")
    assert xlb["effort"][0] in iv.LEVELS
    # The solver's name links to its record, and its effort is a pill shaded
    # by level, with the required interventions on hover.
    href = f"../solvers.qmd#{xlb['anchor']}" if xlb["anchor"] else "../solvers.qmd"
    assert f'<a href="{href}">XLB</a>' in html
    level = xlb["effort"][0]
    assert (
        f'<td class="ck-effort" data-sort="{level[1:]}">'
        f'<span class="eff eff-{level}" title="'
    ) in html
    # A solver with setup deviations is tagged, and the notes list them.
    tagged = [r for r in periodic["rows"] if r["deviations"]]
    assert html.count('class="ck-dev-tag"') == len(tagged)
    if tagged:
        assert "Modified setups (" in html


def test_checks_with_missing_experiments_say_how_many_ran() -> None:
    chk = {
        "state": "pass",
        "n_run": 1,
        "n_gates": 5,
        "blocked_as": "Blocked",
        "metrics": [],
    }
    cell = scoreboard._check_td(chk)
    assert "Pass</span>" in cell
    assert "1 of 5 run" in cell
    assert "1 of 5 run" in scoreboard._check_td({**chk, "state": "flagged"})
    assert "run</span>" not in scoreboard._check_td({**chk, "n_run": 5})


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


def test_cost_compares_solvers_on_their_configured_device(tmp_path: Path) -> None:
    """A CPU solver is compared with the CPU solvers even when the machine that
    measured it had a GPU."""
    _snapshot(tmp_path, [_row("cost/spatial_cost", {s: _cell() for s in SOLVERS})])
    d = tmp_path / "ns-grid" / "cost" / "spatial_cost"
    d.mkdir(parents=True)
    t4 = {"gpus": ["Tesla T4, 16384 MiB"], "cpu": "EPYC"}
    data = {
        "results": [
            _timing(s, t, 0.01) for s, t in zip(SOLVERS, (1.0, 2.0, 4.0), strict=True)
        ],
        "extras": {"hardware": t4},
    }
    (d / "result.json").write_text(json.dumps(data))
    meta = {
        "ns-grid": {
            **META["ns-grid"],
            "PICT": {"display": "PICT", "anchor": "", "gpu": False},
        }
    }
    setup = _setup(scoreboard.compute(tmp_path, meta), "ns-2d-periodic")
    pict = _solver(setup, "PICT")
    assert not pict["gpu"]
    assert pict["cost"]["Forward cost"]["ratio"] == 1.0
    assert _solver(setup, "XLB")["cost"]["Forward cost"]["ratio"] == 2.0
    html = scoreboard.render_setup(setup)
    assert "measured on Tesla T4" in html


def test_cost_compares_every_pass_at_one_size(tmp_path: Path) -> None:
    """One N for the forward and VJP columns; a solver that completed fewer
    than half the sizes is left out of the comparison."""
    _snapshot(tmp_path, [_row("cost/spatial_cost", {s: _cell() for s in SOLVERS})])
    sizes = {"PhiFlow": (16, 32, 64, 128), "XLB": (16, 32, 64), "PICT": (16,)}

    def write(experiment: str, scale: float) -> None:
        d = tmp_path / "ns-grid" / "cost" / experiment
        d.mkdir(parents=True)
        rows = [
            {**_timing(s, scale * n, 0.01), "sweep_value": str(n)}
            for s, ns in sizes.items()
            for n in ns
        ]
        (d / "result.json").write_text(json.dumps({"results": rows}))

    write("spatial_cost", 1.0)
    write("vjp_cost/by_N", 2.0)
    setup = _setup(scoreboard.compute(tmp_path, META), "ns-2d-periodic")
    assert setup["cost_n"][True] == 64
    for name in ("PhiFlow", "XLB"):
        for label in ("Forward cost", "VJP cost"):
            assert _solver(setup, name)["cost"][label]["n"] == 64
    pict = _solver(setup, "PICT")["cost"]["Forward cost"]
    assert pict == {"ratio": None, "reach": 16}
    html = scoreboard.render_setup(setup)
    assert "Forward and VJP at N=64," in html
    assert "stops at N=16" in html


def test_time_to_plateau_has_its_own_column() -> None:
    def row(name: str, seconds: float | None) -> dict:
        checks = {
            "optimization": {
                "state": "pass",
                "n_run": 1,
                "n_gates": 1,
                "blocked_as": "Blocked",
                "metrics": [],
            }
        }
        return {
            "name": name,
            "anchor": "",
            "gpu": True,
            "checks": checks,
            "cost": {},
            "flags": [],
            "limits": [],
            "unsupported": [],
            "deviations": [],
            "effort": ("L0", ""),
            "plateau": {"iters": 30.0, "n_iters": 60, "seconds": seconds},
            "accuracy_cost": None,
        }

    setup = {
        "key": "ns-2d-cylinder",
        "id": "F2",
        "label": "",
        "setup": "",
        "task": "",
        "suites": ["optimization"],
        "has_cost": False,
        "cost_n": {},
        "gpu_model": "",
        "has_effort": True,
        "rows": [row("XLB", 30.0), row("PICT", 90.0)],
    }
    html = scoreboard.render_setup(setup)
    assert ">Time to plateau</th>" in html
    assert "plateau after" not in html
    assert 'data-sort="90.0"' in html and "3.0×" in html
    assert "Multiples of the fastest GPU solver" in html
    # Without any time, the column counts iterations.
    for r in setup["rows"]:
        r["plateau"]["seconds"] = None
    assert ">Iterations to plateau</th>" in scoreboard.render_setup(setup)


def test_cost_to_error_interpolates_between_grids() -> None:
    def runs(*points):
        return [
            {
                "grid": g,
                "median_errors": [e / 4, e],
                "seconds_per_time": c,
                "diverged": c is None,
            }
            for g, e, c in points
        ]

    result = {
        "results": [
            # 10% lies halfway between 40% and 2.5% on a log scale.
            {
                "solver": "A",
                "metrics": {"runs": runs((32, 0.4, 1.0), (64, 0.025, 4.0))},
            },
            {
                "solver": "B",
                "metrics": {"runs": runs((32, 0.05, 1.0), (64, 0.01, 4.0))},
            },
            {"solver": "C", "metrics": {"runs": runs((32, 0.5, 1.0), (64, 0.2, 4.0))}},
            {
                "solver": "D",
                "metrics": {"runs": runs((32, 2.0, None), (64, 2.0, None))},
            },
        ]
    }
    out = scoreboard.cost_to_error(result, 0.1)
    assert out["A"]["state"] == "reached"
    assert out["A"]["seconds"] == pytest.approx(2.0)
    assert out["A"]["grids"] == (32, 64)
    assert out["B"] == {"state": "coarsest", "seconds": 1.0, "grids": (32,)}
    assert out["C"] == {"state": "not_reached", "best": 0.2}
    assert out["D"] == {"state": "diverged"}
    cells = [scoreboard._accuracy_cost_td(out[s], 0.1, 1.0) for s in "ABCD"]
    assert "2.0×" in cells[0]
    assert "≤ " in cells[1]
    assert "not reached" in cells[2] and "best 20%" in cells[2]
    assert "diverges" in cells[3]


def test_check_tables_name_solvers_that_do_not_run_other_setups() -> None:
    def setup(key: str, states: dict[str, str]) -> dict:
        rows = [
            {"name": n, "checks": {"forward": {"state": st}}}
            for n, st in states.items()
        ]
        return {"key": key, "setup": key.title(), "rows": rows}

    periodic = setup("ns-2d-periodic", {"Exponax": "pass", "XLB": "pass"})
    cylinder = setup("ns-2d-cylinder", {"Exponax": "na", "XLB": "flagged"})
    heat = setup("heat", {"FEniCS": "fail"})
    scoreboard._mark_setups_elsewhere([periodic, cylinder, heat])
    assert periodic["elsewhere"] == {"Exponax": [cylinder]}
    assert cylinder["elsewhere"] == {}
    # Setups of another domain don't count.
    assert heat["elsewhere"] == {}
