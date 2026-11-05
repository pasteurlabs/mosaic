# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intervention records on the results pages and the solver reference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docs.scripts import generate, generate_results  # noqa: E402

views = generate_results.views


def _write_result(root: Path, experiment: str, rows: list[dict]) -> None:
    d = root / "ns-grid" / experiment
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"results": rows}))


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_result(
        tmp_path,
        "forward/cylinder",
        [{"solver": "OpenFOAM", "metrics": {}}, {"solver": "XLB", "metrics": {}}],
    )
    _write_result(
        tmp_path,
        "forward/tgv_nu_sweep",
        [
            {"solver": "XLB", "metrics": {"error": 1e-3}},
            {"solver": "Unrecorded", "metrics": {"error": 2e-3}},
        ],
    )
    monkeypatch.setattr(generate_results, "RESULTS_DIR", tmp_path)
    return tmp_path


def test_experiments_map_to_setups() -> None:
    assert views.iv.setup_of("ns-grid", "forward/cylinder").key == "ns-2d-cylinder"
    assert views.iv.setup_of("ns-grid", "optimization/drag_opt").key == (
        "ns-2d-cylinder"
    )
    assert views.iv.setup_of("ns-grid", "forward/agreement/tgv").key == (
        "ns-2d-periodic"
    )
    for exp in ("forward/kolmogorov", "forward/kolmogorov_laminar"):
        assert views.iv.setup_of("ns-grid", exp).key == "ns-2d-kolmogorov"
    assert views.iv.setup_of("ns-grid", "gradient/chain_fd_check").key == (
        "ns-2d-kolmogorov"
    )
    assert views.iv.setup_of("ns-3d-grid", "cost/spatial_cost").key == "ns-3d"
    assert views.iv.setup_of("unknown-problem", "forward/x") is None
    # Initial conditions belong to the setup whose experiments start from them.
    assert views.iv.setup_of("ns-grid", "ics/dfg_inflow").key == "ns-2d-cylinder"
    assert views.iv.setup_of("ns-grid", "ics/rest").key == "ns-2d-kolmogorov"
    assert views.iv.setup_of("ns-grid", "ics/tgv").key == "ns-2d-periodic"


def test_effort_block_covers_one_setup(results: Path) -> None:
    block = "\n".join(generate_results._effort_block("ns-grid", "ns-2d-periodic"))
    assert "## Effort and deviations" in block
    assert "OpenFOAM" not in block  # only ran the cylinder setup
    # A solver without a record says so instead of claiming no effort.
    assert "| Unrecorded | not recorded | not recorded | not recorded |" in block
    cylinder = "\n".join(generate_results._effort_block("ns-grid", "ns-2d-cylinder"))
    # XLB's sub-stepping deviation shows its title on hover.
    xlb = views.records_for("ns-2d-cylinder")["XLB"]
    substepping = next(d for d in xlb.deviations if d.id == "ma-targeted-substepping")
    assert views.title(substepping) in cylinder


def test_ranking_table_links_its_experiment(results: Path) -> None:
    block = "\n".join(
        generate_results._leaderboard_block(
            "ns-grid", "forward", "ns-2d-periodic", frozenset({"tgv_nu_sweep"})
        )
    )
    # Effort and deviations are in the check table at the top of the page.
    assert "| Solver | Mean rel. error |" in block
    # The caption links to the experiment the ranking was measured in, or to
    # the suite when that experiment has no heading on the page.
    assert "(#exp-forward-tgv-nu-sweep)" in block
    unshown = "\n".join(
        generate_results._leaderboard_block("ns-grid", "forward", "ns-2d-periodic")
    )
    assert "(#suite-forward)" in unshown
    # The cylinder setup's own forward experiment has no error metric.
    assert (
        generate_results._leaderboard_block("ns-grid", "forward", "ns-2d-cylinder")
        == []
    )


def test_optimization_table_shows_iterations_to_plateau(results: Path) -> None:
    _write_result(
        results,
        "optimization/drag_opt",
        [
            {"solver": "XLB", "metrics": {"final_drag": 3.1, "iters_to_plateau": 120}},
            {"solver": "PICT", "metrics": {"final_drag": 3.0}},
        ],
    )
    block = "\n".join(
        generate_results._leaderboard_block("ns-grid", "optimization", "ns-2d-cylinder")
    )
    assert "| Solver | Final drag | Iterations to plateau |" in block
    assert "| XLB | 3.10e+00 | 120 |" in block
    # Results recorded before the harness kept the metric show a dash.
    assert "| PICT | 3.00e+00 | — |" in block


def test_time_to_plateau_prorates_the_wall_time(results: Path) -> None:
    d = results / "ns-grid" / "optimization" / "drag_opt"
    d.mkdir(parents=True)
    rows = [
        {
            "solver": "XLB",
            "metrics": {"final_drag": 3.1, "iters_to_plateau": 30, "n_iters": 60},
        },
        {"solver": "PICT", "metrics": {"final_drag": 3.0, "iters_to_plateau": 10}},
    ]
    provenance = {"wall_time_s": {"XLB": 120.0, "PICT": 50.0}}
    (d / "result.json").write_text(
        json.dumps({"results": rows, "provenance": provenance})
    )
    block = "\n".join(
        generate_results._leaderboard_block("ns-grid", "optimization", "ns-2d-cylinder")
    )
    assert "| Solver | Final drag | Iterations to plateau | Time to plateau |" in block
    # Half of the 60 iterations of a 120 s run.
    assert "| XLB | 3.10e+00 | 30 | 60 s |" in block
    # Without the number of iterations run there is no estimate.
    assert "| PICT | 3.00e+00 | 10 | — |" in block


def test_recorded_time_to_plateau_wins_over_the_estimate(results: Path) -> None:
    d = results / "ns-grid" / "optimization" / "drag_opt"
    d.mkdir(parents=True)
    metrics = {"final_drag": 3.1, "iters_to_plateau": 30, "n_iters": 60}
    rows = [
        {"solver": "XLB", "metrics": {**metrics, "seconds_to_plateau": 12.0}},
        {"solver": "PICT", "metrics": {**metrics, "final_drag": 3.0}},
    ]
    provenance = {"wall_time_s": {"XLB": 120.0, "PICT": 50.0}}
    (d / "result.json").write_text(
        json.dumps({"results": rows, "provenance": provenance})
    )
    plateaus = generate_results.scoreboard.plateau(
        json.loads((d / "result.json").read_text())
    )
    assert plateaus["XLB"] == {
        "iters": 30.0,
        "n_iters": 60,
        "seconds": 12.0,
        "estimated": False,
    }
    assert plateaus["PICT"]["estimated"] and plateaus["PICT"]["seconds"] == 25.0


def test_no_results_means_no_effort_block(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(generate_results, "RESULTS_DIR", tmp_path)
    assert generate_results._effort_block("ns-grid", "ns-2d-periodic") == []


def test_every_setup_gets_a_page_without_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(generate_results, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(generate_results, "OUTPUT_DIR", tmp_path)
    for name in ("SCOREBOARD_INCLUDE", "EXPLORER_INCLUDE", "SCORES_JSON"):
        monkeypatch.setattr(generate_results, name, tmp_path / name)
    monkeypatch.setattr(generate_results, "GATES_INCLUDE", tmp_path / "GATES_INCLUDE")
    generate_results.main()
    for setup in generate_results.scoreboard.setups():
        page = (
            tmp_path / "results" / generate_results.setup_page(setup["key"])
        ).read_text()
        assert f'title: "{setup["setup"]}"' in page
        assert "No benchmark results are available" in page


def test_setup_pages_only_show_their_experiments() -> None:
    tree = {
        "forward": {"cylinder": [], "tgv_nu_sweep": []},
        "ics": {"dfg_inflow": [], "tgv": []},
        "cost": {"": []},
    }
    cylinder = generate_results._setup_suites("ns-grid", tree, "ns-2d-cylinder")
    assert cylinder == {"forward": {"cylinder": []}, "ics": {"dfg_inflow": []}}
    periodic = generate_results._setup_suites("ns-grid", tree, "ns-2d-periodic")
    assert set(periodic) == {"forward", "ics", "cost"}
    assert "cylinder" not in periodic["forward"]


def test_no_required_intervention_reads_as_l0() -> None:
    class Record:
        def interventions_for(self, setup: str) -> list:
            return []

        def effort(self, setup: str) -> None:
            return None

    text, detail = views.effort_parts(Record(), "heat")
    assert text == "L0"
    assert "nothing beyond expected tweaks" in detail


def test_effort_cell_names_the_required_interventions() -> None:
    record = views.records_for("ns-2d-cylinder")["PICT"]
    cell = views.effort_cell(record, "ns-2d-cylinder")
    assert cell.startswith("[L2]{.m-tip")
    force = next(i for i in record.interventions if i.id == "body-force-integration")
    assert views.title(force) in cell


def test_attribute_text_cannot_break_a_pipe_table() -> None:
    text = views.attr('a | b "c" {d}\nnext line')
    assert "|" not in text and '"' not in text and "\n" not in text
    assert "{" not in text


def test_linkify_links_pull_requests_and_urls() -> None:
    out = views.linkify("See #222, https://example.org/x. and issues/38#123")
    assert "[#222](https://github.com/pasteurlabs/mosaic/pull/222)" in out
    assert "<https://example.org/x>." in out
    assert "issues/38#123" in out


def test_solver_reference_shows_every_record() -> None:
    categories = generate.collect_solvers()
    solvers = [s for group in categories.values() for s in group]
    assert all(s["record"] is not None for s in solvers)
    qmd = generate.generate_qmd(categories)
    assert qmd.count("#### Interventions and deviations") == len(solvers)
    assert "`mumps-adjoint-monkeypatch`" in qmd
    assert "| Setup | Effort |" in qmd
    assert "**Setup deviations**" in qmd
