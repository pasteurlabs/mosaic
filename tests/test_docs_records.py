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

from docs import generate, generate_results  # noqa: E402

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
    assert views.iv.setup_of("ns-3d-grid", "cost/spatial_cost").key == "ns-3d"
    assert views.iv.setup_of("unknown-problem", "forward/x") is None


def test_effort_block_has_one_table_per_setup(results: Path) -> None:
    block = "\n".join(generate_results._effort_block("ns-grid"))
    assert "## Setup and effort" in block
    assert 'class="m-effort-select"' in block
    assert "**Channel flow past a cylinder**" in block
    assert "**Periodic box (2D)**" in block
    # XLB's sub-stepping deviation shows on hover; a solver without a record
    # says so instead of claiming no effort.
    assert "ma-targeted-substepping" in block
    assert "| Unrecorded | not recorded | not recorded | not recorded |" in block


def test_ranking_table_gains_effort_columns(results: Path) -> None:
    block = "\n".join(generate_results._leaderboard_block("ns-grid", "forward"))
    assert "| Solver | Mean rel. error | Effort | Deviations |" in block
    assert "{.sortable-table .effort-table}" in block


def test_no_results_means_no_effort_block(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(generate_results, "RESULTS_DIR", tmp_path)
    assert generate_results._effort_block("ns-grid") == []


def test_effort_cell_names_the_required_interventions() -> None:
    record = views.records_for("ns-2d-cylinder")["PICT"]
    cell = views.effort_cell(record, "ns-2d-cylinder")
    assert cell.startswith("[L2]{.m-tip")
    assert "body-force-integration" in cell


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
    assert "**Reason unknown.**" in qmd
    assert "**Setup deviations**" in qmd
