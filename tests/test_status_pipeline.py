# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the status classification pipeline.

Exercises collect_status, scoring, and snapshot export using mock result files.
No Docker or solver builds required.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def mock_results(tmp_path, monkeypatch):
    """Create a minimal results directory with mock data for two solvers."""
    results_dir = tmp_path / "mosaic-results"

    # Create a mock forward/agreement result with two solvers
    exp_dir = results_dir / "ns-grid" / "forward" / "agreement"
    exp_dir.mkdir(parents=True)

    result = {
        "by_param": {
            "64": {
                "solver_a": {"error": 0.01, "valid": True},
                "solver_b": {"error": float("nan"), "valid": False},
            },
            "128": {
                "solver_a": {"error": 0.005, "valid": True},
                # solver_c not present -> should be NOT_RUN
            },
        },
        "tesseract_hashes": {
            "solver_a": "abcd1234abcd1234",
            "solver_b": "efgh5678efgh5678",
        },
        "harness_hash": "0000111100001111",
        "harness_fn": "mosaic.benchmarks.problems.shared.forward.run_agreement",
    }
    (exp_dir / "result.json").write_text(json.dumps(result))

    # Create a mock gradient/fd_check result
    grad_dir = results_dir / "ns-grid" / "gradient" / "fd_check"
    grad_dir.mkdir(parents=True)

    grad_result = {
        "by_solver": {
            "solver_a": {
                "0.01": {"rel_error": [0.001], "cosine": 0.999},
            },
        },
        "tesseract_hashes": {"solver_a": "abcd1234abcd1234"},
        "harness_hash": "2222333322223333",
        "harness_fn": "mosaic.benchmarks.problems.shared.gradient.run_fd_check",
    }
    (grad_dir / "result.json").write_text(json.dumps(grad_result))

    # Point the results_dir to our mock
    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(results_dir))

    return results_dir


def test_collect_status_classifies_solvers(mock_results):
    """Solvers are classified correctly based on result data."""
    from mosaic.benchmarks.core.status import (
        collect_status,
    )
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-grid")
    st = collect_status(cfg, suites=["forward", "gradient"])

    # Find the forward/agreement row
    agreement_rows = [r for r in st.rows if "agreement" in r.label]
    assert len(agreement_rows) >= 1, (
        f"No agreement row found. Rows: {[r.label for r in st.rows]}"
    )


def test_snapshot_json_is_valid(mock_results):
    """JSON snapshot produced by snapshot_to_dict has the right shape end-to-end.

    Round-trips through ``json.dumps`` to guarantee it's serialisable, then
    asserts on the concrete structure consumers downstream rely on:
    a per-problem entry with ``problem``, ``solvers``, ``rows`` (list of
    ``{label, cells: {solver: {status, ...}}}``), and a numeric or null ``score``.
    """
    from mosaic.benchmarks.core.status import collect_status, snapshot_to_dict
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-grid")
    st = collect_status(cfg, suites=["forward"])
    snapshot = snapshot_to_dict([st])

    parsed = json.loads(json.dumps(snapshot))

    assert set(parsed.keys()) >= {"problems"}
    assert list(parsed["problems"].keys()) == ["ns-grid"]

    problem_data = parsed["problems"]["ns-grid"]
    assert problem_data["problem"] == "ns-grid"
    assert isinstance(problem_data["solvers"], list)
    assert isinstance(problem_data["rows"], list)
    assert problem_data["rows"], (
        "snapshot rows must not be empty for a configured problem"
    )
    # score is either a float in [0, 1] or null when no cells contribute.
    score = problem_data["score"]
    assert score is None or (isinstance(score, int | float) and 0.0 <= score <= 1.0)

    # Every row carries a label and a per-solver cell dict, where each cell
    # has at least a string ``status``. These are the fields the dashboard
    # template and ``compare`` command both depend on.
    for row in problem_data["rows"]:
        assert isinstance(row["label"], str) and row["label"]
        assert isinstance(row["cells"], dict)
        for solver, cell in row["cells"].items():
            assert isinstance(solver, str)
            assert isinstance(cell["status"], str)


def test_score_computation():
    """The mixed-cell score formula matches the published weights.

    Not covered by ``core/test_status.py``: that file tests extremes and
    single-cell transitions; this anchors the specific arithmetic for a
    representative mixed bag (2 ok + 1 fail + 1 missing → 2.33/4 = 0.5825).
    """
    from mosaic.benchmarks.core.status import (
        FAILED,
        NOT_RUN,
        OK,
        Cell,
        compute_score,
    )

    cells = [
        Cell(status=OK),
        Cell(status=OK),
        Cell(status=FAILED, reason="NaN"),
        Cell(status=NOT_RUN),
    ]
    score, n = compute_score(cells)
    assert n == 4
    assert score == pytest.approx(0.5825, abs=0.001)


def test_cell_weight_key_mapping():
    """cell_weight_key returns the correct SCORE_WEIGHTS key for each status."""
    from mosaic.benchmarks.core.config import ExclusionCategory
    from mosaic.benchmarks.core.status import (
        ANOMALY,
        EXCLUDED,
        FAILED,
        NOT_RUN,
        OK,
        Cell,
        cell_weight_key,
    )

    assert cell_weight_key(Cell(status=OK)) == "ok"
    assert cell_weight_key(Cell(status=OK, stale=True)) == "ok*"
    assert cell_weight_key(Cell(status=ANOMALY)) == "anom"
    assert cell_weight_key(Cell(status=ANOMALY, stale=True)) == "anom*"
    assert cell_weight_key(Cell(status=FAILED)) == "fail"
    assert cell_weight_key(Cell(status=FAILED, stale=True)) == "fail*"
    assert cell_weight_key(Cell(status=NOT_RUN)) == "missing"
    assert (
        cell_weight_key(
            Cell(status=EXCLUDED, category=ExclusionCategory.UNSPECIFIED.value)
        )
        == "excl"
    )
    assert (
        cell_weight_key(
            Cell(status=EXCLUDED, category=ExclusionCategory.CATEGORICAL.value)
        )
        is None
    )


def test_status_to_dict_roundtrip():
    """status_to_dict output is JSON-serialisable and retains structure."""
    from mosaic.benchmarks.core.status import (
        FAILED,
        OK,
        Cell,
        ExperimentRow,
        ProblemStatus,
        status_to_dict,
    )

    st = ProblemStatus(
        problem="test-problem",
        solvers=["a", "b"],
        rows=[
            ExperimentRow(
                suite="forward",
                experiment="agreement",
                result_path=None,
                cells={
                    "a": Cell(status=OK),
                    "b": Cell(status=FAILED, reason="diverged"),
                },
            ),
        ],
    )
    d = status_to_dict(st)
    json_str = json.dumps(d)
    parsed = json.loads(json_str)

    assert parsed["problem"] == "test-problem"
    assert parsed["solvers"] == ["a", "b"]
    assert len(parsed["rows"]) == 1
    row = parsed["rows"][0]
    assert row["label"] == "forward/agreement"
    assert row["cells"]["a"]["status"] == "ok"
    assert row["cells"]["b"]["status"] == "failed"
    assert row["cells"]["b"]["reason"] == "diverged"
    assert parsed["score"] is not None
