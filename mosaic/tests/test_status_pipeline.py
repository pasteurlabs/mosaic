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
        "harness_fn": "mosaic.benchmarks.suites.forward.run_agreement",
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
        "harness_fn": "mosaic.benchmarks.suites.gradient.run_fd_check",
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
    """JSON snapshot produced by snapshot_to_dict is valid and contains expected fields."""
    from mosaic.benchmarks.core.status import collect_status, snapshot_to_dict
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-grid")
    st = collect_status(cfg, suites=["forward"])
    snapshot = snapshot_to_dict([st])

    # Snapshot must be JSON-serializable
    json_str = json.dumps(snapshot)
    parsed = json.loads(json_str)

    assert isinstance(parsed, dict)
    assert "problems" in parsed
    # snapshot_to_dict keys problems by name (dict), not a list
    assert isinstance(parsed["problems"], dict)
    assert len(parsed["problems"]) >= 1

    problem_data = parsed["problems"]["ns-grid"]
    assert "problem" in problem_data
    assert "score" in problem_data
    assert "rows" in problem_data


def test_score_computation():
    """Score weights are applied correctly for different cell types."""
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
    assert score is not None
    assert 0.0 < score < 1.0  # not all ok, not all failed
    # 2 ok (1.0 each) + 1 fail (0.0) + 1 missing (0.33) = 2.33/4 = 0.5825
    assert 0.5 < score < 0.65


def test_score_all_ok():
    """All-ok cells should give score 1.0."""
    from mosaic.benchmarks.core.status import OK, Cell, compute_score

    cells = [Cell(status=OK) for _ in range(5)]
    score, n = compute_score(cells)
    assert score == 1.0
    assert n == 5


def test_score_all_failed():
    """All-failed cells should give score 0.0."""
    from mosaic.benchmarks.core.status import FAILED, Cell, compute_score

    cells = [Cell(status=FAILED) for _ in range(3)]
    score, n = compute_score(cells)
    assert score == 0.0
    assert n == 3


def test_score_empty():
    """Empty cell list returns None score."""
    from mosaic.benchmarks.core.status import compute_score

    score, n = compute_score([])
    assert score is None
    assert n == 0


def test_score_categorical_excluded_not_counted():
    """Categorical exclusions are excluded from the denominator."""
    from mosaic.benchmarks.core.status import (
        EXCL_CATEGORICAL,
        EXCLUDED,
        OK,
        Cell,
        compute_score,
    )

    cells = [
        Cell(status=OK),
        Cell(status=EXCLUDED, category=EXCL_CATEGORICAL),
        Cell(status=EXCLUDED, category=EXCL_CATEGORICAL),
    ]
    score, n = compute_score(cells)
    # Only the OK cell contributes
    assert n == 1
    assert score == 1.0


def test_score_stale_ok_penalised():
    """Stale OK cells receive a lower weight than fresh OK cells."""
    from mosaic.benchmarks.core.status import OK, SCORE_WEIGHTS, Cell, compute_score

    fresh = [Cell(status=OK)]
    stale = [Cell(status=OK, stale=True)]
    fresh_score, _ = compute_score(fresh)
    stale_score, _ = compute_score(stale)
    assert fresh_score is not None
    assert stale_score is not None
    assert fresh_score > stale_score
    assert fresh_score == SCORE_WEIGHTS["ok"]
    assert stale_score == SCORE_WEIGHTS["ok*"]


def test_cell_weight_key_mapping():
    """cell_weight_key returns the correct SCORE_WEIGHTS key for each status."""
    from mosaic.benchmarks.core.status import (
        ANOMALY,
        EXCL_CATEGORICAL,
        EXCL_UNSPECIFIED,
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
    assert cell_weight_key(Cell(status=EXCLUDED, category=EXCL_UNSPECIFIED)) == "excl"
    assert cell_weight_key(Cell(status=EXCLUDED, category=EXCL_CATEGORICAL)) is None


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
