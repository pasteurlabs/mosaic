"""Use the real classifier to prevent a reporting-only CI success."""

import importlib.util
import json
from pathlib import Path

import pytest

from mosaic.benchmarks.core.status import _resolve_harness_hash, _resolve_tesseract_hash
from mosaic.benchmarks.problems import get_config
from tests.test_corrector_status import valid_metrics

SCRIPT = Path(__file__).parents[1] / ".github/scripts/check-corrector-result.py"
spec = importlib.util.spec_from_file_location("corrector_ci_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg = get_config("ns-grid")
    name = next(s.name for s in cfg.solvers if s.uses_gpu)
    path = tmp_path / cfg.name / "optimization/solver_in_loop/result.json"
    path.parent.mkdir(parents=True)
    return cfg, name, path


def save_result(path, name, metrics):
    fn = "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop.solver_in_loop"
    data = {
        "schema_version": 1,
        "results": [{"solver": name, "sweep_value": None, "metrics": metrics}],
        "provenance": {
            "harness_fn": fn,
            "harness_hash": _resolve_harness_hash(fn, {}),
            "tesseract_hashes": {
                name: _resolve_tesseract_hash(get_config("ns-grid"), name, {})
            },
        },
    }
    path.write_text(json.dumps(data))


def test_valid_run_passes_even_when_supervision_wins(results):
    cfg, name, path = results
    save_result(path, name, valid_metrics())
    assert gate.check_results(cfg, "gpu", name) == []


@pytest.mark.parametrize(
    "change",
    [
        {"eligible_for_corrector_training": False, "total_optimizer_updates": 0},
        {"supervised_completed": False},
        {"status": "failed", "failure_type": "error"},
        {"status": "failed", "failure_type": "timeout"},
        {"stop_gradient_total_optimizer_updates": 15},
        {"end_to_end_fd_checks_by_seed": []},
        {"end_to_end_fd_checks_by_seed": [[{"relative_error": 0.1}]]},
        {"final_grad_norm": float("nan")},
    ],
)
def test_invalid_execution_fails_real_classifier(results, change):
    cfg, name, path = results
    save_result(path, name, valid_metrics() | change)
    assert gate.check_results(cfg, "gpu", name)


def test_missing_result_fails(results):
    cfg, name, _ = results
    assert gate.check_results(cfg, "gpu", name)


def test_missing_other_requested_solver_is_not_hidden(results):
    cfg, name, path = results
    save_result(path, name, valid_metrics())
    assert gate.check_results(cfg, "gpu")


def test_hardware_and_categorical_exclusions_are_respected(results):
    cfg, name, _ = results
    assert gate.check_results(cfg, "cpu", name) == []
    openfoam = next(s.name for s in cfg.solvers if "foam" in s.name.lower())
    assert gate.check_results(cfg, "gpu", openfoam) == []
    assert gate.check_results(cfg, "cpu", openfoam) == []


def test_process_exit_status_matches_classifier(results, monkeypatch):
    _cfg, name, path = results
    monkeypatch.setattr(
        "sys.argv", [str(SCRIPT), "--hardware", "gpu", "--solvers", name]
    )
    assert gate.main() == 1
    save_result(path, name, valid_metrics())
    assert gate.main() == 0


def test_unrelated_legacy_anomaly_does_not_block_corrector(results):
    cfg, name, path = results
    save_result(path, name, valid_metrics())
    other = path.parents[1] / "drag_opt" / "result.json"
    other.parent.mkdir()
    save_result(other, name, {"status": "failed", "failure_type": "error"})
    assert gate.check_results(cfg, "gpu", name) == []


def test_stale_valid_result_is_not_a_fresh_ci_pass(results):
    cfg, name, path = results
    save_result(path, name, valid_metrics())
    data = json.loads(path.read_text())
    data["provenance"]["harness_hash"] = "obsolete"
    path.write_text(json.dumps(data))
    failures = gate.check_results(cfg, "gpu", name)
    assert failures and "stale" in failures[0]


def test_solver_filter_matches_cli_case_and_problem_map(results):
    cfg, name, path = results
    save_result(path, name, valid_metrics())
    assert gate.check_results(cfg, "gpu", name.lower()) == []
    assert gate.check_results(cfg, "gpu", f"ns-grid={name.lower()}") == []
    path.unlink()
    assert gate.check_results(cfg, "gpu", f"ns-grid={name}")


@pytest.mark.parametrize(
    "selection",
    ["NotASolver", "ns-grid=NotASolver", "ns-grid=", "ns-grid=XLB;malformed"],
)
def test_invalid_solver_selection_cannot_skip_validation(results, selection):
    cfg, _name, _path = results
    assert gate.check_results(cfg, "gpu", selection)
