"""CI must reject skipped or partial corrector runs without demanding superiority."""

import pytest

from mosaic.benchmarks.core.status import Cell, _refine_recovery
from mosaic.benchmarks.core.status_checks import OptimizationSummary
from mosaic.benchmarks.problems.navier_stokes_grid.corrector_status import (
    corrector_execution,
)


def valid_metrics():
    return {
        "reference_kind": "solver_self_refined",
        "eligible_for_corrector_training": True,
        "reference_convergence_passed": True,
        "valid_for_vjp_ranking": True,
        "completed": True,
        "supervised_completed": True,
        "model_seeds": [0],
        "total_optimizer_updates": 16,
        "stop_gradient_total_optimizer_updates": 16,
        "supervised_total_optimizer_updates": 16,
        "end_to_end_fd_checks_by_seed": [[{"relative_error": 0.001}]],
        "final_grad_norm": 0.2,
        "stop_gradient_final_grad_norm": 0.2,
        "mean_rollout_error": 0.2,
        "stop_gradient_mean_rollout_error": 0.1,
        "supervised_mean_rollout_error": 0.1,
        "final_train_loss": 0.5,
        "stop_gradient_final_train_loss": 0.4,
    }


def test_short_run_need_not_outperform_supervision():
    assert corrector_execution(16)(OptimizationSummary(metrics=valid_metrics())) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("eligible_for_corrector_training", False),
        ("reference_convergence_passed", False),
        ("valid_for_vjp_ranking", False),
        ("completed", False),
        ("status", "failed"),
        ("failure_type", "timeout"),
        ("supervised_completed", False),
        ("supervised_total_optimizer_updates", 15),
        ("stop_gradient_total_optimizer_updates", 0),
        ("end_to_end_fd_checks_by_seed", []),
        ("end_to_end_fd_checks_by_seed", [[{"relative_error": 0.1}]]),
        ("final_grad_norm", float("nan")),
    ],
)
def test_partial_or_invalid_run_is_anomalous(field, value):
    metrics = valid_metrics()
    metrics[field] = value
    assert corrector_execution(16)(OptimizationSummary(metrics=metrics))[0] == "anomaly"


def test_actual_status_classifier_checks_every_run():
    good, bad = valid_metrics(), valid_metrics()
    bad["supervised_completed"] = False
    cells = {"solver": Cell("ok")}
    _refine_recovery(
        {"by_solver": {"solver": {"1": good, "2": bad}}},
        cells,
        [corrector_execution(16)],
    )
    assert cells["solver"].status == "anomaly"


def test_missing_metrics_cannot_pass():
    assert corrector_execution(16)(OptimizationSummary())[0] == "anomaly"


def test_failed_nested_run_cannot_hide_behind_successful_run():
    metrics = {
        "1": valid_metrics(),
        "2": {"status": "failed", "failure_type": "timeout", "elapsed_s": 30},
    }
    assert corrector_execution(16)(OptimizationSummary(metrics=metrics))[0] == "anomaly"
