# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict execution checks survive the core optimization-summary pipeline."""

from copy import deepcopy

import pytest

from mosaic.benchmarks.core.status import (
    OK,
    Cell,
    _collect_recovery_metrics,
    _refine_recovery,
)
from mosaic.benchmarks.core.status_checks import OptimizationSummary
from mosaic.benchmarks.problems.buoyancy_grid.status import corrector_execution


def valid_metrics():
    row = {
        "status": "ok",
        "updates": 2,
        "unroll": 1,
        "train_frames": 2,
        "window_seed": 0,
        "window_starts": [0, 1],
        "validation_interval": 1,
        "eval_frames": 2,
        "seeds": {"train": 0, "validation": 1, "test": 100},
        "gradient_check": {"relative_error": 0.001},
        "gradient_check_nonzero": {"relative_error": 0.002},
        "reference": {"time_refinement_error": 0.001, "audit_gap_ratio": 0.1},
        "recurrence_error": 0.0,
        "learning_rates": [0.003],
        "divergence_rms": {},
    }
    row["reference"]["splits"] = {
        split: {
            "time_refinement_error": 0.001,
            "coarse_gap": 0.01,
            "audit_gap_ratio": 0.1,
        }
        for split in ("train", "validation", "test")
    }
    for arm in (
        "reference",
        "uncorrected",
        "solver_in_loop",
    ):
        row["divergence_rms"][arm] = [0.0] * 3
        if arm == "reference":
            continue
        row[arm] = {
            key: [0.0, 0.1, 0.2] for key in ("loss", "velocity_mse", "scalar_mse")
        }
        if arm != "uncorrected":
            row[arm].update(
                {
                    "selected_candidate": 0,
                    "selected_update": 1,
                    "total_training_seconds": 1.0,
                    "total_validation_seconds": 0.5,
                    "total_seconds": 1.5,
                    "candidates": [
                        {
                            "learning_rate": 0.003,
                            "validation_loss": 0.1,
                            "selected_update": 1,
                            "validation_checkpoints": [
                                {"update": 0, "loss": 0.2},
                                {"update": 1, "loss": 0.1},
                                {"update": 2, "loss": 0.2},
                            ],
                            "training_loss": [0.2, 0.1],
                            "completed_updates": 2,
                            "training_seconds": 1.0,
                            "validation_seconds": 0.5,
                        }
                    ],
                }
            )
    return row


def test_complete_execution_passes_even_without_superiority():
    row = valid_metrics()
    row["solver_in_loop"]["loss"] = [0.0, 1.0, 2.0]
    assert corrector_execution(OptimizationSummary(metrics=row)) is None


def test_identity_selection_still_requires_complete_training_budget():
    row = valid_metrics()
    result = row["solver_in_loop"]
    candidate = result["candidates"][0]
    candidate["validation_checkpoints"][0]["loss"] = 0.1
    candidate["selected_update"] = result["selected_update"] = 0
    assert corrector_execution(OptimizationSummary(metrics=row)) is None
    candidate["completed_updates"] = 0
    assert corrector_execution(OptimizationSummary(metrics=row)) is not None


def test_reported_final_loss_uses_trained_rollout_not_baseline():
    row = valid_metrics()
    row["uncorrected"]["loss"][-1] = 0.9
    row["final_loss"] = row["solver_in_loop"]["loss"][-1]
    cells = {"phiflow": Cell(OK)}
    _collect_recovery_metrics({"by_solver": {"phiflow": row}}, cells)
    assert cells["phiflow"].metrics["final_loss"] == 0.2


@pytest.mark.parametrize("key", ["solver_in_loop", "uncorrected"])
def test_missing_training_or_baseline_rollout_rejected(key):
    row = valid_metrics()
    row.pop(key)
    assert corrector_execution(OptimizationSummary(metrics=row)) is not None


@pytest.mark.parametrize(
    "path,value",
    [
        (("gradient_check", "relative_error"), -1.0),
        (("gradient_check_nonzero", "relative_error"), 0.1),
        (("reference", "time_refinement_error"), 0.06),
        (("reference", "audit_gap_ratio"), 0.3),
        (("recurrence_error",), 0.001),
        (("seeds", "test"), 0),
        (("learning_rates",), [float("nan")]),
        (("window_starts",), [0, 0]),
        (("validation_interval",), 0),
        (("solver_in_loop", "selected_update"), 2),
        (("solver_in_loop", "total_seconds"), 1.0),
        (("solver_in_loop", "total_training_seconds"), float("nan")),
        (("solver_in_loop", "loss"), [0.0, 0.1]),
        (("divergence_rms", "solver_in_loop"), [0.0, float("inf"), 0.0]),
    ],
)
def test_bad_metrics_rejected_through_summary_pipeline(path, value):
    row = valid_metrics()
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    cells = {"phiflow": Cell(OK)}
    _refine_recovery(
        {"by_solver": {"phiflow": {"default": row}}}, cells, [corrector_execution]
    )
    assert cells["phiflow"].status != OK


def test_missing_updates_or_mismatched_candidate_rejected():
    for key, value in (("completed_updates", 1), ("learning_rate", 0.1)):
        row = deepcopy(valid_metrics())
        row["solver_in_loop"]["candidates"][0][key] = value
        assert corrector_execution(OptimizationSummary(metrics=row)) is not None


def test_missing_or_explicitly_failed_record_rejected():
    for row in ({}, {"status": "failed"}):
        assert corrector_execution(OptimizationSummary(metrics=row)) is not None


def test_heldout_reference_cannot_be_omitted_or_hidden_by_summary():
    row = valid_metrics()
    row["reference"]["splits"].pop("test")
    assert corrector_execution(OptimizationSummary(metrics=row)) is not None
    row = valid_metrics()
    row["reference"]["splits"]["test"]["audit_gap_ratio"] = 0.5
    assert corrector_execution(OptimizationSummary(metrics=row)) is not None


@pytest.mark.parametrize("change", ["missing_final", "nonfinite", "late_tie"])
def test_validation_checkpoint_selection_is_strict(change):
    row = valid_metrics()
    candidate = row["solver_in_loop"]["candidates"][0]
    if change == "missing_final":
        candidate["validation_checkpoints"].pop()
    elif change == "nonfinite":
        candidate["validation_checkpoints"][2]["loss"] = float("nan")
    else:
        candidate["validation_checkpoints"][0]["loss"] = 0.1
    assert corrector_execution(OptimizationSummary(metrics=row)) is not None
