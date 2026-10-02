# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit pilot split isolation, failure gates and cost accounting with mock solvers."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.flow_control import control, pilot
from experiments.flow_control.baselines import CostLedger, Work
from experiments.flow_control.control import ControlConfig, Task


@pytest.fixture
def harness(monkeypatch):
    config = ControlConfig(control_slots=2)
    state = {"events": [], "bad_seed": None, "training_completed": True}

    def task(seed):
        initial = np.full((2, 2, 1, 2), seed / 1000, dtype=np.float32)
        goal = initial + 0.01
        return Task(
            seed=seed,
            initial=initial,
            goal=goal,
            fine_initial=initial.copy(),
            fine_goal=goal.copy(),
            generating_controls=np.full((2, 8), seed / 1000, dtype=np.float32),
            initial_seed=seed,
            goal_seed=10_000_000 + seed,
            fine_goal_rollout=np.stack([initial, goal, goal]),
        )

    def prepare(_t, _ctx, _config, seeds, work):
        state["events"].append(("prepare", tuple(seeds)))
        tasks = [task(seed) for seed in seeds]
        work.wall_time_s += 10 * len(seeds)
        work.rollout_count += 2 * len(seeds)
        return (
            tasks,
            [
                {"task_seed": seed, "passed": seed != state["bad_seed"]}
                for seed in seeds
            ],
            np.stack([value.fine_goal for value in tasks]),
        )

    def fit(tasks, controls, _config, *, label_costs, **kwargs):
        state["events"].append(("fit_start", tuple(value.seed for value in tasks)))
        state["fit_labels"] = [np.array(value, copy=True) for value in controls]
        state["fit_tasks"] = tasks
        if state.get("raise_training"):
            raise FloatingPointError("mock numerical training failure")
        costs = copy.deepcopy(label_costs)
        costs.phases["policy_training"] = Work(wall_time_s=7, optimizer_updates=1000)
        state["events"].append(("fit_finished", tuple(kwargs["checkpoint_updates"])))
        return {
            "model": "final-model",
            "completed": state["training_completed"],
            "updates": 1000 if state["training_completed"] else 1,
            "trace": [],
            "costs": costs.to_dict(),
            "checkpoint_models": {
                250: "model250",
                500: "model500",
                1000: "final-model",
            },
        }

    def improved(_t, _ctx, tasks, _config, ledger):
        state["events"].append(("expert", tuple(value.seed for value in tasks)))
        ledger.phases["label_generation"] = Work(
            wall_time_s=11,
            rollout_count=6,
            gradient_rollout_count=4,
            optimizer_updates=4,
            label_examples=len(tasks),
        )
        return (
            [value.generating_controls + 0.001 for value in tasks],
            [{"task_seed": value.seed, "label_source": "shooting"} for value in tasks],
        )

    def rollout(_t, _ctx, initial, _controls, _config, **kwargs):
        state["events"].append(("validation_rollout", None))
        return np.stack([initial] * 3) if kwargs.get("return_states") else initial

    def evaluate(_t, _ctx, model, tasks, _config, work, zero_fields):
        state["events"].append(
            ("evaluate", model, tuple(value.seed for value in tasks))
        )
        assert any(event[0] == "fit_finished" for event in state["events"])
        work.wall_time_s += 2
        work.rollout_count += 2 * len(tasks)
        return {"admitted": True, "mean_objective": 0.1}, {"fine_rollouts": zero_fields}

    def linear(_t, _ctx, tasks, _config, work):
        state["events"].append(
            ("linear_validation", tuple(value.seed for value in tasks))
        )
        work.wall_time_s += 3
        work.rollout_count += 3 * len(tasks)
        return {"admitted": True, "mean_objective": 0.1}, {}

    monkeypatch.setattr(pilot, "_prepare", prepare)
    monkeypatch.setattr(pilot, "init_policy", lambda *_: "initial-model")
    monkeypatch.setattr(pilot, "_serialized", lambda model: str(model).encode())
    monkeypatch.setattr(pilot, "train_imitation_policy", fit)
    monkeypatch.setattr(pilot, "_improved_labels", improved)
    monkeypatch.setattr(pilot, "rollout", rollout)
    monkeypatch.setattr(pilot, "_evaluate", evaluate)
    monkeypatch.setattr(pilot, "_linear_validation", linear)
    return config, state, task


@pytest.mark.parametrize("method", ["demonstration_imitation", "improved_imitation"])
def test_pilot_isolates_labels_and_charges_common_and_extra_work(harness, method):
    config, state, _ = harness
    result = pilot.run_pilot(
        None,
        SimpleNamespace(),
        config,
        method=method,
        model_seed=0,
        train_seeds=(10, 11),
        validation_seeds=(100,),
    )
    metrics = result["metrics"]
    assert metrics["completed"], metrics.get("failure")
    assert [task.seed for task in state["fit_tasks"]] == [10, 11]
    expected = [task.generating_controls for task in state["fit_tasks"]]
    if method == "improved_imitation":
        expected = [value + 0.001 for value in expected]
        assert ("expert", (10, 11)) in state["events"]
    else:
        assert not any(event[0] == "expert" for event in state["events"])
    np.testing.assert_array_equal(state["fit_labels"], expected)
    np.testing.assert_array_equal(result["arrays"]["training_labels"], expected)
    phases = metrics["costs"]["phases"]
    assert phases["common_data_preparation"]["wall_time_s"] == 30
    assert phases["common_data_preparation"]["rollout_count"] == 6
    assert phases["policy_training"]["wall_time_s"] == 7
    if method == "improved_imitation":
        assert phases["label_generation"]["wall_time_s"] == 11
        assert phases["label_generation"]["rollout_count"] == 6
    else:
        assert phases.get("label_generation", {}).get("wall_time_s", 0) == 0
    assert metrics["costs"]["total"]["wall_time_s"] == pytest.approx(
        sum(value["wall_time_s"] for value in phases.values())
    )
    assert not metrics["held_out_test_used"]
    assert [row["update"] for row in metrics["validation_checkpoints"]] == [
        250,
        500,
        1000,
    ]
    finished = next(
        index
        for index, event in enumerate(state["events"])
        if event[0] == "fit_finished"
    )
    assert all(
        index > finished
        for index, event in enumerate(state["events"])
        if event[0] in {"evaluate", "validation_rollout"}
    )
    assert all(
        event[2] == (100,) for event in state["events"] if event[0] == "evaluate"
    )
    assert set(result["model_checkpoints"]) == {250, 500, 1000}


@pytest.mark.parametrize("failure", ["incomplete", "exception"])
def test_failed_training_never_becomes_a_completed_comparison(harness, failure):
    config, state, _ = harness
    state["training_completed"] = False
    state["raise_training"] = failure == "exception"
    result = pilot.run_pilot(
        None,
        SimpleNamespace(),
        config,
        method="demonstration_imitation",
        model_seed=0,
        train_seeds=(10, 11),
        validation_seeds=(100,),
    )
    assert not result["metrics"]["completed"]
    assert not result["metrics"]["admitted"]
    assert result["metrics"]["failure_stage"] == "policy_training"
    assert not any(event[0] == "evaluate" for event in state["events"])
    assert not result["model_checkpoints"]
    if failure == "exception":
        assert not result["metrics"]["query_counts_complete"]
        assert (
            result["metrics"]["costs"]["phases"]["policy_training"][
                "failed_rollout_count"
            ]
            == 1
        )


def test_validation_goal_failure_stops_before_any_fitting_or_expert_labels(harness):
    config, state, _ = harness
    state["bad_seed"] = 100
    result = pilot.run_pilot(
        None,
        SimpleNamespace(),
        config,
        method="improved_imitation",
        model_seed=0,
        train_seeds=(10, 11),
        validation_seeds=(100,),
    )
    assert not result["metrics"]["completed"]
    assert result["metrics"]["failure_stage"] == "data_preparation"
    assert not any(
        event[0] in {"fit_start", "expert", "evaluate"} for event in state["events"]
    )
    assert (
        result["metrics"]["costs"]["phases"]["common_data_preparation"]["rollout_count"]
        == 6
    )


def test_overlapping_splits_are_rejected_before_work(harness):
    config, state, _ = harness
    with pytest.raises(ValueError, match="disjoint"):
        pilot.run_pilot(
            None,
            SimpleNamespace(),
            config,
            method="full",
            model_seed=0,
            train_seeds=(10, 11),
            validation_seeds=(11,),
        )
    assert not state["events"]


@pytest.mark.parametrize("shooting_loss", [1.0, 3.0])
def test_expert_keeps_the_better_demonstration_and_charges_all_queries(
    monkeypatch, shooting_loss
):
    config = ControlConfig(control_slots=2)
    demonstration = np.full((2, 8), 0.03, dtype=np.float32)
    candidate = np.full((2, 8), 0.12, dtype=np.float32)
    task = SimpleNamespace(
        seed=10, initial=None, goal=None, generating_controls=demonstration
    )
    monkeypatch.setattr(
        pilot,
        "direct_shooting",
        lambda *_: {
            "completed": True,
            "controls": candidate,
            "final_loss": shooting_loss,
            "rollout_count": 3,
            "gradient_rollout_count": 2,
            "updates": 2,
            "trace": [4.0, shooting_loss],
        },
    )

    def demo_objective(_t, _ctx, _initial, _goal, controls, _config):
        np.testing.assert_array_equal(controls, demonstration)
        return 2.0

    monkeypatch.setattr(pilot, "objective", demo_objective)
    ledger = CostLedger()
    labels, rows = pilot._improved_labels(None, None, [task], config, ledger)
    np.testing.assert_array_equal(
        labels[0], candidate if shooting_loss < 2 else demonstration
    )
    assert rows[0]["label_source"] == (
        "shooting" if shooting_loss < 2 else "demonstration"
    )
    work = ledger.phase("label_generation")
    assert (
        work.rollout_count,
        work.gradient_rollout_count,
        work.optimizer_updates,
        work.label_examples,
    ) == (4, 2, 2, 1)


def test_warm_start_uses_its_realized_controls_without_changing_default(monkeypatch):
    config = ControlConfig(control_slots=2, shooting_updates=1)
    task = SimpleNamespace(initial=None, goal=None)
    monkeypatch.setattr(
        control,
        "objective",
        lambda _t, _ctx, _i, _g, controls, _c: jnp.sum((controls - 0.05) ** 2),
    )
    requested = np.full((2, 8), 0.1, dtype=np.float32)
    warm = control.direct_shooting(None, None, task, config, initial_controls=requested)
    zero = control.direct_shooting(None, None, task, config)
    assert warm["initialization"] == "provided_controls"
    np.testing.assert_allclose(warm["initial_controls"], requested, atol=2e-8)
    assert warm["initial_loss"] == pytest.approx(
        float(np.sum((warm["initial_controls"] - 0.05) ** 2)), rel=2e-6
    )
    assert zero["initialization"] == "zero"
    np.testing.assert_array_equal(zero["initial_controls"], np.zeros((2, 8)))
    assert zero["initial_loss"] == pytest.approx(16 * 0.05**2, rel=2e-6)
    with pytest.raises(ValueError):
        control.direct_shooting(
            None, None, task, config, initial_controls=np.ones((2, 8))
        )
