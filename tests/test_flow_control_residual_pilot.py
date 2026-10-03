"""Residual development wrapper isolation, model injection and cost checks."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.flow_control import residual_pilot as pilot
from experiments.flow_control.baselines import CostLedger, Work
from experiments.flow_control.control import ControlConfig, Task


@pytest.fixture
def harness(monkeypatch):
    cfg = ControlConfig(control_slots=2)
    state = {"events": [], "complete": True, "failure": None, "admitted": True}
    train_ids, val_ids = list(range(1000, 1064)), list(range(2000, 2016))

    def task(seed):
        field = np.full((2, 2, 1, 2), seed / 1000, dtype=np.float32)
        return Task(
            seed,
            field,
            field,
            field,
            field,
            np.full((2, 8), seed),
            initial_seed=seed,
            goal_seed=seed + 10000000,
            fine_goal_rollout=np.stack([field] * 3),
        )

    tasks = [task(seed) for seed in train_ids + val_ids]
    shared_ledger = CostLedger()
    shared_ledger.phases["common_data_preparation"] = Work(
        wall_time_s=100, rollout_count=160
    )
    shared_ledger.phases["label_generation"] = Work(wall_time_s=200, rollout_count=640)
    shared_costs = shared_ledger.to_dict()

    def load(*args, **kwargs):
        state["events"].append("load")
        if state["failure"] == "load":
            raise ValueError("dataset admission rejected")
        assert (
            kwargs["train_seeds"] == train_ids and kwargs["validation_seeds"] == val_ids
        )
        return (
            tasks[:64],
            tasks[64:],
            {
                "task_coarse_zero_terminal": np.stack([t.initial for t in tasks]),
                "task_improved_controls": np.stack(
                    [t.generating_controls + 1 for t in tasks]
                ),
            },
            {"dataset_sha256": "canonical", "costs": shared_costs},
        )

    def features(task, zero, config):
        return replace(
            task, policy_features=np.zeros(64), baseline_latents=np.zeros((2, 8))
        )

    def init(seed, config, *, architecture):
        assert architecture == "linear_residual"
        return "initial"

    def fit(tasks, config, **kwargs):
        assert kwargs["model"] == "initial"
        assert [t.seed for t in tasks] == train_ids
        assert all(t.policy_features is not None for t in tasks)
        state["events"].append("fit")
        if state["failure"] == "fit":
            raise RuntimeError("training interrupted")
        state["fit_kwargs"] = kwargs
        cost = CostLedger()
        cost.phases["policy_training"] = Work(
            wall_time_s=7,
            rollout_count=9,
            gradient_rollout_count=3,
            optimizer_updates=kwargs["updates"],
        )
        snapshots = {n: f"model{n}" for n in kwargs["checkpoint_updates"]}
        if state["failure"] == "missing_checkpoint":
            snapshots.pop(kwargs["updates"])
        return {
            "model": f"model{kwargs['updates']}",
            "completed": state["complete"],
            "updates": kwargs["updates"] if state["complete"] else 1,
            "trace": [],
            "checkpoint_models": snapshots,
            "costs": cost.to_dict(),
        }

    def imitation(tasks, labels, config, *, label_costs, **kwargs):
        assert label_costs.total().rollout_count == 0
        state["labels"] = np.asarray(labels)
        return fit(tasks, config, **kwargs)

    def evaluate(t, ctx, model, tasks, config, work, zero):
        assert "fit" in state["events"]
        assert [t.seed for t in tasks] == val_ids
        state["events"].append(("evaluate", model))
        work.rollout_count += 2 * len(tasks)
        return {"admitted": state["admitted"], "mean_objective": 0.1}, {
            "fine_rollouts": zero
        }

    def objective(*args):
        state["events"].append("train_fit_diagnostic")
        return 0.2

    monkeypatch.setattr(pilot, "load_dataset", load)
    monkeypatch.setattr(pilot, "attach_features", features)
    monkeypatch.setattr(pilot, "init_policy", init)
    monkeypatch.setattr(pilot, "_serialized", lambda model: str(model).encode())
    monkeypatch.setattr(
        pilot, "train_policy", lambda t, c, tasks, cfg, **kw: fit(tasks, cfg, **kw)
    )
    monkeypatch.setattr(
        pilot, "train_spsa_policy", lambda t, c, tasks, cfg, **kw: fit(tasks, cfg, **kw)
    )
    monkeypatch.setattr(pilot, "train_imitation_policy", imitation)
    monkeypatch.setattr(pilot, "policy_controls", lambda *args: np.zeros((2, 8)))
    monkeypatch.setattr(pilot, "objective", objective)
    monkeypatch.setattr(
        pilot, "rollout", lambda t, c, initial, *a, **kw: np.stack([initial] * 3)
    )
    monkeypatch.setattr(pilot, "_evaluate", evaluate)
    monkeypatch.setattr(
        pilot, "_linear_validation", lambda *args: ({"admitted": True}, {})
    )
    payload = {
        "method": "full",
        "model_seed": 0,
        "train_seeds": train_ids,
        "validation_seeds": val_ids,
        "dataset_path": "unused",
        "image_sha256": "image",
    }
    return cfg, state, payload, tasks, shared_costs


@pytest.mark.parametrize(
    "method", ["full", "spsa", "demonstration_imitation", "improved_imitation"]
)
def test_injected_initial_checkpoint_splits_and_separate_costs(
    harness, tmp_path, method
):
    cfg, state, payload, tasks, shared_costs = harness
    payload["method"] = method
    result = pilot.run_residual_pilot(
        None, None, cfg, payload=payload, out_dir=tmp_path
    )
    metrics = result["metrics"]
    assert metrics["completed"], metrics.get("failure")
    expected = [0, 250, 500, 1000] if "imitation" in method else [0, 100, 300]
    assert [r["update"] for r in metrics["validation_checkpoints"]] == expected
    assert set(result["model_checkpoints"]) == set(expected)
    assert result["model_checkpoints"][0] == b"initial"
    assert metrics["shared_dataset_costs"] == shared_costs
    standalone = metrics["standalone_training_costs"]["total"]
    assert standalone["wall_time_s"] == (307 if method == "improved_imitation" else 107)
    assert standalone["rollout_count"] == (
        809 if method == "improved_imitation" else 169
    )
    phases = metrics["costs"]["phases"]
    assert phases["policy_training"]["rollout_count"] == 9
    assert phases["training_fit_diagnostic"]["rollout_count"] == 64
    assert "label_generation" not in phases and "common_data_preparation" not in phases
    assert phases["validation"]["rollout_count"] == 16 + 32 * len(expected)
    assert state["events"].index("fit") < state["events"].index("train_fit_diagnostic")
    assert all(
        state["events"].index("fit") < index
        for index, event in enumerate(state["events"])
        if isinstance(event, tuple) and event[0] == "evaluate"
    )
    if "imitation" in method:
        labels = np.stack([t.generating_controls for t in tasks[:64]])
        if method == "improved_imitation":
            labels += 1
        np.testing.assert_array_equal(state["labels"], labels)
    assert not metrics["held_out_test_used"]


@pytest.mark.parametrize("failure", ["load", "fit", "incomplete", "missing_checkpoint"])
def test_failure_never_reaches_validation(harness, tmp_path, failure):
    cfg, state, payload, _, _ = harness
    state["failure"] = failure
    state["complete"] = failure != "incomplete"
    result = pilot.run_residual_pilot(
        None, None, cfg, payload=payload, out_dir=tmp_path
    )
    assert not result["metrics"]["completed"]
    assert not result["metrics"]["admitted"]
    assert not any(isinstance(event, tuple) for event in state["events"])
    if failure == "load":
        assert "fit" not in state["events"]
    if failure == "fit":
        assert result["metrics"]["query_counts_complete"] is False


def test_finite_completed_training_can_fail_validation_admission(harness, tmp_path):
    cfg, state, payload, _, _ = harness
    state["admitted"] = False
    result = pilot.run_residual_pilot(
        None, None, cfg, payload=payload, out_dir=tmp_path
    )
    assert result["metrics"]["completed"]
    assert not result["metrics"]["admitted"]
