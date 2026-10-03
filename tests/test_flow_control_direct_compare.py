"""Matched task, timing separation and missing-candidate reporting invariants."""

import json

import numpy as np
import pytest

from experiments.flow_control import direct_compare as compare
from experiments.flow_control.control import ControlConfig, Task


@pytest.fixture
def harness(monkeypatch):
    cfg = ControlConfig(control_slots=2)
    field = np.ones((2, 2, 1, 2), dtype=np.float32)
    task = Task(
        3000,
        field,
        field,
        field,
        field,
        np.zeros((2, 8), dtype=np.float32),
        initial_seed=3000,
        goal_seed=10003000,
        fine_goal_rollout=np.stack([field] * 3),
    )
    state = {
        "optimizers": 0,
        "rollouts": 0,
        "failure": False,
        "gradient": True,
        "unavailable": False,
    }

    def optimize(loss, initial, config, method, **kwargs):
        state["optimizers"] += 1
        assert kwargs["optimizer_settings"]["max_queries"] > 512
        np.testing.assert_array_equal(initial, np.zeros((2, 8)))
        if state["failure"] and state["optimizers"] == 1:
            raise RuntimeError("optimizerfailed")
        snapshots = [
            {
                "budget_kind": "wall_time_s",
                "budget": b,
                "available": True,
                "reached": True,
                "controls": initial.copy(),
                "best_loss": 0.1,
            }
            for b in (30, 60, 120)
        ]
        snapshots.append(
            {
                "budget_kind": "queries",
                "budget": 128,
                "available": not state["unavailable"],
                "reached": not state["unavailable"],
                "controls": None if state["unavailable"] else initial.copy(),
            }
        )
        return {
            "snapshots": snapshots,
            "failure": None,
            "accounting": {"forward_count": 4, "vjp_count": 1},
        }

    def rollout(t, ctx, initial, actions, config, **kwargs):
        state["rollouts"] += 1
        if kwargs.get("return_states"):
            assert state["optimizers"] == 11
            return np.stack([initial] * 3)
        return initial

    def gradient(*args):
        assert state["optimizers"] == 11
        return {"passed": state["gradient"]}

    monkeypatch.setattr(compare, "generate_tasks", lambda *args: [task])
    monkeypatch.setattr(compare, "rollout", rollout)
    monkeypatch.setattr(compare, "gradient_check", gradient)
    monkeypatch.setattr(
        compare, "actuator_basis", lambda *args: np.zeros((8, 2, 2, 1, 2))
    )
    monkeypatch.setattr(
        compare,
        "linear_action_baseline",
        lambda *args, **kwargs: np.zeros((2, 8), dtype=np.float32),
    )
    monkeypatch.setattr(compare, "optimize_actions", optimize)
    return cfg, state


def test_matched_tasks_no_fine_interleave_dedup_and_budget_coverage(harness):
    cfg, state = harness
    state["unavailable"] = True
    result = compare.run_direct_compare(None, None, cfg, payload={"task_seed": 3000})
    metrics = result["metrics"]
    assert metrics["completed"] and metrics["admitted"], metrics.get("failure")
    assert len(metrics["settings"]) == 11 and state["optimizers"] == 11
    assert len({r["input_sha256"] for r in metrics["settings"]}) == 1
    assert len(set(metrics["optimizer_order"])) == 11
    assert "zero_fine_rollout" in result["arrays"]
    assert metrics["common_admitted"]
    assert metrics["unique_fine_candidates"] == 1
    assert (
        metrics["common_and_evaluation_costs"]["phases"]["fine_evaluation"][
            "rollout_count"
        ]
        == 2
    )
    assert (
        metrics["common_and_evaluation_costs"]["phases"]["gradient_admission"][
            "rollout_count"
        ]
        == 19
    )
    for row in metrics["settings"]:
        assert len(row["snapshots"]) == 4
        assert row["snapshots"][-1]["fine_objective"] is None
        assert not row["snapshots"][-1]["admitted"]
        assert row["snapshots"][0]["fine_evaluation"]["cache_hit"]
        assert f"snapshots_{row['setting_id']}_0_fine_rollout" in result["arrays"]
    json.dumps(metrics)


@pytest.mark.parametrize("failure", ["optimizer", "gradient"])
def test_failure_retained_all_other_arms_still_evaluated(harness, failure):
    cfg, state = harness
    state["failure"] = failure == "optimizer"
    state["gradient"] = failure != "gradient"
    result = compare.run_direct_compare(None, None, cfg, payload={"task_seed": 3000})
    assert result["metrics"]["completed"]
    assert not result["metrics"]["admitted"]
    assert state["optimizers"] == 11
    assert len(result["metrics"]["settings"]) == 11
    if failure == "optimizer":
        assert (
            sum(
                r.get("query_counts_complete") is False
                for r in result["metrics"]["settings"]
            )
            == 1
        )


def test_no_unfrozen_test_or_development_grid(harness):
    cfg, _ = harness
    with pytest.raises(ValueError, match="selection provenance"):
        compare.run_direct_compare(
            None, None, cfg, payload={"task_seed": 5000, "comparison_stage": "test"}
        )
    with pytest.raises(ValueError, match="full11"):
        compare.run_direct_compare(
            None,
            None,
            cfg,
            payload={"task_seed": 3000, "settings": compare.optimizer_cells()[:1]},
        )
    with pytest.raises(ValueError, match="one selected"):
        compare.run_direct_compare(
            None,
            None,
            cfg,
            payload={
                "task_seed": 5000,
                "comparison_stage": "test",
                "selection_sha256": "frozen",
                "settings": compare.optimizer_cells()[:3],
            },
        )
