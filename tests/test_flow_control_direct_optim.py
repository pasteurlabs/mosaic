"""Budget boundaries and meaningful analytic direct-control comparisons."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.flow_control.direct_optim import optimize_actions


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, amount):
        self.value += amount


def test_indivisible_overshoot_never_improves_earlier_checkpoint():
    clock = Clock()
    calls = []

    def loss(controls):
        clock.advance(2.0)
        calls.append(np.asarray(controls).copy())
        return jnp.asarray(10.0 - len(calls))

    result = optimize_actions(
        loss,
        np.zeros((1, 2)),
        SimpleNamespace(control_bound=1),
        "spsa",
        {"max_queries": 20, "query_budgets": (1, 2, 3)},
        budgets=(1, 3, 5),
        clock=clock,
    )
    snapshots = result["snapshots"]
    assert [s["available"] for s in snapshots[:3]] == [False, True, True]
    assert [s["selected_query"] for s in snapshots[:3]] == [None, 1, 2]
    assert [s["available"] for s in snapshots[3:]] == [True, True, False]
    assert result["query_events"][-1]["completed_s"] == 6
    assert result["query_events"][-1]["overshoot_s"] == 1
    assert result["accounting"]["forward_count"] == 3
    assert result["accounting"]["vjp_count"] == 0
    assert result["accounting"]["optimizer_updates"] == 1
    np.testing.assert_array_equal(snapshots[2]["controls"], calls[1])


@pytest.mark.parametrize("method", ["adam", "spsa", "powell"])
def test_convex_optimization_improves_and_records_costs(method):
    clock = Clock()

    def loss(controls):
        clock.advance(1)
        return jnp.sum((controls - 0.3) ** 2)

    result = optimize_actions(
        loss,
        np.zeros((1, 1)),
        SimpleNamespace(control_bound=1),
        method,
        {"lr": 0.05, "max_queries": 65, "query_budgets": (32, 64, 128)},
        budgets=(200,),
        seed=4,
        clock=clock,
    )
    best = result["snapshots"][0]
    assert best["available"] and best["best_loss"] < 0.005
    assert result["failure"] is None
    assert result["accounting"]["forward_count"] == len(result["query_events"])
    assert result["accounting"]["vjp_count"] == (
        len(result["query_events"]) - 1 if method == "adam" else 0
    )
    assert not result["snapshots"][-1]["available"]
    assert np.max(np.abs(best["controls"])) <= 1
    assert (
        best["controls_sha256"]
        == result["query_events"][best["selected_query"] - 1]["controls_sha256"]
    )


def test_failure_retains_previous_snapshot_and_failed_call_cost():
    clock = Clock()
    queries = 0

    def loss(controls):
        nonlocal queries
        queries += 1
        clock.advance(1)
        if queries == 3:
            raise RuntimeError("representative solver failure")
        return jnp.sum(controls**2)

    result = optimize_actions(
        loss,
        np.zeros((1, 1)),
        SimpleNamespace(control_bound=1),
        "spsa",
        {"query_budgets": (2, 3)},
        budgets=(20,),
        clock=clock,
    )
    assert not result["completed"]
    assert "representative solver failure" in result["failure"]
    assert result["accounting"]["forward_count"] == 3
    assert result["accounting"]["failed_queries"] == 1
    assert result["accounting"]["wall_time_s"] == 3
    assert result["snapshots"][0]["best_loss"] == 0
    assert result["query_events"][-1]["completed_s"] == 3


def test_spsa_directions_are_charged_individually_and_reproducible():
    def run():
        clock = Clock()

        def loss(controls):
            clock.advance(1)
            return jnp.sum((controls - 0.2) ** 2)

        return optimize_actions(
            loss,
            np.zeros((2, 2)),
            SimpleNamespace(control_bound=0.25),
            "spsa",
            {"directions": 4, "max_queries": 9, "query_budgets": (9,)},
            budgets=(100,),
            seed=7,
            clock=clock,
        )

    a, b = run(), run()
    assert a["accounting"]["forward_count"] == 9
    assert a["accounting"]["optimizer_updates"] == 1
    assert [e["controls_sha256"] for e in a["query_events"]] == [
        e["controls_sha256"] for e in b["query_events"]
    ]
    assert a["snapshots"][1]["available"]


def test_nonfinite_objective_is_preserved_as_failure():
    clock = Clock()

    def loss(controls):
        clock.advance(2)
        return jnp.asarray(float("nan"))

    result = optimize_actions(
        loss,
        np.zeros((1, 1)),
        SimpleNamespace(control_bound=1),
        "adam",
        budgets=(5,),
        clock=clock,
    )
    assert result["failure"].startswith("FloatingPointError")
    assert result["accounting"]["failed_queries"] == 1
    assert not any(s["available"] for s in result["snapshots"])


def test_nonfinite_ad_gradient_keeps_finite_initial_candidate():
    clock = Clock()

    def loss(controls):
        clock.advance(1)
        return jnp.sum(jnp.sqrt(controls))

    result = optimize_actions(
        loss,
        np.zeros((1, 1)),
        SimpleNamespace(control_bound=1),
        "adam",
        budgets=(5,),
        clock=clock,
    )
    assert "nonfinite objective gradient" in result["failure"]
    assert result["accounting"]["forward_count"] == 2
    assert result["accounting"]["vjp_count"] == 1
    assert result["accounting"]["failed_queries"] == 1
    assert result["snapshots"][0]["best_loss"] == 0
    assert result["snapshots"][0]["selected_query"] == 1
    assert result["accounting"]["first_ad_call_including_compilation_s"] == 1
