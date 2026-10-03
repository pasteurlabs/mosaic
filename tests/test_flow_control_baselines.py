"""Analytic checks for action estimators, cheap controls and baseline accounting."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.flow_control.baselines import (
    CostLedger,
    CountedObjective,
    antithetic_action_gradient,
    linear_action_baseline,
    optimise_spsa_actions,
)


def test_antithetic_latent_gradient_matches_analytic_bounded_objective():
    seen = []

    def objective(controls):
        seen.append(np.asarray(controls))
        return float(3 * jnp.sum(controls))

    latents = jnp.array([[2.0]], dtype=jnp.float32)
    gradient, diagnostic = antithetic_action_gradient(
        latents,
        objective,
        control_bound=0.25,
        perturbation=0.01,
        key=jax.random.PRNGKey(116),
        directions=3,
    )
    expected = 3 * 0.25 * (1 - np.tanh(2.0) ** 2)
    np.testing.assert_allclose(gradient, expected, rtol=3e-4, atol=1e-6)
    assert len(seen) == diagnostic["rollout_count"] == 6
    assert all(np.max(np.abs(value)) < 0.25 for value in seen)


def test_action_teacher_charges_all_probes_and_returns_best_queried_action():
    queried = []

    def objective(controls):
        loss = jnp.sum((controls - 0.1) ** 2)
        queried.append((np.asarray(controls).copy(), float(loss)))
        return loss

    ledger = CostLedger()
    result = optimise_spsa_actions(
        objective,
        jnp.zeros((1, 1)),
        control_bound=0.25,
        seed=7,
        updates=3,
        lr=0.1,
        perturbation=0.05,
        directions=2,
        ledger=ledger,
    )
    best_controls, best_value = min(queried, key=lambda item: item[1])
    np.testing.assert_array_equal(result["controls"], best_controls)
    assert result["loss"] == best_value
    assert ledger.total().rollout_count == len(queried) == 14
    assert ledger.total().gradient_rollout_count == 0
    assert ledger.total().optimizer_updates == 3
    assert ledger.total().label_examples == 1
    assert ledger.total().wall_time_s >= ledger.total().oracle_wall_time_s > 0


def test_failed_objective_is_still_charged():
    ledger = CostLedger()
    counted = CountedObjective(
        lambda controls: jnp.sum(controls) * jnp.nan, ledger, phase="training"
    )
    with pytest.raises(FloatingPointError, match="nonfinite"):
        counted(jnp.zeros((2, 3)))
    total = ledger.total()
    assert total.rollout_count == total.failed_rollout_count == 1
    assert total.wall_time_s >= total.oracle_wall_time_s > 0


def test_linear_baseline_recovers_orthogonal_actions_and_clips_active_bound():
    basis = np.eye(3, dtype=np.float32).reshape(3, 3, 1)
    terminal = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    desired = np.array([0.1, -0.2, 0.5], dtype=np.float32)
    goal = terminal + 2 * np.einsum("m,mij->ij", desired, basis)
    result = linear_action_baseline(
        goal,
        terminal,
        basis,
        control_slots=8,
        control_bound=0.25,
        duration=2,
    )
    np.testing.assert_allclose(result, np.tile([0.1, -0.2, 0.25], (8, 1)), atol=1e-7)
    with pytest.raises(ValueError, match="orthogonal"):
        linear_action_baseline(
            goal,
            terminal,
            basis + 0.1,
            control_slots=8,
            control_bound=0.25,
            duration=2,
        )


def test_spsa_nonfinite_gradient_stops_with_completed_work_retained(monkeypatch):
    import sys
    from types import SimpleNamespace

    from experiments.flow_control import baselines

    fake_control = SimpleNamespace(
        init_policy=lambda seed, config: jnp.zeros((1, 8)),
        policy_latents=lambda model, task: model,
        objective=lambda t, ctx, initial, goal, controls, config: jnp.sum(controls**2),
    )
    monkeypatch.setitem(sys.modules, "experiments.flow_control.control", fake_control)

    def failing_estimate(latents, objective, **kwargs):
        objective(jnp.ones_like(latents) * 0.1)
        objective(jnp.ones_like(latents) * -0.1)
        return jnp.full_like(latents, jnp.inf), {}

    monkeypatch.setattr(baselines, "antithetic_action_gradient", failing_estimate)
    task = SimpleNamespace(initial=np.zeros(1), goal=np.zeros(1))
    result = baselines.train_spsa_policy(
        None,
        None,
        [task],
        SimpleNamespace(control_bound=0.25),
        updates=2,
    )
    assert not result["completed"]
    assert result["updates"] == 0
    assert result["rollout_count"] == 2
    assert result["failure"] == "nonfinite SPSA policy gradient"
    np.testing.assert_array_equal(result["model"], np.zeros((1, 8)))


def test_extra_expert_cost_exhausts_imitation_budget_before_fitting(monkeypatch):
    import sys
    from types import SimpleNamespace

    from experiments.flow_control import baselines

    fake_control = SimpleNamespace(
        init_policy=lambda seed, config: jnp.zeros((1, 8)),
        policy_controls=lambda model, task, config: (
            config.control_bound * jnp.tanh(model)
        ),
    )
    monkeypatch.setitem(sys.modules, "experiments.flow_control.control", fake_control)
    labels = CostLedger()
    labels.phase("label_generation").wall_time_s = 10
    labels.phase("label_generation").rollout_count = 20
    result = baselines.train_imitation_policy(
        [object()],
        [np.zeros((1, 8))],
        SimpleNamespace(control_slots=1, control_bound=0.25),
        label_costs=labels,
        updates=2,
        wall_time_budget_s=1,
    )
    assert result["updates"] == 0
    assert result["rollout_count"] == 20
    assert result["label_generation_wall_time_s"] == 10
    assert result["wall_time_s"] >= 10
    assert "policy_training" not in labels.phases


def test_imitation_checkpoints_keep_earlier_models_without_restarting(monkeypatch):
    import sys
    from types import SimpleNamespace

    from experiments.flow_control import baselines

    fake_control = SimpleNamespace(
        init_policy=lambda seed, config: jnp.zeros((1, 8)),
        policy_controls=lambda model, task, config: (
            config.control_bound * jnp.tanh(model)
        ),
    )
    monkeypatch.setitem(sys.modules, "experiments.flow_control.control", fake_control)
    kwargs = {
        "tasks": [object()],
        "controls": [np.full((1, 8), 0.1)],
        "config": SimpleNamespace(control_slots=1, control_bound=0.25),
        "label_costs": CostLedger(),
        "lr": 0.01,
    }
    result = baselines.train_imitation_policy(
        **kwargs, updates=3, checkpoint_updates=(1, 2, 3)
    )
    for update in [1, 2]:
        prefix = baselines.train_imitation_policy(**kwargs, updates=update)
        np.testing.assert_array_equal(
            result["checkpoint_models"][update], prefix["model"]
        )
    np.testing.assert_array_equal(result["checkpoint_models"][3], result["model"])
    assert not np.array_equal(result["checkpoint_models"][1], result["model"])
    assert result["rollout_count"] == result["gradient_rollout_count"] == 0
