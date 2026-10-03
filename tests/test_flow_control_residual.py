"""Physical-feature, initialization and model-injection invariants (Slurm only)."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.flow_control import baselines, control
from experiments.flow_control.residual_policy import (
    MODES,
    attach_features,
    init_residual_policy,
)


def _task():
    cfg = control.ControlConfig()
    basis = np.asarray(control.actuator_basis(64))
    initial = 0.15 * basis[3]
    zero = 0.1 * basis[3]
    goal = zero + 0.05 * basis[0]
    return (
        cfg,
        control.Task(1, initial, goal, initial, goal, np.full((8, 8), np.nan)),
        zero,
    )


def test_features_read_observations_only_and_start_from_linear():
    cfg, task, zero = _task()
    task = attach_features(task, zero, cfg)
    poisoned = attach_features(
        replace(
            task,
            fine_goal=np.full_like(task.goal, np.nan),
            generating_controls=np.ones((8, 8)) * 999,
        ),
        zero,
        cfg,
    )
    assert len(MODES) == 24
    assert task.policy_features.shape == (64,)
    np.testing.assert_array_equal(task.policy_features, poisoned.policy_features)
    model = init_residual_policy(0, cfg)
    controls = np.asarray(control.policy_controls(model, task, cfg))
    expected = baselines.linear_action_baseline(
        task.goal,
        zero,
        np.asarray(control.actuator_basis(64)),
        control_slots=8,
        control_bound=cfg.control_bound,
        duration=cfg.dt * cfg.steps_per_slot * cfg.control_slots,
    )
    np.testing.assert_allclose(controls, expected, atol=3e-7)
    assert sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))) == 4192
    changed = attach_features(replace(task, initial=2 * task.initial), zero, cfg)
    np.testing.assert_allclose(
        changed.policy_features[:48], 2 * task.policy_features[:48], atol=1e-6
    )
    np.testing.assert_array_equal(
        changed.policy_features[48:], task.policy_features[48:]
    )


def test_missing_features_and_invalid_fields_rejected():
    cfg, task, zero = _task()
    with pytest.raises(ValueError, match="cached"):
        control.policy_latents(init_residual_policy(0, cfg), task)
    with pytest.raises(ValueError, match="finite canonical"):
        attach_features(task, zero * np.nan, cfg)


@pytest.mark.parametrize("method", ["full", "spsa", "imitation"])
def test_injected_model_checkpoint_training_without_solver(monkeypatch, method):
    cfg, task, zero = _task()
    task = attach_features(task, zero, cfg)
    model = init_residual_policy(4, cfg)
    original = np.asarray(model.output.weight).copy()

    def loss(t, ctx, initial, goal, actions, config):
        return jnp.mean((actions - 0.03) ** 2)

    monkeypatch.setattr(control, "objective", loss)
    # A supplied model must bypass the legacy default constructor entirely.
    monkeypatch.setattr(
        control,
        "init_policy",
        lambda *a, **k: pytest.fail("unexpected reinitialization"),
    )
    kwargs = {"model": model, "updates": 2, "lr": 0.01, "checkpoint_updates": (1, 2)}
    if method == "full":
        result = control.train_policy(None, None, [task], cfg, **kwargs)
    elif method == "spsa":
        result = baselines.train_spsa_policy(None, None, [task], cfg, **kwargs)
    else:
        result = baselines.train_imitation_policy(
            [task],
            [np.full((8, 8), 0.03)],
            cfg,
            label_costs=baselines.CostLedger(),
            **kwargs,
        )
    assert result["completed"]
    assert set(result["checkpoint_models"]) == {1, 2}
    np.testing.assert_array_equal(model.output.weight, original)
    assert not np.array_equal(
        result["checkpoint_models"][1].output.weight,
        result["checkpoint_models"][2].output.weight,
    )
    assert np.isfinite(
        np.asarray(control.policy_controls(result["model"], task, cfg))
    ).all()
