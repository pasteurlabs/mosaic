# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and configuration tests for the 2D solver-in-the-loop task."""

from __future__ import annotations

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.benchmarks.problems.navier_stokes_grid.corrector import (
    PeriodicResidualCNN,
    apply_corrector,
    centered_divergence_rms,
    divergence_rms,
    init_corrector,
    project_periodic_correction,
    spectral_prolong,
    spectral_restrict,
)
from mosaic.benchmarks.problems.navier_stokes_grid.ics import _tgv
from mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop import (
    _evaluate_rollout,
    _make_solver_self_reference_datasets,
    _make_supervised_inputs,
    _rollout_log_gain,
    _solver_advance,
    _stop_recurrent_gradient,
    _supervised_loss,
    _window_loss,
    solver_in_loop,
)


def test_periodic_corrector_is_translation_equivariant():
    model = init_corrector(jax.random.PRNGKey(0), hidden_channels=4, kernel_size=3)
    model = eqx.tree_at(
        lambda value: value.layers[-1].weight,
        model,
        jax.random.normal(jax.random.PRNGKey(2), model.layers[-1].weight.shape),
    )
    velocity = jax.random.normal(jax.random.PRNGKey(1), (8, 8, 1, 2))
    shifted = jnp.roll(velocity, shift=(2, -1), axis=(0, 1))

    assert isinstance(model, PeriodicResidualCNN)
    assert isinstance(model, eqx.Module)
    assert model.architecture == "periodic_residual_cnn"
    expected = jnp.roll(
        apply_corrector(model, velocity, velocity_scale=1.0),
        shift=(2, -1),
        axis=(0, 1),
    )
    actual = apply_corrector(model, shifted, velocity_scale=1.0)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_corrector_starts_from_the_uncorrected_solver():
    model = init_corrector(jax.random.PRNGKey(0), hidden_channels=4, kernel_size=3)
    velocity = jax.random.normal(jax.random.PRNGKey(1), (8, 8, 1, 2))

    np.testing.assert_array_equal(
        apply_corrector(model, velocity, velocity_scale=1.0),
        jnp.zeros_like(velocity),
    )


def test_periodic_projection_is_divergence_free_and_zero_mean():
    delta = jax.random.normal(jax.random.PRNGKey(2), (16, 16, 1, 2))
    projected = project_periodic_correction(delta, 2.0 * np.pi)

    assert divergence_rms(projected, 2.0 * np.pi) < 1e-5
    np.testing.assert_allclose(
        np.asarray(projected).mean(axis=(0, 1, 2)),
        np.zeros(2),
        atol=1e-6,
    )
    assert centered_divergence_rms(_tgv(16), 2.0 * np.pi) < 1e-6


def test_spectral_restriction_preserves_low_mode_tgv():
    fine = _tgv(32)
    coarse = spectral_restrict(fine, 16)

    np.testing.assert_allclose(coarse, _tgv(16), rtol=1e-5, atol=1e-5)


def test_spectral_prolongation_preserves_low_mode_tgv():
    coarse = _tgv(16)
    fine = spectral_prolong(coarse, 32)

    np.testing.assert_allclose(fine, _tgv(32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        spectral_restrict(fine, 16),
        coarse,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "convergence_tolerance, expected", [(0.01, True), (1e-8, False)]
)
@pytest.mark.parametrize("audit_factor", [2, 4])
def test_solver_self_reference_matches_physical_time_and_passes_closure(
    monkeypatch,
    convergence_tolerance,
    expected,
    audit_factor,
):
    calls: list[tuple[int, float, int, object | None]] = []

    def _closed_refined_step(
        _t,
        _ctx,
        velocity,
        *,
        dt,
        steps,
        native_state=None,
    ):
        calls.append((velocity.shape[0], dt, steps, native_state))
        per_step_decay = 1.0 - dt / velocity.shape[0]
        next_native_state = (
            jnp.asarray([1.0])
            if native_state is None
            else jnp.asarray(native_state) + 1.0
        )
        return jnp.asarray(velocity) * per_step_decay**steps, next_native_state

    monkeypatch.setattr(
        "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop."
        "_unforced_solver_advance",
        _closed_refined_step,
    )
    ctx = SimpleNamespace(
        name="closed-dummy",
        phys={"N": 8, "nu": 0.001, "dt": 0.02, "steps": 1},
        domain_extent=2.0 * np.pi,
    )
    train, train_rollouts, test, dataset_hash, audit = (
        _make_solver_self_reference_datasets(
            None,
            ctx,
            dataset={
                "reference_factor": 2,
                "reference_temporal_factor": 2,
                "reference_audit_factor": audit_factor,
                "reference_audit_temporal_factor": 4,
                "reference_convergence_tolerance": convergence_tolerance,
                "train_seeds": [0, 1],
                "test_seeds": [100],
                "train_frames": 2,
                "prefix_audit_seeds": [0, 100],
                "prefix_audit_frames": [1, 2],
                "k0": 2.0,
                "minimum_refinement_signal": 1e-5,
            },
            evaluation={"rollout_frames": 3},
            training={"unroll": 2},
        )
    )

    assert train.shape == (2, 3, 8, 8, 1, 2)
    assert train_rollouts.shape == (2, 4, 8, 8, 1, 2)
    assert test.shape == (1, 4, 8, 8, 1, 2)
    assert len(dataset_hash) == 16
    assert audit["eligible_for_corrector_training"] is expected
    assert audit["reference_convergence_passed"] is expected
    assert audit["reference_audit_grid_size"] == 8 * audit_factor
    assert audit["reference_convergence_scope"] == (
        "time" if audit_factor == 2 else "space_time"
    )
    assert audit["max_coarse_closure_error"] < 1e-5
    assert audit["max_fine_closure_error"] < 1e-5
    assert audit["max_coarse_closure_to_signal_ratio"] < 1e-4
    assert audit["max_fine_closure_to_signal_ratio"] < 1e-4
    assert audit["mean_refinement_signal"] > 1e-5
    assert any(call[:3] == (16, 0.01, 2) for call in calls)
    assert any(call[:3] == (8 * audit_factor, 0.005, 4) for call in calls)
    assert any(call[:3] == (8, 0.02, 1) for call in calls)
    assert any(call[3] is not None for call in calls)
    assert 0.01 * 2 == 0.02 * 1


def test_solver_self_reference_rejects_coarse_closure_larger_than_signal(
    monkeypatch,
):
    def _coarse_call_biased_step(
        _t,
        _ctx,
        velocity,
        *,
        dt,
        steps,
        native_state=None,
    ):
        per_step_decay = 1.0 - dt / velocity.shape[0]
        call_bias = 5e-4 if velocity.shape[0] == 8 else 0.0
        next_native_state = (
            jnp.asarray([1.0])
            if native_state is None
            else jnp.asarray(native_state) + 1.0
        )
        return (
            jnp.asarray(velocity) * per_step_decay**steps + call_bias,
            next_native_state,
        )

    monkeypatch.setattr(
        "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop."
        "_unforced_solver_advance",
        _coarse_call_biased_step,
    )
    ctx = SimpleNamespace(
        name="coarse-call-biased-dummy",
        phys={"N": 8, "nu": 0.001, "dt": 0.02, "steps": 1},
        domain_extent=2.0 * np.pi,
    )

    *_datasets, audit = _make_solver_self_reference_datasets(
        None,
        ctx,
        dataset={
            "reference_factor": 2,
            "reference_temporal_factor": 2,
            "train_seeds": [0],
            "test_seeds": [100],
            "train_frames": 2,
            "prefix_audit_seeds": [0, 100],
            "prefix_audit_frames": [1, 2],
            "k0": 2.0,
            "minimum_refinement_signal": 1e-5,
        },
        evaluation={"rollout_frames": 2},
        training={"unroll": 2},
    )

    assert audit["max_coarse_closure_error"] < audit["closure_relative_tolerance"]
    assert (
        audit["max_coarse_closure_to_signal_ratio"]
        > audit["closure_to_signal_tolerance"]
    )
    assert audit["max_fine_closure_to_signal_ratio"] < 1e-5
    assert audit["eligible_for_corrector_training"] is False


def test_solver_advance_threads_optional_native_state(monkeypatch):
    calls: list[dict] = []

    def _apply(_t, inputs):
        calls.append(inputs)
        velocity = jnp.asarray(inputs["v0"])
        native_state = inputs.get("state")
        if native_state is None:
            native_state = jnp.zeros_like(velocity)
        return {
            "result": velocity + native_state + 1.0,
            "state": native_state + 2.0,
        }

    monkeypatch.setattr(
        "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop.apply_tesseract",
        _apply,
    )
    ctx = SimpleNamespace(
        name="stateful-dummy",
        phys={"N": 2, "nu": 0.001, "dt": 0.02, "steps": 1},
        output_key="result",
        make_inputs=lambda _name, velocity, **_physics: {"v0": velocity},
    )
    t = SimpleNamespace(
        openapi_schema={
            "components": {
                "schemas": {
                    "Apply_InputSchema": {
                        "properties": {
                            "v0": {},
                            "state": {},
                            "return_state": {},
                        }
                    },
                    "ApplyInputSchema": {
                        "differentiable_arrays": {"v0": {}, "state": {}}
                    },
                    "ApplyOutputSchema": {
                        "differentiable_arrays": {"result": {}, "state": {}}
                    },
                }
            }
        }
    )
    initial = jnp.ones((2, 2, 1, 2))

    first_velocity, first_native_state = _solver_advance(
        t,
        ctx,
        initial,
        frame_steps=1,
    )
    second_velocity, second_native_state = _solver_advance(
        t,
        ctx,
        first_velocity,
        frame_steps=1,
        native_state=first_native_state,
    )

    assert "state" not in calls[0]
    assert calls[0]["return_state"] is True
    assert calls[1]["return_state"] is True
    np.testing.assert_array_equal(calls[1]["state"], first_native_state)
    np.testing.assert_array_equal(first_velocity, 2.0 * jnp.ones_like(initial))
    np.testing.assert_array_equal(first_native_state, 2.0 * jnp.ones_like(initial))
    np.testing.assert_array_equal(second_velocity, 5.0 * jnp.ones_like(initial))
    np.testing.assert_array_equal(second_native_state, 4.0 * jnp.ones_like(initial))


def test_solver_advance_leaves_stateless_schema_unchanged(monkeypatch):
    calls: list[dict] = []

    def _apply(_t, inputs):
        calls.append(inputs)
        return {"result": jnp.asarray(inputs["v0"]) + 1.0}

    monkeypatch.setattr(
        "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop.apply_tesseract",
        _apply,
    )
    t = SimpleNamespace(
        openapi_schema={
            "components": {
                "schemas": {
                    "Apply_InputSchema": {
                        "properties": {
                            "v0": {},
                            "state": {},
                            "return_state": {},
                        }
                    },
                    "ApplyInputSchema": {"differentiable_arrays": {"v0": {}}},
                    "ApplyOutputSchema": {"differentiable_arrays": {"result": {}}},
                }
            }
        }
    )
    ctx = SimpleNamespace(
        name="stateless-dummy",
        phys={"N": 2, "nu": 0.001, "dt": 0.02, "steps": 1},
        output_key="result",
        make_inputs=lambda _name, velocity, **_physics: {"v0": velocity},
    )

    velocity, native_state = _solver_advance(
        t,
        ctx,
        jnp.zeros((2, 2, 1, 2)),
        frame_steps=1,
    )

    assert set(calls[0]) == {"v0"}
    assert native_state is None
    np.testing.assert_array_equal(velocity, jnp.ones_like(velocity))


def test_corrected_rollout_threads_velocity_and_native_state(monkeypatch):
    calls: list[tuple[np.ndarray, int | None]] = []

    def _advance(_t, _ctx, velocity, *, frame_steps, native_state=None):
        del frame_steps
        calls.append(
            (
                np.asarray(velocity),
                None if native_state is None else int(native_state),
            )
        )
        next_native_state = 1 if native_state is None else native_state + 1
        return jnp.asarray(velocity) + next_native_state, next_native_state

    monkeypatch.setattr(
        "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop._solver_advance",
        _advance,
    )
    monkeypatch.setattr(
        "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop."
        "corrected_velocity",
        lambda _model, velocity, **_kwargs: velocity + 10.0,
    )
    reference = np.zeros((3, 2, 2, 1, 2), dtype=np.float32)
    ctx = SimpleNamespace(domain_extent=2.0 * np.pi)

    rollout, _ = _evaluate_rollout(
        None,
        ctx,
        None,
        reference,
        frame_steps=1,
        velocity_scale=1.0,
        corrected=True,
    )

    assert calls[0][1] is None
    assert calls[1][1] == 1
    np.testing.assert_array_equal(calls[1][0], 11.0 * np.ones_like(reference[0]))
    np.testing.assert_array_equal(rollout[-1], 23.0 * np.ones_like(reference[0]))


def test_training_window_threads_corrected_velocity_and_native_state(monkeypatch):
    calls: list[tuple[np.ndarray, int | None]] = []

    def _advance(_t, _ctx, velocity, *, frame_steps, native_state=None):
        del frame_steps
        calls.append(
            (
                np.asarray(velocity),
                None if native_state is None else int(native_state),
            )
        )
        next_native_state = 1 if native_state is None else native_state + 1
        return jnp.asarray(velocity) + next_native_state, next_native_state

    monkeypatch.setattr(
        "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop._solver_advance",
        _advance,
    )
    monkeypatch.setattr(
        "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop."
        "corrected_velocity",
        lambda model, velocity, **_kwargs: velocity + model,
    )
    targets = jnp.ones((3, 2, 2, 1, 2))
    ctx = SimpleNamespace(domain_extent=2.0 * np.pi)

    loss = _window_loss(
        jnp.asarray(10.0),
        targets,
        t=None,
        ctx=ctx,
        frame_steps=1,
        velocity_scale=1.0,
        differentiate_solver=True,
        loss_scale=1.0,
    )

    assert np.isfinite(float(loss))
    assert calls[0][1] is None
    assert calls[1][1] == 1
    np.testing.assert_array_equal(calls[1][0], 12.0 * np.ones_like(targets[0]))


def test_stop_gradient_cuts_velocity_and_native_state():
    def stopped(value):
        velocity, native_state = _stop_recurrent_gradient(
            value,
            {"memory": 2.0 * value},
        )
        return velocity + native_state["memory"]

    primal, tangent = jax.jvp(
        stopped,
        (jnp.asarray(3.0),),
        (jnp.asarray(1.0),),
    )

    assert float(primal) == 9.0
    assert float(tangent) == 0.0


def test_rollout_log_gain_is_geometric_and_ignores_initial_frame():
    baseline = np.asarray([0.0, 4.0, 8.0])
    corrected = np.asarray([0.0, 2.0, 2.0])

    gain = _rollout_log_gain(baseline, corrected)

    np.testing.assert_allclose(np.exp(gain), np.sqrt(8.0))


def test_supervised_pairs_teacher_force_velocity_and_preserve_native_state(monkeypatch):
    """Fixed pairs use reference inputs, never the preceding solver prediction."""
    import importlib

    module = importlib.import_module(
        "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
    )
    calls = []

    def advance(t, ctx, velocity, *, frame_steps, native_state=None):
        calls.append((np.asarray(velocity).copy(), native_state))
        return velocity + 7, 1 if native_state is None else native_state + 1

    monkeypatch.setattr(module, "_solver_advance", advance)
    train = np.arange(6, dtype=np.float32).reshape(2, 3, 1, 1, 1, 1)
    inputs = _make_supervised_inputs(None, None, train, frame_steps=4)
    np.testing.assert_array_equal(inputs, train[:, :-1] + 7)
    assert [state for _, state in calls] == [None, 1, None, 1]
    np.testing.assert_array_equal(calls[1][0], train[0, 1])


def test_supervised_loss_has_independent_pairs_and_correct_parameter_gradient(
    monkeypatch,
):
    """A scalar independent reference checks normalization and gradient exactly."""
    import importlib

    module = importlib.import_module(
        "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
    )
    monkeypatch.setattr(
        module, "corrected_velocity", lambda model, velocity, **kwargs: model * velocity
    )
    inputs = jnp.asarray([2.0, 3.0])
    targets = jnp.asarray([4.0, 5.0])

    def loss(model):
        return _supervised_loss(
            model,
            inputs,
            targets,
            velocity_scale=1.0,
            domain_extent=1.0,
            loss_scale=2.0,
        )

    expected = (((2 * 1.5 - 4) ** 2 / 16) + ((3 * 1.5 - 5) ** 2 / 25)) / 4
    expected_grad = (2 * 2 * (2 * 1.5 - 4) / 16 + 2 * 3 * (3 * 1.5 - 5) / 25) / 4
    np.testing.assert_allclose(loss(1.5), expected, atol=1e-7)
    np.testing.assert_allclose(jax.grad(loss)(1.5), expected_grad, atol=1e-7)


def test_fd_refinement_uses_one_model_and_does_not_change_training(monkeypatch):
    """A known nonlinear loss refines at one point before the unchanged update."""
    import importlib

    module = importlib.import_module(solver_in_loop.__module__)
    monkeypatch.setattr(module, "_window_loss", lambda model, **kwargs: model**4)
    monkeypatch.setattr(
        module, "init_corrector", lambda *args, **kwargs: jnp.asarray(0.7)
    )
    train = np.zeros((1, 2, 1), dtype=np.float32)
    arguments = {
        "frame_steps": 1,
        "velocity_scale": 1.0,
        "loss_scale": 1.0,
        "differentiate_solver": True,
        "model_seed": 7,
    }
    training = {"max_updates": 1, "unroll": 1, "fd_epsilon": 0.1}
    baseline = module._train_corrector(
        None,
        SimpleNamespace(domain_extent=2 * np.pi),
        train,
        training=training,
        **arguments,
    )
    checks = []
    refined = module._train_corrector(
        None,
        SimpleNamespace(domain_extent=2 * np.pi),
        train,
        training={**training, "fd_epsilons": [0.03, 0.01, 0.03]},
        fd_checks=checks,
        **arguments,
    )
    assert float(baseline[0]) == float(refined[0])
    assert baseline[1:3] == refined[1:3]
    assert baseline[4] == refined[4] == checks[0]["relative_error"]
    assert [check["epsilon"] for check in checks] == [0.1, 0.03, 0.01]
    assert len({check["autodiff"] for check in checks}) == 1
    assert abs(checks[0]["autodiff"]) == pytest.approx(4 * 0.7**3, abs=1e-6)
    assert (
        checks[0]["relative_error"]
        > checks[1]["relative_error"]
        > checks[2]["relative_error"]
    )


def test_three_arms_train_with_same_budget_and_check_each_model_seed(monkeypatch):
    """Exercise the real CNN, optimizer, recurrent loss and reference admission."""
    import importlib

    module = importlib.import_module(solver_in_loop.__module__)

    def advance(_t, _ctx, velocity, *, dt, steps, native_state=None):
        del native_state
        # A closed, differentiable transition with a genuine spatial-refinement
        # signal, unlike an identity fixture that must fail reference admission.
        rate = 0.2 + 120.0 / velocity.shape[0]
        return velocity * jnp.exp(-rate * dt * steps), None

    monkeypatch.setattr(module, "_unforced_solver_advance", advance)
    monkeypatch.setattr(module, "_supports_native_state", lambda _: False)
    ctx = SimpleNamespace(
        name="closed-dummy",
        domain_extent=2 * np.pi,
        phys={"N": 8, "dt": 0.1, "steps": 1, "nu": 0.001},
        run={
            "dataset": {
                "train_seeds": [0],
                "test_seeds": [100],
                "train_frames": 2,
                "reference_factor": 2,
                "reference_temporal_factor": 2,
                "reference_audit_factor": 2,
                "reference_audit_temporal_factor": 4,
                "prefix_audit_frames": [1, 2],
                "k0": 2.0,
            },
            "training": {
                "max_updates": 1,
                "unroll": 2,
                "hidden_channels": 4,
                "kernel_size": 3,
                "model_seeds": [0, 1],
                "fd_epsilon": 0.001,
                "include_supervised_baseline": True,
            },
            "evaluation": {"rollout_frames": 2},
        },
    )
    result = solver_in_loop(None, ctx)
    metrics = result["metrics"]
    assert metrics["completed"] is True
    assert metrics["eligible_for_corrector_training"] is True
    assert metrics["reference_convergence_passed"] is True
    assert metrics["valid_for_vjp_ranking"] is True
    # Large coarse-grid error is a learning signal, not failed recurrence.
    assert metrics["native_final_rollout_error"] > 0.5
    for prefix in ["", "stop_gradient_", "supervised_"]:
        assert metrics[f"{prefix}total_optimizer_updates"] == 2
    assert all(len(checks) == 1 for checks in metrics["end_to_end_fd_checks_by_seed"])
    for arm in ["corrected", "stop_gradient", "supervised"]:
        errors = result["snapshots"][f"error_{arm}_samples"]
        assert errors.shape == (2, 1, 3)
        assert np.isfinite(errors).all()
