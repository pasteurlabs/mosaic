# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and configuration tests for the 2D solver-in-the-loop task."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.utils import _debug_run, active_solvers
from mosaic.benchmarks.problems.navier_stokes_grid.corrector import (
    PeriodicResidualCNN,
    apply_corrector,
    centered_divergence_rms,
    divergence_rms,
    init_corrector,
    project_periodic_correction,
    reference_trajectory,
    relative_l2,
    spectral_restrict,
)
from mosaic.benchmarks.problems.navier_stokes_grid.ics import _tgv, _tgv_analytic
from mosaic.benchmarks.problems.navier_stokes_grid.plots import (
    _periodic_vorticity_2d,
    _plot_solver_in_loop_diagnostics,
    _plot_solver_in_loop_fairness,
    _plot_solver_in_loop_fields,
    _plot_solver_in_loop_physics,
    _save_solver_in_loop_animation,
)
from mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop import (
    _evaluate_rollout,
    _make_solver_self_reference_datasets,
    _passes_reference_accuracy_gate,
    _rollout_log_gain,
    _solver_advance,
    _stop_recurrent_gradient,
    _window_loss,
    make_reference_dataset,
    solver_in_loop,
)

_IDENTITY_DUMMY = (
    Path(__file__).parent
    / "dummy_tesseracts"
    / "navier_stokes_grid_identity"
    / "tesseract_api.py"
).resolve()


def test_tgv_control_uses_forward_baseline_with_mach_safe_xlb_budget():
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-grid")
    forward = cfg.experiments["forward/baseline"]
    control = cfg.experiments["optimization/solver_in_loop_tgv"]
    forward_run = inspect.signature(forward.fn).parameters["_kw"].default["runs"][0]
    control_run = inspect.signature(control.fn).parameters["_kw"].default["runs"][0]
    forward_physics = forward_run["physics"]
    control_physics = control_run["physics"]
    control_dataset = control_run["dataset"]
    control_evaluation = control_run["evaluation"]

    for key in ("nu", "dt", "steps"):
        assert control_physics[key] == forward_physics[key]
    assert forward_physics["lbm_N_base"] == 64
    assert control_physics["lbm_N_base"] == 16
    assert forward_run["sweep"]["key"] == "N"
    assert control_physics["N"] in forward_run["sweep"]["values"]
    initial = _tgv(control_physics["N"])
    by_key = {solver.key: solver for solver in cfg.solvers}
    xlb_inputs = cfg.make_inputs(
        by_key["xlb"].name,
        initial,
        domain_extent=cfg.domain_extent,
        **control_physics,
    )
    pict_inputs = cfg.make_inputs(
        by_key["pict"].name,
        initial,
        domain_extent=cfg.domain_extent,
        **control_physics,
    )
    assert xlb_inputs["steps"] == 4
    assert np.isclose(float(xlb_inputs["dt"][0]), 0.0025)
    assert pict_inputs["steps"] == 1
    assert np.isclose(float(pict_inputs["dt"][0]), 0.01)
    differentiable = {"jax_cfd", "ins_jl", "phiflow", "pict", "warp_ns", "xlb"}
    active_forward = {
        cfg.solver(name).key for name in active_solvers(cfg, "forward", "baseline")
    }
    assert differentiable <= active_forward
    for solver in differentiable:
        assert "optimization/solver_in_loop_tgv" not in cfg.exclusions.get(solver, {})
    assert control_evaluation["rollout_frames"] == 100
    assert control_dataset["long_closure_tolerance"] == 0.01
    assert control_evaluation["native_long_error_tolerance"] == 0.01


def test_multimode_forward_agreement_does_not_use_tgv_reference():
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-grid")
    experiment = cfg.experiments["forward/agreement/multimode"]
    run = inspect.signature(experiment.fn).parameters["_kw"].default["runs"][0]

    assert run["reference"] == "consensus"


def test_shared_solver_loop_ranking_declares_forward_admission_bounds():
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-grid")
    experiment = cfg.experiments["optimization/solver_in_loop"]
    run = inspect.signature(experiment.fn).parameters["_kw"].default["runs"][0]
    assert run["dataset"]["k0"] == 2.0
    assert run["dataset"]["sigma_k"] == 0.5
    assert run["evaluation"]["first_interval_error_tolerance"] == 0.05
    assert run["evaluation"]["native_long_error_tolerance"] == 0.5
    self_reference = cfg.experiments["optimization/solver_in_loop_self_reference"]
    self_reference_run = (
        inspect.signature(self_reference.fn).parameters["_kw"].default["runs"][0]
    )
    assert self_reference_run["dataset"]["k0"] == run["dataset"]["k0"]
    assert self_reference_run["dataset"]["sigma_k"] == run["dataset"]["sigma_k"]


def test_self_reference_does_not_gate_the_learnable_refinement_signal():
    assert _passes_reference_accuracy_gate(
        "solver_self_refined",
        first_interval_error=0.6,
        first_interval_tolerance=0.15,
        native_long_error=0.7,
        native_long_tolerance=0.5,
    )
    assert not _passes_reference_accuracy_gate(
        "pseudo_spectral_multimode",
        first_interval_error=0.6,
        first_interval_tolerance=0.15,
        native_long_error=0.7,
        native_long_tolerance=0.5,
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


def test_spectral_reference_matches_tgv_decay():
    viscosity = 0.05
    dt = 0.01
    steps = 10
    initial = _tgv(16)
    trajectory = reference_trajectory(
        initial,
        viscosity=viscosity,
        dt=dt,
        frame_steps=steps,
        n_frames=1,
        substeps=1,
        domain_extent=2.0 * np.pi,
    )
    expected = _tgv_analytic(
        initial,
        nu=viscosity,
        t=dt * steps,
        L=2.0 * np.pi,
    )

    assert relative_l2(trajectory[-1], expected) < 1e-5


def test_analytic_tgv_reference_has_exact_decay_and_distinct_phases():
    viscosity = 0.05
    dt = 0.02
    frame_steps = 4
    train, test, dataset_hash = make_reference_dataset(
        physics={"N": 16, "nu": viscosity, "dt": dt, "steps": frame_steps},
        dataset={
            "reference_kind": "analytic_tgv",
            "reference_factor": 2,
            "train_seeds": [0],
            "test_seeds": [100],
            "train_frames": 2,
        },
        evaluation={"rollout_frames": 2},
        training={"unroll": 1},
        domain_extent=2.0 * np.pi,
    )

    expected_decay = np.exp(-2.0 * viscosity * dt * frame_steps * 2)
    actual_decay = np.linalg.norm(test[0, -1]) / np.linalg.norm(test[0, 0])
    np.testing.assert_allclose(actual_decay, expected_decay, rtol=1e-6)
    assert not np.allclose(train[0, 0], test[0, 0])
    assert len(dataset_hash) == 16


def test_solver_self_reference_matches_physical_time_and_passes_closure(
    monkeypatch,
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
        "_solver_advance_with_physics",
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
    assert audit["eligible_for_corrector_training"] is True
    assert audit["max_coarse_closure_error"] < 1e-5
    assert audit["max_fine_closure_error"] < 1e-5
    assert audit["max_coarse_closure_to_signal_ratio"] < 1e-4
    assert audit["max_fine_closure_to_signal_ratio"] < 1e-4
    assert audit["mean_refinement_signal"] > 1e-5
    assert any(call[:3] == (16, 0.01, 2) for call in calls)
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
        "_solver_advance_with_physics",
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
        loss_mode="mean",
        solver_loss_weight=0.1,
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


def test_debug_run_caps_corrector_training():
    run = {
        "physics": {"N": 32, "steps": 4},
        "training": {
            "max_updates": 100,
            "unroll": 8,
            "hidden_channels": 32,
            "kernel_size": 5,
            "model_seeds": [0, 1, 2],
            "check_grad": True,
        },
        "dataset": {
            "train_seeds": [0, 1, 2],
            "test_seeds": [100, 101],
            "train_frames": 16,
        },
        "evaluation": {"rollout_frames": 24},
    }

    _debug_run(run)

    assert run["physics"]["N"] == 16
    assert run["physics"]["steps"] == 4
    assert run["training"] == {
        "max_updates": 2,
        "unroll": 2,
        "hidden_channels": 8,
        "kernel_size": 3,
        "model_seeds": [0],
        "check_grad": False,
    }
    assert run["dataset"]["train_seeds"] == [0]
    assert run["dataset"]["test_seeds"] == [100]
    assert run["dataset"]["train_frames"] == 3
    assert run["evaluation"]["rollout_frames"] == 3


def test_solver_in_loop_fields_render_reference_raw_and_corrected(tmp_path):
    n = 16
    coordinates = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    reference = np.zeros((n, n, 1, 2), dtype=np.float32)
    reference[:, :, 0, 0] = np.sin(y)
    reference[:, :, 0, 1] = -np.sin(x)

    expected_vorticity = -np.cos(x) - np.cos(y)
    np.testing.assert_allclose(
        _periodic_vorticity_2d(reference),
        expected_vorticity,
        rtol=3e-2,
        atol=3e-2,
    )

    arrays = {
        "reference_rollout_0": np.stack((reference, reference)),
        "evaluation_times": np.asarray([0.0, 0.1]),
        "rollout_uncorrected_0": np.stack((reference, 0.8 * reference)),
        "rollout_corrected_0": np.stack((reference, 0.95 * reference)),
    }
    fig = _plot_solver_in_loop_fields(
        arrays,
        ["jax-cfd"],
        tmp_path,
        save=True,
    )

    assert fig is not None
    assert [axis.get_title() for axis in fig.axes[:3]] == [
        "Reference",
        "Solver only",
        "Solver + corrector",
    ]
    rendered = tmp_path / "solver_in_loop_fields.png"
    assert rendered.exists()
    assert rendered.stat().st_size > 0


def test_solver_in_loop_fairness_physics_and_animation_render(tmp_path):
    n = 8
    reference = np.asarray(_tgv(n))
    reference_rollout = np.stack((reference, 0.98 * reference, 0.96 * reference))
    arrays = {
        "reference_rollout": reference_rollout,
        "evaluation_times": np.asarray([0.0, 0.1, 0.2]),
        "rollout_uncorrected_0": np.stack(
            (reference, 0.85 * reference, 0.7 * reference)
        ),
        "rollout_corrected_0": np.stack((reference, 0.95 * reference, 0.9 * reference)),
        "rollout_uncorrected_1": np.stack(
            (reference, 0.9 * reference, 0.8 * reference)
        ),
        "rollout_corrected_1": np.stack(
            (reference, 0.97 * reference, 0.94 * reference)
        ),
    }
    data = {
        "by_solver": {
            "jax-cfd": {
                "native_final_rollout_error": 0.1,
                "first_interval_rollout_error": 0.08,
                "first_interval_rollout_error_ic_std": 0.01,
                "uncorrected_rollout_error": 0.3,
                "recurrent_to_native_error_ratio": 3.0,
                "uncorrected_mean_rollout_error": 0.25,
                "mean_rollout_error": 0.12,
                "geometric_error_reduction": 2.0,
                "rollout_log_gain_seed_std": 0.1,
                "stop_gradient_geometric_error_reduction": 1.4,
                "stop_gradient_rollout_log_gain_seed_std": 0.08,
                "solver_vjp_geometric_lift": 2.0 / 1.4,
                "solver_vjp_log_lift_seed_std": 0.06,
                "solver_vjp_log_lift_ic_std": 0.03,
                "seen_ic_matched_horizon_error": 0.1,
                "heldout_ic_matched_horizon_error": 0.12,
                "seen_ic_long_horizon_error": 0.15,
                "heldout_ic_long_horizon_error": 0.2,
                "ic_generalization_ratio_at_matched_horizon": 1.2,
                "seen_ic_temporal_extrapolation_ratio": 1.5,
                "heldout_ic_temporal_extrapolation_ratio": 5.0 / 3.0,
                "matched_horizon_time": 0.1,
                "rollout_final_time": 0.2,
            }
        }
    }

    fairness = _plot_solver_in_loop_fairness(
        data,
        ["jax-cfd"],
        tmp_path,
        save=True,
    )
    physics = _plot_solver_in_loop_physics(
        arrays,
        ["jax-cfd", "warp-ns"],
        tmp_path,
        save=True,
    )
    diagnostics = _plot_solver_in_loop_diagnostics(
        data,
        ["jax-cfd"],
        tmp_path,
        save=True,
    )
    _save_solver_in_loop_animation(arrays, ["jax-cfd"], tmp_path)

    assert fairness is not None
    assert physics is not None
    assert diagnostics is not None
    # One shared reference plus raw/corrected curves for each solver.
    assert all(len(axis.lines) == 5 for axis in physics.axes[:4])
    for filename in (
        "solver_in_loop_diagnostics.png",
        "solver_in_loop_fairness.png",
        "solver_in_loop_physics.png",
        "solver_in_loop_trajectory.gif",
    ):
        rendered = tmp_path / filename
        assert rendered.exists()
        assert rendered.stat().st_size > 0


def test_self_reference_fairness_normalizes_solver_specific_targets(tmp_path):
    data = {
        "by_solver": {
            "pict": {
                "uncorrected_mean_rollout_error": 0.4,
                "mean_rollout_error": 0.2,
                "stop_gradient_mean_rollout_error": 0.3,
                "geometric_error_reduction": 2.0,
                "stop_gradient_geometric_error_reduction": 4.0 / 3.0,
                "solver_vjp_geometric_lift": 1.5,
            }
        }
    }

    figure = _plot_solver_in_loop_fairness(
        data,
        ["pict"],
        tmp_path,
        save=False,
        solver_specific_reference=True,
    )

    assert figure is not None
    np.testing.assert_allclose(
        [patch.get_height() for patch in figure.axes[0].patches],
        [0.5, 0.75],
    )
    assert figure.axes[0].get_title() == "Target-normalized quality"


def test_solver_vjp_panels_only_show_recurrence_admitted_cells(tmp_path):
    common = {
        "uncorrected_mean_rollout_error": 0.4,
        "mean_rollout_error": 0.2,
        "stop_gradient_mean_rollout_error": 0.22,
        "geometric_error_reduction": 2.0,
        "stop_gradient_geometric_error_reduction": 1.8,
        "solver_vjp_geometric_lift": 2.0 / 1.8,
        "median_update_time_s": 1.0,
    }
    data = {
        "by_solver": {
            "jax-cfd": {**common, "valid_for_vjp_ranking": False},
            "pict": {**common, "valid_for_vjp_ranking": True},
        }
    }

    figure = _plot_solver_in_loop_fairness(
        data,
        ["jax-cfd", "pict"],
        tmp_path,
        save=False,
    )

    assert figure is not None
    assert len(figure.axes[2].patches) == 1
    assert [tick.get_text() for tick in figure.axes[2].get_xticklabels()] == ["PICT"]
    assert figure.axes[2].get_title() == "Benefit from solver VJP (admitted only)"


def test_solver_in_loop_runs_recurrently_through_dummy(tmp_path, monkeypatch):
    """One update crosses the apply/VJP boundary and writes canonical artifacts."""
    from mosaic.benchmarks.problems import get_config

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    base = get_config("ns-grid")
    jax_cfd = next(spec for spec in base.solvers if spec.key == "jax_cfd")
    cfg = dataclasses.replace(base, solvers=[jax_cfd])
    cfg.add_experiment(
        "optimization/solver_in_loop_smoke",
        solver_in_loop,
        runs=[
            {
                "ic": {"name": "multimode", "seed": 0},
                "physics": {"N": 8, "nu": 0.001, "dt": 0.02, "steps": 1},
                "dataset": {
                    "reference_factor": 2,
                    "reference_substeps": 1,
                    "train_seeds": [0],
                    "test_seeds": [100],
                    "train_frames": 2,
                    "k0": 2.0,
                },
                "training": {
                    "max_updates": 1,
                    "unroll": 2,
                    "loss_mode": "solver_terminal",
                    "solver_loss_weight": 0.1,
                    "loss_normalization": "solver_baseline",
                    "hidden_channels": 4,
                    "kernel_size": 3,
                    "model_seeds": [0, 1],
                    "check_grad": True,
                    "fd_epsilon": 1e-3,
                },
                "evaluation": {
                    "rollout_frames": 2,
                    "stable_error_threshold": 1.0,
                },
            }
        ],
    )
    result = cfg.experiments["optimization/solver_in_loop_smoke"].fn(
        cfg,
        {jax_cfd.name: f"inprocess:{_IDENTITY_DUMMY}"},
    )

    assert len(result["results"]) == 1
    metrics = result["results"][0]["metrics"]
    assert metrics["n_updates"] == 1
    assert metrics["total_optimizer_updates"] == 2
    assert metrics["stop_gradient_total_optimizer_updates"] == 2
    assert metrics["n_model_seeds"] == 2
    assert metrics["completed"] is True
    assert metrics["final_grad_norm"] > 0
    assert metrics["end_to_end_fd_rel_error"] < 5e-2
    assert metrics["native_final_rollout_error"] >= 0
    assert metrics["native_final_rollout_error_p95"] >= 0
    assert metrics["long_closure_error_p95"] < 1e-6
    assert metrics["first_interval_error_p95"] >= 0
    assert metrics["solver_vjp_geometric_lift"] > 0
    assert metrics["solver_vjp_update_overhead_ratio"] > 0
    assert metrics["corrector_architecture"] == "periodic_residual_cnn"
    assert metrics["training_loss_mode"] == "solver_terminal"
    assert metrics["training_solver_loss_weight"] == 0.1
    assert metrics["training_loss_normalization"] == "solver_baseline"
    assert metrics["training_loss_scale"] > 0
    assert metrics["n_train_trajectories"] == 1
    assert metrics["n_test_trajectories"] == 1
    assert metrics["seen_ic_matched_horizon_error"] >= 0
    assert metrics["heldout_ic_long_horizon_error"] >= 0
    assert metrics["final_rollout_error_ic_std"] == 0
    out_dir = tmp_path / "ns-grid" / "optimization" / "solver_in_loop_smoke"
    assert (out_dir / "result.json").exists()
    with np.load(out_dir / "corrector_fields.npz") as snapshots:
        assert snapshots["solver_vjp_log_lift_samples_0"].shape == (2, 1)


def test_solver_self_reference_skips_training_below_refinement_floor(
    tmp_path,
    monkeypatch,
):
    """An admitted interface still skips training when refinement has no signal."""
    from mosaic.benchmarks.problems import get_config

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    base = get_config("ns-grid")
    pict = next(spec for spec in base.solvers if spec.key == "pict")
    cfg = dataclasses.replace(base, solvers=[pict])
    cfg.add_experiment(
        "optimization/solver_self_reference_ineligible_smoke",
        solver_in_loop,
        runs=[
            {
                "ic": {"name": "multimode", "seed": 0},
                "physics": {"N": 8, "nu": 0.001, "dt": 0.02, "steps": 1},
                "dataset": {
                    "reference_kind": "solver_self_refined",
                    "reference_factor": 2,
                    "reference_temporal_factor": 2,
                    "train_seeds": [0],
                    "test_seeds": [100],
                    "train_frames": 1,
                    "prefix_audit_seeds": [0, 100],
                    "prefix_audit_frames": [1],
                    "k0": 2.0,
                },
                "training": {
                    "max_updates": 1,
                    "unroll": 1,
                    "hidden_channels": 4,
                    "kernel_size": 3,
                },
                "evaluation": {"rollout_frames": 1},
            }
        ],
    )
    result = cfg.experiments["optimization/solver_self_reference_ineligible_smoke"].fn(
        cfg,
        {pict.name: f"inprocess:{_IDENTITY_DUMMY}"},
    )

    metrics = result["results"][0]["metrics"]
    assert metrics["completed"] is True
    assert metrics["eligible_for_corrector_training"] is False
    assert metrics["mean_refinement_signal"] == 0
    assert metrics["n_updates"] == 0
