# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and configuration tests for the 2D solver-in-the-loop task."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.utils import _debug_run
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
    _plot_solver_in_loop_fairness,
    _plot_solver_in_loop_fields,
    _plot_solver_in_loop_physics,
    _save_solver_in_loop_animation,
)
from mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop import (
    _rollout_log_gain,
    make_reference_dataset,
    solver_in_loop,
)

_IDENTITY_DUMMY = (
    Path(__file__).parent
    / "dummy_tesseracts"
    / "navier_stokes_grid_identity"
    / "tesseract_api.py"
).resolve()


def test_periodic_corrector_is_translation_equivariant():
    model = init_corrector(jax.random.PRNGKey(0), hidden_channels=4, kernel_size=3)
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
        "reference_rollout": np.stack((reference, reference)),
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
    }
    data = {
        "by_solver": {
            "jax-cfd": {
                "native_final_rollout_error": 0.1,
                "uncorrected_rollout_error": 0.3,
                "uncorrected_mean_rollout_error": 0.25,
                "mean_rollout_error": 0.12,
                "geometric_error_reduction": 2.0,
                "rollout_log_gain_seed_std": 0.1,
                "stop_gradient_geometric_error_reduction": 1.4,
                "stop_gradient_rollout_log_gain_seed_std": 0.08,
                "solver_vjp_geometric_lift": 2.0 / 1.4,
                "solver_vjp_log_lift_seed_std": 0.06,
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
        ["jax-cfd"],
        tmp_path,
        save=True,
    )
    _save_solver_in_loop_animation(arrays, ["jax-cfd"], tmp_path)

    assert fairness is not None
    assert physics is not None
    for filename in (
        "solver_in_loop_fairness.png",
        "solver_in_loop_physics.png",
        "solver_in_loop_trajectory.gif",
    ):
        rendered = tmp_path / filename
        assert rendered.exists()
        assert rendered.stat().st_size > 0


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
    assert metrics["solver_vjp_geometric_lift"] > 0
    assert metrics["solver_vjp_update_overhead_ratio"] > 0
    assert metrics["corrector_architecture"] == "periodic_residual_cnn"
    out_dir = tmp_path / "ns-grid" / "optimization" / "solver_in_loop_smoke"
    assert (out_dir / "result.json").exists()
    assert (out_dir / "corrector_fields.npz").exists()
