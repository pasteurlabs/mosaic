# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and configuration tests for the 2D solver-in-the-loop task."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.utils import _debug_run
from mosaic.benchmarks.problems.navier_stokes_grid.corrector import (
    apply_corrector,
    divergence_rms,
    init_corrector,
    project_periodic_correction,
    reference_trajectory,
    relative_l2,
    spectral_restrict,
)
from mosaic.benchmarks.problems.navier_stokes_grid.ics import _tgv, _tgv_analytic
from mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop import (
    solver_in_loop,
)

_IDENTITY_DUMMY = (
    Path(__file__).parent
    / "dummy_tesseracts"
    / "navier_stokes_grid_identity"
    / "tesseract_api.py"
).resolve()


def test_periodic_corrector_is_translation_equivariant():
    params = init_corrector(jax.random.PRNGKey(0), hidden_channels=4, kernel_size=3)
    velocity = jax.random.normal(jax.random.PRNGKey(1), (8, 8, 1, 2))
    shifted = jnp.roll(velocity, shift=(2, -1), axis=(0, 1))

    expected = jnp.roll(
        apply_corrector(params, velocity, velocity_scale=1.0),
        shift=(2, -1),
        axis=(0, 1),
    )
    actual = apply_corrector(params, shifted, velocity_scale=1.0)

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


def test_debug_run_caps_corrector_training():
    run = {
        "physics": {"N": 32, "steps": 4},
        "training": {
            "max_updates": 100,
            "unroll": 8,
            "hidden_channels": 32,
            "kernel_size": 5,
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
        "check_grad": False,
    }
    assert run["dataset"]["train_seeds"] == [0]
    assert run["dataset"]["test_seeds"] == [100]
    assert run["dataset"]["train_frames"] == 3
    assert run["evaluation"]["rollout_frames"] == 3


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
    assert metrics["completed"] is True
    assert metrics["final_grad_norm"] > 0
    assert metrics["end_to_end_fd_rel_error"] < 5e-2
    out_dir = tmp_path / "ns-grid" / "optimization" / "solver_in_loop_smoke"
    assert (out_dir / "result.json").exists()
    assert (out_dir / "corrector_fields.npz").exists()
