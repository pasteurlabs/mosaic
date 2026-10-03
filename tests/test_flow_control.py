# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical invariants and analytic gradient checks for the control benchmark."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.flow_control import control
from mosaic.benchmarks.problems.navier_stokes_grid.corrector import spectral_restrict


def _identity_drift(monkeypatch):
    calls = []

    def advance(_t, ctx, velocity, *, dt, steps, native_state=None):
        calls.append((dt, steps, native_state, dict(ctx.phys)))
        return velocity, 1 if native_state is None else native_state + 1

    monkeypatch.setattr(control, "_unforced_solver_advance", advance)
    return calls


def _context(config):
    return SimpleNamespace(
        name="toy", phys={"nu": config.nu}, domain_extent=config.domain_extent, run={}
    )


@pytest.mark.parametrize("n", [8, 24, 64, 192])
def test_actuators_are_orthonormal_zero_mean_and_divergence_free(n):
    # Measure the float32 physical fields with float64 reductions/FFT so the
    # check does not add another low-precision accumulation error at 192².
    basis = np.asarray(control.actuator_basis(n), dtype=np.float64)
    # The physical vector inner product sums velocity components, then averages space.
    matrix = basis.reshape(8, -1)
    gram = matrix @ matrix.T / n**2
    np.testing.assert_allclose(gram, np.eye(8), atol=6e-7)
    np.testing.assert_allclose(basis.mean(axis=(1, 2, 3)), 0, atol=2e-7)
    waves = np.fft.fftfreq(n, d=1 / n)
    transformed = np.fft.fft2(basis[:, :, :, 0], axes=(1, 2))
    divergence = (
        1j * waves[None, :, None] * transformed[..., 0]
        + 1j * waves[None, None, :] * transformed[..., 1]
    )
    assert np.max(np.abs(np.fft.ifft2(divergence, axes=(1, 2)))) < 6e-6


def test_actuators_represent_identical_physical_forces_on_both_grids():
    fine = control.actuator_basis(24)
    coarse = control.actuator_basis(8)
    restricted = jnp.stack([spectral_restrict(mode, 8) for mode in fine])
    np.testing.assert_allclose(restricted, coarse, atol=8e-7)


def test_bound_and_effort_match_normalized_integrated_physical_energy():
    config = control.ControlConfig(control_slots=3, steps_per_slot=2, dt=0.07)
    latent = jnp.linspace(-15, 15, 24).reshape(3, 8)
    coefficients = control.controls_from_latent(latent, config)
    assert np.max(np.abs(coefficients)) <= config.control_bound
    assert np.isfinite(
        np.asarray(
            jax.grad(lambda z: control.controls_from_latent(z, config).sum())(latent)
        )
    ).all()
    energies = []
    for n in (8, 24):
        force = np.einsum("sm,mxyzc->sxyzc", coefficients, control.actuator_basis(n))
        slot_energy = np.mean(np.sum(force**2, axis=-1), axis=(1, 2, 3))
        duration = config.dt * config.steps_per_slot * config.control_slots
        integrated = slot_energy.sum() * config.dt * config.steps_per_slot
        energies.append(integrated / (duration * 8 * config.control_bound**2))
    np.testing.assert_allclose(
        energies, control.control_effort(coefficients, config), rtol=2e-6
    )
    doubled = replace(config, control_slots=6)
    assert float(
        control.control_effort(jnp.repeat(coefficients, 2, axis=0), doubled)
    ) == pytest.approx(float(control.control_effort(coefficients, config)), rel=2e-7)


@pytest.mark.parametrize("temporal_factor", [1, 3, 6])
def test_rollout_preserves_physical_duration_and_native_state(
    monkeypatch, temporal_factor
):
    calls = _identity_drift(monkeypatch)
    config = control.ControlConfig(control_slots=3, steps_per_slot=2, dt=0.07)
    initial = jnp.zeros((8, 8, 1, 2))
    coefficients = jnp.linspace(-0.2, 0.2, 24).reshape(3, 8)
    states = control.rollout(
        None,
        _context(config),
        initial,
        coefficients,
        config,
        temporal_factor=temporal_factor,
        return_states=True,
    )
    expected = (
        config.dt
        * config.steps_per_slot
        * jnp.einsum(
            "m,mxyzc->xyzc", coefficients.sum(axis=0), control.actuator_basis(8)
        )
    )
    np.testing.assert_allclose(states[-1], expected, rtol=3e-6, atol=1e-7)
    assert states.shape == (4, 8, 8, 1, 2)
    assert len(calls) == 3 * 2 * temporal_factor
    assert sum(dt for dt, *_ in calls) == pytest.approx(0.42)
    assert all(steps == 1 for _, steps, *_ in calls)
    assert [native for _, _, native, _ in calls] == [None, *range(1, len(calls))]


def test_terminal_gradient_matches_known_forced_linear_system(monkeypatch):
    _identity_drift(monkeypatch)
    config = control.ControlConfig(control_slots=2, steps_per_slot=2, dt=0.07)
    initial = jnp.zeros((8, 8, 1, 2))
    basis = control.actuator_basis(8)
    latent = jnp.linspace(-0.4, 0.4, 16).reshape(2, 8)
    goal_coefficients = jnp.linspace(-0.03, 0.03, 8)
    goal = jnp.einsum("m,mxyzc->xyzc", goal_coefficients, basis)

    def loss(value):
        coefficients = control.controls_from_latent(value, config)
        final = control.rollout(None, _context(config), initial, coefficients, config)
        return control.terminal_loss(final, goal, coefficients, config)

    actual = jax.grad(loss)(latent)
    bounded = np.asarray(control.controls_from_latent(latent, config))
    slot_time = config.dt * config.steps_per_slot
    residual = slot_time * bounded.sum(axis=0) - np.asarray(goal_coefficients)
    # Each vector-RMS-normalized mode has mean square 1/2 over both components.
    dloss_dcontrol = slot_time * residual[None, :] / config.velocity_scale**2
    dloss_dcontrol = dloss_dcontrol + 2 * config.effort_weight * bounded / (
        bounded.size * config.control_bound**2
    )
    expected = (
        dloss_dcontrol * config.control_bound * (1 - np.tanh(np.asarray(latent)) ** 2)
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-7)


def test_task_generation_keeps_fine_goals_and_public_demonstrations(monkeypatch):
    _identity_drift(monkeypatch)
    config = control.ControlConfig(control_slots=2, steps_per_slot=1)
    first = control.generate_tasks(None, _context(config), config, [4100, 4101])
    replay = control.generate_tasks(None, _context(config), config, [4100])[0]
    np.testing.assert_array_equal(first[0].fine_initial, replay.fine_initial)
    np.testing.assert_array_equal(
        first[0].generating_controls, replay.generating_controls
    )
    np.testing.assert_array_equal(first[0].fine_goal, replay.fine_goal)
    assert not np.array_equal(first[0].fine_initial, first[1].fine_initial)
    assert not np.array_equal(
        first[0].generating_controls, first[1].generating_controls
    )
    assert {task.initial_seed for task in first} == {4100, 4101}
    assert {task.goal_seed for task in first} == {10_004_100, 10_004_101}
    assert not {task.initial_seed for task in first}.intersection(
        task.goal_seed for task in first
    )
    for task in first:
        assert task.fine_initial.shape == task.fine_goal.shape == (192, 192, 1, 2)
        assert task.initial.shape == task.goal.shape == (64, 64, 1, 2)
        expected = task.fine_initial + config.dt * config.steps_per_slot * np.einsum(
            "m,mxyzc->xyzc",
            task.generating_controls.sum(axis=0),
            control.actuator_basis(192),
        )
        np.testing.assert_allclose(task.fine_goal, expected, atol=1e-6)
        np.testing.assert_array_equal(task.fine_goal_rollout[0], task.fine_initial)
        np.testing.assert_array_equal(task.fine_goal_rollout[-1], task.fine_goal)
        np.testing.assert_array_equal(task.goal, spectral_restrict(task.fine_goal, 64))
        assert np.max(np.abs(task.generating_controls)) <= config.control_bound
    with pytest.raises(ValueError, match="unique"):
        control.generate_tasks(None, _context(config), config, [4100, 4100])


@pytest.mark.parametrize("field", ["viscosity", "domain"])
def test_rollout_rejects_mismatched_solver_physics(monkeypatch, field):
    calls = _identity_drift(monkeypatch)
    config = control.ControlConfig(control_slots=1)
    ctx = _context(config)
    if field == "viscosity":
        ctx.phys["nu"] = config.nu * 2
    else:
        ctx.domain_extent = config.domain_extent * 2
    with pytest.raises(ValueError, match=field):
        control.rollout(None, ctx, jnp.zeros((8, 8, 1, 2)), jnp.zeros((1, 8)), config)
    assert not calls
