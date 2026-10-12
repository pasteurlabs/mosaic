# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small mathematical checks; numerical solver validation runs on Slurm/CI."""

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.problems.buoyancy_grid import training
from mosaic.benchmarks.problems.buoyancy_grid.corrector import (
    apply_corrector,
    init_corrector,
    restrict_state,
)


def test_restriction_preserves_faces_and_conserves_scalar():
    state = np.zeros((8, 8, 3), dtype=np.float32)
    state[..., 0] = np.arange(1, 9)[:, None]
    state[..., 1] = np.arange(1, 9)[None, :]
    state[..., 2] = np.arange(64).reshape(8, 8)
    state[-1, :, 0] = 0
    state[:, -1, 1] = 0
    coarse = restrict_state(state, 2)
    np.testing.assert_array_equal(coarse[:, 0, 0], [2, 4, 6, 0])
    np.testing.assert_array_equal(coarse[0, :, 1], [2, 4, 6, 0])
    assert coarse[..., 2].mean() == state[..., 2].mean()


def test_corrector_starts_at_identity_and_preserves_scalar_and_walls():
    state = jnp.ones((6, 6, 3)).at[-1, :, 0].set(0).at[:, -1, 1].set(0)
    params = init_corrector(0, 2)
    np.testing.assert_array_equal(apply_corrector(params, state, jnp.ones(3)), state)
    params[-1]["b"] = jnp.ones(1)
    corrected = np.asarray(apply_corrector(params, state, jnp.ones(3)))
    np.testing.assert_array_equal(corrected[..., 2], state[..., 2])
    np.testing.assert_array_equal(corrected[-1, :, 0], 0)
    np.testing.assert_array_equal(corrected[:, -1, 1], 0)
    assert corrected[0, 0, 0] > state[0, 0, 0]
    delta = corrected - np.asarray(state)
    ux = np.pad(delta[..., 0], ((1, 0), (0, 0)))
    uy = np.pad(delta[..., 1], ((0, 0), (1, 0)))
    np.testing.assert_allclose(np.diff(ux, axis=0) + np.diff(uy, axis=1), 0, atol=1e-6)


def test_stopping_solver_gradient_keeps_values_but_removes_temporal_derivative(
    monkeypatch,
):
    monkeypatch.setattr(training, "apply_corrector", lambda p, state, scales: state + p)
    targets = jnp.zeros((3, 2, 2, 3))

    def loss(p, stop):
        return training.window_loss(
            p,
            targets,
            lambda x: 2 * x,
            jnp.ones(3),
            stop_solver_gradient=stop,
        )

    assert float(loss(1.0, False)) == float(loss(1.0, True))
    np.testing.assert_allclose(jax.grad(lambda p: loss(p, False))(1.0), 10.0, rtol=1e-6)
    np.testing.assert_allclose(jax.grad(lambda p: loss(p, True))(1.0), 4.0, rtol=1e-6)


def test_directional_fd_checks_recurrent_derivative(monkeypatch):
    monkeypatch.setattr(training, "apply_corrector", lambda p, state, scales: state + p)
    targets = jnp.zeros((3, 2, 2, 3))
    check = training.directional_fd(
        lambda p: training.window_loss(p, targets, lambda x: 2 * x, jnp.ones(3)),
        jnp.asarray(0.5),
    )
    assert check["relative_error"] < 1e-3
