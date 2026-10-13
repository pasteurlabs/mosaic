# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small mathematical checks; numerical solver validation runs on Slurm/CI."""

from itertools import pairwise
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(params))
    np.testing.assert_array_equal(apply_corrector(params, state, jnp.ones(3)), state)
    np.testing.assert_array_equal(
        apply_corrector(params, np.asarray(state), jnp.ones(3)), state
    )
    params[-1]["b"] = jnp.ones(1, dtype=jnp.float32)
    corrected = np.asarray(apply_corrector(params, state, jnp.ones(3)))
    np.testing.assert_array_equal(corrected[..., 2], state[..., 2])
    np.testing.assert_array_equal(corrected[-1, :, 0], 0)
    np.testing.assert_array_equal(corrected[:, -1, 1], 0)
    assert corrected[0, 0, 0] > state[0, 0, 0]
    delta = corrected - np.asarray(state)
    ux = np.pad(delta[..., 0], ((1, 0), (0, 0)))
    uy = np.pad(delta[..., 1], ((0, 0), (1, 0)))
    np.testing.assert_allclose(np.diff(ux, axis=0) + np.diff(uy, axis=1), 0, atol=1e-6)


def test_training_gradient_includes_recurrent_solver_derivative(monkeypatch):
    monkeypatch.setattr(training, "apply_corrector", lambda p, state, scales: state + p)
    targets = jnp.zeros((3, 2, 2, 3))

    def loss(p):
        return training.window_loss(p, targets, lambda x: 2 * x, jnp.ones(3))

    np.testing.assert_allclose(jax.grad(loss)(1.0), 10.0, rtol=1e-6)


def test_directional_fd_checks_recurrent_derivative(monkeypatch):
    monkeypatch.setattr(training, "apply_corrector", lambda p, state, scales: state + p)
    targets = jnp.zeros((3, 2, 2, 3))
    check = training.directional_fd(
        lambda p: training.window_loss(p, targets, lambda x: 2 * x, jnp.ones(3)),
        jnp.asarray(0.5),
    )
    assert check["relative_error"] < 1e-3


@pytest.mark.parametrize("ic", [{}, {"name": "interacting_plumes"}])
def test_training_uses_configured_ic_for_each_split(monkeypatch, ic):
    calls = []

    class InitialsCaptured(Exception):
        pass

    def initial(**kwargs):
        calls.append(kwargs)
        if len(calls) == 3:
            raise InitialsCaptured

    def wrong_initial(**kwargs):
        raise AssertionError("selected the wrong initial-condition family")

    family = ic.get("name", "plume")
    registry = {"plume": wrong_initial, "interacting_plumes": wrong_initial}
    registry[family] = initial
    ctx = SimpleNamespace(
        run={
            "ic": ic,
            "training": {"train_seed": 7, "validation_seed": 8, "test_seed": 9},
        },
        phys={"N": 32},
        domain_extent=2.0,
        cfg=SimpleNamespace(make_ic=registry),
    )
    # Stop after initial construction; neither a solver nor JAX execution is needed.
    monkeypatch.setattr(training, "trajectory", lambda *args: [None])
    monkeypatch.setattr(training, "restrict_state", lambda state, factor: state)
    with pytest.raises(InitialsCaptured):
        training.solver_in_loop(None, ctx)
    assert calls == [{"N": 64, "L": 2.0, "seed": seed} for seed in (7, 8, 9)]


@pytest.mark.parametrize("updates,count", [(64, 93), (128, 85), (1, 1), (4, 1)])
def test_training_windows_cover_full_trajectory_evenly(updates, count):
    starts = [training.window_start(i, updates, count) for i in range(updates)]
    assert starts[0] == 0
    assert starts[-1] == count - 1
    assert all(0 <= start < count for start in starts)
    gaps = [b - a for a, b in pairwise(starts)]
    if gaps:
        assert max(gaps) - min(gaps) <= 1
    if updates >= count:
        assert set(starts) == set(range(count))
