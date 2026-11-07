# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MMA optimizer of the structural topology optimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.benchmarks.problems.structural_mesh.optimization import _run_mma

# A compliance-like objective, sum(w / x): with mean(x) <= v_frac its minimum
# is at x proportional to sqrt(w), which stays inside the bounds for these w.
_W = jnp.array([1.0, 2.0, 3.0, 4.0])
_V_FRAC = 0.5


def _compliance(x):
    return jnp.sum(_W / x)


def _optimum():
    s = np.sqrt(np.asarray(_W))
    return _V_FRAC * s.size * s / s.sum()


def test_mma_reaches_the_constrained_optimum():
    x0 = jnp.full(4, _V_FRAC, dtype=jnp.float32)
    x, compliances, vol_fracs, diag = _run_mma(
        _compliance, x0, v_frac=_V_FRAC, x_min=1e-3, max_evals=500, patience=500
    )
    assert x.dtype == x0.dtype and x.shape == x0.shape
    np.testing.assert_allclose(np.asarray(x), _optimum(), rtol=1e-3)
    assert float(jnp.mean(x)) <= _V_FRAC + 1e-6
    assert diag["converged"]
    assert diag["n_evals"] == diag["n_iters"] == len(compliances) == len(vol_fracs)
    assert len(diag["grad_norms"]) == len(compliances)
    assert compliances[-1] < compliances[0]


def test_mma_stops_on_the_evaluation_budget():
    x0 = jnp.full(4, _V_FRAC)
    _, compliances, _, diag = _run_mma(
        _compliance, x0, v_frac=_V_FRAC, x_min=1e-3, max_evals=3, patience=500
    )
    assert len(compliances) == 3
    assert not diag["converged"]


def test_mma_stops_after_patience_evaluations_without_improvement():
    """MMA follows a gradient that leads away from the minimum at the start."""
    x0 = jnp.full(4, 0.3)

    def uphill(x):
        value = 1.0 + jnp.sum((x - x0) ** 2)
        return jax.lax.stop_gradient(value + jnp.sum(x)) - jnp.sum(x)

    history: list = []
    _, compliances, _, diag = _run_mma(
        uphill,
        x0,
        v_frac=_V_FRAC,
        x_min=1e-3,
        max_evals=100,
        patience=5,
        snap_interval=2,
        history=history,
    )
    assert len(compliances) == 6
    assert diag["converged"]
    assert len(history) == 3


def test_mma_raises_on_a_nonfinite_compliance():
    with pytest.raises(FloatingPointError):
        _run_mma(
            lambda x: jnp.sum(x) * jnp.nan,
            jnp.full(4, _V_FRAC),
            v_frac=_V_FRAC,
            x_min=1e-3,
            max_evals=10,
            patience=10,
        )
