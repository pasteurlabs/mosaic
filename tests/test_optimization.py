# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared optimizer primitives' non-finite loss handling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from mosaic.benchmarks.core.harness import classify_failure
from mosaic.benchmarks.problems.shared.optimization import _run_lbfgs, _run_optim


@pytest.mark.parametrize("optim_fn", [_run_optim, _run_lbfgs])
def test_nonfinite_loss_raises(optim_fn):
    """A NaN loss fails the run instead of returning a NaN trajectory."""

    def loss_fn(x):
        return jnp.sum(x**2) * jnp.nan

    with pytest.raises(FloatingPointError) as exc_info:
        optim_fn(loss_fn, jnp.ones(3), 1e-2, 5, 5)
    msg = str(exc_info.value)
    assert "loss is not finite (nan) at iteration 0" in msg
    assert classify_failure(type(exc_info.value).__name__, msg) == "nan"


@pytest.mark.parametrize("optim_fn", [_run_optim, _run_lbfgs])
def test_finite_loss_runs(optim_fn):
    _, losses, _ = optim_fn(lambda x: jnp.sum(x**2), jnp.ones(3), 1e-2, 5, 5)
    assert losses


def _loss_with_bad_gradient_shell(bad: float):
    """Smooth bowl whose gradient is ``bad`` for 0.5 < |x - 1| < 7; the value stays finite."""

    @jax.custom_vjp
    def loss(x):
        return jnp.sum(jnp.cosh(x - 1.0) - 1.0)

    def fwd(x):
        return loss(x), x

    def bwd(x, ct):
        r = jnp.linalg.norm(x - 1.0)
        return (jnp.where((r > 0.5) & (r < 7.0), bad, jnp.sinh(x - 1.0) * ct),)

    loss.defvjp(fwd, bwd)
    return loss


@pytest.mark.parametrize("bad", [jnp.inf, -jnp.inf, 3e38])
def test_lbfgs_rejects_trial_steps_with_unusable_gradient(bad):
    """A trial step with a finite value but an infinite or overflowing gradient is
    rejected like a non-finite value. Before, it entered L-BFGS's history and every
    later step was NaN.
    """
    x0 = jnp.full((4,), 5.0)  # |x0 - 1| = 8, just outside the shell
    x, losses, _ = _run_lbfgs(_loss_with_bad_gradient_shell(bad), x0, max_iters=20)
    assert jnp.all(jnp.isfinite(x))
    assert losses[-1] <= losses[0]


def test_lbfgs_unchanged_where_finite(monkeypatch):
    """Where values and gradients are finite, the iterates are bitwise unchanged."""
    from mosaic.benchmarks.problems.shared import optimization

    def loss(x):
        return jnp.sum((x - jnp.arange(4.0)) ** 4) + jnp.sum(jnp.sin(x))

    x, losses, _ = _run_lbfgs(loss, jnp.zeros(4), max_iters=15)
    monkeypatch.setattr(optimization, "_outside_domain_if_nonfinite", lambda f: f)
    x_ref, losses_ref, _ = _run_lbfgs(loss, jnp.zeros(4), max_iters=15)
    assert losses == losses_ref
    assert jnp.array_equal(x, x_ref)
