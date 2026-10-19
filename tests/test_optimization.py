# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared optimizer primitives' non-finite loss handling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
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
    rejected like a non-finite value, so no iterate is NaN.
    """
    x0 = jnp.full((4,), 5.0)  # |x0 - 1| = 8, just outside the shell
    x, losses, _ = _run_lbfgs(_loss_with_bad_gradient_shell(bad), x0, max_iters=20)
    assert jnp.all(jnp.isfinite(x))
    assert losses[-1] <= losses[0]


def _bowl(x):
    return jnp.sum((x - jnp.arange(4.0)) ** 2) + 0.1 * jnp.sum(x**4)


def _counting_grad_proj(nan_calls: set[int] = frozenset()):
    """A ``grad_proj_fn`` that records each evaluated ``x``'s gradient and returns
    a NaN gradient on the given 0-based calls.
    """
    seen: list = []

    def host(g):
        seen.append(np.asarray(g).copy())
        return g * np.nan if len(seen) - 1 in nan_calls else g

    def proj(g):
        return jax.pure_callback(host, jax.ShapeDtypeStruct(g.shape, g.dtype), g)

    return proj, seen


@pytest.mark.parametrize("nan_calls", [{2}, {2, 3}, {4}])
def test_lbfgs_recovers_from_nonfinite_trial_points(nan_calls):
    """After non-finite trial points the run still reaches the same minimum."""
    proj, _ = _counting_grad_proj(nan_calls)
    x, losses, diag = _run_lbfgs(_bowl, jnp.zeros(4), max_iters=50, grad_proj_fn=proj)
    _, losses_ref, _ = _run_lbfgs(_bowl, jnp.zeros(4), max_iters=50)
    assert jnp.all(jnp.isfinite(x))
    assert losses[-1] == pytest.approx(losses_ref[-1], rel=1e-5)
    assert diag["converged"]


def test_lbfgs_evaluates_each_point_once():
    proj, seen = _counting_grad_proj()
    _, losses, _ = _run_lbfgs(_bowl, jnp.zeros(4), max_iters=50, grad_proj_fn=proj)
    assert len(losses) > 2
    assert len({g.tobytes() for g in seen}) == len(seen)


def test_lbfgs_records_init_and_final_iterate():
    x, losses, diag = _run_lbfgs(_bowl, jnp.zeros(4), max_iters=50)
    assert losses[0] == pytest.approx(float(_bowl(jnp.zeros(4))))
    assert losses[-1] == pytest.approx(float(_bowl(x)))
    assert len(diag["grad_norms"]) == len(losses)


def test_lbfgs_respects_bounds():
    """The unconstrained minimum lies outside the box, so some entries end on a bound."""
    x, _, _ = _run_lbfgs(_bowl, jnp.full(4, 0.5), max_iters=50, bounds=(0.25, 1.0))
    assert jnp.all((x >= 0.25) & (x <= 1.0))
    assert jnp.any(x == 1.0)


def test_lbfgs_reports_convergence():
    _, losses, diag = _run_lbfgs(_bowl, jnp.zeros(4), max_iters=50)
    assert diag["converged"] and len(losses) < 50
    _, losses, diag = _run_lbfgs(_bowl, jnp.zeros(4), max_iters=2)
    assert not diag["converged"] and len(losses) == 3
