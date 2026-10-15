# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared optimizer primitives' non-finite loss handling."""

from __future__ import annotations

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
