# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for shared optimisation diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.problems.shared.optimization import _run_lbfgs


def test_lbfgs_records_iterate_and_projected_gradient_diagnostics():
    init = jnp.asarray([2.0, -1.0], dtype=jnp.float32)

    _, losses, diagnostics = _run_lbfgs(
        lambda value: jnp.sum((value - 0.25) ** 2),
        init,
        max_iters=3,
        record_diagnostics=True,
        div_fn=lambda value: float(np.max(np.abs(value))),
        grad_proj_fn=lambda gradient: gradient.at[1].set(0.0),
    )

    assert len(losses) == 3
    assert len(diagnostics["grad_norms"]) == len(losses)
    assert len(diagnostics["grad_divs"]) == len(losses)
    assert len(diagnostics["ic_divs"]) == len(losses)
    assert diagnostics["grad_divs"][0] == 3.5
    assert all(np.isfinite(diagnostics["ic_divs"]))
