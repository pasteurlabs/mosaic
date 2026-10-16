# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warm-state wrappers preserve shared gradient science and sweep metadata."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.benchmarks.core.experiment import KernelContext
from mosaic.benchmarks.problems.buoyancy_grid import gradient
from mosaic.benchmarks.problems.shared import gradient as shared


@pytest.mark.parametrize("wrapper_name", ["fd_check", "param_sweep"])
@pytest.mark.parametrize("steps", [1, 4, 16])
def test_warm_gradient_preparation_is_outside_requested_transition(
    monkeypatch, wrapper_name, steps
):
    calls = []
    state = jnp.ones((2, 2, 3), dtype=jnp.float32)
    physics = {"steps": steps, "dt": 0.01, "nu": 0.03, "domain_extent": 1.0}
    run = {"fd": {"eps_values": [0.1, 0.01, 0.001], "n_dirs": 3}}

    def make_inputs(name, ic, **phys):
        calls.append((name, ic, phys))
        return {"state": ic, **phys}

    def apply(t, inputs):
        assert t == "solver-service"
        return {"out": inputs["state"] * (inputs["steps"] + 1)}

    ctx = KernelContext(
        name="backend",
        cfg=None,
        ic=state,
        phys=physics,
        sweep_key="steps",
        sweep_value=steps,
        ic_key="state",
        output_key="out",
        make_inputs=make_inputs,
        domain_extent=1.0,
        run=run,
        seed=7,
    )
    seen = []

    def shared_kernel(t, current):
        seen.append(current)
        # Like shared FD kernels, differentiate the new transition with respect
        # to its developed initial state, not through the preparation map.
        inputs = current.make_inputs(current.name, current.ic, **current.phys)
        derivative = jax.grad(lambda x: jnp.sum(x * (inputs["steps"] + 1)))(current.ic)
        return {"metrics": {"derivative": derivative}}

    monkeypatch.setattr(gradient, "apply_tesseract", apply)
    monkeypatch.setattr(gradient, f"shared_{wrapper_name}", shared_kernel)
    result = getattr(gradient, wrapper_name)("solver-service", ctx)
    assert len(calls) == 2
    assert calls[0][2] == {**physics, "steps": 4}
    assert calls[1][2] == physics
    np.testing.assert_array_equal(seen[0].ic, state * 5)
    np.testing.assert_array_equal(
        result["metrics"]["derivative"], np.full(state.shape, steps + 1)
    )
    assert seen[0].run is run and seen[0].phys is physics
    assert (
        seen[0].sweep_key == "steps"
        and seen[0].sweep_value == steps
        and seen[0].seed == 7
    )
    assert ctx.ic is state and ctx.phys == physics


@pytest.mark.parametrize("name", ["fd_check", "param_sweep"])
def test_buoyancy_gradient_wrapper_preserves_shared_registration(name):
    assert (
        getattr(gradient, name)._mosaic_kernel_config
        == getattr(shared, name)._mosaic_kernel_config
    )
