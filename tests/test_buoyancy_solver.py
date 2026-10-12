# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native solver checks, run in the PhiFlow environment on a cluster worker."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def solver():
    pytest.importorskip("phi.jax.flow")
    path = (
        Path(__file__).resolve().parents[1]
        / "mosaic/tesseracts/buoyancy-grid/phiflow/tesseract_api.py"
    )
    spec = importlib.util.spec_from_file_location("buoyancy_phiflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs():
    import jax.numpy as jnp
    from mosaic_shared.problems.buoyancy_grid import make_plume_ic

    return {
        "state0": jnp.asarray(make_plume_ic(8)),
        "viscosity": jnp.array([0.001]),
        "diffusivity": jnp.array([0.0001]),
        "buoyancy": jnp.array([1.0]),
        "dt": jnp.array([0.02]),
        "steps": 1,
        "domain_extent": 1.0,
    }


def test_resume_and_wall_faces(solver):
    args = _inputs()
    first = solver.apply_jit(args)["result"]
    resumed = solver.apply_jit({**args, "state0": first})["result"]
    together = solver.apply_jit({**args, "steps": 2})["result"]
    np.testing.assert_allclose(resumed, together, atol=1e-6, rtol=1e-6)
    assert np.isfinite(together).all()
    np.testing.assert_array_equal(together[-1, :, 0], 0)
    np.testing.assert_array_equal(together[:, -1, 1], 0)
    assert np.linalg.norm(together[..., :2]) > 0


def test_zero_buoyancy_preserves_rest(solver):
    import jax.numpy as jnp

    args = _inputs()
    result = solver.apply_jit({**args, "buoyancy": jnp.array([0.0])})["result"]
    np.testing.assert_allclose(result[..., :2], 0, atol=1e-7)


def test_recurrent_state_directional_gradient(solver):
    import jax
    import jax.numpy as jnp

    args = _inputs()
    # Nonzero velocity avoids evaluating interpolation gradients at a kink.
    state = solver.apply_jit({**args, "steps": 3})["result"]
    rng = np.random.default_rng(4)
    direction = rng.normal(size=state.shape).astype(np.float32)
    direction[-1, :, 0] = 0
    direction[:, -1, 1] = 0
    direction = jnp.asarray(direction / np.linalg.norm(direction))
    weights = jnp.asarray(rng.normal(size=state.shape), dtype=jnp.float32)

    def objective(value):
        return jnp.sum(
            solver.apply_jit({**args, "state0": value, "steps": 2})["result"] * weights
        )

    gradient = jax.grad(objective)(state)
    analytic = float(jnp.sum(gradient * direction))
    eps = 1e-3
    finite_difference = float(
        (objective(state + eps * direction) - objective(state - eps * direction))
        / (2 * eps)
    )
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(analytic, finite_difference, rtol=0.03, atol=0.003)
    np.testing.assert_array_equal(gradient[-1, :, 0], 0)
    np.testing.assert_array_equal(gradient[:, -1, 1], 0)


def test_runtime_vjp_physics_parameters(solver):
    import jax.numpy as jnp

    args = _inputs()
    args["state0"] = solver.apply_jit({**args, "steps": 3})["result"]
    weights = jnp.asarray(
        np.random.default_rng(6).normal(size=args["state0"].shape), dtype=jnp.float32
    )
    directions = {"viscosity": 0.01, "diffusivity": 0.01, "buoyancy": 0.2, "dt": 0.01}
    gradients = solver.vector_jacobian_product(
        solver.InputSchema(**args), set(directions), {"result"}, {"result": weights}
    )
    analytic = sum(
        float(np.asarray(gradients[key])[0]) * scale
        for key, scale in directions.items()
    )
    epsilon = 1e-3
    plus = {
        **args,
        **{key: args[key] + epsilon * scale for key, scale in directions.items()},
    }
    minus = {
        **args,
        **{key: args[key] - epsilon * scale for key, scale in directions.items()},
    }
    finite_difference = float(
        jnp.sum(
            (solver.apply_jit(plus)["result"] - solver.apply_jit(minus)["result"])
            * weights
        )
        / (2 * epsilon)
    )
    np.testing.assert_allclose(analytic, finite_difference, rtol=0.03, atol=0.003)


def test_diffusion_stability_bound(solver):
    args = _inputs()
    args["dt"] = np.array([100.0], np.float32)
    with pytest.raises(ValueError, match="Explicit diffusion"):
        solver.InputSchema(**args)


def test_abstract_eval_matches_input_shape(solver):
    inputs = solver.InputSchema(**_inputs())
    assert solver.abstract_eval(inputs) == {
        "result": {"shape": (8, 8, 3), "dtype": "float32"}
    }
    abstract = solver.InputSchema.model_construct(
        state0={"shape": (16, 16, 3), "dtype": "float32"}
    )
    assert solver.abstract_eval(abstract) == {
        "result": {"shape": (16, 16, 3), "dtype": "float32"}
    }
