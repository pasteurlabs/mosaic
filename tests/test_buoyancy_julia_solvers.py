# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in contracts for preinstalled Julia adapters; never bootstrap Julia in pytest."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module", params=["waterlily", "oceananigans"])
def solver(request):
    selected = os.environ.get("MOSAIC_TEST_JULIA_SOLVERS", "").split(",")
    if request.param not in selected:
        pytest.skip("Set MOSAIC_TEST_JULIA_SOLVERS to explicitly opt in")
    executable = os.environ.get("PYTHON_JULIACALL_EXE")
    project = os.environ.get("PYTHON_JULIACALL_PROJECT")
    if not executable or not Path(executable).is_file() or not project:
        pytest.skip("A preinstalled Julia executable and project must be configured")
    pytest.importorskip("juliacall")
    path = (
        Path(__file__).resolve().parents[1]
        / "mosaic/tesseracts/buoyancy-grid"
        / request.param
        / "tesseract_api.py"
    )
    spec = importlib.util.spec_from_file_location(f"buoyancy_{request.param}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    n = 8
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    state = np.zeros((n, n, 3), np.float32)
    state[..., 2] = np.exp(-((xx - 0.45) ** 2 + (yy - 0.25) ** 2) / 0.025)
    return {
        "state0": state,
        "viscosity": np.array([0.001], np.float32),
        "diffusivity": np.array([0.0001], np.float32),
        "buoyancy": np.array([1.0], np.float32),
        "dt": np.array([0.01], np.float32),
        "steps": 1,
        "domain_extent": 1.0,
    }


def apply(solver, args):
    return np.asarray(solver.apply(solver.InputSchema(**args))["result"])


def test_restart_and_ignored_wall_faces(solver):
    args = inputs()
    first = apply(solver, args)
    resumed = apply(solver, {**args, "state0": first})
    together = apply(solver, {**args, "steps": 2})
    assert np.isfinite(together).all()
    np.testing.assert_allclose(resumed, together, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(together[-1, :, 0], 0)
    np.testing.assert_array_equal(together[:, -1, 1], 0)
    padded = args["state0"].copy()
    padded[-1, :, 0] = 7
    padded[:, -1, 1] = -4
    np.testing.assert_allclose(
        apply(solver, {**args, "state0": padded}), first, rtol=0, atol=1e-7
    )
    assert np.linalg.norm(first[..., :2]) > 0


def test_no_buoyancy_preserves_rest(solver):
    args = {**inputs(), "buoyancy": np.array([0.0], np.float32)}
    np.testing.assert_allclose(apply(solver, args)[..., :2], 0, atol=1e-6)


def test_runtime_schema_and_apply(solver):
    from tesseract_core import Tesseract

    args = inputs()
    with Tesseract.from_tesseract_api(solver.__file__) as runtime:
        output = runtime.apply(args)
        np.testing.assert_allclose(
            output["result"], apply(solver, args), rtol=1e-6, atol=1e-6
        )


def test_state_vjp_when_supported(solver):
    if not hasattr(solver, "vector_jacobian_product"):
        pytest.skip("Adapter does not advertise derivatives")
    args = inputs()
    args["state0"] = apply(solver, {**args, "steps": 4})
    args["steps"] = 2
    rng = np.random.default_rng(9)
    cotangent = rng.normal(size=args["state0"].shape).astype(np.float32)
    direction = rng.normal(size=args["state0"].shape).astype(np.float32)
    direction[-1, :, 0] = 0
    direction[:, -1, 1] = 0
    direction /= np.linalg.norm(direction)
    gradient = np.asarray(
        solver.vector_jacobian_product(
            solver.InputSchema(**args),
            {"state0"},
            {"result"},
            {"result": cotangent},
        )["state0"]
    )
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient[-1, :, 0], 0)
    np.testing.assert_array_equal(gradient[:, -1, 1], 0)
    epsilon = 1e-3
    plus = apply(solver, {**args, "state0": args["state0"] + epsilon * direction})
    minus = apply(solver, {**args, "state0": args["state0"] - epsilon * direction})
    finite_difference = np.sum((plus - minus) * cotangent) / (2 * epsilon)
    np.testing.assert_allclose(
        np.sum(gradient * direction), finite_difference, rtol=0.05, atol=0.003
    )


def test_explicit_diffusion_rejects_unstable_step(solver):
    args = {**inputs(), "dt": np.array([100.0], np.float32)}
    with pytest.raises(ValueError):
        solver.InputSchema(**args)
