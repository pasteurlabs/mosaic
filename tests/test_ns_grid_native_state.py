# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solvers that continue from their own state, on the forced periodic box.

A chain of short calls that carries ``state`` from one call to the next must
reproduce one long call, and the VJP must reach through the state. Every test
runs against the solver's ``:latest`` image built from this checkout and is
skipped where that image or Docker is missing.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from .helpers import image_available

_L = 2 * np.pi

_SOLVERS = {
    "jax-cfd": (
        "jax_cfd_navier_stokes_grid:latest",
        {"density": np.array([1.0], np.float32), "inner_steps": 1},
    ),
    "xlb": ("xlb_navier_stokes_grid:latest", {}),
    "ins-jl": ("ins_navier_stokes_grid:latest", {}),
}


@pytest.fixture(scope="module", params=sorted(_SOLVERS))
def solver(request):
    image, extra = _SOLVERS[request.param]
    if not image_available(image):
        pytest.skip(f"{image} not built")
    from tesseract_core import Tesseract

    gpus = ["all"] if shutil.which("nvidia-smi") else None
    with Tesseract.from_image(image, gpus=gpus) as t:
        yield request.param, t, extra


def _flow(n: int, seed: int = 0) -> np.ndarray:
    """A divergence-free field of a few modes, with speeds up to 2."""
    rng = np.random.default_rng(seed)
    x = (np.arange(n) + 0.5) * _L / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    v = np.zeros((n, n, 1, 2))
    for kx, ky in ((1, 2), (3, 1), (2, 3)):
        c = rng.normal() * np.cos(kx * xx + ky * yy + rng.uniform(0, 2 * np.pi))
        v[..., 0, 0] += ky * c
        v[..., 0, 1] -= kx * c
    return (2.0 * v / np.abs(v).max()).astype(np.float32)


def _inputs(v0, steps, extra, **more) -> dict:
    n = v0.shape[0]
    force = np.zeros((n, n, 1, 2), np.float32)
    force[..., 0, 0] = np.sin(4 * (np.arange(n) + 0.5) * _L / n)[None, :]
    return {
        "v0": np.asarray(v0, np.float32),
        "viscosity": np.array([0.01], np.float32),
        "dt": np.array([0.01], np.float32),
        "steps": steps,
        "domain_extent": float(_L),
        "forcing": force,
        "linear_drag": np.array([0.1], np.float32),
        **extra,
        **more,
    }


def test_a_chain_of_calls_continues_exactly(solver):
    _, t, extra = solver
    v0 = _flow(32)
    one = np.asarray(t.apply(_inputs(v0, 40, extra))["result"])
    v, state = v0, None
    for _ in range(4):
        more = {"return_state": True} if state is None else {"state": state}
        out = t.apply(_inputs(v, 10, extra, **more))
        v, state = np.asarray(out["result"]), np.asarray(out["state"])
    np.testing.assert_allclose(v, one, rtol=0, atol=1e-5)


def test_vjp_reaches_through_the_state(solver):
    """⟨c, result⟩ + ⟨s, state⟩ along directions in v0 and the state, against FD."""
    name, t, extra = solver
    v0 = _flow(16)
    first = t.apply(_inputs(v0, 5, extra, return_state=True))
    v1, state = np.asarray(first["result"]), np.asarray(first["state"])
    rng = np.random.default_rng(1)
    c_result = _flow(16, seed=2)
    c_state = rng.normal(size=state.shape).astype(np.float32)
    # Directions: a divergence-free change of v0, and a change of the state
    # proportional to itself so that it stays a state the solver can continue.
    d_v0 = 0.1 * _flow(16, seed=3)
    d_state = 0.01 * state

    def objective(v, s):
        out = t.apply(_inputs(v, 5, extra, state=s.astype(np.float32)))
        return float(
            np.sum(np.asarray(out["result"], np.float64) * c_result)
            + np.sum(np.asarray(out["state"], np.float64) * c_state)
        )

    grads = t.vector_jacobian_product(
        _inputs(v1, 5, extra, state=state),
        ["v0", "state"],
        ["result", "state"],
        {"result": c_result, "state": c_state},
    )
    ad = float(np.sum(grads["v0"] * d_v0) + np.sum(grads["state"] * d_state))
    eps = 1e-2
    fd = (
        objective(v1 + eps * d_v0, state + eps * d_state)
        - objective(v1 - eps * d_v0, state - eps * d_state)
    ) / (2 * eps)
    assert ad == pytest.approx(fd, rel=2e-2), f"{name}: AD {ad:.6g} vs FD {fd:.6g}"
