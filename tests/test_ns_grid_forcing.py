# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Body force and linear drag in the periodic ns-grid solvers.

Started from rest, the Kolmogorov force (A sin(κy), 0) keeps the flow a shear
flow u(y, t). Its advection term vanishes, so the Navier–Stokes equations with
drag α have the exact solution

    u = (A/λ)(1 − e^{−λt}) sin(κy),  v = 0,  λ = νκ² + α.

One run against it checks the amplitude, sign and wavenumber of the force, its
placement on staggered grids, the sign of the drag and any unit conversion.

Every test runs against the solver's ``:latest`` image built from this
checkout and is skipped where that image or Docker is missing. They are never
built here.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from .helpers import image_available

_L = 2 * np.pi

# Solver image and the extra inputs the harness passes it.
_SOLVERS = {
    "jax-cfd": (
        "jax_cfd_navier_stokes_grid:latest",
        {"density": np.array([1.0], np.float32), "inner_steps": 1},
    ),
    "phiflow": ("phiflow_navier_stokes_grid:latest", {}),
    "warp-ns": ("warp_ns_navier_stokes_grid:latest", {}),
    "xlb": ("xlb_navier_stokes_grid:latest", {}),
    "ins-jl": ("ins_navier_stokes_grid:latest", {}),
    "pict": ("pict_navier_stokes_grid:latest", {}),
    "exponax": ("exponax_navier_stokes_grid:latest", {}),
}

_OPENFOAM = "openfoam_navier_stokes_grid:latest"


@pytest.fixture(scope="module", params=sorted(_SOLVERS))
def solver(request):
    image, extra = _SOLVERS[request.param]
    if not image_available(image):
        pytest.skip(f"{image} not built")
    from tesseract_core import Tesseract

    # CI builds some of these images on runners without GPUs, where the JAX
    # solvers fall back to the CPU.
    gpus = ["all"] if shutil.which("nvidia-smi") else None
    with Tesseract.from_image(image, gpus=gpus) as t:
        yield request.param, t, extra


def _kolmogorov_force(n: int, k: int, amplitude: float = 1.0) -> np.ndarray:
    """(A sin(κy), 0) at the cell centres, shape (n, n, 1, 2)."""
    y = (np.arange(n) + 0.5) * _L / n
    force = np.zeros((n, n, 1, 2), np.float32)
    force[..., 0, 0] = amplitude * np.sin(2 * np.pi * k * y / _L)[None, :]
    return force


def _inputs(v0, *, nu, dt, steps, extra, **more) -> dict:
    return {
        "v0": np.asarray(v0, np.float32),
        "viscosity": np.array([nu], np.float32),
        "dt": np.array([dt], np.float32),
        "steps": steps,
        "domain_extent": float(_L),
        **extra,
        **more,
    }


def _smooth_field(n: int, seed: int = 0, amplitude: float = 0.3) -> np.ndarray:
    """A divergence-free field from a few low modes of a stream function."""
    rng = np.random.default_rng(seed)
    x = (np.arange(n) + 0.5) * _L / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    for kx, ky in ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)):
        a, phase = rng.normal(), rng.uniform(0, 2 * np.pi)
        # ψ = a sin(kx x + ky y + φ); u = ∂ψ/∂y, v = −∂ψ/∂x
        c = a * np.cos(kx * xx + ky * yy + phase)
        u += ky * c
        v -= kx * c
    field = np.stack([u, v], -1)[:, :, None, :]
    return (amplitude * field / np.abs(field).max()).astype(np.float32)


def test_laminar_kolmogorov_transient(solver):
    """Within 5% of the exact shear flow at N = 64 after two time units."""
    _, t, extra = solver
    n, k, nu, alpha, dt, steps = 64, 4, 0.05, 0.1, 0.01, 200
    force = _kolmogorov_force(n, k)
    out = t.apply(
        _inputs(
            np.zeros_like(force),
            nu=nu,
            dt=dt,
            steps=steps,
            extra=extra,
            forcing=force,
            linear_drag=np.array([alpha], np.float32),
        )
    )
    rate = nu * k**2 + alpha
    exact = force * (1 - np.exp(-rate * dt * steps)) / rate
    result = np.asarray(out["result"], np.float64)
    error = np.linalg.norm(result - exact) / np.linalg.norm(exact)
    assert error < 0.05, f"relative error {error:.3g}"


def test_zero_force_and_drag_change_nothing(solver):
    """A zero force and zero drag give the unforced result."""
    _, t, extra = solver
    v0 = _smooth_field(16)
    common = {"nu": 0.01, "dt": 0.05, "steps": 10, "extra": extra}
    plain = np.asarray(t.apply(_inputs(v0, **common))["result"])
    zero = np.asarray(
        t.apply(
            _inputs(
                v0,
                **common,
                forcing=np.zeros_like(v0),
                linear_drag=np.zeros(1, np.float32),
            )
        )["result"]
    )
    np.testing.assert_allclose(zero, plain, rtol=0, atol=1e-6)


def test_forced_vjp_matches_central_difference(solver):
    """d⟨c, u(T)⟩/dv0 along a direction, with force and drag, against FD."""
    name, t, extra = solver
    n = 16
    v0 = _smooth_field(n)
    force = _kolmogorov_force(n, 2)
    inputs = _inputs(
        v0,
        nu=0.02,
        dt=0.02,
        steps=10,
        extra=extra,
        forcing=force,
        linear_drag=np.array([0.1], np.float32),
    )
    cotangent = _smooth_field(n, seed=1, amplitude=1.0)
    direction = _smooth_field(n, seed=2, amplitude=1.0)

    def objective(v):
        out = t.apply({**inputs, "v0": v.astype(np.float32)})["result"]
        return float(np.sum(np.asarray(out, np.float64) * cotangent))

    grad = t.vector_jacobian_product(inputs, ["v0"], ["result"], {"result": cotangent})[
        "v0"
    ]
    eps = 1e-2
    fd = (objective(v0 + eps * direction) - objective(v0 - eps * direction)) / (2 * eps)
    ad = float(np.sum(np.asarray(grad, np.float64) * direction))
    assert ad == pytest.approx(fd, rel=1e-2), f"{name}: AD {ad:.6g} vs FD {fd:.6g}"


@pytest.mark.skipif(not image_available(_OPENFOAM), reason=f"{_OPENFOAM} not built")
def test_openfoam_rejects_forcing():
    from tesseract_core import Tesseract

    v0 = np.zeros((8, 8, 1, 2), np.float32)
    with (
        Tesseract.from_image(_OPENFOAM) as t,
        pytest.raises(Exception, match="icoFoam"),
    ):
        t.apply(_inputs(v0, nu=0.01, dt=0.01, steps=1, extra={}, forcing=v0))
