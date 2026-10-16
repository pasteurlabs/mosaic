# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""INS.jl: ``ns_apply`` must match the rollout that ``ns_vjp`` differentiates.

``ns_apply`` uses the library's in-place time stepper and ``ns_vjp`` a
separate non-mutating rollout. Any difference between the two biases
finite-difference gradient checks, which compare ``apply`` against the VJP.
The input is deliberately not divergence-free: the initial projection then
changes the field, which is where the two paths used to diverge.

Runs ``ns_solver.jl`` from this checkout inside an existing INS.jl image
(building one takes too long for a unit test); skipped when there is none.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .helpers import image_available

_IMAGE = "ins_navier_stokes_grid:latest"
_SRC_DIR = (
    Path(__file__).parents[1]
    / "mosaic"
    / "mosaic_shared"
    / "problems"
    / "navier_stokes_grid"
)

_SCRIPT = """
using IncompressibleNavierStokes, LinearAlgebra, Random
include("/src/ns_solver.jl")
n, L, nu, dt, steps = 8, 2pi, 0.01, 0.05, 3
for (ndim, fwd) in ((2, ns_forward_2d), (3, ns_forward_3d))
    v0 = randn(Xoshiro(0), Float32, ntuple(_ -> n, ndim)..., ndim)
    setup, psolver = get_setup_and_psolver(n, L, ndim)
    rhs = create_right_hand_side(setup, psolver)
    a = ns_apply(v0, nu, dt, steps, n, L)
    b = fwd(v0, rhs, setup, psolver, Float32(nu), Float32(dt), steps, n)
    println("REL_DIFF ", ndim, " ", norm(a - b) / norm(b))
end
"""


@pytest.mark.skipif(not image_available(_IMAGE), reason=f"{_IMAGE} not available")
def test_apply_matches_vjp_rollout():
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "julia",
            "-v",
            f"{_SRC_DIR}:/src:ro",
            _IMAGE,
            "--project=/app/julia_env",
            "-e",
            _SCRIPT,
        ],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    rel = {
        int(ndim): float(val)
        for _, ndim, val in (
            line.split()
            for line in result.stdout.splitlines()
            if line.startswith("REL_DIFF")
        )
    }
    assert set(rel) == {2, 3}, result.stdout
    # Float32 round-off; the stale-ghost mismatch this guards against is ~1e-3.
    assert all(v < 1e-5 for v in rel.values()), rel
