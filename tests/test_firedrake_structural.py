# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural compliance VJP in a native Firedrake environment."""

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("prescribed", [False, True], ids=["clamp", "moving-support"])
def test_compliance_vjp(prescribed: bool) -> None:
    """Check the solve count and directional derivative for both boundary conditions."""
    pytest.importorskip("firedrake")
    from firedrake.petsc import PETSc
    from mosaic_shared.problems.structural_mesh.schemas import make_default_inputs
    from tesseract_core import Tesseract

    inputs = make_default_inputs(nx=4, ny=2, nz=2)
    if prescribed:
        # A second, moving support produces strain even without the traction.
        points = inputs["hex_mesh"]["points"]
        bc = inputs["boundary_conditions"]["dirichlet"]
        support = (points[:, 2] == 1) & (points[:, 0] >= 1)
        bc["mask"] = np.where(support, 2, bc["mask"])
        bc["values"] = np.float32([[0, 0, 0], [0.002, -0.001, 0.003]])

    rho = np.linspace(0.35, 0.85, inputs["rho"].size, dtype=np.float32)
    inputs["rho"] = rho
    directory = (
        Path(__file__).resolve().parents[1] / "mosaic/tesseracts/structural-mesh"
    )
    PETSc.Log.begin()
    solves = PETSc.Log.Event("KSPSolve")
    expected_solves = 2 if prescribed else 1

    with Tesseract.from_tesseract_api(
        directory / "firedrake-structural/tesseract_api.py"
    ) as solver:

        def gradient():
            before = solves.getPerfInfo()["count"]
            grad = solver.vector_jacobian_product(
                inputs, ["rho"], ["compliance"], {"compliance": np.float32(1)}
            )["rho"]
            assert solves.getPerfInfo()["count"] - before == expected_solves
            return np.asarray(grad)

        # A standalone VJP and a VJP following apply both solve at their own input.
        grad = gradient()
        solver.apply(inputs)
        np.testing.assert_array_equal(gradient(), grad)

        direction = np.linspace(1.0, 0.2, rho.size, dtype=np.float32)
        h = 0.002
        plus = solver.apply({**inputs, "rho": rho + h * direction})["compliance"]
        minus = solver.apply({**inputs, "rho": rho - h * direction})["compliance"]
        fd = (float(plus) - float(minus)) / (2 * h)
        assert np.dot(grad, direction) == pytest.approx(fd, rel=1e-3)
