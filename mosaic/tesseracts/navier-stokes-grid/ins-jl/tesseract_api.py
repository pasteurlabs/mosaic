# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import juliacall
import numpy as np
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.problems.navier_stokes_grid.channel import check_periodic
from mosaic_shared.schema_types import make_differentiable
from pydantic import model_validator


class InputSchema(
    make_differentiable(_CanonicalInputSchema, ["v0", "viscosity", "dt"])
):
    """Input schema for the incompressible Navier-Stokes Julia solver."""

    @model_validator(mode="after")
    def _check_periodic(self) -> "InputSchema":
        check_periodic(self, "INS.jl")
        return self


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result"])):
    """Output schema for the incompressible Navier-Stokes Julia solver."""


# ---------------------------------------------------------------------------
# Julia module initialisation
# ---------------------------------------------------------------------------

jl = juliacall.newmodule("ins_ns")
jl.seval('using Pkg; Pkg.activate(ENV["JULIA_PROJECT"])')
jl.seval("using IncompressibleNavierStokes, Zygote")
jl.include(str(Path(__file__).parent / "solver.jl"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_julia(arr: np.ndarray):
    """Convert a numpy array to a Julia array (zero-copy if contiguous)."""
    return juliacall.convert(jl.Array, np.ascontiguousarray(arr))


def _to_numpy(jl_arr: Any) -> np.ndarray:
    """Convert a Julia array back to numpy."""
    return np.asarray(jl_arr).copy()


def _body_force(inputs: InputSchema) -> dict[str, Any]:
    """Keyword arguments for the body force and drag, empty when there are none.

    Both are 2-D only (the canonical schema rejects them in 3-D).
    """
    kwargs: dict[str, Any] = {}
    if inputs.forcing is not None:
        kwargs["force"] = _to_julia(np.asarray(inputs.forcing)[:, :, 0, :])
    if inputs.linear_drag is not None:
        kwargs["drag"] = float(inputs.linear_drag[0])
    return kwargs


def _run(
    v0: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    **body_force: Any,
) -> np.ndarray:
    """Forward pass.

    2-D: v0 (nx, ny, 1, 2) → result (nx, ny, 1, 2)
    3-D: v0 (nx, ny, nz, 3) → result (nx, ny, nz, 3)
    """
    nx = v0.shape[0]
    ndim = v0.shape[-1]
    nz = v0.shape[2]

    if ndim == 2 and nz == 1:
        # 2-D path: squeeze the z=1 dimension before passing to Julia
        v0_in = v0[:, :, 0, :]  # (nx, ny, 2)
        result = _to_numpy(
            jl.ns_apply(
                _to_julia(v0_in),
                float(viscosity),
                float(dt),
                int(steps),
                int(nx),
                float(domain_extent),
                **body_force,
            )
        )  # (nx, ny, 2)
        return result[:, :, np.newaxis, :]  # (nx, ny, 1, 2)
    else:
        # 3-D path: pass the full (nx, ny, nz, 3) array
        return _to_numpy(
            jl.ns_apply(
                _to_julia(v0),
                float(viscosity),
                float(dt),
                int(steps),
                int(nx),
                float(domain_extent),
            )
        )  # (nx, ny, nz, 3)


def _vjp(
    v0: np.ndarray,
    cotangent: np.ndarray,
    viscosity: float,
    dt: float,
    steps: int,
    domain_extent: float,
    **body_force: Any,
) -> dict:
    """VJP: cotangent w.r.t. result -> gradients w.r.t. v0, viscosity, dt, domain_extent."""
    nx = v0.shape[0]
    ndim = v0.shape[-1]
    nz = v0.shape[2]

    if ndim == 2 and nz == 1:
        v0_in = v0[:, :, 0, :]
        cot_in = cotangent[:, :, 0, :]
        result = jl.ns_vjp(
            _to_julia(v0_in),
            _to_julia(cot_in),
            float(viscosity),
            float(dt),
            int(steps),
            int(nx),
            float(domain_extent),
            **body_force,
        )
        grad_v0_2d = _to_numpy(result[0])
        grad_nu = float(result[1])
        grad_dt_val = float(result[2])
        grad_L = float(result[3])
        return {
            "v0": grad_v0_2d[:, :, np.newaxis, :].astype(np.float32),
            "viscosity": np.array([grad_nu], dtype=np.float32),
            "dt": np.array([grad_dt_val], dtype=np.float32),
            "domain_extent": np.float32(grad_L),
        }
    else:
        result = jl.ns_vjp(
            _to_julia(v0),
            _to_julia(cotangent),
            float(viscosity),
            float(dt),
            int(steps),
            int(nx),
            float(domain_extent),
        )
        grad_v0 = _to_numpy(result[0])
        grad_nu = float(result[1])
        grad_dt_val = float(result[2])
        grad_L = float(result[3])
        return {
            "v0": grad_v0.astype(np.float32),
            "viscosity": np.array([grad_nu], dtype=np.float32),
            "dt": np.array([grad_dt_val], dtype=np.float32),
            "domain_extent": np.float32(grad_L),
        }


# ---------------------------------------------------------------------------
# Tesseract API endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Run the forward incompressible Navier-Stokes simulation."""
    result = _run(
        np.asarray(inputs.v0),
        float(inputs.viscosity[0]),
        float(inputs.dt[0]),
        inputs.steps,
        inputs.domain_extent,
        **_body_force(inputs),
    )
    return {"result": result.astype(np.float32)}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Compute vector-Jacobian products for the Navier-Stokes solver."""
    ct_result = cotangent_vector.get("result")
    if ct_result is None:
        return {}

    grads = _vjp(
        np.asarray(inputs.v0),
        np.asarray(ct_result, dtype=np.float32),
        float(inputs.viscosity[0]),
        float(inputs.dt[0]),
        inputs.steps,
        inputs.domain_extent,
        **_body_force(inputs),
    )
    return {key: grads[key] for key in vjp_inputs if key in grads}


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, Any]:
    """Return output shapes and dtypes without running the solver."""
    raw = abstract_inputs.model_dump()

    v0 = raw["v0"]
    if isinstance(v0, dict) and "shape" in v0 and "dtype" in v0:
        return {"result": {"shape": v0["shape"], "dtype": v0["dtype"]}}
    arr = np.asarray(v0)
    return {"result": {"shape": list(arr.shape), "dtype": str(arr.dtype)}}
