# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402 — sys.path bootstrap must precede the workspace imports below
"""Dummy Navier-Stokes tesseract — constant outputs for end-to-end framework tests.

Imports the canonical :class:`InputSchema` / :class:`OutputSchema` from
:mod:`mosaic_shared.problems.navier_stokes_grid` and wraps them with
``make_differentiable`` for the same fields the real solvers expose
(``v0`` on input; ``result``, ``body_force`` on output). ``apply`` returns a
zero ``result`` array of the same shape as the incoming ``v0``; with an
obstacle it adds a zero ``pressure`` and a constant ``body_force`` history.
``vector_jacobian_product`` returns zeros. Output is therefore *independent
of the input*, so any gradient through the chain comes out zero — that's the
right answer for a "constant field" dummy.

Like every ns-grid solver, it raises on boundary conditions it does not
implement. It accepts the two configurations the benchmark uses: fully
periodic, and the DFG 2D-1 channel (profile inlet at x_lo, outflow at x_hi,
no-slip walls in y).

The point of this stub is to exercise the framework end-to-end (kernel +
sweep loop + per_solver_loop + apply_tesseract VJP plumbing + result.json
writers) without Docker / real solvers in the loop. Tests can plug it
in via ``tags = {solver: "inprocess:<this_file>"}`` and the runner
recognises the ``inprocess:`` prefix to use
:meth:`tesseract_core.Tesseract.from_tesseract_api`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ``mosaic_shared`` lives under ``mosaic/tesseracts/`` as a uv-workspace
# package. ``Tesseract.from_tesseract_api`` loads this file via
# ``load_module_from_path``, which bypasses the import machinery that uv
# wires up — so we have to put the workspace dir on ``sys.path`` ourselves
# to make the canonical-schema imports below resolve in any environment.
_TESSERACTS_DIR = Path(__file__).resolve().parents[3] / "tesseracts"
if str(_TESSERACTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESSERACTS_DIR))

import numpy as np
from mosaic_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.schema_types import make_differentiable
from pydantic import ConfigDict, model_validator
from tesseract_core.runtime import ShapeDType

# The one non-periodic configuration the dummy accepts: the DFG 2D-1 channel.
_CHANNEL_BCS = {
    "x_lo": "dirichlet",
    "x_hi": "neumann",
    "y_lo": "no_slip",
    "y_hi": "no_slip",
}


# Match the differentiable surface of the real ns-grid solvers (e.g. jax_cfd).
# Subclassing keeps the type name ``InputSchema`` so the OpenAPI schema
# tesseract-jax's :class:`Jaxeract` introspects lands at the expected key
# ``Apply_InputSchema`` rather than ``Apply_DifferentiableInputSchema``.
class InputSchema(
    make_differentiable(
        _CanonicalInputSchema, ["v0", "viscosity", "dt", "inflow_profile"]
    )
):
    # Real solvers accept extra per-solver tuning kwargs via
    # ``input_overrides`` (``density``, ``inner_steps``, ``order``, …).
    # Accept-and-ignore them on the dummy.
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def _check_supported_bcs(self) -> InputSchema:
        bc = self.boundary_conditions
        if bc.is_fully_periodic:
            return self
        faces = {face: getattr(bc, face).type.value for face in _CHANNEL_BCS}
        if faces != _CHANNEL_BCS or self.inflow_profile is None:
            raise ValueError(
                "the dummy solver supports fully periodic BCs or the DFG channel "
                f"{_CHANNEL_BCS} with an inflow_profile; got {faces}"
            )
        return self


class OutputSchema(
    make_differentiable(_CanonicalOutputSchema, ["result", "body_force"])
):
    pass


# Constant output value — small enough that downstream FD tests with
# eps_values ~ 1.0 don't hit any spurious overflow even when v0 has unit
# magnitude.
_RESULT_VALUE = 0.0
# Force on the obstacle: the DFG 2D-1 reference drag F_x = Cd Ū² D / 2 with
# Cd = 5.5795, Ū = 0.2, D = 0.1, so the harness's plausibility checks on the
# initial drag pass.
_FORCE_VALUE = np.asarray([5.5795 * 0.2**2 * 0.1 / 2, 0.0], dtype=np.float32)


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: zero velocity field shaped like v0; with an obstacle, also
    zero pressure and a constant force history.
    """
    v0 = np.asarray(inputs.v0)
    result = np.full(v0.shape, _RESULT_VALUE, dtype=np.float32)
    if inputs.obstacle is None:
        return OutputSchema(result=result)
    return OutputSchema(
        result=result,
        pressure=np.zeros(v0.shape[:3], dtype=np.float32),
        body_force=np.tile(_FORCE_VALUE, (inputs.steps, 1)),
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """VJP: ∂(constant)/∂(anything) = 0 — return zeros of the right shape."""
    out: dict[str, Any] = {}
    if "v0" in vjp_inputs:
        out["v0"] = np.zeros_like(np.asarray(inputs.v0), dtype=np.float32)
    if "viscosity" in vjp_inputs:
        out["viscosity"] = np.zeros((1,), dtype=np.float32)
    if "dt" in vjp_inputs:
        out["dt"] = np.zeros((1,), dtype=np.float32)
    if "inflow_profile" in vjp_inputs and inputs.inflow_profile is not None:
        out["inflow_profile"] = np.zeros_like(
            np.asarray(inputs.inflow_profile), dtype=np.float32
        )
    return out


def abstract_eval(abstract_inputs):
    """Shape inference: ``result`` matches ``v0``; with an obstacle, ``pressure``
    is (nx, ny, nz) and ``body_force`` (steps, 2).
    """
    raw = abstract_inputs.model_dump()
    v0 = raw["v0"]
    if isinstance(v0, dict) and "shape" in v0:
        v0_shape = tuple(v0["shape"])
        v0_dtype = v0.get("dtype", "float32")
    else:
        arr = np.asarray(v0)
        v0_shape = arr.shape
        v0_dtype = str(arr.dtype)
    out = {"result": ShapeDType(shape=v0_shape, dtype=v0_dtype)}
    if raw.get("obstacle") is not None:
        out["pressure"] = ShapeDType(shape=v0_shape[:3], dtype="float32")
        out["body_force"] = ShapeDType(shape=(raw["steps"], 2), dtype="float32")
    return out
