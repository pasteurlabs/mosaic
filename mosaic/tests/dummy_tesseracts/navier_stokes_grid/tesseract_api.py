# ruff: noqa: E402 — sys.path bootstrap must precede the workspace imports below
"""Dummy Navier-Stokes tesseract — constant outputs for end-to-end framework tests.

Imports the canonical :class:`InputSchema` / :class:`OutputSchema` from
:mod:`tesseract_shared.problems.navier_stokes_grid` and wraps them with
``make_differentiable`` for the same fields the real solvers expose
(``v0`` on input; ``result``, ``drag`` on output). ``apply`` returns a
zero ``result`` array of the same shape as the incoming ``v0`` and a
fixed scalar ``drag``; ``vector_jacobian_product`` returns zeros. Output
is therefore *independent of the input*, so any gradient through the
chain comes out zero — that's the right answer for a "constant field"
dummy.

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

# ``tesseract_shared`` lives under ``mosaic/tesseracts/`` as a uv-workspace
# package. ``Tesseract.from_tesseract_api`` loads this file via
# ``load_module_from_path``, which bypasses the import machinery that uv
# wires up — so we have to put the workspace dir on ``sys.path`` ourselves
# to make the canonical-schema imports below resolve in any environment.
_TESSERACTS_DIR = Path(__file__).resolve().parents[3] / "tesseracts"
if str(_TESSERACTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESSERACTS_DIR))

import numpy as np
from pydantic import ConfigDict
from tesseract_core.runtime import ShapeDType
from tesseract_shared.problems.navier_stokes_grid import (
    InputSchema as _CanonicalInputSchema,
)
from tesseract_shared.problems.navier_stokes_grid import (
    OutputSchema as _CanonicalOutputSchema,
)
from tesseract_shared.types import make_differentiable


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


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["result", "drag"])):
    pass


# Constant output value — small enough that downstream FD tests with
# eps_values ~ 1.0 don't hit any spurious overflow even when v0 has unit
# magnitude.
_RESULT_VALUE = 0.0
_DRAG_VALUE = np.asarray([0.0], dtype=np.float32)


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: zero velocity field shaped like v0, fixed scalar drag."""
    v0 = np.asarray(inputs.v0)
    result = np.full(v0.shape, _RESULT_VALUE, dtype=np.float32)
    return OutputSchema(result=result, drag=_DRAG_VALUE.copy())


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
    """Shape inference: ``result`` matches ``v0``; ``drag`` is always (1,)."""
    raw = abstract_inputs.model_dump()
    v0 = raw["v0"]
    if isinstance(v0, dict) and "shape" in v0:
        v0_shape = tuple(v0["shape"])
        v0_dtype = v0.get("dtype", "float32")
    else:
        arr = np.asarray(v0)
        v0_shape = arr.shape
        v0_dtype = str(arr.dtype)
    return {
        "result": ShapeDType(shape=v0_shape, dtype=v0_dtype),
        "drag": ShapeDType(shape=(1,), dtype="float32"),
    }
