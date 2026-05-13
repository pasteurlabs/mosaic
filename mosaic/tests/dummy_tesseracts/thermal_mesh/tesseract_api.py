"""Dummy thermal-mesh tesseract — constant outputs for end-to-end framework tests.

Imports the canonical :class:`InputSchema` / :class:`OutputSchema` from
:mod:`tesseract_shared.problems.thermal_mesh` and wraps them with
``make_differentiable`` for the same fields the real solvers expose
(``rho`` / ``source`` on input; ``thermal_compliance`` /
``identification_error`` on output). ``apply`` returns zero scalars for
both outputs; ``vector_jacobian_product`` returns zeros. The output is
therefore *independent of the input*, so any gradient through the chain
comes out zero — that's the right answer for a "constant field" dummy.

The point of this stub is to exercise the framework end-to-end (kernel +
sweep loop + per_solver_loop + apply_tesseract VJP plumbing + result.json
writers) without Docker / real solvers in the loop. Tests plug it in via
``tags = {solver: "inprocess:<this_file>"}`` and the runner recognises
the ``inprocess:`` prefix to use
:meth:`tesseract_core.Tesseract.from_tesseract_api`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tesseract_core.runtime import ShapeDType
from tesseract_shared.problems.thermal_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from tesseract_shared.problems.thermal_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from tesseract_shared.types import make_differentiable

# Match the differentiable surface of the real thermal-mesh solvers
# (jax_fem, fenics_heat, dealii_heat, firedrake_heat, torch_fem_thermal).
InputSchema = make_differentiable(_CanonicalInputSchema, ["rho", "source"])
OutputSchema = make_differentiable(
    _CanonicalOutputSchema, ["thermal_compliance", "identification_error"]
)


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: zero scalars for both outputs."""
    return OutputSchema(
        thermal_compliance=np.float32(0.0),
        identification_error=np.float32(0.0),
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """VJP: ∂(constant)/∂(anything) = 0 — return zeros of the right shape."""
    out: dict[str, Any] = {}
    if "rho" in vjp_inputs:
        out["rho"] = np.zeros_like(np.asarray(inputs.rho), dtype=np.float32)
    if "source" in vjp_inputs:
        out["source"] = np.zeros_like(np.asarray(inputs.source), dtype=np.float32)
    return out


def abstract_eval(abstract_inputs):
    """Shape inference: both outputs are scalars."""
    return {
        "thermal_compliance": ShapeDType(shape=(), dtype="float32"),
        "identification_error": ShapeDType(shape=(), dtype="float32"),
    }
