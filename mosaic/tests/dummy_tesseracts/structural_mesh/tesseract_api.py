# ruff: noqa: E402 — sys.path bootstrap must precede the workspace imports below
"""Dummy structural-mesh tesseract — constant scalar compliance for end-to-end tests.

Imports the canonical :class:`InputSchema` / :class:`OutputSchema` from
:mod:`tesseract_shared.problems.structural_mesh` and wraps them with
``make_differentiable`` over the same fields the real SIMP solvers
expose (``rho`` on input; ``compliance`` on output). ``apply`` returns a
zero scalar ``compliance``; ``vector_jacobian_product`` returns zeros.

Output is *independent of the input*, so any gradient through the chain
comes out zero — the right answer for a "constant scalar" dummy.

Like the ns-grid stub, this exists to exercise the framework end-to-end
(kernel + sweep loop + per_solver_loop + apply_tesseract VJP plumbing +
result.json writers) without Docker / real FEM solvers in the loop.
Plug it in via ``tags = {solver: "inprocess:<this_file>"}`` and the
runner's ``inprocess:`` prefix routes to
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
from tesseract_core.runtime import ShapeDType
from tesseract_shared.problems.structural_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from tesseract_shared.problems.structural_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from tesseract_shared.types import make_differentiable


# Match the differentiable surface of the real structural-mesh solvers
# (jax-fem, topopt-jl, …): only `rho` in / `compliance` out are diff'd.
# Subclassing keeps the type name ``InputSchema`` so the OpenAPI schema
# tesseract-jax's :class:`Jaxeract` introspects lands at the expected key
# ``Apply_InputSchema`` rather than ``Apply_DifferentiableInputSchema``.
class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho"])):
    pass


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["compliance"])):
    pass


# Scalar zero — matches the canonical OutputSchema ``compliance: Array[(), Float32]``.
_COMPLIANCE_VALUE = np.asarray(0.0, dtype=np.float32)


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: constant scalar zero compliance, independent of inputs."""
    return OutputSchema(compliance=_COMPLIANCE_VALUE.copy())


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
    return out


def abstract_eval(abstract_inputs):
    """Shape inference: ``compliance`` is always a Float32 scalar."""
    return {"compliance": ShapeDType(shape=(), dtype="float32")}
