import glob
import os
import sys

os.environ.setdefault("JULIA_PROJECT", "/app/julia_env")
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", "/app/julia_env")

# PythonCall (Julia side) needs to find the Python shared library.
# When Julia precompiles extensions in a subprocess it cannot auto-detect it,
# so we set JULIA_PYTHONCALL_LIB explicitly before importing juliacall.
if "JULIA_PYTHONCALL_LIB" not in os.environ:
    import sysconfig

    _libdir = sysconfig.get_config_var("LIBDIR") or os.path.join(sys.exec_prefix, "lib")
    _ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    for _pattern in [
        f"{_libdir}/libpython{_ver}.so.1.0",
        f"{_libdir}/libpython{_ver}.so",
        f"{_libdir}/libpython3.so",
    ]:
        _matches = glob.glob(_pattern)
        if _matches:
            os.environ["JULIA_PYTHONCALL_LIB"] = _matches[0]
            break

# ---------------------------------------------------------------------------
# Julia module initialisation (synchronous, spawn-safe)
# ---------------------------------------------------------------------------
#
# uvicorn's multi-worker mode uses Python's "spawn" start method (not "fork"),
# so each worker is a fresh process that re-imports tesseract_api.py cleanly.
# The parent process never imports this module, so fork-safety is not a concern.
#
# The original background-thread approach caused a fatal liveness issue:
#   - import juliacall holds the Python GIL for several seconds (Julia runtime
#     init is a C-extension operation).
#   - uvicorn's Multiprocess supervisor pings each worker via a Pipe every 0.5 s
#     with a 5-second timeout; if the worker's pong thread cannot respond (because
#     the GIL is held by the Julia init thread), the supervisor considers the
#     worker "hung" and SIGKILL's it.
#   - This causes the continuous worker-restart loop seen in container logs.
#
# Fix: perform Julia initialisation SYNCHRONOUSLY during module import, inside
# a cross-process filelock so at most one worker initialises at a time.  The
# import of this module completes only after Julia is ready, which happens
# before uvicorn registers the worker as started and before the pong thread
# needs to respond to any liveness checks.
from pathlib import Path
from typing import Any

import filelock
import mosaic_shared
import numpy as np
from mosaic_shared.problems.structural_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.structural_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.types import make_differentiable
from pydantic import Field  # still needed for InputSchema fields
from tesseract_core.runtime import ShapeDType

_jl = None
_julia_init_lock = filelock.FileLock("/tmp/julia_init.lock", timeout=600)

# Initialise Julia synchronously at import time (serialised across workers).
with _julia_init_lock:
    import juliacall

    _jl_mod = juliacall.newmodule("topopt_jl")
    _jl_mod.seval('using Pkg; Pkg.activate(ENV["JULIA_PROJECT"])')
    _jl_mod.seval("using TopOpt, Printf")
    _jl_mod.include(
        str(
            Path(mosaic_shared.__file__).parent
            / "problems"
            / "structural_mesh"
            / "topopt_solver.jl"
        )
    )
    _jl = _jl_mod


def _get_jl():  # mosaic:util
    """Return the initialised Julia module."""
    return _jl


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho"])):
    """Inputs for TopOpt.jl SIMP solver, extended with material parameters."""

    E: float = Field(default=1.0, description="Young's modulus of the solid material.")
    nu: float = Field(default=0.3, description="Poisson's ratio.")
    xmin: float = Field(
        default=0.001,
        description="Minimum density (void stiffness) to prevent singular stiffness matrix.",
    )


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["compliance"])):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_julia(arr: np.ndarray):  # mosaic:io
    """Convert a numpy array to a Julia array (zero-copy when contiguous)."""
    import juliacall

    jl = _get_jl()
    return juliacall.convert(jl.Array, np.ascontiguousarray(arr))


def _to_numpy(jl_arr) -> np.ndarray:  # mosaic:io
    """Convert a Julia array to a numpy array."""
    return np.asarray(jl_arr).copy()


def _unpack(inputs: InputSchema):  # mosaic:io
    """Extract active mesh/BC arrays from the padded InputSchema."""
    hm = inputs.hex_mesh
    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float64)
    cells = np.asarray(hm.faces[: hm.n_faces], dtype=np.int64)
    rho = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float64)
    bc = inputs.boundary_conditions
    dm = np.asarray(bc.dirichlet.mask if bc.dirichlet else [], dtype=np.int32)
    if bc.neumann:
        vm = np.asarray(bc.neumann.mask, dtype=np.int32)
        vv = np.asarray(bc.neumann.values, dtype=np.float64)
    else:
        vm = np.zeros(len(pts), dtype=np.int32)
        vv = np.zeros((0, 3), dtype=np.float64)
    return pts, cells, rho, dm, vm, vv


# ---------------------------------------------------------------------------
# Tesseract API endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve FEA and return structural compliance."""
    jl = _get_jl()
    pts, cells, rho, dm, vm, vv = _unpack(inputs)

    result = jl.topopt_forward(
        _to_julia(rho),
        _to_julia(pts),
        _to_julia(cells),
        _to_julia(dm),
        _to_julia(vm),
        _to_julia(vv),
        float(inputs.E),
        float(inputs.nu),
        float(inputs.xmin),
    )
    c = float(result[0])
    return OutputSchema(compliance=np.float32(c))


def vector_jacobian_product(  # mosaic:grad:rho:adjoint
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP for compliance w.r.t. rho via analytical SIMP adjoint."""
    assert vjp_inputs <= {"rho"}
    assert vjp_outputs <= {"compliance"}

    if "rho" not in vjp_inputs:
        return {}

    cot_c = float(cotangent_vector.get("compliance", 0.0))
    hm = inputs.hex_mesh
    grad_rho = np.zeros(len(np.asarray(inputs.rho)), dtype=np.float32)
    if abs(cot_c) == 0.0:
        return {"rho": grad_rho}

    jl = _get_jl()
    pts, cells, rho, dm, vm_mask, vv = _unpack(inputs)
    E, nu, xmin = float(inputs.E), float(inputs.nu), float(inputs.xmin)

    result = jl.topopt_forward(
        _to_julia(rho),
        _to_julia(pts),
        _to_julia(cells),
        _to_julia(dm),
        _to_julia(vm_mask),
        _to_julia(vv),
        E,
        nu,
        xmin,
    )
    c_grad = _to_numpy(result[1])  # ∂C/∂ρ_e  (n_active_cells,)
    grad_rho[: hm.n_faces] = (cot_c * c_grad).astype(np.float32)
    return {"rho": grad_rho}


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, Any]:
    """Shape inference without running the solver."""
    return {"compliance": ShapeDType(shape=(), dtype="float32")}
