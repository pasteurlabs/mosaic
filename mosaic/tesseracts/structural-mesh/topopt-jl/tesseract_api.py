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
from pydantic import Field  # still needed for InputSchema fields
from tesseract_core.runtime import ShapeDType

_jl = None
_julia_init_lock = filelock.FileLock("/tmp/julia_init.lock", timeout=600)

# Initialise Julia synchronously at import time (serialised across workers).
with _julia_init_lock:
    import juliacall  # noqa: PLC0415

    _jl_mod = juliacall.newmodule("topopt_jl")
    _jl_mod.seval('using Pkg; Pkg.activate(ENV["JULIA_PROJECT"])')
    _jl_mod.seval("using TopOpt, Zygote, Printf")
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


class InputSchema(_CanonicalInputSchema):
    """Inputs for TopOpt.jl SIMP solver, extended with material parameters."""

    E: float = Field(default=1.0, description="Young's modulus of the solid material.")
    nu: float = Field(default=0.3, description="Poisson's ratio.")
    xmin: float = Field(
        default=0.001,
        description="Minimum density (void stiffness) to prevent singular stiffness matrix.",
    )


class OutputSchema(_CanonicalOutputSchema):
    """Outputs for TopOpt.jl SIMP solver (canonical interface)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# HEX8 node parametric coordinates in Abaqus ordering
# (xi=±1 along x, eta=±1 along y, zeta=±1 along z)
_HEX8_XI = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=np.float64)
_HEX8_ETA = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=np.float64)
_HEX8_ZETA = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float64)


def _von_mises_hex8(  # mosaic:physics
    pts: np.ndarray,
    cells: np.ndarray,
    u_dofs: np.ndarray,
    rho: np.ndarray,
    E: float,
    nu: float,
    xmin: float,
    penal: float = 3.0,
) -> np.ndarray:  # mosaic:physics
    """Compute per-cell von Mises stress for a structured HEX8 mesh.

    Uses shape function derivatives at element centroids (xi=eta=zeta=0):
        dN_i/dx = xi_i / (4 * dx_e),  etc.

    Args:
        pts:    (n_nodes, 3)  float64 node coordinates
        cells:  (n_cells, 8)  int64   0-based node indices (Abaqus ordering)
        u_dofs: (n_nodes*3,)  float64 DOF vector, interleaved [ux1,uy1,uz1, ux2,…]
        rho:    (n_cells,)    float64 density field in [0,1]
        E, nu, xmin: SIMP material parameters
        penal:  SIMP penalization exponent

    Returns:
        (n_cells,) float32 von Mises stress
    """
    n_nodes = len(pts)
    n_cells = len(cells)

    # Reshape DOF vector to (n_nodes, 3)
    u = u_dofs.reshape(n_nodes, 3)

    # Element node coordinates: (n_cells, 8, 3)
    cell_pts = pts[cells]

    # Element sizes from bounding box (rectangular hex assumed)
    dx = cell_pts[:, :, 0].max(axis=1) - cell_pts[:, :, 0].min(axis=1)  # (n_cells,)
    dy = cell_pts[:, :, 1].max(axis=1) - cell_pts[:, :, 1].min(axis=1)
    dz = cell_pts[:, :, 2].max(axis=1) - cell_pts[:, :, 2].min(axis=1)

    # Shape function derivatives at centroid:  dN_i/dx = xi_i / (4*dx_e)
    dNdx = _HEX8_XI[None, :] / (4.0 * dx[:, None])  # (n_cells, 8)
    dNdy = _HEX8_ETA[None, :] / (4.0 * dy[:, None])
    dNdz = _HEX8_ZETA[None, :] / (4.0 * dz[:, None])

    # Build strain-displacement matrix B: (n_cells, 6, 24)
    # DOF layout per element: [u1x, u1y, u1z,  u2x, u2y, u2z,  …  u8x, u8y, u8z]
    B = np.zeros((n_cells, 6, 24), dtype=np.float64)
    for i in range(8):
        col = 3 * i
        B[:, 0, col + 0] = dNdx[:, i]  # eps_xx
        B[:, 1, col + 1] = dNdy[:, i]  # eps_yy
        B[:, 2, col + 2] = dNdz[:, i]  # eps_zz
        B[:, 3, col + 0] = dNdy[:, i]  # gamma_xy (row 0 part)
        B[:, 3, col + 1] = dNdx[:, i]  # gamma_xy (row 1 part)
        B[:, 4, col + 1] = dNdz[:, i]  # gamma_yz (row 1 part)
        B[:, 4, col + 2] = dNdy[:, i]  # gamma_yz (row 2 part)
        B[:, 5, col + 0] = dNdz[:, i]  # gamma_xz (row 0 part)
        B[:, 5, col + 2] = dNdx[:, i]  # gamma_xz (row 2 part)

    # Element displacement vector: (n_cells, 24)
    u_e = u[cells].reshape(n_cells, 24)

    # Strain at centroid: (n_cells, 6)
    eps = np.einsum("cij,cj->ci", B, u_e)

    # SIMP effective Young's modulus per cell
    E_eff = xmin * E + (E - xmin * E) * rho**penal  # (n_cells,)

    # Isotropic constitutive matrix D (normalised; scale by E_eff below)
    c1 = (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c3 = 0.5 / (1.0 + nu)
    D_base = np.array(
        [
            [c1, c2, c2, 0, 0, 0],
            [c2, c1, c2, 0, 0, 0],
            [c2, c2, c1, 0, 0, 0],
            [0, 0, 0, c3, 0, 0],
            [0, 0, 0, 0, c3, 0],
            [0, 0, 0, 0, 0, c3],
        ],
        dtype=np.float64,
    )

    # Stress: (n_cells, 6)
    sigma = E_eff[:, None] * np.einsum("ij,cj->ci", D_base, eps)

    sxx, syy, szz = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    sxy, syz, sxz = sigma[:, 3], sigma[:, 4], sigma[:, 5]

    von_mises = np.sqrt(
        0.5
        * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (sxy**2 + syz**2 + sxz**2)
        )
    )
    return von_mises.astype(np.float32)


def _to_julia(arr: np.ndarray):  # mosaic:io
    """Convert a numpy array to a Julia array (zero-copy when contiguous)."""
    import juliacall  # noqa: PLC0415 — deferred until worker is initialised

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


def _von_mises_direct_grad(  # mosaic:grad:rho:adjoint
    pts: np.ndarray,
    cells: np.ndarray,
    u_dofs: np.ndarray,
    rho: np.ndarray,
    E: float,
    nu: float,
    xmin: float,
    w: np.ndarray,
    penal: float = 3.0,
) -> np.ndarray:
    """Direct ∂(w·σ_vm)/∂ρ_e from the E_eff(ρ) dependence (no adjoint solve).

    Since σ_vm_e = E_eff_e * σ̃_vm_e  (σ̃ uses D at E=1):
        ∂(w_e * σ_vm_e)/∂ρ_e = w_e * (σ_vm_e / E_eff_e) * dE_eff_e/dρ_e
    """
    vm = _von_mises_hex8(pts, cells, u_dofs, rho, E, nu, xmin, penal)
    E_eff = xmin * E + (E - xmin * E) * rho**penal
    dE_drho = penal * (E - xmin * E) * rho ** (penal - 1)
    E_eff = np.where(E_eff > 0, E_eff, 1e-30)
    return (w * vm / E_eff * dE_drho).astype(np.float64)


def _von_mises_adjoint_rhs(  # mosaic:grad:rho:adjoint
    pts: np.ndarray,
    cells: np.ndarray,
    u_dofs: np.ndarray,
    rho: np.ndarray,
    E: float,
    nu: float,
    xmin: float,
    w: np.ndarray,
    penal: float = 3.0,
) -> np.ndarray:
    """Assemble (∂(w·σ_vm)/∂u) — the adjoint RHS for the u-path of the VJP.

    Uses the same centroid B-matrix approximation as _von_mises_hex8.
    Returns a (n_nodes * 3,) float64 vector suitable for topopt_general_vjp.
    """
    n_nodes = len(pts)
    n_cells = len(cells)

    u = u_dofs.reshape(n_nodes, 3)
    cell_pts = pts[cells]
    dx = cell_pts[:, :, 0].max(axis=1) - cell_pts[:, :, 0].min(axis=1)
    dy = cell_pts[:, :, 1].max(axis=1) - cell_pts[:, :, 1].min(axis=1)
    dz = cell_pts[:, :, 2].max(axis=1) - cell_pts[:, :, 2].min(axis=1)

    dNdx = _HEX8_XI[None, :] / (4.0 * dx[:, None])
    dNdy = _HEX8_ETA[None, :] / (4.0 * dy[:, None])
    dNdz = _HEX8_ZETA[None, :] / (4.0 * dz[:, None])

    B = np.zeros((n_cells, 6, 24))
    for i in range(8):
        col = 3 * i
        B[:, 0, col + 0] = dNdx[:, i]
        B[:, 1, col + 1] = dNdy[:, i]
        B[:, 2, col + 2] = dNdz[:, i]
        B[:, 3, col + 0] = dNdy[:, i]
        B[:, 3, col + 1] = dNdx[:, i]
        B[:, 4, col + 1] = dNdz[:, i]
        B[:, 4, col + 2] = dNdy[:, i]
        B[:, 5, col + 0] = dNdz[:, i]
        B[:, 5, col + 2] = dNdx[:, i]

    u_e = u[cells].reshape(n_cells, 24)
    eps = np.einsum("cij,cj->ci", B, u_e)

    E_eff = xmin * E + (E - xmin * E) * rho**penal
    c1 = (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c3 = 0.5 / (1.0 + nu)
    D_base = np.array(
        [
            [c1, c2, c2, 0, 0, 0],
            [c2, c1, c2, 0, 0, 0],
            [c2, c2, c1, 0, 0, 0],
            [0, 0, 0, c3, 0, 0],
            [0, 0, 0, 0, c3, 0],
            [0, 0, 0, 0, 0, c3],
        ]
    )

    sigma = E_eff[:, None] * np.einsum("ij,cj->ci", D_base, eps)
    sxx, syy, szz = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    sxy, syz, sxz = sigma[:, 3], sigma[:, 4], sigma[:, 5]
    vm = np.sqrt(
        0.5
        * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (sxy**2 + syz**2 + sxz**2)
        )
    )

    # ∂vm_e/∂σ_e  (Voigt 6-vector, normalised by 2·vm)
    dvm_dsigma = np.stack(
        [
            2 * sxx - syy - szz,
            -sxx + 2 * syy - szz,
            -sxx - syy + 2 * szz,
            6.0 * sxy,
            6.0 * syz,
            6.0 * sxz,
        ],
        axis=1,
    )
    vm_safe = np.where(vm > 1e-10, vm, 1e-10)
    dvm_dsigma /= 2.0 * vm_safe[:, None]

    # ∂vm_e/∂u_e = E_eff_e · Bᵀ D (∂vm/∂σ)_e   shape (n_cells, 24)
    D_dvm = np.einsum("ij,cj->ci", D_base, dvm_dsigma)
    dvm_du_e = np.einsum("cjk,cj->ck", B, D_dvm) * E_eff[:, None]

    # Scatter cotangent-scaled contributions into global DOF vector
    adj_e = (w[:, None] * dvm_du_e).reshape(n_cells, 8, 3)
    adj_rhs = np.zeros((n_nodes, 3), dtype=np.float64)
    np.add.at(adj_rhs, cells, adj_e)
    return adj_rhs.flatten()


# ---------------------------------------------------------------------------
# Tesseract API endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve FEA and return compliance, von Mises stress, and displacement."""
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
    u_dofs = _to_numpy(result[2])  # (n_nodes * 3,) interleaved DOFs

    n_nodes = len(pts)
    displacement = u_dofs.reshape(n_nodes, 3).astype(np.float32)

    von_mises = _von_mises_hex8(
        pts,
        cells,
        u_dofs,
        rho,
        E=float(inputs.E),
        nu=float(inputs.nu),
        xmin=float(inputs.xmin),
    )

    return OutputSchema(
        compliance=np.float32(c),
        von_mises_stress=von_mises,
        displacement=displacement,
    )


def vector_jacobian_product(  # mosaic:grad:rho:adjoint
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP for compliance, von_mises_stress, and displacement w.r.t. rho.

    - compliance: analytical SIMP adjoint (comp.grad), no extra linear solve.
    - displacement: adjoint solve K λ = cotan_disp, sensitivity formula.
    - von_mises_stress: direct ρ-path (through E_eff) + adjoint solve for u-path.

    All three outputs can be requested simultaneously; the adjoint system is
    solved once for the combined RHS.
    """
    assert vjp_inputs <= {"rho"}
    assert vjp_outputs <= {"compliance", "von_mises_stress", "displacement"}

    jl = _get_jl()
    pts, cells, rho, dm, vm_mask, vv = _unpack(inputs)
    E, nu, xmin = float(inputs.E), float(inputs.nu), float(inputs.xmin)

    # Forward pass — needed for compliance gradient and u (for other outputs).
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
    u_dofs = _to_numpy(result[2])  # (n_dofs,) interleaved nodal DOFs

    grad_rho_active = np.zeros(len(rho), dtype=np.float64)

    if "compliance" in vjp_outputs:
        cot_c = float(cotangent_vector.get("compliance", 0.0))
        grad_rho_active += cot_c * c_grad

    # Build combined adjoint RHS for displacement and von_mises_stress.
    need_adjoint = "displacement" in vjp_outputs or "von_mises_stress" in vjp_outputs
    if need_adjoint:
        adj_rhs = np.zeros(len(pts) * 3, dtype=np.float64)

        if "displacement" in vjp_outputs:
            cot_disp = np.asarray(
                cotangent_vector.get("displacement", 0.0), dtype=np.float64
            )
            adj_rhs += cot_disp.flatten()

        if "von_mises_stress" in vjp_outputs:
            cot_vm = np.asarray(
                cotangent_vector.get("von_mises_stress", 0.0), dtype=np.float64
            )
            # Direct ρ-path: ∂(w·σ_vm)/∂ρ through E_eff
            grad_rho_active += _von_mises_direct_grad(
                pts, cells, u_dofs, rho, E, nu, xmin, cot_vm
            )
            # u-path: accumulate adjoint RHS from (∂σ_vm/∂u)ᵀ w
            adj_rhs += _von_mises_adjoint_rhs(
                pts, cells, u_dofs, rho, E, nu, xmin, cot_vm
            )

        # Adjoint solve + element sensitivities (Julia).
        # topopt_general_vjp zeroes fixed DOFs in adj_rhs, solves K λ = adj_rhs,
        # and returns ∂f/∂ρ_e = −(dρ̃_e/dρ_e) λ_eᵀ Kₑ uₑ per element.
        grad_rho_active += _to_numpy(
            _get_jl().topopt_general_vjp(
                _to_julia(adj_rhs),
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
        )

    # Pad back to the full capacity-padded rho shape.
    hm = inputs.hex_mesh
    grad_rho = np.zeros(len(np.asarray(inputs.rho)), dtype=np.float32)
    grad_rho[: hm.n_faces] = grad_rho_active.astype(np.float32)
    return {"rho": grad_rho}


def abstract_eval(abstract_inputs: InputSchema) -> dict[str, Any]:
    """Shape inference without running the solver."""
    d = abstract_inputs.model_dump()
    faces = d["hex_mesh"]["faces"]
    points = d["hex_mesh"]["points"]
    n_cells = faces["shape"][0] if isinstance(faces, dict) else len(faces)
    n_nodes = points["shape"][0] if isinstance(points, dict) else len(points)
    return {
        "compliance": ShapeDType(shape=(), dtype="float32"),
        "von_mises_stress": ShapeDType(shape=(n_cells,), dtype="float32"),
        "displacement": ShapeDType(shape=(n_nodes, 3), dtype="float32"),
    }
