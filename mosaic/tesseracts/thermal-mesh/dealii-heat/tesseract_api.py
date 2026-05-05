
"""Thermal topology optimisation on a structured hexahedral mesh.

Uses deal.II Q1 finite elements as a C++ subprocess.  Python writes JSON + rho.npy
to a tempdir, runs the compiled heat_solver binary, and reads back temperature.npy
and compliance.txt (plus gradient.npy for the VJP).

SIMP conductivity:
    k(ρ) = k_min + (k_max − k_min) · ρ^p    (k_min = 1e-3 · k_max)

Objective:
    C = ∮_Γ_N q_n · T dΓ

Gradient (analytic, self-adjoint):
    dC/dρ_e = -dk/dρ_e · ∫_e |∇T|² dΩ  (negative: more conductivity → lower C)

Source gradient — identification_error (adjoint):
    E = sum_i (T_i - T_target_i)^2
    Adjoint solve:  K · λ = 2(T - T_target)
    dE/dsource_e = vol_e / 8 * sum(λ_i  for nodes i of cell e)

Source gradient — thermal_compliance (adjoint):
    C = T^T f_Neumann  (boundary work by Neumann flux)
    PDE:  K T = f_Neumann + f_source,  f_source_i = source_e * vol_e / 8
    Adjoint solve:  K · λ = f_Neumann = K T − f_source
    dC/dsource_e = vol_e / 8 * sum(λ_i  for nodes i of cell e)
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mosaic_shared.problems.thermal_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.thermal_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from pydantic import Field
from tesseract_core.runtime import ShapeDType

# ---------------------------------------------------------------------------
# Binary path
# ---------------------------------------------------------------------------

_DEALII_SOLVER = os.environ.get(
    "DEALII_HEAT_SOLVER", "/opt/dealii_heat/build/heat_solver"
)


# ---------------------------------------------------------------------------
# Schema (subclass with material parameters)
# ---------------------------------------------------------------------------


class InputSchema(_CanonicalInputSchema):
    """Inputs for deal.II heat solver, extended with SIMP material parameters."""

    k_max: float = Field(
        default=1.0,
        description="Maximum thermal conductivity (fully solid material).",
    )
    p_exp: float = Field(
        default=3.0,
        description="SIMP penalisation exponent p (k(ρ) = k_min + (k_max−k_min)·ρ^p).",
    )


class OutputSchema(_CanonicalOutputSchema):
    """Outputs for deal.II heat solver (canonical interface)."""


# ---------------------------------------------------------------------------
# Mesh helper: infer nx, ny, nz from HexMesh
# ---------------------------------------------------------------------------


def _infer_grid_dims(
    inputs: InputSchema,
) -> tuple[int, int, int, float, float, float]:  # mosaic:io
    """Infer structured grid dimensions from the HexMesh point array.

    The benchmark always builds a structured hex mesh from
    ``np.linspace(0, L, n+1)`` in each direction.  Given n_nodes =
    (nx+1)*(ny+1)*(nz+1) and the unique coordinate counts we can recover
    nx, ny, nz exactly.

    Returns:
        (nx, ny, nz, Lx, Ly, Lz)
    """
    hm = inputs.hex_mesh
    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float32)

    xs = np.unique(np.round(pts[:, 0], 6))
    ys = np.unique(np.round(pts[:, 1], 6))
    zs = np.unique(np.round(pts[:, 2], 6))

    nx = len(xs) - 1
    ny = len(ys) - 1
    nz = len(zs) - 1

    Lx = float(xs[-1] - xs[0])
    Ly = float(ys[-1] - ys[0])
    Lz = float(zs[-1] - zs[0])

    return nx, ny, nz, Lx, Ly, Lz


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_inputs(inputs: InputSchema, wd: Path) -> None:  # mosaic:io
    """Serialise inputs to ``input.json``, ``rho.npy``, and ``source.npy`` in *wd*."""
    nx, ny, nz, Lx, Ly, Lz = _infer_grid_dims(inputs)

    hm = inputs.hex_mesh
    bc = inputs.boundary_conditions

    # Active density slice
    rho_active = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float32)
    np.save(str(wd / "rho.npy"), rho_active)

    # Active source slice (new)
    source_active = np.asarray(inputs.source[: hm.n_faces], dtype=np.float32)
    np.save(str(wd / "source.npy"), source_active)

    # Dirichlet BC
    dm = np.asarray(bc.dirichlet.mask if bc.dirichlet else [], dtype=np.int32)
    dv = (
        np.asarray(bc.dirichlet.values, dtype=np.float32)
        if bc.dirichlet and bc.dirichlet.values is not None
        else np.zeros((0, 1), dtype=np.float32)
    )

    # Neumann BC
    nm = np.asarray(bc.neumann.mask if bc.neumann else [], dtype=np.int32)
    nv = (
        np.asarray(bc.neumann.values, dtype=np.float32)
        if bc.neumann
        else np.zeros((0, 1), dtype=np.float32)
    )

    payload = {
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "Lz": float(Lz),
        "k_max": float(inputs.k_max),
        "p_exp": float(inputs.p_exp),
        "rho_file": "rho.npy",
        "source_file": "source.npy",
        "dirichlet_mask": dm.tolist(),
        "dirichlet_values": dv.tolist(),
        "neumann_mask": nm.tolist(),
        "neumann_values": nv.tolist(),
    }

    with open(wd / "input.json", "w") as f:
        json.dump(payload, f)


def _run_solver(wd: Path, compute_gradient: bool = False) -> None:  # mosaic:physics
    """Invoke the deal.II heat_solver binary."""
    cmd = [_DEALII_SOLVER, str(wd / "input.json")]
    if compute_gradient:
        cmd.append("--gradient")
    result = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"deal.II solver failed:\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-500:]}"
        )


# ---------------------------------------------------------------------------
# Python FEM helpers for source VJP (adjoint approach)
# ---------------------------------------------------------------------------

# HEX8 reference node coordinates on [-1,1]^3 (Abaqus ordering)
_HEX8_NODES_REF = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=np.float64,
)


def _compute_ref_stiffness(
    dx: float, dy: float, dz: float
) -> np.ndarray:  # mosaic:physics
    """Compute 8x8 HEX8 reference element stiffness matrix for unit conductivity."""
    gp = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    gw = np.array([1.0, 1.0])
    Jinv = np.diag([2.0 / dx, 2.0 / dy, 2.0 / dz])
    detJ = (dx / 2.0) * (dy / 2.0) * (dz / 2.0)
    nodes = _HEX8_NODES_REF
    xi_i, eta_i, zeta_i = nodes[:, 0], nodes[:, 1], nodes[:, 2]

    K_ref = np.zeros((8, 8), dtype=np.float64)
    for wi, xi in zip(gw, gp):
        for wj, eta in zip(gw, gp):
            for wk, zeta in zip(gw, gp):
                dN = np.zeros((8, 3), dtype=np.float64)
                dN[:, 0] = xi_i * (1.0 + eta_i * eta) * (1.0 + zeta_i * zeta) / 8.0
                dN[:, 1] = (1.0 + xi_i * xi) * eta_i * (1.0 + zeta_i * zeta) / 8.0
                dN[:, 2] = (1.0 + xi_i * xi) * (1.0 + eta_i * eta) * zeta_i / 8.0
                B = dN @ Jinv.T
                K_ref += (wi * wj * wk) * detJ * (B @ B.T)
    return K_ref


def _assemble_K(  # mosaic:physics
    inputs: "InputSchema",
    cells: np.ndarray,
    points: np.ndarray,
    n_nodes: int,
    dx: float,
    dy: float,
    dz: float,
) -> sp.csr_matrix:
    """Assemble global stiffness matrix using SIMP conductivity."""
    n_cells = len(cells)
    k_min = 1e-3 * float(inputs.k_max)
    k_max = float(inputs.k_max)
    p_exp = float(inputs.p_exp)

    rho = np.clip(np.asarray(inputs.rho[:n_cells], dtype=np.float64), 0.0, 1.0)
    k_elem = k_min + (k_max - k_min) * rho**p_exp

    K_ref = _compute_ref_stiffness(dx, dy, dz)

    ii, jj = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    ii_flat, jj_flat = ii.ravel(), jj.ravel()
    K_ref_flat = K_ref.ravel()

    rows = cells[:, ii_flat].ravel()
    cols = cells[:, jj_flat].ravel()
    vals = (k_elem[:, None] * K_ref_flat[None, :]).ravel()

    K = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    return K.tocsr()


def _apply_dirichlet_to_system(  # mosaic:physics
    K: sp.csr_matrix,
    rhs: np.ndarray,
    d_mask: np.ndarray,
    d_values: np.ndarray,
    n_nodes: int,
    homogeneous: bool = False,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Apply Dirichlet BCs by elimination (symmetric).

    If homogeneous=True, prescribed values are 0 (for adjoint solve).
    """
    rhs = rhs.copy()
    mask = np.asarray(d_mask, dtype=np.int32)[:n_nodes]
    constrained = np.where(mask > 0)[0]
    if len(constrained) == 0:
        return K, rhs

    if homogeneous:
        prescribed = np.zeros(len(constrained), dtype=np.float64)
    else:
        d_vals = np.asarray(d_values, dtype=np.float64)
        groups = mask[constrained] - 1
        prescribed = (
            d_vals[groups, 0] if d_vals.size > 0 else np.zeros(len(constrained))
        )

    T_p = np.zeros(n_nodes, dtype=np.float64)
    T_p[constrained] = prescribed
    rhs -= K @ T_p

    K_coo = K.tocoo().astype(np.float64)
    mask_row = np.isin(K_coo.row, constrained)
    mask_col = np.isin(K_coo.col, constrained)
    keep = ~(mask_row | mask_col)

    rows_new = np.concatenate([K_coo.row[keep], constrained])
    cols_new = np.concatenate([K_coo.col[keep], constrained])
    vals_new = np.concatenate([K_coo.data[keep], np.ones(len(constrained))])

    K_bc = sp.coo_matrix((vals_new, (rows_new, cols_new)), shape=K.shape).tocsr()
    rhs[constrained] = prescribed
    return K_bc, rhs


def _compute_source_vjp(  # mosaic:grad:source:adjoint
    inputs: "InputSchema",
    T_nodal: np.ndarray,
    cot_id: float,
) -> np.ndarray:
    """Compute VJP of identification_error w.r.t. source via Python adjoint.

    E = sum_i (T_i - T_target_i)^2
    Adjoint:  K · λ = 2 * cot_id * (T - T_target)
    Source gradient:  dE/dsource_e = vol_e / 8 * sum_{i in e}(λ_i)

    Returns:
        grad_source: (n_cells,) float32
    """
    hm = inputs.hex_mesh
    n_cells = int(hm.n_faces)
    n_nodes = int(hm.n_points)

    d = hm.model_dump()
    cells = np.asarray(d["faces"][:n_cells], dtype=np.int32)
    points = np.asarray(d["points"][:n_nodes], dtype=np.float64)

    xs = np.unique(np.round(points[:, 0], 7))
    ys = np.unique(np.round(points[:, 1], 7))
    zs = np.unique(np.round(points[:, 2], 7))
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    dz = float(zs[1] - zs[0]) if len(zs) > 1 else 1.0
    vol_e = dx * dy * dz

    # Assemble K
    K = _assemble_K(inputs, cells, points, n_nodes, dx, dy, dz)

    # Build adjoint RHS: 2 * cot_id * (T - T_target)
    target_temp = np.asarray(inputs.target_temperature, dtype=np.float64)
    T = np.asarray(T_nodal, dtype=np.float64)
    n_match = min(n_nodes, len(target_temp))
    adj_rhs = np.zeros(n_nodes, dtype=np.float64)
    adj_rhs[:n_match] = 2.0 * cot_id * (T[:n_match] - target_temp[:n_match])

    # Apply Dirichlet BCs (homogeneous on adjoint)
    bc_dict = inputs.boundary_conditions.model_dump()
    d_bc = bc_dict.get("dirichlet") or {}
    d_mask = np.asarray(d_bc.get("mask", []), dtype=np.int32)
    d_vals_raw = d_bc.get("values")
    d_vals = (
        np.asarray(d_vals_raw, dtype=np.float64)
        if d_vals_raw is not None
        else np.zeros((0, 1), dtype=np.float64)
    )

    K_bc, rhs_bc = _apply_dirichlet_to_system(
        K, adj_rhs, d_mask, d_vals, n_nodes, homogeneous=True
    )

    # Solve adjoint
    try:
        lam = spla.spsolve(K_bc.tocsc(), rhs_bc)
    except Exception:
        lam = np.zeros(n_nodes, dtype=np.float64)

    # Source gradient: transpose of source-to-RHS map
    # f_i += source_e * vol_e / 8  (for each node i of cell e)
    # → dE/dsource_e = vol_e / 8 * sum_{i in e}(λ_i)
    lam_elem = lam[cells]  # (n_cells, 8)
    grad_source = (vol_e / 8.0) * lam_elem.sum(axis=1)  # (n_cells,)

    return grad_source.astype(np.float32)


def _compute_compliance_source_vjp(  # mosaic:grad:source:adjoint
    inputs: "InputSchema",
    T_nodal: np.ndarray,
    cot_tc: float,
) -> np.ndarray:
    """Compute VJP of thermal_compliance w.r.t. source via adjoint solve.

    The compliance is C = T^T f_Neumann  (work done by Neumann flux on temperature).
    The PDE is K T = f_Neumann + f_source  where  f_source_i = source_e * vol_e/8.

    Adjoint equation:  K λ = ∂C/∂T = f_Neumann = K T − f_source
    Source gradient:  dC/dsource_e = cot_tc * vol_e/8 * sum_{i in e}(λ_i)

    For source=0 (default): f_source=0, so K λ = K T → λ = T (no solve needed).
    For general source:  λ = T − T_source_only  where K T_source_only = f_source,
    or equivalently λ is obtained by solving K λ = K T − f_source.

    Returns:
        grad_source: (n_cells,) float32
    """
    hm = inputs.hex_mesh
    n_cells = int(hm.n_faces)
    n_nodes = int(hm.n_points)

    d = hm.model_dump()
    cells = np.asarray(d["faces"][:n_cells], dtype=np.int32)
    points = np.asarray(d["points"][:n_nodes], dtype=np.float64)

    xs = np.unique(np.round(points[:, 0], 7))
    ys = np.unique(np.round(points[:, 1], 7))
    zs = np.unique(np.round(points[:, 2], 7))
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    dz = float(zs[1] - zs[0]) if len(zs) > 1 else 1.0
    vol_e = dx * dy * dz

    T = np.asarray(T_nodal, dtype=np.float64)

    # Assemble K and compute f_source to build adjoint RHS = K T − f_source
    K = _assemble_K(inputs, cells, points, n_nodes, dx, dy, dz)

    # Build source RHS vector:  f_source_i = sum_{e containing i}(source_e * vol_e / 8)
    source_arr = np.asarray(inputs.source[:n_cells], dtype=np.float64)
    f_source = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(f_source, cells.ravel(), np.repeat(source_arr * vol_e / 8.0, 8))

    # Adjoint RHS: f_Neumann = K T − f_source
    adj_rhs = K @ T - f_source  # (n_nodes,)
    # Scale by cotangent
    adj_rhs *= cot_tc

    # Apply homogeneous Dirichlet BCs to adjoint
    bc_dict = inputs.boundary_conditions.model_dump()
    d_bc = bc_dict.get("dirichlet") or {}
    d_mask = np.asarray(d_bc.get("mask", []), dtype=np.int32)
    d_vals_raw = d_bc.get("values")
    d_vals = (
        np.asarray(d_vals_raw, dtype=np.float64)
        if d_vals_raw is not None
        else np.zeros((0, 1), dtype=np.float64)
    )

    K_bc, rhs_bc = _apply_dirichlet_to_system(
        K, adj_rhs, d_mask, d_vals, n_nodes, homogeneous=True
    )

    # Solve adjoint system K λ = cot_tc * f_Neumann
    try:
        lam = spla.spsolve(K_bc.tocsc(), rhs_bc)
    except Exception:
        lam = np.zeros(n_nodes, dtype=np.float64)

    # Source gradient: transpose of source-to-RHS map
    # f_i += source_e * vol_e / 8  (for each node i of cell e)
    # → dC/dsource_e = vol_e / 8 * sum_{i in e}(λ_i)
    lam_elem = lam[cells]  # (n_cells, 8)
    grad_source = (vol_e / 8.0) * lam_elem.sum(axis=1)  # (n_cells,)

    return grad_source.astype(np.float32)


def _parse_outputs(inputs: InputSchema, wd: Path) -> OutputSchema:  # mosaic:io
    """Read temperature.npy and compliance.txt written by the C++ solver."""
    temperature = np.load(str(wd / "temperature.npy")).astype(np.float32)
    with open(wd / "compliance.txt") as f:
        compliance = float(f.read().strip())

    target_temp = np.asarray(inputs.target_temperature, dtype=np.float32)
    n = min(len(temperature), len(target_temp))
    id_error = np.float32(np.sum((temperature[:n] - target_temp[:n]) ** 2))

    return OutputSchema(
        thermal_compliance=np.float32(compliance),
        temperature=temperature,
        identification_error=id_error,
    )


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve heat conduction and return compliance + temperature.

    Args:
        inputs: Validated InputSchema with density field, mesh, BCs, material params.

    Returns:
        OutputSchema with thermal_compliance (scalar) and temperature (n_nodes,).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        wd = Path(tmpdir)
        _write_inputs(inputs, wd)
        _run_solver(wd)
        return _parse_outputs(inputs, wd)


def vector_jacobian_product(  # mosaic:grad:rho,source
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP for rho (analytic SIMP sensitivity) and source (Python adjoint).

    rho path:
        Runs the forward solve with ``--gradient`` flag so the C++ binary also
        writes ``gradient.npy`` (shape: n_active_cells, analytic ∂C/∂ρ per cell).
        Only differentiation through rho → thermal_compliance is supported.

    source path:
        Uses Python adjoint solves.  Both cotangents are supported and summed.
        identification_error: K·λ = 2·cot_id·(T − T_target)
        thermal_compliance:   K·λ = cot_tc·(KT − f_source)

    Args:
        inputs: Validated InputSchema.
        vjp_inputs: Names of inputs for which gradients are requested.
        vjp_outputs: Names of outputs whose cotangents are provided.
        cotangent_vector: Dict of output-name → cotangent scalar/array.

    Returns:
        Dict mapping input names → gradient arrays matching input shapes.
    """
    want_rho = "rho" in vjp_inputs
    want_source = "source" in vjp_inputs

    if not want_rho and not want_source:
        return {}

    result: dict[str, Any] = {}

    # --- rho VJP: C++ analytic SIMP sensitivity ---
    # mosaic:grad:rho:analytic
    if want_rho:
        cot = float(cotangent_vector.get("thermal_compliance", 1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            wd = Path(tmpdir)
            _write_inputs(inputs, wd)
            _run_solver(wd, compute_gradient=True)
            gradient = np.load(str(wd / "gradient.npy")).astype(np.float32)

        hm = inputs.hex_mesh
        n_active = hm.n_faces
        grad_rho = np.zeros(len(np.asarray(inputs.rho)), dtype=np.float32)
        grad_rho[:n_active] = (gradient[:n_active] * cot).astype(np.float32)
        result["rho"] = grad_rho

    # --- source VJP: Python adjoint via identification_error and/or thermal_compliance ---
    # mosaic:grad:source:adjoint
    if want_source:
        cot_id = float(cotangent_vector.get("identification_error", 0.0))
        cot_tc = float(cotangent_vector.get("thermal_compliance", 0.0))

        n_source = len(np.asarray(inputs.source))
        if cot_id == 0.0 and cot_tc == 0.0:
            # No relevant cotangent → zero source gradient
            result["source"] = np.zeros(n_source, dtype=np.float32)
        else:
            # Forward solve to get temperature (needed for both paths)
            with tempfile.TemporaryDirectory() as tmpdir:
                wd = Path(tmpdir)
                _write_inputs(inputs, wd)
                _run_solver(wd, compute_gradient=False)
                T_nodal = np.load(str(wd / "temperature.npy")).astype(np.float32)

            hm = inputs.hex_mesh
            n_active = hm.n_faces
            grad_source = np.zeros(n_source, dtype=np.float32)

            if cot_id != 0.0:
                grad_id = _compute_source_vjp(inputs, T_nodal, cot_id)
                grad_source[:n_active] = grad_source[:n_active] + grad_id[:n_active]

            if cot_tc != 0.0:
                grad_tc = _compute_compliance_source_vjp(inputs, T_nodal, cot_tc)
                grad_source[:n_active] = grad_source[:n_active] + grad_tc[:n_active]

            result["source"] = grad_source

    return result


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver.

    Args:
        abstract_inputs: InputSchema with shape/dtype metadata (no values).

    Returns:
        Dict mapping output names to ShapeDType descriptors.
    """
    d = abstract_inputs.model_dump()
    points = d["hex_mesh"]["points"]
    n_nodes = points["shape"][0] if isinstance(points, dict) else len(points)
    return {
        "thermal_compliance": ShapeDType(shape=(), dtype="float32"),
        "temperature": ShapeDType(shape=(n_nodes,), dtype="float32"),
        "identification_error": ShapeDType(shape=(), dtype="float32"),
    }
