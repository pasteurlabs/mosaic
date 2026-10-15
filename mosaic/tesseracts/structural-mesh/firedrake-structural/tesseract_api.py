# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F405

"""Linear elasticity + SIMP topology optimisation on an arbitrary hexahedral mesh.

Uses Firedrake + firedrake-adjoint to solve 3-D linear elasticity with SIMP
material interpolation and compute the exact adjoint gradient of the structural
compliance objective.

CRITICAL import order: firedrake.adjoint must immediately follow firedrake so
that it can monkey-patch solve/assemble and record operations on the adjoint
tape.
"""

import hashlib
import os
import tempfile
from typing import Any

import meshio
import numpy as np
from firedrake import *  # noqa: F403
from firedrake.adjoint import *  # noqa: F403
from mosaic_shared.problems.structural_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.structural_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.schema_types import make_differentiable
from pydantic import Field
from scipy.spatial import cKDTree
from tesseract_core.runtime import ShapeDType

# ---------------------------------------------------------------------------
# Schema — extends canonical with material parameters
# ---------------------------------------------------------------------------


class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho"])):
    """Inputs for Firedrake structural solver, extended with material parameters."""

    E_max: float = Field(
        default=70000.0,
        description="Young's modulus of the fully solid material.",
    )
    nu: float = Field(
        default=0.3,
        description="Poisson's ratio.",
    )
    xmin: float = Field(
        default=1e-3,
        description="Void stiffness ratio (E_min = xmin * E_max).",
    )
    penal: float = Field(
        default=3.0,
        description="SIMP penalisation exponent (E(rho) = E_min + (E_max-E_min)*rho^penal).",
    )


class OutputSchema(make_differentiable(_CanonicalOutputSchema, ["compliance"])):
    """Firedrake structural solver output schema."""


# ---------------------------------------------------------------------------
# Mesh construction with tagged boundary groups
# ---------------------------------------------------------------------------


def _build_firedrake_mesh(
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask: np.ndarray,
    neumann_mask: np.ndarray,
):
    """Convert numpy hex mesh arrays to a Firedrake Mesh via GMSH .msh file.

    Boundary facets are tagged with physical group IDs so that Firedrake can
    reference them as ``ds(tag)`` in Neumann integrals and as subdomain IDs
    in ``DirichletBC`` constructors.

    Tag convention (matching the 1-indexed group convention of the schema):
        Dirichlet group k → tag k
        Neumann group k   → tag 100 + k  (offset to avoid collision)

    Args:
        pts: Node coordinates, shape (n_nodes, 3), float64.
        cells: Hex cell connectivity, shape (n_cells, 8), int64.
        dirichlet_mask: Per-node Dirichlet group index (0=free, k>=1 → group k).
        neumann_mask: Per-node Neumann group index (0=free, k>=1 → group k).

    Returns:
        Tuple (mesh, neumann_offset) where:
            mesh: Firedrake Mesh with tagged boundary facets.
            neumann_offset: Integer offset added to Neumann group IDs in the mesh
                            tags (100 by default).
    """
    neumann_offset = 100

    # Build boundary facets from cell faces and tag them.
    # For a hex mesh, each cell has 6 faces; boundary faces appear exactly once.

    # HEX8 face connectivity: 6 faces × 4 vertices (Abaqus ordering)
    # Face vertex indices within a cell (local HEX8 face definitions):
    #   Face 0: nodes 0,1,2,3 (z-min)
    #   Face 1: nodes 4,5,6,7 (z-max)
    #   Face 2: nodes 0,1,5,4 (y-min)
    #   Face 3: nodes 2,3,7,6 (y-max)
    #   Face 4: nodes 0,3,7,4 (x-min)
    #   Face 5: nodes 1,2,6,5 (x-max)
    hex_local_faces = [
        [0, 1, 2, 3],  # z-min
        [4, 5, 6, 7],  # z-max
        [0, 1, 5, 4],  # y-min
        [2, 3, 7, 6],  # y-max
        [0, 3, 7, 4],  # x-min
        [1, 2, 6, 5],  # x-max
    ]

    # Collect all faces (sorted node tuples → count occurrences).
    face_count: dict[tuple, int] = {}
    face_nodes: dict[tuple, tuple] = {}  # sorted → original node ordering
    for cell in cells:
        for local_f in hex_local_faces:
            raw = tuple(int(cell[i]) for i in local_f)
            key = tuple(sorted(raw))
            face_count[key] = face_count.get(key, 0) + 1
            face_nodes[key] = raw

    # Boundary facets are those that appear exactly once.
    boundary_quads: list[tuple] = []
    boundary_tags: list[int] = []

    for key, count in face_count.items():
        if count != 1:
            continue
        raw = face_nodes[key]
        # Determine tag: majority vote of vertex group assignments.
        d_groups = [int(dirichlet_mask[v]) for v in raw if v < len(dirichlet_mask)]
        n_groups_ = [int(neumann_mask[v]) for v in raw if v < len(neumann_mask)]

        d_uniq = set(d_groups)
        n_uniq = set(n_groups_)

        if len(d_uniq) == 1 and d_uniq != {0}:
            tag = next(iter(d_uniq))
        elif len(n_uniq) == 1 and n_uniq != {0}:
            tag = neumann_offset + next(iter(n_uniq))
        else:
            tag = 0  # untagged boundary face

        boundary_quads.append(raw)
        boundary_tags.append(tag)

    # Build meshio mesh with boundary quads as tagged "quad" cells.
    # Firedrake reads physical tags from ``gmsh:physical`` cell data.
    # Each unique tag must be its own cell block with the matching physical tag.
    if boundary_quads:
        boundary_arr = np.array(boundary_quads, dtype=np.int64)
        boundary_tag_arr = np.array(boundary_tags, dtype=np.int32)

        # One cell block per unique non-zero tag so that each block carries a
        # single physical tag value (meshio writes one block per tag group).
        unique_tags = sorted(set(boundary_tags) - {0})
        extra_cells = []
        extra_phys: list[np.ndarray] = []
        for t in unique_tags:
            mask = boundary_tag_arr == t
            block_quads = boundary_arr[mask]
            n_block = int(mask.sum())
            extra_cells.append(("quad", block_quads))
            extra_phys.append(np.full(n_block, t, dtype=np.int32))

        # Untagged boundary faces — give them tag 0 (not referenced by BCs).
        mask0 = boundary_tag_arr == 0
        if mask0.any():
            extra_cells.append(("quad", boundary_arr[mask0]))
            extra_phys.append(np.zeros(int(mask0.sum()), dtype=np.int32))

        # gmsh:physical cell data: one array per cell block.
        # Volume (hex) block gets physical tag 999 (unused by BCs).
        hex_phys = np.full(len(cells), 999, dtype=np.int32)
        phys_data = [hex_phys, *extra_phys]

        mio_mesh = meshio.Mesh(
            points=pts.astype(np.float64),
            cells=[("hexahedron", cells.astype(np.int64)), *extra_cells],
            cell_data={"gmsh:physical": phys_data},
        )
    else:
        mio_mesh = meshio.Mesh(
            points=pts.astype(np.float64),
            cells=[("hexahedron", cells.astype(np.int64))],
            cell_data={"gmsh:physical": [np.full(len(cells), 999, dtype=np.int32)]},
        )

    fd, msh_path = tempfile.mkstemp(suffix=".msh")
    os.close(fd)
    try:
        meshio.write(msh_path, mio_mesh, file_format="gmsh22")
        mesh = Mesh(msh_path)
    finally:
        if os.path.exists(msh_path):
            os.unlink(msh_path)

    return mesh, neumann_offset


# ---------------------------------------------------------------------------
# Cell reorder map  (Firedrake may renumber cells on load)
# ---------------------------------------------------------------------------


def _cell_reorder_map(
    pts: np.ndarray, input_cells: np.ndarray, fd_mesh: Any
) -> np.ndarray:
    """Build Firedrake-cell-index → input-cell-index permutation via centroid matching.

    Firedrake may reorder cells when loading from a GMSH file.  This function
    recovers the permutation so that ``rho_values[input_idx]`` is assigned to
    the correct Firedrake DG0 DOF, and the adjoint gradient can be mapped back.

    Args:
        pts: Input mesh node coordinates, shape (n_nodes, 3).
        input_cells: Input cell connectivity, shape (n_input_cells, 8).
        fd_mesh: Firedrake Mesh built from the same data.

    Returns:
        Array of shape (n_fd_cells,) where entry j gives the input cell index
        that corresponds to Firedrake cell j.
    """
    input_centroids = pts[input_cells].mean(axis=1)  # (n_input_cells, 3)

    # Firedrake cell centroids via coordinate field.
    coord_arr = fd_mesh.coordinates.dat.data_ro  # (n_fd_nodes, 3)
    cell_node_map = fd_mesh.coordinates.cell_node_map().values  # (n_fd_cells, 8)
    fd_centroids = coord_arr[cell_node_map].mean(axis=1)  # (n_fd_cells, 3)

    tree = cKDTree(input_centroids)
    _, fd_to_input = tree.query(fd_centroids)
    return fd_to_input  # shape (n_fd_cells,)


# ---------------------------------------------------------------------------
# BC extraction helpers
# ---------------------------------------------------------------------------


def _extract_dirichlet(bc: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract Dirichlet mask and values arrays from a MeshBC object.

    When ``dirichlet.values`` is ``None`` (homogeneous BCs), the number of
    groups is inferred from the maximum group index in the mask so that the
    correct number of zero-displacement DirichletBC objects are constructed.

    Returns:
        Tuple (mask, values) where:
            mask: Per-node group index, shape (n_nodes,), int32.
            values: Per-group displacement, shape (n_groups, 3), float64.
                    Zero displacement for all groups when values was None.
    """
    if bc.dirichlet is None:
        return np.zeros(0, dtype=np.int32), np.zeros((0, 3), dtype=np.float64)

    mask = np.asarray(bc.dirichlet.mask, dtype=np.int32)
    if bc.dirichlet.values is not None:
        values = np.asarray(bc.dirichlet.values, dtype=np.float64)
    else:
        # Infer group count from mask; prescribed value is zero displacement.
        n_groups = int(mask.max()) if mask.size > 0 else 0
        values = np.zeros((n_groups, 3), dtype=np.float64)

    return mask, values


def _extract_neumann(bc: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract Neumann mask and values arrays from a MeshBC object.

    Returns:
        Tuple (mask, values) where:
            mask: Per-node group index, shape (n_nodes,), int32.
            values: Per-group traction, shape (n_groups, 3), float64.
    """
    if bc.neumann is None:
        return np.zeros(0, dtype=np.int32), np.zeros((0, 3), dtype=np.float64)

    mask = np.asarray(bc.neumann.mask, dtype=np.int32)
    values = np.asarray(bc.neumann.values, dtype=np.float64)
    return mask, values


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


# `_SETUP_CACHE` holds one replayable `ReducedFunctional` for the most recent
# (mesh, BC, material), i.e. every input except rho. Recording it pays for the
# gmsh mesh round-trip, function spaces, forms and one annotated solve. Later
# calls replay it at a new rho, which also reuses pyadjoint's forward and
# adjoint solvers and the MUMPS symbolic factorization.
#
# The VJP replays the tape at its own rho instead of reusing the forward
# solution of a preceding `apply`. Every other solver's VJP endpoint also
# re-runs the forward pass, so this keeps VJP costs comparable.
#
# The first two replay-then-derivative sweeps on a tape compile different TSFC
# kernels; the second one's (~1.5 s) would otherwise land on the second VJP
# call. Kernels are cached across tapes, so the first tape a process records
# runs both sweeps up front.

# Sparse direct LU via MUMPS (Firedrake's default, made explicit) for the
# forward solve. pyadjoint copies these parameters to the adjoint solve.
_SOLVER_PARAMETERS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

_SETUP_CACHE: dict[str, dict[str, Any]] = {}
_ADJOINT_KERNELS_COMPILED = False


def _setup_cache_key(*arrays: np.ndarray, params: tuple[float, ...]) -> str:
    """Hash everything the recorded tape depends on, i.e. all inputs except rho."""
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.array(params, dtype=np.float64).tobytes())
    return h.hexdigest()


def _build_reduced_functional(
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    E_max: float,
    nu: float,
    xmin: float,
    penal: float,
) -> dict[str, Any]:
    """Record 3-D linear elasticity with SIMP on a fresh tape.

    Solves:
        -∇·σ(u) = 0    in Ω

    with SIMP stiffness:
        E(ρ) = E_min + (E_max − E_min) · ρ^p    (E_min = xmin·E_max, p=penal)
        σ = λ tr(ε) I + 2μ ε
        ε = ½(∇u + ∇u^T)

    Compliance objective:
        C = ∫_Γ_N f · u dΓ  (= F^T U for linear elasticity)

    The tape is recorded at an arbitrary uniform density; callers evaluate it
    at the real rho with ``Jhat(rho)``.

    Args:
        pts: Mesh node coordinates, shape (n_nodes, 3).
        cells: Hex cell connectivity, shape (n_cells, 8).
        dirichlet_mask_vals: Per-node Dirichlet group index, shape (n_nodes,).
        dirichlet_values_vals: Per-group prescribed displacement, shape (n_groups, 3).
        neumann_mask_vals: Per-node Neumann group index, shape (n_nodes,).
        neumann_values_vals: Per-group surface traction, shape (n_neumann_groups, 3).
        E_max: Young's modulus of the fully solid material.
        nu: Poisson's ratio.
        xmin: Void stiffness ratio (E_min = xmin * E_max).
        penal: SIMP penalisation exponent.

    Returns:
        Dict with the compliance ``ReducedFunctional`` ("Jhat"), the DG0
        density space ("rho_space") and the Firedrake → input cell permutation
        ("fd_to_input_cells").
    """
    # ---- Mesh with boundary tags -----------------------------------------
    mesh, neumann_offset = _build_firedrake_mesh(
        pts, cells, dirichlet_mask_vals, neumann_mask_vals
    )
    fd_to_input_cells = _cell_reorder_map(pts, cells, mesh)

    # ---- Function spaces -------------------------------------------------
    # CG1 vector for displacement (3 DOFs per node); DG0 for density.
    V = VectorFunctionSpace(mesh, "CG", 1)
    DG0 = FunctionSpace(mesh, "DG", 0)

    tape = Tape()
    set_working_tape(tape)
    continue_annotation()

    # ---- Density field ---------------------------------------------------
    rho_fn = Function(DG0, name="rho")
    rho_fn.dat.data[:] = 0.5

    # ---- SIMP stiffness --------------------------------------------------
    E_min_val = Constant(xmin * E_max)
    E_max_val = Constant(E_max)
    E_field = E_min_val + (E_max_val - E_min_val) * rho_fn ** Constant(penal)

    nu_c = Constant(nu)
    lam = E_field * nu_c / ((1 + nu_c) * (1 - 2 * nu_c))
    mu_c = E_field / (2 * (1 + nu_c))

    # ---- Strain and stress operators -------------------------------------
    def epsilon(w: Any) -> Any:
        return 0.5 * (grad(w) + grad(w).T)

    def sigma(w: Any) -> Any:
        return lam * tr(epsilon(w)) * Identity(3) + 2 * mu_c * epsilon(w)

    # ---- Variational form ------------------------------------------------
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(sigma(u), epsilon(v)) * dx

    # ---- Neumann BCs (surface tractions, via tagged boundary IDs) --------
    n_neumann_groups = neumann_values_vals.shape[0]

    # ds() uses the boundary tags embedded in the GMSH file.
    # Neumann group k was written with tag = neumann_offset + k.
    L = inner(Constant((0.0, 0.0, 0.0)), v) * dx  # zero initialiser
    for k in range(n_neumann_groups):
        traction = Constant(tuple(float(x) for x in neumann_values_vals[k]))
        tag = neumann_offset + (k + 1)
        L = L + inner(traction, v) * ds(tag)

    # ---- Dirichlet BCs (prescribed displacement, via tagged boundary IDs) -
    n_dirichlet_groups = dirichlet_values_vals.shape[0]
    bcs = []
    for k in range(n_dirichlet_groups):
        val = tuple(float(x) for x in dirichlet_values_vals[k])
        tag = k + 1  # Dirichlet group k+1 written directly as tag k+1
        bc = DirichletBC(V, Constant(val), tag)
        bcs.append(bc)

    # ---- Solve -----------------------------------------------------------
    u_sol = Function(V)
    problem = LinearVariationalProblem(a, L, u_sol, bcs=bcs)
    LinearVariationalSolver(problem, solver_parameters=_SOLVER_PARAMETERS).solve()

    # ---- Objective: structural compliance --------------------------------
    # C = ∫_Γ_N f · u dΓ  (recorded on the pyadjoint tape via assemble)
    J_form = inner(Constant((0.0, 0.0, 0.0)), u_sol) * dx
    for k in range(n_neumann_groups):
        traction = Constant(tuple(float(x) for x in neumann_values_vals[k]))
        tag = neumann_offset + (k + 1)
        J_form = J_form + inner(traction, u_sol) * ds(tag)
    J = assemble(J_form)
    pause_annotation()

    return {
        "Jhat": ReducedFunctional(J, Control(rho_fn), tape=tape),
        "rho_space": DG0,
        "fd_to_input_cells": fd_to_input_cells,
    }


def _replay_elasticity(
    rho_values: np.ndarray,
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    E_max: float = 70000.0,
    nu: float = 0.3,
    xmin: float = 1e-3,
    penal: float = 3.0,
) -> tuple[float, dict[str, Any]]:
    """Evaluate compliance at rho on the cached tape, recording it on first use.

    Returns:
        Tuple (J_val, entry) where entry is the cache entry built by
        `_build_reduced_functional`, now evaluated at rho.
    """
    global _ADJOINT_KERNELS_COMPILED
    args = (
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
    )
    params = (E_max, nu, xmin, penal)
    key = _setup_cache_key(*args, params=params)
    if key not in _SETUP_CACHE:
        _SETUP_CACHE.clear()
        _SETUP_CACHE[key] = _build_reduced_functional(*args, *params)
        if not _ADJOINT_KERNELS_COMPILED:
            Jhat = _SETUP_CACHE[key]["Jhat"]
            for _ in range(2):
                Jhat(Jhat.controls[0].control)
                Jhat.derivative()
            _ADJOINT_KERNELS_COMPILED = True
    entry = _SETUP_CACHE[key]

    rho_fn = Function(entry["rho_space"])
    rho_fn.dat.data[:] = np.clip(rho_values[entry["fd_to_input_cells"]], 0.0, 1.0)
    return float(entry["Jhat"](rho_fn)), entry


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve linear elasticity and return compliance.

    Args:
        inputs: Validated InputSchema containing the density field, mesh, and BCs.

    Returns:
        OutputSchema with compliance (scalar).
    """
    hm = inputs.hex_mesh
    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float64)
    cells = np.asarray(hm.faces[: hm.n_faces], dtype=np.int64)
    rho_values = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float64)

    bc = inputs.boundary_conditions
    dm, dv = _extract_dirichlet(bc)
    nm, nv = _extract_neumann(bc)

    J_val, _ = _replay_elasticity(
        rho_values,
        pts,
        cells,
        dm,
        dv,
        nm,
        nv,
        E_max=inputs.E_max,
        nu=inputs.nu,
        xmin=inputs.xmin,
        penal=inputs.penal,
    )

    return OutputSchema(compliance=np.float32(J_val))


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via firedrake-adjoint: gradient of compliance objective.

    Args:
        inputs: Validated InputSchema.
        vjp_inputs: Names of inputs for which gradients are requested.
        vjp_outputs: Names of outputs whose cotangents are provided.
        cotangent_vector: Dict of output-name -> cotangent scalar.

    Returns:
        Dict mapping "rho" -> gradient array of the same shape as inputs.rho.
    """
    assert vjp_inputs <= {"rho"}
    assert vjp_outputs <= {"compliance"}

    if "rho" not in vjp_inputs:
        return {}

    cot_c = float(cotangent_vector.get("compliance", 0.0))
    hm = inputs.hex_mesh
    grad_rho = np.zeros(len(np.asarray(inputs.rho)), dtype=np.float32)
    if abs(cot_c) == 0.0:
        return {"rho": grad_rho}

    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float64)
    cells = np.asarray(hm.faces[: hm.n_faces], dtype=np.int64)
    rho_values = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float64)

    bc = inputs.boundary_conditions
    dm, dv = _extract_dirichlet(bc)
    nm, nv = _extract_neumann(bc)

    _, entry = _replay_elasticity(
        rho_values,
        pts,
        cells,
        dm,
        dv,
        nm,
        nv,
        E_max=inputs.E_max,
        nu=inputs.nu,
        xmin=inputs.xmin,
        penal=inputs.penal,
    )

    # Map Firedrake DG0 DOF order → input cell order.
    dJ_drho = np.zeros(len(rho_values))
    dJ_drho[entry["fd_to_input_cells"]] = entry["Jhat"].derivative().dat.data_ro
    grad_rho[: hm.n_faces] = (cot_c * dJ_drho).astype(np.float32)
    return {"rho": grad_rho}


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver."""
    return {"compliance": ShapeDType(shape=(), dtype="float32")}
