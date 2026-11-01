# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F405

"""Thermal heat conduction on an arbitrary hexahedral mesh.

Uses Firedrake + firedrake-adjoint to solve steady-state heat conduction with
SIMP material interpolation and a volumetric heat source, then compute the
exact adjoint gradient of both the thermal compliance and the source-identification
objectives.

CRITICAL import order: firedrake.adjoint must immediately follow firedrake so
that it can monkey-patch solve/assemble and record operations on the adjoint tape.
"""

import hashlib
import os
import tempfile
from typing import Any

import meshio
import numpy as np
from firedrake import *  # noqa: F403
from firedrake.adjoint import *  # noqa: F403
from mosaic_shared.problems.thermal_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.thermal_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.schema_types import make_differentiable
from pydantic import Field
from scipy.spatial import cKDTree
from tesseract_core.runtime import ShapeDType

# ---------------------------------------------------------------------------
# Schema — extends canonical with material parameters
# ---------------------------------------------------------------------------


class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho", "source"])):
    """Inputs for Firedrake thermal solver, extended with material parameters."""

    k_max: float = Field(
        default=1.0,
        description="Maximum thermal conductivity (fully solid/conducting material).",
    )
    p_exp: float = Field(
        default=3.0,
        description="SIMP penalisation exponent p (k(ρ) = k_min + (k_max−k_min)·ρ^p).",
    )


class OutputSchema(
    make_differentiable(
        _CanonicalOutputSchema, ["thermal_compliance", "identification_error"]
    )
):
    """Firedrake thermal solver output schema."""


# ---------------------------------------------------------------------------
# Mesh construction with tagged boundary groups  (reused from structural-mesh)
# ---------------------------------------------------------------------------


def _build_firedrake_mesh(
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask: np.ndarray,
    neumann_mask: np.ndarray,
):
    """Convert numpy hex mesh arrays to a Firedrake Mesh via GMSH .msh file.

    Tag convention:
        Dirichlet group k → tag k
        Neumann group k   → tag 100 + k  (offset to avoid collision)
    """
    neumann_offset = 100

    hex_local_faces = [
        [0, 1, 2, 3],  # z-min
        [4, 5, 6, 7],  # z-max
        [0, 1, 5, 4],  # y-min
        [2, 3, 7, 6],  # y-max
        [0, 3, 7, 4],  # x-min
        [1, 2, 6, 5],  # x-max
    ]

    face_count: dict[tuple, int] = {}
    face_nodes: dict[tuple, tuple] = {}
    for cell in cells:
        for local_f in hex_local_faces:
            raw = tuple(int(cell[i]) for i in local_f)
            key = tuple(sorted(raw))
            face_count[key] = face_count.get(key, 0) + 1
            face_nodes[key] = raw

    boundary_quads: list[tuple] = []
    boundary_tags: list[int] = []

    for key, count in face_count.items():
        if count != 1:
            continue
        raw = face_nodes[key]
        d_groups = [int(dirichlet_mask[v]) for v in raw if v < len(dirichlet_mask)]
        n_groups_ = [int(neumann_mask[v]) for v in raw if v < len(neumann_mask)]
        d_uniq = set(d_groups)
        n_uniq = set(n_groups_)
        if len(d_uniq) == 1 and d_uniq != {0}:
            tag = next(iter(d_uniq))
        elif len(n_uniq) == 1 and n_uniq != {0}:
            tag = neumann_offset + next(iter(n_uniq))
        else:
            tag = 0
        boundary_quads.append(raw)
        boundary_tags.append(tag)

    if boundary_quads:
        boundary_arr = np.array(boundary_quads, dtype=np.int64)
        boundary_tag_arr = np.array(boundary_tags, dtype=np.int32)
        unique_tags = sorted(set(boundary_tags) - {0})
        extra_cells = []
        extra_phys: list[np.ndarray] = []
        for t in unique_tags:
            mask = boundary_tag_arr == t
            block_quads = boundary_arr[mask]
            n_block = int(mask.sum())
            extra_cells.append(("quad", block_quads))
            extra_phys.append(np.full(n_block, t, dtype=np.int32))
        mask0 = boundary_tag_arr == 0
        if mask0.any():
            extra_cells.append(("quad", boundary_arr[mask0]))
            extra_phys.append(np.zeros(int(mask0.sum()), dtype=np.int32))
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

    # Record which Neumann tags were actually embedded in the mesh so that
    # callers can guard ds(tag) calls.  When the hot-spot stripe is narrower
    # than one mesh element (coarse debug runs), no face satisfies the
    # all-nodes-same-group criterion and the set will be empty — the caller
    # must skip those ds(tag) terms to avoid LookupError from Firedrake.
    active_neumann_tags: set[int] = set()
    for t in boundary_tags:
        if t >= neumann_offset:
            active_neumann_tags.add(t)

    return mesh, neumann_offset, active_neumann_tags


# ---------------------------------------------------------------------------
# Cell / node reorder maps  (Firedrake may renumber on load)
# ---------------------------------------------------------------------------


def _cell_reorder_map(
    pts: np.ndarray, input_cells: np.ndarray, fd_mesh: Any
) -> np.ndarray:
    """Build Firedrake-cell-index → input-cell-index permutation via centroid matching."""
    input_centroids = pts[input_cells].mean(axis=1)
    coord_arr = fd_mesh.coordinates.dat.data_ro
    cell_node_map = fd_mesh.coordinates.cell_node_map().values
    fd_centroids = coord_arr[cell_node_map].mean(axis=1)
    tree = cKDTree(input_centroids)
    _, fd_to_input = tree.query(fd_centroids)
    return fd_to_input


def _node_reorder_map(pts: np.ndarray, fd_mesh: Any) -> np.ndarray:
    """Build Firedrake-node-index → input-node-index permutation via coordinate matching."""
    fd_coords = fd_mesh.coordinates.dat.data_ro
    tree = cKDTree(pts)
    _, fd_to_input = tree.query(fd_coords)
    return fd_to_input


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


# `_SETUP_CACHE` holds the replayable reduced functionals for the most recent
# (mesh, BC, material), i.e. every input except rho and source. Recording them
# pays for the gmsh mesh round-trip, function spaces, forms and one annotated
# solve. Later calls replay them at a new (rho, source), which also reuses
# pyadjoint's forward and adjoint solvers and the MUMPS symbolic factorization.
#
# `Chat` maps (rho, source) to the thermal compliance and `That` to the
# temperature field; seeding `That`'s derivative with dI/dT gives the
# identification-error gradient. They share one tape, so one adjoint sweep
# yields d/drho and d/dsource together.
#
# The VJP replays the tape at its own inputs instead of reusing the forward
# solution of a preceding `apply`. Every other solver's VJP endpoint also
# re-runs the forward pass, so this keeps VJP costs comparable.
#
# The first two replay-then-derivative sweeps on a tape compile different TSFC
# kernels; the second one's (~1 s) would otherwise land on the second VJP
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
    """Hash everything the recorded tape depends on, i.e. all inputs but rho/source."""
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.array(params, dtype=np.float64).tobytes())
    return h.hexdigest()


def _build_reduced_functionals(
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    k_max: float,
    p_exp: float,
) -> dict[str, Any]:
    """Record 3-D steady-state heat conduction with SIMP + body source on a fresh tape.

    Solves:
        -∇·(k(ρ) ∇T) = f    in Ω

    with SIMP conductivity:
        k(ρ) = k_min + (k_max − k_min) · ρ^p    (k_min = 1e-3·k_max)

    Thermal compliance:
        C = ∮_Γ_N q_n · T dΓ

    The tape is recorded at an arbitrary (rho, source); callers evaluate it at
    the real values with ``Chat([rho, source])``.

    Returns:
        Dict with the compliance and temperature reduced functionals ("Chat",
        "That"), the DG0 control space ("dg0"), the CG1 temperature space
        ("V") and the Firedrake → input cell and node permutations
        ("fd_to_input_cells", "fd_to_input_nodes").
    """
    mesh, neumann_offset, active_neumann_tags = _build_firedrake_mesh(
        pts, cells, dirichlet_mask_vals, neumann_mask_vals
    )
    fd_to_input_cells = _cell_reorder_map(pts, cells, mesh)
    fd_to_input_nodes = _node_reorder_map(pts, mesh)

    V = FunctionSpace(mesh, "CG", 1)
    DG0 = FunctionSpace(mesh, "DG", 0)

    tape = Tape()
    set_working_tape(tape)
    continue_annotation()

    # ---- Density and source fields (DG0) -------------------------------
    rho_fn = Function(DG0, name="rho")
    rho_fn.dat.data[:] = 0.5
    source_fn = Function(DG0, name="source")

    # ---- SIMP conductivity ---------------------------------------------
    k_min_val = Constant(1e-3 * k_max)
    k_field = k_min_val + (Constant(k_max) - k_min_val) * rho_fn ** Constant(p_exp)

    # ---- Neumann facet markers -----------------------------------------
    n_neumann_groups = neumann_values_vals.shape[0]
    # Build boundary measure from GMSH tags in the mesh.
    # ds: Firedrake will use the mesh tags; ds(tag) selects the tagged facets

    # ---- Variational problem -------------------------------------------
    T = TrialFunction(V)
    v = TestFunction(V)
    a = inner(k_field * grad(T), grad(v)) * dx

    # Neumann RHS — only include ds(tag) terms for tags that were actually
    # embedded in the mesh.  When the hot-spot stripe is too narrow for the
    # current mesh resolution (coarse debug runs), no boundary face satisfies
    # the all-nodes-same-group criterion and the tag will be absent from the
    # mesh.  Skipping those terms replicates FEniCS behaviour (empty integral
    # = 0) and prevents LookupError from Firedrake.
    L = inner(Constant(0.0), v) * dx
    for k in range(n_neumann_groups):
        tag = neumann_offset + (k + 1)
        if tag not in active_neumann_tags:
            continue
        q_n = Constant(float(neumann_values_vals[k, 0]))
        L = L + q_n * v * ds(tag)

    # Body heat source: ∫_Ω f · v dΩ
    L = L + source_fn * v * dx

    # ---- Dirichlet BCs -------------------------------------------------
    n_dirichlet_groups = dirichlet_values_vals.shape[0]
    bcs = []
    for k in range(n_dirichlet_groups):
        T_prescribed = Constant(float(dirichlet_values_vals[k, 0]))
        tag = k + 1
        bcs.append(DirichletBC(V, T_prescribed, tag))

    # ---- Solve ---------------------------------------------------------
    T_sol = Function(V)
    problem = LinearVariationalProblem(a, L, T_sol, bcs=bcs)
    LinearVariationalSolver(problem, solver_parameters=_SOLVER_PARAMETERS).solve()

    # ---- Thermal compliance --------------------------------------------
    J_form = inner(Constant(0.0), T_sol) * dx
    for k in range(n_neumann_groups):
        tag = neumann_offset + (k + 1)
        if tag not in active_neumann_tags:
            continue
        q_n = Constant(float(neumann_values_vals[k, 0]))
        J_form = J_form + q_n * T_sol * ds(tag)
    J = assemble(J_form)
    pause_annotation()

    controls = [Control(rho_fn), Control(source_fn)]
    return {
        "Chat": ReducedFunctional(J, controls, tape=tape),
        "That": ReducedFunctional(T_sol, controls, tape=tape),
        "dg0": DG0,
        "V": V,
        "fd_to_input_cells": fd_to_input_cells,
        "fd_to_input_nodes": fd_to_input_nodes,
    }


def _replay_heat(
    rho_values: np.ndarray,
    source_values: np.ndarray,
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    k_max: float = 1.0,
    p_exp: float = 3.0,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Evaluate the heat problem at (rho, source) on the cached tape.

    Records the tape on first use for a given mesh, BC and material.

    Returns:
        Tuple (J_val, T_nodes, entry) where:
            J_val:   Scalar thermal compliance.
            T_nodes: Temperature at input mesh nodes, shape (n_input_nodes,).
            entry:   Cache entry built by `_build_reduced_functionals`, now
                     evaluated at (rho, source).
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
    params = (k_max, p_exp)
    key = _setup_cache_key(*args, params=params)
    if key not in _SETUP_CACHE:
        _SETUP_CACHE.clear()
        _SETUP_CACHE[key] = _build_reduced_functionals(*args, *params)
        if not _ADJOINT_KERNELS_COMPILED:
            Chat = _SETUP_CACHE[key]["Chat"]
            for _ in range(2):
                Chat([c.control for c in Chat.controls])
                Chat.derivative()
            _ADJOINT_KERNELS_COMPILED = True
    entry = _SETUP_CACHE[key]
    fd_to_input_cells = entry["fd_to_input_cells"]

    rho_fn = Function(entry["dg0"])
    rho_fn.dat.data[:] = np.clip(rho_values[fd_to_input_cells], 0.0, 1.0)
    source_fn = Function(entry["dg0"])
    has_source = fd_to_input_cells < len(source_values)
    source_fn.dat.data[has_source] = source_values[fd_to_input_cells[has_source]]

    # Replaying the tape recomputes every block, so the temperature is read
    # back from the solve's output without a second replay.
    J = entry["Chat"]([rho_fn, source_fn])
    T_fd = entry["That"].functional.block_variable.saved_output.dat.data_ro
    input_to_fd = np.zeros(len(pts), dtype=np.int64)
    input_to_fd[entry["fd_to_input_nodes"]] = np.arange(len(T_fd))
    return float(J), T_fd[input_to_fd], entry


def _gradient_pair(
    reduced_functional: Any, entry: dict[str, Any], n_cells: int, adj_input: Any = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """(d/drho, d/dsource) from one adjoint sweep, mapped to input cell order."""
    d_rho, d_source = reduced_functional.derivative(adj_input=adj_input)
    fd_to_input_cells = entry["fd_to_input_cells"]
    d_rho_input = np.zeros(n_cells)
    d_rho_input[fd_to_input_cells] = d_rho.dat.data_ro
    d_source_input = np.zeros(n_cells)
    d_source_input[fd_to_input_cells] = d_source.dat.data_ro
    return d_rho_input, d_source_input


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve heat conduction and return compliance + identification error.

    Args:
        inputs: Validated InputSchema containing the density field, source,
                mesh, boundary conditions, and material parameters.

    Returns:
        OutputSchema with thermal_compliance (scalar) and identification_error (scalar).
    """
    hm = inputs.hex_mesh
    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float64)
    cells = np.asarray(hm.faces[: hm.n_faces], dtype=np.int64)
    rho_values = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float64)
    source_values = np.asarray(inputs.source[: hm.n_faces], dtype=np.float64)
    target_temp = np.asarray(inputs.target_temperature, dtype=np.float64)

    bc = inputs.boundary_conditions
    dm = np.asarray(bc.dirichlet.mask if bc.dirichlet else [])
    dv = np.asarray(
        bc.dirichlet.values
        if bc.dirichlet and bc.dirichlet.values is not None
        else np.zeros((0, 1)),
        dtype=np.float64,
    )
    vm = np.asarray(bc.neumann.mask if bc.neumann else [])
    vv = np.asarray(
        bc.neumann.values if bc.neumann else np.zeros((0, 1)), dtype=np.float64
    )

    J_val, T_nodes, _ = _replay_heat(
        rho_values,
        source_values,
        pts,
        cells,
        dm,
        dv,
        vm,
        vv,
        k_max=inputs.k_max,
        p_exp=inputs.p_exp,
    )

    n = min(len(T_nodes), len(target_temp))
    id_error = np.float32(np.sum((T_nodes[:n] - target_temp[:n]) ** 2))

    return OutputSchema(
        thermal_compliance=np.float32(J_val),
        identification_error=id_error,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via firedrake-adjoint: gradients of both objectives scaled by cotangents.

    Supports differentiation through:
        rho, source → thermal_compliance    (one adjoint sweep of ``Chat``)
        rho, source → identification_error  (one adjoint sweep of ``That``,
                                             seeded with dI/dT)

    Args:
        inputs: Validated InputSchema.
        vjp_inputs: Names of inputs for which gradients are requested.
        vjp_outputs: Names of outputs whose cotangents are provided.
        cotangent_vector: Dict of output-name → cotangent scalar/array.

    Returns:
        Dict with "rho" and/or "source" gradient arrays matching input shapes.
    """
    want_rho = "rho" in vjp_inputs
    want_source = "source" in vjp_inputs
    if not (want_rho or want_source):
        return {}

    hm = inputs.hex_mesh
    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float64)
    cells = np.asarray(hm.faces[: hm.n_faces], dtype=np.int64)
    rho_values = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float64)
    source_values = np.asarray(inputs.source[: hm.n_faces], dtype=np.float64)

    bc = inputs.boundary_conditions
    dm = np.asarray(bc.dirichlet.mask if bc.dirichlet else [])
    dv = np.asarray(
        bc.dirichlet.values
        if bc.dirichlet and bc.dirichlet.values is not None
        else np.zeros((0, 1)),
        dtype=np.float64,
    )
    vm = np.asarray(bc.neumann.mask if bc.neumann else [])
    vv = np.asarray(
        bc.neumann.values if bc.neumann else np.zeros((0, 1)), dtype=np.float64
    )

    _, _, entry = _replay_heat(
        rho_values,
        source_values,
        pts,
        cells,
        dm,
        dv,
        vm,
        vv,
        k_max=inputs.k_max,
        p_exp=inputs.p_exp,
    )

    n_cells = hm.n_faces
    grad_rho = np.zeros(n_cells)
    grad_source = np.zeros(n_cells)

    cot_compliance = float(cotangent_vector.get("thermal_compliance", 0.0))
    if cot_compliance != 0.0:
        dC_drho, dC_dsource = _gradient_pair(entry["Chat"], entry, n_cells)
        grad_rho += dC_drho * cot_compliance
        grad_source += dC_dsource * cot_compliance

    cot_id_error = float(cotangent_vector.get("identification_error", 0.0))
    if cot_id_error != 0.0:
        # I = Σ_nodes (T - T_target)², so the temperature adjoint is seeded
        # with dI/dT = 2(T - T_target) at the nodes that have a target value.
        T_tgt = np.asarray(inputs.target_temperature, dtype=np.float64)
        fd_to_input_nodes = entry["fd_to_input_nodes"]
        has_target = fd_to_input_nodes < len(T_tgt)
        T_fd = entry["That"].functional.block_variable.saved_output.dat.data_ro
        dI_dT = Cofunction(entry["V"].dual())
        dI_dT.dat.data[has_target] = 2.0 * (
            T_fd[has_target] - T_tgt[fd_to_input_nodes[has_target]]
        )
        dI_drho, dI_dsource = _gradient_pair(entry["That"], entry, n_cells, dI_dT)
        grad_rho += dI_drho * cot_id_error
        grad_source += dI_dsource * cot_id_error

    result: dict[str, Any] = {}
    if want_rho:
        result["rho"] = np.zeros(len(np.asarray(inputs.rho)), dtype=np.float32)
        result["rho"][:n_cells] = grad_rho.astype(np.float32)
    if want_source:
        result["source"] = np.zeros(len(np.asarray(inputs.source)), dtype=np.float32)
        result["source"][:n_cells] = grad_source.astype(np.float32)
    return result


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver."""
    return {
        "thermal_compliance": ShapeDType(shape=(), dtype="float32"),
        "identification_error": ShapeDType(shape=(), dtype="float32"),
    }
