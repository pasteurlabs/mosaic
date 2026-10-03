# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F405

"""Linear elasticity topology optimisation on an arbitrary hexahedral mesh.

Uses FEniCS (DOLFIN 2019.1.0) + dolfin-adjoint to solve 3-D linear elasticity
with SIMP material interpolation and compute the exact adjoint gradient of the
structural compliance objective.

CRITICAL import order: dolfin_adjoint must immediately follow dolfin so that
it can monkey-patch solve/assemble and record operations on the adjoint tape.
This is fixed on DOLFIN-adjoint main branch.
"""

import hashlib
import os
import tempfile
from typing import Any

import meshio
import numpy as np
from dolfin import *  # noqa: F403
from dolfin_adjoint import *  # noqa: F403

# Legacy dolfin-adjoint (e.g. dolfin-adjoint==2019.1.0 from conda-forge, as
# pinned in tesseract_environment.yaml) does not propagate the forward
# solve's `solver_parameters` to the adjoint linear solve, which then
# defaults to UMFPACK and runs out of memory well before mumps would (see
# pasteurlabs/mosaic#180).

try:
    import fenics_adjoint.types.compat as _fa_compat

    def _adjoint_linalg_solve_mumps(*args: Any, **kwargs: Any) -> Any:
        return _fa_compat.backend.solve(*args, "mumps")

    _fa_compat.linalg_solve = _adjoint_linalg_solve_mumps
except ImportError:
    pass

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


class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho"])):
    """Inputs for FEniCS linear elasticity solver, extended with material parameters."""

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
    """FEniCS structural solver output schema."""


# ---------------------------------------------------------------------------
# Mesh conversion helpers  (copied verbatim from fenics-heat)
# ---------------------------------------------------------------------------


def _build_fenics_mesh(pts: np.ndarray, cells: np.ndarray) -> Mesh:
    """Convert numpy hex mesh arrays to a FEniCS Mesh via meshio XDMF.

    DOLFIN XML only supports triangles/tetrahedra; XDMF supports hexahedra.

    Args:
        pts: Node coordinates, shape (n_nodes, 3), float64.
        cells: Hex cell connectivity, shape (n_cells, 8), int64.

    Returns:
        FEniCS Mesh object.
    """
    mio_mesh = meshio.Mesh(
        points=pts.astype(np.float64),
        cells=[("hexahedron", cells.astype(np.int64))],
    )
    fd, xdmf_path = tempfile.mkstemp(suffix=".xdmf")
    os.close(fd)
    h5_path = xdmf_path.replace(".xdmf", ".h5")
    try:
        meshio.write(xdmf_path, mio_mesh, file_format="xdmf")
        mesh = Mesh()
        with XDMFFile(xdmf_path) as xf:
            xf.read(mesh)
    finally:
        for p in (xdmf_path, h5_path):
            if os.path.exists(p):
                os.unlink(p)
    return mesh


def _cell_reorder_map(
    pts: np.ndarray, input_cells: np.ndarray, fenics_mesh: Mesh
) -> np.ndarray:
    """Build FEniCS-cell-index to input-cell-index permutation via centroid matching.

    FEniCS may reorder cells when loading a mesh. This function recovers the
    mapping so that rho_values[input_idx] can be assigned to the correct
    FEniCS DG0 DOF, and the adjoint gradient can be mapped back.

    Args:
        pts: Input mesh node coordinates, shape (n_nodes, 3).
        input_cells: Input cell connectivity, shape (n_input_cells, 8).
        fenics_mesh: The FEniCS Mesh built from the same data.

    Returns:
        Array of shape (n_fenics_cells,) where entry j gives the input cell
        index that corresponds to FEniCS cell j.
    """
    input_centroids = pts[input_cells].mean(axis=1)  # (n_cells, 3)

    n_cells_f = fenics_mesh.num_cells()
    fenics_centroids = np.array(
        [Cell(fenics_mesh, i).midpoint().array() for i in range(n_cells_f)]
    )  # (n_cells_f, 3)

    tree = cKDTree(input_centroids)
    _, fenics_to_input = tree.query(fenics_centroids)
    return fenics_to_input


# ---------------------------------------------------------------------------
# Neumann / Dirichlet facet marker helper
# ---------------------------------------------------------------------------


def _mark_neumann_facets(mesh: Mesh, neumann_mask_vals: np.ndarray) -> MeshFunction:
    """Mark boundary facets by group from a per-node mask.

    A boundary facet is assigned group k if ALL of its vertices carry
    mask == k (with k > 0).  Facets on the opposite boundary or interior
    facets remain unmarked (tag = 0).

    Works for both Neumann (traction) and Dirichlet (displacement) groups.

    Args:
        mesh: FEniCS Mesh object.
        neumann_mask_vals: Integer array of length >= n_vertices.  Entry i gives
            the group (1-indexed) of vertex i; 0 means no BC.

    Returns:
        MeshFunction of size_t defined on facets, with tag k > 0 for every
        boundary facet whose vertices all belong to group k.
    """
    facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_markers.set_all(0)
    # Build facet-to-vertex connectivity (required before iterating).
    mesh.init(mesh.topology().dim() - 1, 0)
    for facet in facets(mesh):
        if not facet.exterior():
            continue
        verts = facet.entities(0)  # vertex global indices
        groups = [
            int(neumann_mask_vals[v]) for v in verts if v < len(neumann_mask_vals)
        ]
        if groups and len(set(groups)) == 1 and groups[0] > 0:
            facet_markers[facet.index()] = groups[0]
    return facet_markers


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------
#
# `_SETUP_CACHE` holds a persistent, replayable `ReducedFunctional` per
# (mesh, BC, material) combination — everything the problem depends on
# *except* rho.
# Building it does ONE real solve: mesh build, function
# spaces, facet marking, UFL form assembly and `solve()`. Every later
# evaluation at a different rho,  the entire point of a topology-optimisation,
# run calls `Jhat(new_rho)` instead.
# dolfin-adjoint updates the control's checkpoint and replays each
# already-recorded Block's `recompute()` in place.
# `vector_jacobian_product` is self-contained: it replays `Jhat(rho)` at the
# point it was handed and then takes `Jhat.derivative()` (dolfin-adjoint's
# derivative is defined "around the last supplied value of the control" —
# see `ReducedFunctional.derivative`). It deliberately does NOT reuse a
# forward solution cached by a preceding `apply` call, even though
# tesseract-jax always invokes the two back-to-back at the same point: every
# other differentiable solver in the suite re-evaluates the forward pass
# inside its own VJP endpoint (jax-fem calls `jax.vjp` fresh each time, and
# tesseract-core's shared `jax_recipes` residual cache is disabled
# everywhere), so reusing state across the two endpoints would make this
# solver's measured VJP cost incomparable with the rest of the benchmark.

_SETUP_CACHE: dict[str, dict[str, Any]] = {}


def _setup_cache_key(
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
) -> str:
    """Hash everything the ReducedFunctional's graph depends on.

    That is everything except rho, which `Jhat(rho)` replays the graph at.
    """
    h = hashlib.sha256()
    for arr in (
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
    ):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.array([E_max, nu, xmin, penal], dtype=np.float64).tobytes())
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
    """One-time setup for a (mesh, BC, material) combination.

    Builds the mesh, function spaces, and ONE annotated forward solve,
    wrapped as a `ReducedFunctional`.

    Solves 3-D linear elasticity topology optimisation problem:
        -div(sigma(u)) = 0    in Omega

    with SIMP stiffness:
        E(rho) = E_min + (E_max - E_min) * rho^penal    (E_min = xmin * E_max)

    Lame parameters:
        lambda = E(rho) * nu / ((1+nu)(1-2nu))
        mu     = E(rho) / (2*(1+nu))

    Boundary conditions:
        u = u_prescribed         on Gamma_D  (Dirichlet groups)
        sigma * n = t            on Gamma_N  (Neumann / traction groups)

    Structural compliance objective:
        C = F^T U = assemble(action(L, u_sol))

    Returns:
        ``{"Jhat", "rho_space", "fenics_to_input"}`` — see module docstring
        above for how ``Jhat`` gets reused at every later rho.
    """
    mesh = _build_fenics_mesh(pts, cells)
    fenics_to_input = _cell_reorder_map(pts, cells, mesh)

    # CG1 vector space for displacement; DG0 for piecewise-constant density.
    V = VectorFunctionSpace(mesh, "CG", 1)
    DG0 = FunctionSpace(mesh, "DG", 0)
    neumann_facet_markers = _mark_neumann_facets(mesh, neumann_mask_vals)
    dirichlet_facet_markers = _mark_neumann_facets(mesh, dirichlet_mask_vals)

    set_working_tape(Tape())

    # ---- Density field (arbitrary initial value — every `apply` replays
    # this graph at the real rho via `Jhat(rho)`) --------------------------
    rho_fn = Function(DG0, name="rho")
    rho_fn.vector()[:] = 0.5

    # ---- SIMP stiffness ---------------------------------------------------
    # E(rho) = E_min + (E_max - E_min) * rho^penal,  E_min = xmin * E_max
    E_min = Constant(xmin * E_max)
    E = E_min + (Constant(E_max) - E_min) * rho_fn**penal

    nu_c = Constant(nu)
    lam = E * nu_c / ((1 + nu_c) * (1 - 2 * nu_c))
    mu = E / (2 * (1 + nu_c))

    # ---- Strain / stress helpers -----------------------------------------
    def eps(w: Any) -> Any:
        return 0.5 * (grad(w) + grad(w).T)

    def sig(w: Any) -> Any:
        return lam * tr(eps(w)) * Identity(3) + 2 * mu * eps(w)

    # ---- Neumann facet markers and traction linear form ------------------
    ds_N = Measure("ds", domain=mesh, subdomain_data=neumann_facet_markers)

    u, v = TrialFunction(V), TestFunction(V)

    a = inner(sig(u), eps(v)) * dx

    n_neumann_groups = neumann_values_vals.shape[0]
    L = dot(Constant((0.0, 0.0, 0.0)), v) * dx  # zero initialiser
    for k in range(n_neumann_groups):
        t_k = Constant(tuple(float(x) for x in neumann_values_vals[k]))
        L = L + dot(t_k, v) * ds_N(k + 1)

    # ---- Dirichlet BCs ---------------------------------------------------
    bcs = []
    for k in range(dirichlet_values_vals.shape[0]):
        u_k = Constant(tuple(float(x) for x in dirichlet_values_vals[k]))
        bc = DirichletBC(V, u_k, dirichlet_facet_markers, k + 1)
        bcs.append(bc)

    # ---- Solve -------------------------------------------------------------
    # mumps: UMFPACK (the FEniCS default) runs out of memory at moderate mesh
    # sizes; the adjoint solve picks this up too (see the mumps patch above).
    # This `solver_parameters` choice is captured on the Block and reused by
    # every later `Jhat(rho)` replay automatically.
    u_sol = Function(V)
    solve(a == L, u_sol, bcs, solver_parameters={"linear_solver": "mumps"})

    # ---- Compliance: C = F^T U = assemble(action(L, u_sol)) -------------
    # action(L, u_sol) substitutes u_sol into the linear form L, giving the
    # scalar F^T U.  This keeps the compliance on the dolfin-adjoint tape.
    J = assemble(action(L, u_sol))

    Jhat = ReducedFunctional(J, Control(rho_fn))

    return {
        "Jhat": Jhat,
        "rho_space": DG0,
        "fenics_to_input": fenics_to_input,
    }


def _get_reduced_functional(
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
    """Build (or fetch) the cached `ReducedFunctional` for this combination.

    Keyed on (mesh, BC, material).
    """
    key = _setup_cache_key(
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
        E_max,
        nu,
        xmin,
        penal,
    )
    entry = _SETUP_CACHE.get(key)
    if entry is not None:
        return entry

    entry = _build_reduced_functional(
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
        E_max,
        nu,
        xmin,
        penal,
    )
    _SETUP_CACHE.clear()
    _SETUP_CACHE[key] = entry
    return entry


def _gradient_from_state(state: dict[str, Any]) -> np.ndarray:
    """Adjoint gradient dC/drho, around the point of the last `Jhat(rho)` call.

    No mesh rebuild, no form reconstruction, no forward re-solve: dolfin-
    adjoint's `ReducedFunctional.derivative()` walks the blocks already
    recorded on the cached tape, using whichever value was last replayed in.
    """
    dJ_fenics_vec = state["Jhat"].derivative().vector().get_local().copy()

    # Map FEniCS DG0 DOF order to input cell order.
    dJ_input = np.zeros(state["n_input_cells"])
    dJ_input[state["fenics_to_input"]] = dJ_fenics_vec
    return dJ_input


def _solve_forward(
    rho_values: np.ndarray,
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
    """Evaluate compliance at this rho by replaying the cached `Jhat`."""
    entry = _get_reduced_functional(
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
        E_max,
        nu,
        xmin,
        penal,
    )
    fenics_to_input = entry["fenics_to_input"]
    rho_fn = Function(entry["rho_space"])
    rho_fn.vector()[:] = np.clip(rho_values[fenics_to_input], 0.0, 1.0)

    J = entry["Jhat"](rho_fn)

    return {
        "Jhat": entry["Jhat"],
        "J": J,
        "fenics_to_input": fenics_to_input,
        "n_input_cells": len(rho_values),
    }


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve linear elasticity and return compliance.

    Args:
        inputs: Validated InputSchema containing the density field, mesh,
                boundary conditions, and material parameters.

    Returns:
        OutputSchema with compliance (scalar).
    """
    hm = inputs.hex_mesh
    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float64)
    cells = np.asarray(hm.faces[: hm.n_faces], dtype=np.int64)
    rho_values = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float64)

    bc = inputs.boundary_conditions
    dm = np.asarray(bc.dirichlet.mask if bc.dirichlet else [], dtype=np.int64)
    dv = np.asarray(
        bc.dirichlet.values
        if bc.dirichlet and bc.dirichlet.values is not None
        else np.zeros((1, 3)),  # shape (1,3): one group with zero displacement
        dtype=np.float64,
    )
    nm = np.asarray(bc.neumann.mask if bc.neumann else [], dtype=np.int64)
    nv = np.asarray(
        bc.neumann.values if bc.neumann else np.zeros((0, 3)), dtype=np.float64
    )

    state = _solve_forward(
        rho_values,
        pts,
        cells,
        dm,
        dv,
        nm,
        nv,
        inputs.E_max,
        inputs.nu,
        inputs.xmin,
        inputs.penal,
    )
    return OutputSchema(compliance=np.float32(float(state["J"])))


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via dolfin-adjoint: gradient of compliance objective.

    Self-contained, matching every other differentiable solver in the suite:
    replays the forward problem at the rho it is given (``Jhat(rho)``) and
    then computes the adjoint (``Jhat.derivative()``).
    The replay reuses the cached mesh/function spaces and the
    compiled UFL forms, so this is one forward linear solve plus one adjoint
    linear solve, with no mesh rebuild or form reconstruction.

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
    dm = np.asarray(bc.dirichlet.mask if bc.dirichlet else [], dtype=np.int64)
    dv = np.asarray(
        bc.dirichlet.values
        if bc.dirichlet and bc.dirichlet.values is not None
        else np.zeros((1, 3)),  # shape (1,3): one group with zero displacement
        dtype=np.float64,
    )
    nm = np.asarray(bc.neumann.mask if bc.neumann else [], dtype=np.int64)
    nv = np.asarray(
        bc.neumann.values if bc.neumann else np.zeros((0, 3)), dtype=np.float64
    )
    state = _solve_forward(
        rho_values,
        pts,
        cells,
        dm,
        dv,
        nm,
        nv,
        inputs.E_max,
        inputs.nu,
        inputs.xmin,
        inputs.penal,
    )

    dJ_drho = _gradient_from_state(state)
    grad_rho[: hm.n_faces] = (cot_c * dJ_drho).astype(np.float32)
    return {"rho": grad_rho}


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver."""
    return {"compliance": ShapeDType(shape=(), dtype="float32")}
