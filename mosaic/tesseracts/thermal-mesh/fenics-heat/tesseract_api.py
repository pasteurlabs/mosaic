# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F405

"""Thermal topology optimisation on an arbitrary hexahedral mesh.

Uses FEniCS (DOLFIN 2019.1.0) + dolfin-adjoint to solve steady-state heat
conduction with SIMP material interpolation and compute the exact adjoint
gradient of the thermal compliance objective.

CRITICAL import order: dolfin_adjoint must immediately follow dolfin so that
it can monkey-patch solve/assemble and record operations on the adjoint tape.
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
# pasteurlabs/mosaic#180). Force it to mumps too. Newer dolfin-adjoint
# checkouts fixed this upstream (SolveVarFormBlock now forwards
# `linear_solver` into the adjoint solve automatically) and no longer expose
# this compat shim, so skip the patch if it's absent.
try:
    import fenics_adjoint.types.compat as _fa_compat

    def _adjoint_linalg_solve_mumps(*args: Any, **kwargs: Any) -> Any:
        return _fa_compat.backend.solve(*args, "mumps")

    _fa_compat.linalg_solve = _adjoint_linalg_solve_mumps
except ImportError:
    pass

from mosaic_shared.problems.thermal_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.thermal_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from mosaic_shared.schema_types import make_differentiable

try:
    from pyadjoint import Block, create_overloaded_object
    from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating
except ImportError:
    from pyadjoint.block import Block
    from pyadjoint.overloaded_type import create_overloaded_object
    from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating
from pydantic import Field
from scipy.spatial import cKDTree
from tesseract_core.runtime import ShapeDType


class InputSchema(make_differentiable(_CanonicalInputSchema, ["rho", "source"])):
    """Inputs for FEniCS heat solver, extended with material parameters."""

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
    """FEniCS thermal solver output schema."""


# ---------------------------------------------------------------------------
# Mesh conversion helpers  (copied verbatim from fenics-brinkman)
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
    """Build FEniCS-cell-index → input-cell-index permutation via centroid matching.

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
# Neumann facet marker helper
# ---------------------------------------------------------------------------


def _mark_neumann_facets(mesh: Mesh, neumann_mask_vals: np.ndarray) -> MeshFunction:
    """Mark boundary facets by Neumann group from a per-node mask.

    A boundary facet is assigned group k if ALL of its vertices carry
    neumann_mask == k (with k > 0).  Facets on the Dirichlet boundary or
    interior facets remain unmarked (tag = 0).

    Args:
        mesh: FEniCS Mesh object.
        neumann_mask_vals: Integer array of length ≥ n_vertices.  Entry i gives
            the Neumann group (1-indexed) of vertex i; 0 means no flux.

    Returns:
        MeshFunction of size_t defined on facets, with tag k > 0 for every
        boundary facet whose vertices all belong to Neumann group k.
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
# Identification error as a tape-recorded operation
# ---------------------------------------------------------------------------
#
# Hex grids in legacy FEniCS does not support vertex quadrature, so
# the nodal sum Σ_v (T_v − T_target,v)² cannot be expressed as an integral
# that is divided by the lumped mass matrix. This works in DOLFINx.
# Instead we use the Pyadjoint blocking system to add the NodalSquaredError
# to the computational tape.
#
# A `Block` puts the reduction back on the tape. pyadjoint then owns the whole
# chain, so `Ihat.derivative()` gives d/drho and d/dsource with no hand-written
# adjoint: the recorded `SolveVarFormBlock` performs the adjoint solve (with
# the forward's homogenised BCs and its recorded `solver_parameters`, i.e.
# mumps via the fix at the top of this file) and UFL differentiates k(ρ)
# itself. The value and the gradient then come from one definition of the
# functional rather than two that have to be kept in step by hand.


def _nodal_residual(T: Function, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked residual mask · (T − target), in dof order."""
    if MPI.comm_world.size > 1:
        raise RuntimeError(
            "Nodal residual is not implemented for MPI parallel runs; "
            "run with a single process (mpirun -n 1) instead."
        )
    return mask * (T.vector().get_local() - target)


class NodalSquaredErrorBlock(Block):
    """Tape block for I(T) = Σ_v mask_v (T_v − target_v)², T in a CG1 space.

    `target` and `mask` are constants of the problem, not controls, so `T` is
    the block's only dependency. `mask` is 0 at nodes the caller gave no
    target value for, dropping them from the value and the adjoint seed alike.
    """

    def __init__(self, T: Function, target: np.ndarray, mask: np.ndarray) -> None:
        """Record `T` as the block's single dependency; target/mask are fixed."""
        super().__init__()
        self.add_dependency(T)
        self._target = target
        self._mask = mask

    def __str__(self) -> str:
        """Label used in tape visualisations."""
        return "NodalSquaredError"

    def recompute_component(
        self, inputs: list, block_variable: Any, idx: int, prepared: Any = None
    ) -> Any:
        """Re-evaluate the nodal sum during a tape replay."""
        r = _nodal_residual(inputs[0], self._target, self._mask)
        return create_overloaded_object(float(np.dot(r, r)))

    def evaluate_adj_component(
        self,
        inputs: list,
        adj_inputs: list,
        block_variable: Any,
        idx: int,
        prepared: Any = None,
    ) -> Any:
        """Adjoint seed dI/dT, handed on to the recorded solve block."""
        # dI/dT is the bare nodal residual — no mass matrix, matching the
        # nodal functional. Returned as a dual vector, as the other blocks'
        # `Function` adjoint components are.
        seed = inputs[0].vector().copy()
        seed.set_local(
            2.0
            * float(adj_inputs[0])
            * _nodal_residual(inputs[0], self._target, self._mask)
        )
        seed.apply("insert")
        return seed

    def evaluate_tlm_component(
        self,
        inputs: list,
        tlm_inputs: list,
        block_variable: Any,
        idx: int,
        prepared: Any = None,
    ) -> Any:
        """Directional derivative dI/dT · T_dot, for `taylor_test`."""
        T_dot = tlm_inputs[0]
        if T_dot is None:
            return None
        r = _nodal_residual(inputs[0], self._target, self._mask)
        return 2.0 * float(np.dot(r, T_dot.vector().get_local()))


def nodal_squared_error(T: Function, target: np.ndarray, mask: np.ndarray) -> Any:
    """Σ_v mask_v (T_v − target_v)², recorded on the tape as an AdjFloat."""
    annotate = annotate_tape()
    with stop_annotating():
        r = _nodal_residual(T, target, mask)
    output = create_overloaded_object(float(np.dot(r, r)))
    if annotate:
        block = NodalSquaredErrorBlock(T, target, mask)
        get_working_tape().add_block(block)
        block.add_output(output.block_variable)
    return output


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------
#
# `_SETUP_CACHE` holds a persistent, replayable thermal-compliance
# `ReducedFunctional` (`Chat`, with controls ``[rho, source]``) per (mesh, BC,
# material, target_temperature) combination, i.e. everything the problem
# depends on *except* rho/source.
# Building it does ONE real solve. Every later evaluation at a different
# rho/source — the entire point of a topology-optimisation or source-
# identification run, which calls `apply` thousands of times against the
# same mesh — calls `Chat([new_rho, new_source])` instead: dolfin-adjoint
# updates both controls' checkpoints and replays every already-recorded
# Block's `recompute()` in place (see `pyadjoint.ReducedFunctional.__call__`
# / `Tape.reset_blocks`). This reuses the compiled UFL forms and re-solves
# the *linear system* but skips reconstructing Constants/Measures/
# DirichletBCs/forms from Python and re-hitting FFC's JIT-compile cache
# lookup on every call. So rho/source are plain arguments here, never part
# of a cache key: the cached object is a functional that can be replayed at
# any (rho, source), not a solve at one point.
#
# `Chat` carries both controls, so `.derivative()` returns
# `[d/drho, d/dsource]` from a single adjoint sweep. The adjoint method's
# cost is ~independent of the number of controls, so this is one backward
# pass instead of two. `Ihat` is a second ReducedFunctional over the same
# tape and the same controls, for identification_error, and gives that
# functional's two gradients from its own single sweep; see
# `NodalSquaredErrorBlock` above for how a nodal (non-integral) functional
# gets onto the tape at all.
#
# `vector_jacobian_product` is self-contained: it replays
# `Chat([rho, source])` at the point it was handed and then takes the
# adjoint(s) it needs (dolfin-adjoint's derivative is defined "around the
# last supplied value of the control" — see
# `ReducedFunctional.derivative`). It deliberately does NOT reuse a forward
# solution cached by a preceding `apply` call, even though tesseract-jax
# always invokes the two back-to-back at the same point: every other
# differentiable solver in the suite re-evaluates the forward pass inside
# its own VJP endpoint (jax-fem calls `jax.vjp` fresh each time, and
# tesseract-core's shared `jax_recipes` residual cache is disabled
# everywhere), so reusing state across the two endpoints would make this
# solver's measured VJP cost incomparable with the rest of the benchmark.
#
# The source field is always an explicit tape-tracked control (even when
# every value is zero, which leaves the solution unchanged) so a gradient
# w.r.t. source is always well-defined from the one shared solve.

_SETUP_CACHE: dict[str, dict[str, Any]] = {}


def _setup_cache_key(
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    k_max: float,
    p_exp: float,
    target_temperature: np.ndarray,
) -> str:
    """Hash everything the Chat graph and the identification target depend on.

    That is everything except rho/source, which `Chat([rho, source])`
    replays the graph at.
    """
    h = hashlib.sha256()
    for arr in (
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
        target_temperature,
    ):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.array([k_max, p_exp], dtype=np.float64).tobytes())
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
    target_temperature: np.ndarray,
) -> dict[str, Any]:
    """One-time setup for a (mesh, BC, material, target) combination.

    Builds the mesh, function spaces, and ONE annotated forward solve with
    rho/source as controls, wrapped as two ReducedFunctionals sharing that
    tape.

    Solves the 3-D steady-state heat conduction topology optimisation problem:
        -∇·(k(ρ) ∇T) = 0    in Ω

    with SIMP conductivity:
        k(ρ) = k_min + (k_max − k_min) · ρ^p    (k_min = 1e-3 · k_max)

    Boundary conditions:
        T = T_prescribed                  on Γ_D  (Dirichlet groups)
        k(ρ) ∇T · n = q_n               on Γ_N  (Neumann groups)

    Thermal compliance objective:
        C = ∮_Γ_N q_n · T dΓ

    Identification-error objective:
        I = Σ_nodes (T - T_target)²

    Returns:
        ``{"Chat", "Ihat", "I_err", "rho_space", "fenics_to_input", "mesh",
        "T_sol"}``; see the module docstring above for how these get reused at
        every later (rho, source).
    """
    mesh = _build_fenics_mesh(pts, cells)
    fenics_to_input = _cell_reorder_map(pts, cells, mesh)

    # P1 (CG degree 1) for temperature; DG0 for piecewise-constant density.
    V = FunctionSpace(mesh, "CG", 1)
    DG0 = FunctionSpace(mesh, "DG", 0)
    neumann_facet_markers = _mark_neumann_facets(mesh, neumann_mask_vals)
    # DOLFIN 2019.1.0 requires a facet MeshFunction (not vertex) for
    # DirichletBC.  A boundary facet is assigned Dirichlet group k if ALL
    # of its vertices carry dirichlet_mask == k.  This is identical to the
    # Neumann facet marking logic.
    dirichlet_facet_markers = _mark_neumann_facets(mesh, dirichlet_mask_vals)

    set_working_tape(Tape())

    # ---- Density & source fields (arbitrary initial values — every `apply`
    # replays this graph at the real (rho, source) via `Chat([rho, source])`)
    rho_fn = Function(DG0, name="rho")
    rho_fn.vector()[:] = 0.5
    source_fn = Function(DG0, name="source")
    source_fn.vector()[:] = 0.0

    # ---- SIMP conductivity ------------------------------------------------
    # k(ρ) = k_min + (k_max − k_min) · ρ^p,  k_min = 1e-3 · k_max
    k_min = Constant(1e-3 * k_max)
    k_simp = k_min + (Constant(k_max) - k_min) * rho_fn**p_exp

    ds_N = Measure("ds", domain=mesh, subdomain_data=neumann_facet_markers)

    # ---- Variational problem ----------------------------------------------
    T = TrialFunction(V)
    v = TestFunction(V)

    a = inner(k_simp * grad(T), grad(v)) * dx

    # Build Neumann right-hand side: sum of q_n · v integrated over each
    # Neumann group's facets.  Starting from a zero scalar form avoids the
    # need to special-case an empty Neumann set.
    n_neumann_groups = neumann_values_vals.shape[0]
    L = Constant(0.0) * v * dx
    for k in range(n_neumann_groups):
        q_n = Constant(float(neumann_values_vals[k, 0]))
        L = L + q_n * v * ds_N(k + 1)

    # Body heat source: ∫_Ω f · v dΩ.
    L = L + source_fn * v * dx

    # ---- Dirichlet BCs ---------------------------------------------------
    bcs = []
    for k in range(dirichlet_values_vals.shape[0]):
        T_prescribed = Constant(float(dirichlet_values_vals[k, 0]))
        bc = DirichletBC(V, T_prescribed, dirichlet_facet_markers, k + 1)
        bcs.append(bc)

    # ---- Solve -------------------------------------------------------------
    # mumps: UMFPACK (the FEniCS default) runs out of memory at moderate mesh
    # sizes; the adjoint solve picks this up too (see the mumps patch above).
    # This `solver_parameters` choice is captured on the Block and reused by
    # every later replay automatically.
    T_sol = Function(V)
    solve(a == L, T_sol, bcs, solver_parameters={"linear_solver": "mumps"})

    # ---- Objective: thermal compliance ------------------------------------
    # C = ∮_Γ_N q_n · T dΓ
    # assemble is monkey-patched by dolfin_adjoint and recorded on the tape.
    J_form = Constant(0.0) * T_sol * dx
    for k in range(n_neumann_groups):
        q_n = Constant(float(neumann_values_vals[k, 0]))
        J_form = J_form + q_n * T_sol * ds_N(k + 1)
    J = assemble(J_form)

    # ---- Objective: identification error -----------------------------------
    # I = Σ_v mask_v (T_v − T_target,v), recorded on the tape by
    # NodalSquaredErrorBlock. mask is 0 at nodes the caller supplied no target
    # for, so the value and the adjoint seed cover the same node set by
    # construction.
    d2v = dof_to_vertex_map(V)
    T_tgt = np.asarray(target_temperature, dtype=np.float64)
    has_target = d2v < len(T_tgt)
    target_at_dofs = np.zeros(V.dim(), dtype=np.float64)
    target_at_dofs[has_target] = T_tgt[d2v[has_target]]
    mask_at_dofs = has_target.astype(np.float64)

    I_err = nodal_squared_error(T_sol, target_at_dofs, mask_at_dofs)

    controls = [Control(rho_fn), Control(source_fn)]
    Chat = ReducedFunctional(J, controls)
    Ihat = ReducedFunctional(I_err, controls)

    return {
        "Chat": Chat,
        "Ihat": Ihat,
        "I_err": I_err,
        "rho_space": DG0,
        "T_sol": T_sol,
        "fenics_to_input": fenics_to_input,
        "mesh": mesh,
    }


def _get_reduced_functionals(
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    k_max: float,
    p_exp: float,
    target_temperature: np.ndarray,
) -> dict[str, Any]:
    """Build (or fetch) the cached Chat for this combination.

    Keyed on (mesh, BC, material, target_temperature).
    """
    key = _setup_cache_key(
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
        k_max,
        p_exp,
        target_temperature,
    )
    entry = _SETUP_CACHE.get(key)
    if entry is not None:
        return entry

    entry = _build_reduced_functionals(
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
        k_max,
        p_exp,
        target_temperature,
    )
    _SETUP_CACHE.clear()
    _SETUP_CACHE[key] = entry
    return entry


def _gradient_pair(
    reduced_functional: Any, n_input_cells: int, fenics_to_input: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(d/drho, d/dsource) from one adjoint sweep, mapped to input cell order."""
    d_rho_fn, d_source_fn = reduced_functional.derivative()
    d_rho = np.zeros(n_input_cells)
    d_rho[fenics_to_input] = d_rho_fn.vector().get_local()
    d_source = np.zeros(n_input_cells)
    d_source[fenics_to_input] = d_source_fn.vector().get_local()
    return d_rho, d_source


def _solve_forward(
    rho_values: np.ndarray,
    source_values: np.ndarray,
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    k_max: float,
    p_exp: float,
    target_temperature: np.ndarray,
) -> dict[str, Any]:
    """Evaluate both objectives at this (rho, source) by replaying the tape."""
    entry = _get_reduced_functionals(
        pts,
        cells,
        dirichlet_mask_vals,
        dirichlet_values_vals,
        neumann_mask_vals,
        neumann_values_vals,
        k_max,
        p_exp,
        target_temperature,
    )
    fenics_to_input = entry["fenics_to_input"]
    n_cells_f = len(fenics_to_input)

    rho_fn = Function(entry["rho_space"])
    rho_fn.vector()[:] = np.clip(rho_values[fenics_to_input], 0.0, 1.0)

    source_fn = Function(entry["rho_space"])
    src_reordered = np.zeros(n_cells_f, dtype=np.float64)
    for fd_idx in range(n_cells_f):
        inp_idx = int(fenics_to_input[fd_idx])
        if inp_idx < len(source_values):
            src_reordered[fd_idx] = float(source_values[inp_idx])
    source_fn.vector()[:] = src_reordered

    J = entry["Chat"]([rho_fn, source_fn])
    # `ReducedFunctional.__call__` replays EVERY block on the tape, not only
    # those the functional depends on, so the identification-error block has
    # been recomputed by that same call and both objectives come out of one
    # solve. Its value is read off the block variable rather than by calling
    # `Ihat([rho, source])`, which would replay — and re-solve — a second time.
    # (dolfin-adjoint's replay creates a fresh Function each time, see
    # `GenericSolveBlock._create_initial_guess`, and stores results on the
    # ORIGINAL block variables' checkpoints rather than mutating the setup
    # objects in place, so values must be read back through them.)
    I_err = float(entry["I_err"].block_variable.saved_output)

    return {
        "Chat": entry["Chat"],
        "Ihat": entry["Ihat"],
        "J": J,
        "I_err": I_err,
        "fenics_to_input": fenics_to_input,
        "n_input_cells": len(rho_values),
    }


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve heat conduction and return both objectives.

    Args:
        inputs: Validated InputSchema containing the density field, mesh,
                boundary conditions, and material parameters.

    Returns:
        OutputSchema with thermal_compliance and identification_error, both
        scalars, from a single replay of the tape.
    """
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

    state = _solve_forward(
        rho_values,
        source_values,
        pts,
        cells,
        dm,
        dv,
        vm,
        vv,
        inputs.k_max,
        inputs.p_exp,
        np.asarray(inputs.target_temperature, dtype=np.float64),
    )

    return OutputSchema(
        thermal_compliance=np.float32(float(state["J"])),
        identification_error=np.float32(state["I_err"]),
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via dolfin-adjoint: ∂C/∂ρ, ∂C/∂source, ∂id_err/∂ρ, ∂id_err/∂source.

    Self-contained, matching every other differentiable solver in the suite:
    replays the forward problem at the (rho, source) it is given and then
    computes the adjoint(s), rather than reusing a solution cached by a
    preceding ``apply`` call — see the module docstring above for why. The
    replay reuses the cached mesh/function spaces and the compiled UFL forms,
    so no mesh rebuild or form reconstruction happens. ``Chat`` carries both
    controls, so a single ``.derivative()`` sweep yields both ``d/drho`` and
    ``d/dsource``; ``Ihat`` carries the same two controls, so
    identification_error's pair comes from a second single sweep.

    Supports:
        rho    → thermal_compliance   (Chat.derivative())
        rho    → identification_error (Ihat.derivative())
        source → thermal_compliance   (Chat.derivative())
        source → identification_error (Ihat.derivative())

    Args:
        inputs: Validated InputSchema.
        vjp_inputs: Names of inputs for which gradients are requested.
        vjp_outputs: Names of outputs whose cotangents are provided.
        cotangent_vector: Dict of output-name → cotangent scalar/array.

    Returns:
        Dict mapping requested input names to gradient arrays.
    """
    want_rho = "rho" in vjp_inputs
    want_source = "source" in vjp_inputs

    if not want_rho and not want_source:
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
    state = _solve_forward(
        rho_values,
        source_values,
        pts,
        cells,
        dm,
        dv,
        vm,
        vv,
        inputs.k_max,
        inputs.p_exp,
        np.asarray(inputs.target_temperature, dtype=np.float64),
    )

    n_input_cells = state["n_input_cells"]
    fenics_to_input = state["fenics_to_input"]
    result = {}
    grad_rho = (
        np.zeros(len(np.asarray(inputs.rho)), dtype=np.float32) if want_rho else None
    )
    grad_source = (
        np.zeros(len(np.asarray(inputs.source)), dtype=np.float32)
        if want_source
        else None
    )

    # Compliance: one adjoint sweep serves both controls — the ReducedFunctional
    # carries [rho, source], so `.derivative()` yields d/drho and d/dsource at
    # once. (Identification error takes its own sweep over the same tape
    # below; each `.derivative()` calls `Tape.reset_variables()` first, so the
    # two do not contaminate one another.)
    cot_compliance = float(cotangent_vector.get("thermal_compliance", 0.0))
    if cot_compliance != 0.0:
        dC_drho, dC_dsource = _gradient_pair(
            state["Chat"], n_input_cells, fenics_to_input
        )
        if want_rho:
            grad_rho[: hm.n_faces] += (dC_drho * cot_compliance).astype(np.float32)
        if want_source:
            grad_source[: hm.n_faces] += (dC_dsource * cot_compliance).astype(
                np.float32
            )

    cot_id_error = float(cotangent_vector.get("identification_error", 0.0))
    if cot_id_error != 0.0:
        dI_drho, dI_dsource = _gradient_pair(
            state["Ihat"], n_input_cells, fenics_to_input
        )
        if want_rho:
            grad_rho[: hm.n_faces] += (dI_drho * cot_id_error).astype(np.float32)
        if want_source:
            grad_source[: hm.n_faces] += (dI_dsource * cot_id_error).astype(np.float32)

    if want_rho:
        result["rho"] = grad_rho
    if want_source:
        result["source"] = grad_source

    return result


def abstract_eval(abstract_inputs: InputSchema) -> dict:
    """Shape inference without running the solver."""
    return {
        "thermal_compliance": ShapeDType(shape=(), dtype="float32"),
        "identification_error": ShapeDType(shape=(), dtype="float32"),
    }
