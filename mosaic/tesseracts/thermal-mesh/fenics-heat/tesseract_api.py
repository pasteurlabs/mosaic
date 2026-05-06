"""Thermal topology optimisation on an arbitrary hexahedral mesh.

Uses FEniCS (DOLFIN 2019.1.0) + dolfin-adjoint to solve steady-state heat
conduction with SIMP material interpolation and compute the exact adjoint
gradient of the thermal compliance objective.

CRITICAL import order: dolfin_adjoint must immediately follow dolfin so that
it can monkey-patch solve/assemble and record operations on the adjoint tape.
"""

import os
import tempfile
from typing import Any

import meshio
import numpy as np
from dolfin import *
from dolfin_adjoint import *
from mosaic_shared.problems.thermal_mesh import (
    InputSchema as _CanonicalInputSchema,
)
from mosaic_shared.problems.thermal_mesh import (
    OutputSchema as _CanonicalOutputSchema,
)
from pydantic import Field
from scipy.spatial import cKDTree
from tesseract_core.runtime import ShapeDType


class InputSchema(_CanonicalInputSchema):
    """Inputs for FEniCS heat solver, extended with material parameters."""

    k_max: float = Field(
        default=1.0,
        description="Maximum thermal conductivity (fully solid/conducting material).",
    )
    p_exp: float = Field(
        default=3.0,
        description="SIMP penalisation exponent p (k(ρ) = k_min + (k_max−k_min)·ρ^p).",
    )


class OutputSchema(_CanonicalOutputSchema):
    """Outputs for FEniCS heat solver (canonical interface)."""


# ---------------------------------------------------------------------------
# Mesh conversion helpers  (copied verbatim from fenics-brinkman)
# ---------------------------------------------------------------------------


def _build_fenics_mesh(pts: np.ndarray, cells: np.ndarray) -> Mesh:  # mosaic:init
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


def _cell_reorder_map(  # mosaic:util
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


def _mark_neumann_facets(
    mesh: Mesh, neumann_mask_vals: np.ndarray
) -> MeshFunction:  # mosaic:init
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
# Core solver
# ---------------------------------------------------------------------------


def _solve_heat(  # mosaic:physics
    rho_values: np.ndarray,
    pts: np.ndarray,
    cells: np.ndarray,
    dirichlet_mask_vals: np.ndarray,
    dirichlet_values_vals: np.ndarray,
    neumann_mask_vals: np.ndarray,
    neumann_values_vals: np.ndarray,
    k_max: float,
    p_exp: float,
    compute_gradient: bool = False,
    source_values: np.ndarray | None = None,
    compute_rho_id_gradient: bool = False,
    target_temperature: np.ndarray | None = None,
):
    """Solve the 3-D steady-state heat conduction topology optimisation problem.

    Solves:
        -∇·(k(ρ) ∇T) = 0    in Ω

    with SIMP conductivity:
        k(ρ) = k_min + (k_max − k_min) · ρ^p    (k_min = 1e-3 · k_max)

    Boundary conditions:
        T = T_prescribed                  on Γ_D  (Dirichlet groups)
        k(ρ) ∇T · n = q_n               on Γ_N  (Neumann groups)

    Thermal compliance objective:
        C = ∮_Γ_N q_n · T dΓ

    The gradient ∂C/∂ρ is computed via dolfin-adjoint's ReducedFunctional.

    Args:
        rho_values: Active per-cell density, shape (n_cells,), values in [0, 1].
        pts: Mesh node coordinates, shape (n_nodes, 3).
        cells: Hex cell connectivity, shape (n_cells, 8).
        dirichlet_mask_vals: Per-node Dirichlet group index, shape (n_nodes,).
        dirichlet_values_vals: Per-group prescribed temperature, shape (n_groups, 1).
        neumann_mask_vals: Per-node Neumann group index, shape (n_nodes,).
        neumann_values_vals: Per-group heat flux, shape (n_neumann_groups, 1).
        k_max: Maximum thermal conductivity.
        p_exp: SIMP penalisation exponent.
        compute_gradient: If True, compute ∂C/∂ρ via dolfin-adjoint.
        source_values: Per-cell volumetric heat source (W/m³), shape (n_cells,).
            Added to the FEM RHS as ∫_Ω f·v dΩ.  None or zeros → no body source.
        compute_rho_id_gradient: If True, compute ∂I/∂ρ (identification_error)
            via dolfin-adjoint.  Requires target_temperature.
        target_temperature: Nodal target temperature, shape (n_nodes,), used
            when compute_rho_id_gradient=True.

    Returns:
        Tuple (J_val, T_vertices, dJ_drho, dI_drho) where:
            J_val: Scalar thermal compliance value.
            T_vertices: Temperature at mesh vertices, shape (n_vertices,).
            dJ_drho: Gradient ∂C/∂ρ, shape (n_cells,), or None if
                     compute_gradient=False.
            dI_drho: Gradient ∂I/∂ρ, shape (n_cells,), or None if
                     compute_rho_id_gradient=False.
    """
    # Fresh tape for every solve — prevents stale gradient accumulation.
    set_working_tape(Tape())

    mesh = _build_fenics_mesh(pts, cells)
    fenics_to_input = _cell_reorder_map(pts, cells, mesh)

    # ---- Function spaces --------------------------------------------------
    # P1 (CG degree 1) for temperature; DG0 for piecewise-constant density.
    V = FunctionSpace(mesh, "CG", 1)
    DG0 = FunctionSpace(mesh, "DG", 0)

    # ---- Density field ----------------------------------------------------
    rho = Function(DG0, name="rho")
    rho_vec = np.clip(rho_values[fenics_to_input], 0.0, 1.0)
    rho.vector()[:] = rho_vec

    # ---- SIMP conductivity ------------------------------------------------
    # k(ρ) = k_min + (k_max − k_min) · ρ^p,  k_min = 1e-3 · k_max
    k_min = Constant(1e-3 * k_max)
    k_simp = k_min + (Constant(k_max) - k_min) * rho**p_exp

    # ---- Neumann facet markers -------------------------------------------
    facet_markers = _mark_neumann_facets(mesh, neumann_mask_vals)
    ds_N = Measure("ds", domain=mesh, subdomain_data=facet_markers)

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

    # Body heat source: ∫_Ω f · v dΩ
    # source_values is per-element (DG0); project to DG0 and add to RHS.
    if source_values is not None and np.any(source_values != 0.0):
        source_dg0 = Function(DG0, name="source")
        n_cells_f = mesh.num_cells()
        src_reordered = np.zeros(n_cells_f, dtype=np.float64)
        for fd_idx in range(n_cells_f):
            inp_idx = int(fenics_to_input[fd_idx])
            if inp_idx < len(source_values):
                src_reordered[fd_idx] = float(source_values[inp_idx])
        source_dg0.vector()[:] = src_reordered
        L = L + source_dg0 * v * dx

    # ---- Dirichlet BCs ---------------------------------------------------
    # DOLFIN 2019.1.0 requires a facet MeshFunction (not vertex) for
    # DirichletBC.  A boundary facet is assigned Dirichlet group k if ALL
    # of its vertices carry dirichlet_mask == k.  This is identical to the
    # Neumann facet marking logic.
    dirichlet_facet_markers = _mark_neumann_facets(mesh, dirichlet_mask_vals)

    bcs = []
    for k in range(dirichlet_values_vals.shape[0]):
        T_prescribed = Constant(float(dirichlet_values_vals[k, 0]))
        bc = DirichletBC(V, T_prescribed, dirichlet_facet_markers, k + 1)
        bcs.append(bc)

    # ---- Solve -----------------------------------------------------------
    T_sol = Function(V)
    solve(a == L, T_sol, bcs)

    # ---- Objective: thermal compliance ------------------------------------
    # C = ∮_Γ_N q_n · T dΓ
    # assemble is monkey-patched by dolfin_adjoint and recorded on the tape.
    J_form = Constant(0.0) * T_sol * dx
    for k in range(n_neumann_groups):
        q_n = Constant(float(neumann_values_vals[k, 0]))
        J_form = J_form + q_n * T_sol * ds_N(k + 1)
    J = assemble(J_form)

    # ---- Temperature at vertices -----------------------------------------
    # compute_vertex_values returns a flat array of length n_vertices.
    T_vertices = T_sol.compute_vertex_values(mesh)

    # ---- Gradient via adjoint --------------------------------------------
    dJ_drho = None
    if compute_gradient:
        Jhat = ReducedFunctional(J, Control(rho))
        dJ_fenics = Jhat.derivative()
        dJ_fenics_vec = dJ_fenics.vector().get_local().copy()

        # Map FEniCS DG0 DOF order → input cell order.
        dJ_input = np.zeros(len(rho_values))
        dJ_input[fenics_to_input] = dJ_fenics_vec
        dJ_drho = dJ_input

    # ---- Gradient of identification_error w.r.t. rho --------------------
    dI_drho = None
    if compute_rho_id_gradient and target_temperature is not None:
        # Fresh tape — rho2 must be created AFTER set_working_tape.
        set_working_tape(Tape())

        mesh2 = _build_fenics_mesh(pts, cells)
        fenics_to_input2 = _cell_reorder_map(pts, cells, mesh2)

        V2 = FunctionSpace(mesh2, "CG", 1)
        DG0_2 = FunctionSpace(mesh2, "DG", 0)

        rho2 = Function(DG0_2, name="rho2")
        rho_vec2 = np.clip(rho_values[fenics_to_input2], 0.0, 1.0)
        rho2.vector()[:] = rho_vec2

        k_min2 = Constant(1e-3 * k_max)
        k_simp2 = k_min2 + (Constant(k_max) - k_min2) * rho2**p_exp

        facet_markers2 = _mark_neumann_facets(mesh2, neumann_mask_vals)
        ds_N2 = Measure("ds", domain=mesh2, subdomain_data=facet_markers2)

        T2_trial = TrialFunction(V2)
        v2_test = TestFunction(V2)
        a2 = inner(k_simp2 * grad(T2_trial), grad(v2_test)) * dx
        n_neumann_groups2 = neumann_values_vals.shape[0]
        L2 = Constant(0.0) * v2_test * dx
        for k in range(n_neumann_groups2):
            q_n2 = Constant(float(neumann_values_vals[k, 0]))
            L2 = L2 + q_n2 * v2_test * ds_N2(k + 1)
        if source_values is not None and np.any(source_values != 0.0):
            source_dg0_2 = Function(DG0_2, name="source2")
            n_cells_f2 = mesh2.num_cells()
            src_reordered2 = np.zeros(n_cells_f2, dtype=np.float64)
            for fd_idx2 in range(n_cells_f2):
                inp_idx2 = int(fenics_to_input2[fd_idx2])
                if inp_idx2 < len(source_values):
                    src_reordered2[fd_idx2] = float(source_values[inp_idx2])
            source_dg0_2.vector()[:] = src_reordered2
            L2 = L2 + source_dg0_2 * v2_test * dx

        dirichlet_facet_markers2 = _mark_neumann_facets(mesh2, dirichlet_mask_vals)
        bcs2 = []
        for k in range(dirichlet_values_vals.shape[0]):
            T_prescribed2 = Constant(float(dirichlet_values_vals[k, 0]))
            bc2 = DirichletBC(V2, T_prescribed2, dirichlet_facet_markers2, k + 1)
            bcs2.append(bc2)

        T_sol2 = Function(V2)
        solve(a2 == L2, T_sol2, bcs2)

        # Build target-temperature P1 function in FEniCS DOF order.
        T_target_fn = Function(V2)
        T_tgt = np.asarray(target_temperature, dtype=np.float64)
        d2v2 = dof_to_vertex_map(V2)
        target_at_dofs2 = np.zeros(V2.dim(), dtype=np.float64)
        for dof_i in range(V2.dim()):
            vert_i = int(d2v2[dof_i])
            if vert_i < len(T_tgt):
                target_at_dofs2[dof_i] = float(T_tgt[vert_i])
        T_target_fn.vector()[:] = target_at_dofs2

        # Nodal correction: identification_error (forward) = sum(nodal diff^2),
        # while dolfin-adjoint differentiates ∫(T-T_t)² dΩ (area-weighted).
        coords2 = mesh2.coordinates()
        domain_vol2 = float(
            np.prod(
                [
                    coords2[:, i].max() - coords2[:, i].min()
                    for i in range(coords2.shape[1])
                ]
            )
        )
        n_nodes_mesh2 = mesh2.num_vertices()
        nodal_correction2 = float(n_nodes_mesh2) / domain_vol2

        diff2 = T_sol2 - T_target_fn
        I2 = assemble(inner(diff2, diff2) * dx)
        Ihat2 = ReducedFunctional(I2, Control(rho2))
        dI_fenics2 = Ihat2.derivative()
        dI_fenics_vec2 = dI_fenics2.vector().get_local().copy() * nodal_correction2

        dI_input2 = np.zeros(len(rho_values))
        dI_input2[fenics_to_input2] = dI_fenics_vec2
        dI_drho = dI_input2

    return float(J), T_vertices, dJ_drho, dI_drho


# ---------------------------------------------------------------------------
# Tesseract endpoints
# ---------------------------------------------------------------------------


def apply(inputs: InputSchema) -> OutputSchema:
    """Forward pass: solve heat conduction and return compliance + temperature.

    Args:
        inputs: Validated InputSchema containing the density field, mesh,
                boundary conditions, and material parameters.

    Returns:
        OutputSchema with thermal_compliance (scalar), temperature (n_vertices,),
        and identification_error (scalar).
    """
    hm = inputs.hex_mesh
    pts = np.asarray(hm.points[: hm.n_points], dtype=np.float64)
    cells = np.asarray(hm.faces[: hm.n_faces], dtype=np.int64)
    rho_values = np.asarray(inputs.rho[: hm.n_faces], dtype=np.float64)
    source_values = np.asarray(inputs.source[: hm.n_faces], dtype=np.float64)
    target_temp = np.asarray(inputs.target_temperature, dtype=np.float32)
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

    J_val, T_verts, _, _ = _solve_heat(
        rho_values,
        pts,
        cells,
        dm,
        dv,
        vm,
        vv,
        inputs.k_max,
        inputs.p_exp,
        compute_gradient=False,
        source_values=source_values,
    )

    T_f32 = T_verts.astype(np.float32)
    n = min(len(T_f32), len(target_temp))
    id_error = np.float32(np.sum((T_f32[:n] - target_temp[:n]) ** 2))

    return OutputSchema(
        thermal_compliance=np.float32(J_val),
        temperature=T_f32,
        identification_error=id_error,
    )


def vector_jacobian_product(  # mosaic:grad:rho,source
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """VJP via dolfin-adjoint: ∂C/∂ρ and ∂id_err/∂source scaled by cotangents.

    Re-runs the forward solve with gradient tracking enabled and uses
    dolfin-adjoint's ReducedFunctional to compute adjoint sensitivities.
    Results are scaled by incoming cotangents and padded to match the
    capacity of the full input arrays.

    Supports:
        rho    → thermal_compliance  (SIMP adjoint via dolfin-adjoint)
        rho    → identification_error (SIMP adjoint on ||T-T_target||² functional)
        source → identification_error (nodal L2 adjoint with area correction)

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

    result = {}

    # ------------------------------------------------------------------
    # rho → thermal_compliance and/or rho → identification_error gradient
    # ------------------------------------------------------------------
    # mosaic:grad:rho:adjoint
    if want_rho:
        cot_compliance_rho = float(cotangent_vector.get("thermal_compliance", 0.0))
        cot_id_error_rho = float(cotangent_vector.get("identification_error", 0.0))
        want_rho_id = want_rho and cot_id_error_rho != 0.0
        target_temp_rho = (
            np.asarray(inputs.target_temperature, dtype=np.float64)
            if want_rho_id
            else None
        )

        _, _, dJ_drho, dI_drho = _solve_heat(
            rho_values,
            pts,
            cells,
            dm,
            dv,
            vm,
            vv,
            inputs.k_max,
            inputs.p_exp,
            compute_gradient=cot_compliance_rho != 0.0,
            source_values=source_values,
            compute_rho_id_gradient=want_rho_id,
            target_temperature=target_temp_rho,
        )

        grad_rho = np.zeros(len(np.asarray(inputs.rho)), dtype=np.float32)
        if dJ_drho is not None and cot_compliance_rho != 0.0:
            grad_rho[: hm.n_faces] += (dJ_drho * cot_compliance_rho).astype(np.float32)
        if dI_drho is not None and cot_id_error_rho != 0.0:
            grad_rho[: hm.n_faces] += (dI_drho * cot_id_error_rho).astype(np.float32)
        result["rho"] = grad_rho

    # ------------------------------------------------------------------
    # source → identification_error  AND  source → thermal_compliance
    # ------------------------------------------------------------------
    # mosaic:grad:source:adjoint
    if want_source:
        cot_src = float(cotangent_vector.get("identification_error", 0.0))
        cot_tc = float(cotangent_vector.get("thermal_compliance", 0.0))

        grad_source = np.zeros(len(np.asarray(inputs.source)), dtype=np.float32)

        if cot_src != 0.0 or cot_tc != 0.0:
            # Fresh tape — source_dg0 must be created AFTER set_working_tape.
            set_working_tape(Tape())

            mesh = _build_fenics_mesh(pts, cells)
            fenics_to_input = _cell_reorder_map(pts, cells, mesh)

            V = FunctionSpace(mesh, "CG", 1)
            DG0 = FunctionSpace(mesh, "DG", 0)

            # ---- Density field (not a control here) ----------------------
            rho_fn = Function(DG0, name="rho")
            rho_vec = np.clip(rho_values[fenics_to_input], 0.0, 1.0)
            rho_fn.vector()[:] = rho_vec

            # ---- SIMP conductivity ----------------------------------------
            k_min = Constant(1e-3 * inputs.k_max)
            k_simp = k_min + (Constant(inputs.k_max) - k_min) * rho_fn**inputs.p_exp

            # ---- Neumann facet markers ------------------------------------
            facet_markers = _mark_neumann_facets(mesh, vm)
            ds_N = Measure("ds", domain=mesh, subdomain_data=facet_markers)

            # ---- Source field as the adjoint control ----------------------
            source_dg0 = Function(DG0, name="source")
            n_cells_f = mesh.num_cells()
            src_reordered = np.zeros(n_cells_f, dtype=np.float64)
            for fd_idx in range(n_cells_f):
                inp_idx = int(fenics_to_input[fd_idx])
                if inp_idx < len(source_values):
                    src_reordered[fd_idx] = float(source_values[inp_idx])
            source_dg0.vector()[:] = src_reordered
            source_ctrl = Control(source_dg0)

            # ---- Variational problem with source on the tape --------------
            T_trial = TrialFunction(V)
            v_test = TestFunction(V)

            a = inner(k_simp * grad(T_trial), grad(v_test)) * dx

            n_neumann_groups = vv.shape[0]
            L = Constant(0.0) * v_test * dx
            for k in range(n_neumann_groups):
                q_n = Constant(float(vv[k, 0]))
                L = L + q_n * v_test * ds_N(k + 1)
            L = L + source_dg0 * v_test * dx

            # ---- Dirichlet BCs -------------------------------------------
            dirichlet_facet_markers = _mark_neumann_facets(mesh, dm)
            bcs = []
            for k in range(dv.shape[0]):
                T_prescribed = Constant(float(dv[k, 0]))
                bc_obj = DirichletBC(V, T_prescribed, dirichlet_facet_markers, k + 1)
                bcs.append(bc_obj)

            # ---- Solve ---------------------------------------------------
            T_sol = Function(V)
            solve(a == L, T_sol, bcs)

            # ---- Nodal correction factor (matches firedrake fix) -----------
            # identification_error = sum((T_nodes - T_target)^2)  (nodal, no area)
            # J_id (dolfin-adjoint) = integral((T-T_t)^2 dΩ)     (area-weighted)
            # Correction: nodal_correction = n_nodes / domain_vol
            coords = mesh.coordinates()  # numpy array (n_vertices, 3)
            domain_vol = float(
                np.prod(
                    [
                        coords[:, i].max() - coords[:, i].min()
                        for i in range(coords.shape[1])
                    ]
                )
            )
            n_nodes_mesh = mesh.num_vertices()
            nodal_correction = float(n_nodes_mesh) / domain_vol

            # ---- Branch: ∂(identification_error)/∂source -----------------
            if cot_src != 0.0:
                # ---- Target temperature as P1 function -------------------
                target_temp = np.asarray(inputs.target_temperature, dtype=np.float64)
                T_target_fn = Function(V)
                # Map nodal target values: FEniCS vertex ordering may differ from input.
                # Use compute_vertex_values-style assignment via dof_to_vertex_map.
                d2v = dof_to_vertex_map(V)
                target_at_dofs = np.zeros(V.dim(), dtype=np.float64)
                for dof_i in range(V.dim()):
                    vert_i = int(d2v[dof_i])
                    if vert_i < len(target_temp):
                        target_at_dofs[dof_i] = float(target_temp[vert_i])
                T_target_fn.vector()[:] = target_at_dofs

                # ---- Identification error functional (Galerkin) -----------
                diff = T_sol - T_target_fn
                J_id = assemble(inner(diff, diff) * dx)

                # ---- Adjoint differentiation for identification_error -----
                Jhat_id = ReducedFunctional(J_id, source_ctrl)
                dJ_src_fenics = Jhat_id.derivative()
                dJ_src_vec = (
                    dJ_src_fenics.vector().get_local().copy() * nodal_correction
                )

                # ---- Map FEniCS DG0 DOF order → input cell order ---------
                dJ_src_input = np.zeros(hm.n_faces, dtype=np.float64)
                for fenics_cell in range(len(fenics_to_input)):
                    inp_cell = int(fenics_to_input[fenics_cell])
                    if inp_cell < hm.n_faces:
                        dJ_src_input[inp_cell] = dJ_src_vec[fenics_cell]

                grad_source[: hm.n_faces] += (dJ_src_input * cot_src).astype(np.float32)

            # ---- Branch: ∂(thermal_compliance)/∂source -------------------
            # thermal_compliance C = ∮_ΓN q_n T dΓ  (Neumann surface integral,
            # same functional used in _solve_heat).
            # Adjoint: K λ = ∂C/∂T = q_n δ_ΓN  (same as Neumann RHS vector)
            # → adjoint solution λ satisfies K λ = f_Neumann  (same RHS as primal
            #   when source is zero), i.e. λ = K^{-1} f_Neumann.
            # ∂C/∂source_e = λ^T · ∂f/∂source_e = sum_i(λ_i * vol_e / n_nodes)
            # Computed via dolfin-adjoint ReducedFunctional with J_c = ∮_ΓN q_n T dΓ.
            if cot_tc != 0.0:
                J_c_form = Constant(0.0) * T_sol * dx
                for k in range(n_neumann_groups):
                    q_n = Constant(float(vv[k, 0]))
                    J_c_form = J_c_form + q_n * T_sol * ds_N(k + 1)
                J_c = assemble(J_c_form)

                # ---- Adjoint differentiation for thermal_compliance -------
                Jhat_c = ReducedFunctional(J_c, source_ctrl)
                dJ_c_src_fenics = Jhat_c.derivative()
                dJ_c_src_vec = dJ_c_src_fenics.vector().get_local().copy()

                # ---- Map FEniCS DG0 DOF order → input cell order ---------
                dJ_c_src_input = np.zeros(hm.n_faces, dtype=np.float64)
                for fenics_cell in range(len(fenics_to_input)):
                    inp_cell = int(fenics_to_input[fenics_cell])
                    if inp_cell < hm.n_faces:
                        dJ_c_src_input[inp_cell] = dJ_c_src_vec[fenics_cell]

                grad_source[: hm.n_faces] += (dJ_c_src_input * cot_tc).astype(
                    np.float32
                )

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
