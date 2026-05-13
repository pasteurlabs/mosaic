"""Mesh, BC, input factory, and diagnostics for structural-mesh."""

from __future__ import annotations

import numpy as np

from mosaic.benchmarks.core.config import SolverSpec
from mosaic.tesseracts.tesseract_shared.types import (
    HexMesh,
    MeshBC,
    MeshDirichletBC,
    MeshNeumannBC,
)

# ── Mesh builder ──────────────────────────────────────────────────────────────


def _hex_mesh_arrays(
    nx: int,
    ny: int,
    nz: int,
    Lx: float = 2.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Structured hex mesh on [0,Lx]×[0,Ly]×[0,Lz].

    Returns:
        points: (n_nodes, 3) float32   node coordinates
        cells:  (n_cells, 8) int32     HEX8 connectivity, 0-based Abaqus ordering
    """
    xs = np.linspace(0.0, Lx, nx + 1, dtype=np.float32)
    ys = np.linspace(0.0, Ly, ny + 1, dtype=np.float32)
    zs = np.linspace(0.0, Lz, nz + 1, dtype=np.float32)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)  # (n_nodes, 3)

    def _nid(ix: int, iy: int, iz: int) -> int:
        return iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix

    cells = np.array(
        [
            [
                _nid(ix, iy, iz),
                _nid(ix + 1, iy, iz),
                _nid(ix + 1, iy + 1, iz),
                _nid(ix, iy + 1, iz),
                _nid(ix, iy, iz + 1),
                _nid(ix + 1, iy, iz + 1),
                _nid(ix + 1, iy + 1, iz + 1),
                _nid(ix, iy + 1, iz + 1),
            ]
            for iz in range(nz)
            for iy in range(ny)
            for ix in range(nx)
        ],
        dtype=np.int32,
    )

    return points, cells


def _cantilever_bcs(  # noqa: PLR0913 — physics/geometry knobs are intentionally individual kwargs
    points: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    Lx: float = 2.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    solver_name: str = "jax_fem",
    F_total: float = 1.0,
    corner_load: bool = False,
    corner_y_high: bool = False,
    corner_z_high: bool = False,
    load_axis: str = "z",
    **_,
) -> MeshBC:
    """Cantilever BCs: clamp left face, apply Neumann load on right side.

    Dirichlet: all nodes at x=0, zero displacement (group 1).

    corner_load=False:
      Full right face load in load_axis direction (−z default, or ±y).
      jax_fem   → traction F/(Ly·Lz) in load_axis direction
      topopt_jl → nodal force F/n_right in load_axis direction

    corner_load=True:
      One-element patch at a corner of the right face (x=Lx).
      corner_y_high / corner_z_high select which corner (default: y=0, z=0).
      load_axis="z": force ±z (sign based on corner_z_high).
      load_axis="y": force ±y (sign based on corner_y_high).
    """
    n_nodes = len(points)
    tol = 1e-6 * max(Lx, Ly, Lz)

    d_mask = np.zeros(n_nodes, dtype=np.int32)
    d_mask[points[:, 0] < tol] = 1

    n_mask = np.zeros(n_nodes, dtype=np.int32)

    # Solvers that apply Neumann BCs via surface integration in the weak form
    # (FEniCS, Firedrake, JAX-FEM, deal.II) need traction [force/area].
    # Node-force solvers (TopOpt.jl) split force evenly across right-face nodes.
    _traction_solvers = {
        "jax_fem",
        "fenics_structural",
        "firedrake_structural",
        "dealii_structural",
    }
    use_traction = solver_name in _traction_solvers

    if corner_load:
        dy = Ly / ny
        dz = Lz / nz
        y_cond = (
            (points[:, 1] > Ly - dy - tol)
            if corner_y_high
            else (points[:, 1] < dy + tol)
        )
        z_cond = (
            (points[:, 2] > Lz - dz - tol)
            if corner_z_high
            else (points[:, 2] < dz + tol)
        )
        corner = (points[:, 0] > Lx - tol) & y_cond & z_cond
        n_mask[corner] = 1
        n_corner = int(corner.sum())
        force = F_total / (dy * dz) if use_traction else F_total / n_corner
        if load_axis == "y":
            y_sign = -1.0 if corner_y_high else +1.0
            force_vec = np.array([[0.0, y_sign * force, 0.0]], dtype=np.float32)
        else:  # "z"
            z_sign = -1.0 if corner_z_high else +1.0
            force_vec = np.array([[0.0, 0.0, z_sign * force]], dtype=np.float32)
    else:
        right = points[:, 0] > (Lx - tol)
        n_mask[right] = 1
        n_right = int(right.sum())
        force = F_total / (Ly * Lz) if use_traction else F_total / n_right
        if load_axis == "y":
            force_vec = np.array([[0.0, force, 0.0]], dtype=np.float32)  # +y lateral
        else:  # "z"
            force_vec = np.array([[0.0, 0.0, -force]], dtype=np.float32)  # −z downward

    return MeshBC(
        dirichlet=MeshDirichletBC(mask=d_mask, values=None),
        neumann=MeshNeumannBC(mask=n_mask, values=force_vec),
    )


# ── Mesh-dim inference helper ─────────────────────────────────────────────────


def _infer_mesh_dims(n_cells: int) -> tuple[int, int, int]:
    """Infer (nx, ny, nz) from a flat cell count for the thin-slab canonical geometry.

    Canonical geometry: ny=2, nz=nx//2 → n_cells = nx * 2 * (nx//2) = nx²
    so nx = round(sqrt(n_cells)).
    """
    nx = round(float(n_cells) ** 0.5)
    ny = 2
    nz = max(1, nx // 2)
    return nx, ny, nz


# ── Input factory ─────────────────────────────────────────────────────────────


def make_inputs(  # noqa: PLR0913 — physics/geometry knobs are intentionally individual kwargs
    spec: SolverSpec,
    ic: np.ndarray,
    *,
    nx: int = 8,
    ny: int | None = None,
    nz: int | None = None,
    N: int | None = None,
    Lx: float = 2.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    F_total: float = 1.0,
    rho_0: float | None = None,
    corner_load: bool = False,
    corner_y_high: bool = False,
    corner_z_high: bool = False,
    load_axis: str = "z",
    **_,
) -> dict:
    """Build solver input dict from density IC and geometry parameters.

    rho_0, if provided, overrides ic with a uniform density field of that value.
    This allows run_agreement to sweep rho_0 while keeping a fixed IC shape.

    N, if provided, overrides nx (resolution_sweep convention: N = nx).

    When neither N nor explicit nx is given and rho_0 is None, infer nx from
    the IC shape using the canonical thin-slab geometry (ny=2, nz=nx//2).
    This ensures resolution_sweep passes the correct mesh to the solver even
    when the suite does not forward N through the physics kwargs.

    corner_load=True selects a single-element corner patch on the right face
    with an upward (+z) force instead of a full-face downward load.

    jax_fem   expects rho shape (n_cells, 1).
    topopt_jl expects rho shape (n_cells,).
    topopt_jl material params (E, nu, xmin) are injected via SolverSpec.input_overrides.
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = 2
    if nz is None:
        nz = max(1, nx // 2)

    # Infer mesh dims from IC when rho_0 is not specified and IC size disagrees
    # with the default nx×ny×nz.  This makes resolution_sweep work even when the
    # suite does not thread N through the physics kwargs into make_inputs.
    if rho_0 is None and ic is not None and len(ic) != nx * ny * nz:
        nx_inferred, ny_inferred, nz_inferred = _infer_mesh_dims(len(ic))
        # Only accept the inference if the inferred shape exactly matches.
        if nx_inferred * ny_inferred * nz_inferred == len(ic):
            nx, ny, nz = nx_inferred, ny_inferred, nz_inferred

    rho_data = (
        np.full((nx * ny * nz,), float(rho_0), dtype=np.float32)
        if rho_0 is not None
        else ic.astype(np.float32)
    )

    points, cells = _hex_mesh_arrays(nx, ny, nz, Lx, Ly, Lz)
    bc = _cantilever_bcs(
        points,
        nx,
        ny,
        nz,
        Lx,
        Ly,
        Lz,
        # ``_traction_solvers`` keys are slug-form ("jax_fem", "ins_jl"); pass
        # the canonical slug so the traction-vs-node-force dispatch matches.
        solver_name=spec.key,
        F_total=F_total,
        corner_load=corner_load,
        corner_y_high=corner_y_high,
        corner_z_high=corner_z_high,
        load_axis=load_axis,
    )

    hex_mesh = HexMesh(
        points=points.astype(np.float32),
        faces=cells.astype(np.int32),
        n_points=len(points),
        n_faces=len(cells),
    )

    base = {
        "rho": rho_data,
        "hex_mesh": hex_mesh.model_dump(),
        "boundary_conditions": bc.model_dump(),
    }
    return {**base, **spec.input_overrides}


# ── Diagnostics ───────────────────────────────────────────────────────────────


def _get_compliance(compliance: np.ndarray, **_) -> float:
    """Structural compliance C = F^T U (scalar)."""
    return float(compliance)


DIAGNOSTICS = {
    "compliance": _get_compliance,
    # Note: von_mises_stress is also returned by both solvers but computed via
    # different approximations (jax_fem: quadrature-averaged; topopt_jl: centroid
    # B-matrix), giving systematically different magnitudes. Compliance is the
    # primary physically-comparable scalar for the agreement benchmark.
}
