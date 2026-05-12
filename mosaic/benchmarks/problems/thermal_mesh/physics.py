"""Mesh / BC builders, reference FEM solve, input factory, and diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mosaic.benchmarks.core.config import SolverSpec
from mosaic.tesseracts.tesseract_shared.types import (
    HexMesh,
    MeshBC,
    MeshDirichletBC,
    MeshNeumannBC,
)

from .ics import _two_gaussians

# SIMP material parameters
_K_MAX = 1.0  # solid thermal conductivity
_P_EXP = 3.0  # SIMP penalisation exponent
_K_MIN_RATIO = 1e-3  # k_min / k_max


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


def _heated_block_bcs(
    points: np.ndarray,
    Lx: float = 2.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    Q_total: float = 1.0,
    hot_spot: bool = False,
) -> MeshBC:
    """Heated-block BCs: cold left face, heat flux on right face.

    Dirichlet: all nodes at x=0, T=0 (group 1).
    Neumann:   heat flux on right face (group 1).

    When hot_spot=False (default): uniform flux Q_total/(Ly·Lz) over the
    entire right face. Optimal SIMP solution is trivially uniform density;
    use this for forward/gradient benchmarks.

    When hot_spot=True: flux concentrated on the central 1/3 strip in y.
    For nz=1 (quasi-2D) the z-filter is dropped and the full slab thickness
    is used; for nz>1 the patch is also filtered to the central 1/3 in z.
    This breaks y-symmetry and drives topology optimisation toward non-trivial
    branching structures.
    """
    tol = 1e-6 * max(Lx, Ly, Lz)

    d_mask = np.zeros(len(points), dtype=np.int32)
    d_mask[points[:, 0] < tol] = 1

    n_mask = np.zeros(len(points), dtype=np.int32)
    on_right = points[:, 0] > (Lx - tol)
    if hot_spot:
        # Detect single-layer z (nz=1): only z=0 and z=Lz nodes exist on the
        # right face; neither falls in [Lz/3, 2Lz/3], so skip the z-filter.
        z_on_right = np.unique(np.round(points[on_right, 2], 8))
        single_z_layer = len(z_on_right) <= 2

        y_cond = (points[:, 1] >= Ly / 3.0 - tol) & (
            points[:, 1] <= 2.0 * Ly / 3.0 + tol
        )
        if single_z_layer:
            # 2D hot-spot: central 1/3 stripe in y, full slab depth in z
            spot_area = (Ly / 3.0) * Lz
            in_spot = on_right & y_cond
        else:
            spot_area = (Ly / 3.0) * (Lz / 3.0)
            z_cond = (points[:, 2] >= Lz / 3.0 - tol) & (
                points[:, 2] <= 2.0 * Lz / 3.0 + tol
            )
            in_spot = on_right & y_cond & z_cond
        q_n = float(Q_total) / spot_area
        n_mask[in_spot] = 1
    else:
        q_n = float(Q_total) / (Ly * Lz)
        n_mask[on_right] = 1

    return MeshBC(
        dirichlet=MeshDirichletBC(
            mask=d_mask,
            values=np.array([[0.0]], dtype=np.float32),  # T=0 explicitly
        ),
        neumann=MeshNeumannBC(
            mask=n_mask,
            values=np.array([[q_n]], dtype=np.float32),
        ),
    )


# ── Reference FEM solve (inverse-recovery ground truth) ──────────────────────


def _approx_target_temperature(
    rho: np.ndarray,
    source: np.ndarray,
    cells: np.ndarray,
    points: np.ndarray,
    bc_dict: dict,
    k_max: float = _K_MAX,
    p_exp: float = _P_EXP,
) -> np.ndarray:
    """Approximate target temperature using a simple FEM solve.

    Solves -div(k(rho)*grad(T)) = source with a lumped-mass source distribution
    to obtain a reference temperature field for inverse experiments.

    Returns nodal temperature array, shape (n_nodes,).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n_nodes = len(points)
    n_cells = len(cells)

    # Infer element sizes
    xs = np.unique(np.round(points[:, 0], 7))
    ys = np.unique(np.round(points[:, 1], 7))
    zs = np.unique(np.round(points[:, 2], 7))
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    dz = float(zs[1] - zs[0]) if len(zs) > 1 else 1.0
    vol_e = dx * dy * dz  # element volume

    # SIMP conductivity
    k_min = _K_MIN_RATIO * k_max
    rho_c = np.clip(rho, 0.0, 1.0).astype(np.float64)
    k_elem = k_min + (k_max - k_min) * rho_c**p_exp

    # Simple diagonal reference stiffness (Laplacian approximation)

    K_ref = _compute_K_ref_simple(dx, dy, dz)

    # Assemble stiffness matrix (COO)
    ii, jj = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    ii_f, jj_f = ii.ravel(), jj.ravel()
    K_ref_f = K_ref.ravel()

    rows = cells[:, ii_f].ravel()
    cols = cells[:, jj_f].ravel()
    vals = (k_elem[:, None] * K_ref_f[None, :]).ravel()
    K = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    # Neumann RHS from bc_dict
    n_bc = bc_dict.get("neumann") or {}
    mask_raw = np.asarray(n_bc.get("mask", []), dtype=np.int32)
    values_raw = n_bc.get("values")
    f = np.zeros(n_nodes, dtype=np.float64)
    if len(mask_raw) > 0 and values_raw is not None:
        mask = np.zeros(n_nodes, dtype=np.int32)
        mask[: len(mask_raw)] = mask_raw[:n_nodes]
        values_arr = np.asarray(values_raw, dtype=np.float64)
        # Distribute Neumann flux uniformly to marked nodes
        on_right = np.where(mask > 0)[0]
        x_max = points[:, 0].max()
        face_nodes_right = on_right[
            np.abs(points[on_right, 0] - x_max) < 1e-8 * x_max + 1e-10
        ]
        if len(face_nodes_right) > 0:
            q_n = float(values_arr[0, 0]) if values_arr.size > 0 else 0.0
            # Approximate: each right-face node gets q_n * dy * dz / 4 (shared by elements)
            per_node_flux = q_n * dy * dz / 4.0
            np.add.at(f, face_nodes_right, per_node_flux)

    # Source RHS: distribute source uniformly to cell nodes (lumped)
    source_f = np.asarray(source, dtype=np.float64)
    source_rhs = np.zeros(n_nodes, dtype=np.float64)
    for e in range(n_cells):
        np.add.at(source_rhs, cells[e], source_f[e] * vol_e / 8.0)
    f += source_rhs

    # Dirichlet BCs
    d_bc = bc_dict.get("dirichlet") or {}
    d_mask = np.asarray(d_bc.get("mask", []), dtype=np.int32)
    d_vals_raw = d_bc.get("values")
    d_vals = (
        np.asarray(d_vals_raw, dtype=np.float64)
        if d_vals_raw is not None
        else np.zeros((0, 1), dtype=np.float64)
    )
    if len(d_mask) > 0:
        constrained = np.where(d_mask[:n_nodes] > 0)[0]
        groups = d_mask[constrained] - 1
        T_prescribed = (
            d_vals[groups, 0] if d_vals.size > 0 else np.zeros(len(constrained))
        )
        T_full = np.zeros(n_nodes, dtype=np.float64)
        T_full[constrained] = T_prescribed
        f -= K @ T_full
        # Zero out constrained rows/cols
        K_coo = K.tocoo()
        keep = ~(np.isin(K_coo.row, constrained) | np.isin(K_coo.col, constrained))
        r_new = np.concatenate([K_coo.row[keep], constrained])
        c_new = np.concatenate([K_coo.col[keep], constrained])
        v_new = np.concatenate([K_coo.data[keep], np.ones(len(constrained))])
        K = sp.coo_matrix((v_new, (r_new, c_new)), shape=(n_nodes, n_nodes)).tocsr()
        f[constrained] = T_prescribed

    try:
        T_nodal = spla.spsolve(K.tocsc(), f)
    except Exception:
        T_nodal = np.zeros(n_nodes, dtype=np.float64)
    return T_nodal.astype(np.float32)


def _compute_K_ref_simple(dx: float, dy: float, dz: float) -> np.ndarray:
    """Reference HEX8 stiffness matrix via 2×2×2 Gauss quadrature."""
    # HEX8 reference element: 2x2x2 Gauss quadrature
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
    gp = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    gw = np.array([1.0, 1.0])
    Jinv = np.diag([2.0 / dx, 2.0 / dy, 2.0 / dz])
    detJ = (dx / 2.0) * (dy / 2.0) * (dz / 2.0)
    K_ref = np.zeros((8, 8), dtype=np.float64)
    for wi, xi in zip(gw, gp, strict=False):
        for wj, eta in zip(gw, gp, strict=False):
            for wk, zeta in zip(gw, gp, strict=False):
                xi_i, eta_i, zeta_i = (
                    _HEX8_NODES_REF[:, 0],
                    _HEX8_NODES_REF[:, 1],
                    _HEX8_NODES_REF[:, 2],
                )
                dN = np.zeros((8, 3), dtype=np.float64)
                dN[:, 0] = xi_i * (1.0 + eta_i * eta) * (1.0 + zeta_i * zeta) / 8.0
                dN[:, 1] = (1.0 + xi_i * xi) * eta_i * (1.0 + zeta_i * zeta) / 8.0
                dN[:, 2] = (1.0 + xi_i * xi) * (1.0 + eta_i * eta) * zeta_i / 8.0
                B = dN @ Jinv.T
                K_ref += (wi * wj * wk) * detJ * (B @ B.T)
    return K_ref


# ── Diagnostics ───────────────────────────────────────────────────────────────


def _get_thermal_compliance(thermal_compliance: np.ndarray, **_) -> float:
    """Thermal compliance C = ∮ q_n T dΓ (work done by heat flux on temperature field)."""
    return float(thermal_compliance)


# ── IC visualisation helper ──────────────────────────────────────────────────


def _density_to_2d(rho: np.ndarray, **_) -> np.ndarray:
    """Per-cell density → (ny, nx) image for the quasi-2D (nz=1) layout.

    Assumes nx = 2·ny (canonical 2:1 x:y aspect ratio) and nz = 1, so
    n_cells = 2·ny².  Returns shape (ny, nx) — the full x-y cross-section.
    """
    n_cells = len(rho)
    ny = max(1, round((n_cells / 2) ** 0.5))
    nx = max(1, n_cells // ny)
    return rho.reshape(ny, nx)


# ── Input factory ─────────────────────────────────────────────────────────────


def build_make_inputs(solvers: list[SolverSpec]) -> Callable:
    """Return a ``make_inputs(solver_name, ic, **physics) → dict`` closure.

    Captures the solver list so per-solver ``input_overrides`` can be merged
    into the final dict without importing :mod:`.config` (which would create
    a cycle, since ``config`` imports from this module).
    """
    spec_by_name = {s.name: s for s in solvers}

    def _make_inputs(  # noqa: PLR0913 — physics/geometry knobs are intentionally individual kwargs
        solver_name: str,
        ic: np.ndarray,
        *,
        nx: int = 8,
        ny: int | None = None,
        nz: int = 1,
        Lx: float = 2.0,
        Ly: float = 1.0,
        Lz: float = 1.0,
        Q_total: float = 1.0,
        rho_0: float | None = None,
        hot_spot: bool = False,
        N: int | None = None,
        ic_field: str = "rho",
        target_from_two_gaussians: bool = False,
        target_rho_from_two_gaussians: bool = False,
        **_,
    ) -> dict:
        """Build solver input dict from IC and geometry parameters.

        ic_field controls which input field the IC array is placed in:
          - "rho"    (default): topology-optimisation mode; ic is the density field.
          - "source": source-identification mode; ic is the volumetric heat source;
                      rho is set to a uniform rho_0 (default 0.5).

        rho_0, if provided, overrides the rho field with a uniform density of that value.
        hot_spot, if True, concentrates Neumann flux on the central 1/3 stripe in y
        (and also in z for nz > 1).
        target_from_two_gaussians, if True, computes target_temperature by running the
        two-Gaussian source through a reference FEM solve analytically.
        target_rho_from_two_gaussians, if True and ic_field != "source", computes
        target_temperature by solving with a two-Gaussian conductivity field (rho in
        [0, 1]) and zero volumetric source — used for conductivity-recovery experiments.

        N, if provided, overrides nx (resolution_sweep convention: N = nx).
        If ic does not match the expected (nx * ny * nz) shape, nx is inferred
        from ic.size to keep the mesh consistent with the density field.

        jax_fem      expects rho shape (n_cells, 1).
        fenics_heat  expects rho shape (n_cells,) and extra k_max/p_exp params
                     (injected via SolverSpec.input_overrides).
        """
        if N is not None:
            nx = N
        if ny is None:
            ny = max(1, nx // 2)

        n_cells_expected = nx * ny * nz

        if ic_field == "source":
            # Source-identification mode: ic is the source field; rho is uniform.
            n_cells_ic = int(ic.size)
            if n_cells_ic != n_cells_expected:
                nx = max(1, round((n_cells_ic * 2) ** 0.5))
                ny = max(1, nx // 2)
            source_data = ic.astype(np.float32)
            rho_val = float(rho_0) if rho_0 is not None else 0.5
            rho_data = np.full((nx * ny * nz,), rho_val, dtype=np.float32)
        else:
            # Topology-optimisation mode (ic_field == "rho"): ic is the density field.
            n_cells_ic = int(ic.size) if rho_0 is None else n_cells_expected
            if n_cells_ic != n_cells_expected:
                nx = max(1, round((n_cells_ic * 2) ** 0.5))
                ny = max(1, nx // 2)
            rho_data = (
                np.full((nx * ny * nz,), float(rho_0), dtype=np.float32)
                if rho_0 is not None
                else ic.astype(np.float32)
            )
            source_data = np.zeros(nx * ny * nz, dtype=np.float32)

        points, cells = _hex_mesh_arrays(nx, ny, nz, Lx, Ly, Lz)
        bc = _heated_block_bcs(points, Lx, Ly, Lz, Q_total=Q_total, hot_spot=hot_spot)

        n_nodes = len(points)
        if target_rho_from_two_gaussians and ic_field != "source":
            # Conductivity-recovery target: solve with two-Gaussian rho (zero source,
            # Neumann BC only) to get T_target.  The optimiser must recover this rho
            # from the resulting temperature observations.
            rho_gt = np.clip(
                _two_gaussians(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly), 0.0, 1.0
            ).astype(np.float32)
            target_temperature = _approx_target_temperature(
                rho_gt,
                np.zeros(nx * ny * nz, dtype=np.float32),
                cells,
                points,
                bc.model_dump(),
            )
        elif target_from_two_gaussians:
            # Source-recovery target: solve with two-Gaussian source at nominal rho.
            target_source = _two_gaussians(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly)
            target_temperature = _approx_target_temperature(
                rho_data, target_source, cells, points, bc.model_dump()
            )
        else:
            target_temperature = np.zeros(n_nodes, dtype=np.float32)

        hex_mesh = HexMesh(
            points=points.astype(np.float32),
            faces=cells.astype(np.int32),
            n_points=len(points),
            n_faces=len(cells),
        )

        base = {
            "rho": rho_data,
            "source": source_data,
            "target_temperature": target_temperature,
            "hex_mesh": hex_mesh.model_dump(),
            "boundary_conditions": bc.model_dump(),
        }
        spec = spec_by_name[solver_name]
        overrides = dict(spec.input_overrides)
        return {**base, **overrides}

    return _make_inputs


DIAGNOSTICS = {"thermal_compliance": _get_thermal_compliance}
