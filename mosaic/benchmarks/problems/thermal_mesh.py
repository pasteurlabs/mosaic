from __future__ import annotations

from pathlib import Path

import numpy as np

from mosaic.benchmarks.core.config import (
    IcSpec,
    ProblemConfig,
    SolverSpec,
    discover_solvers,
)
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.plots.solver_styles import apply_styles
from mosaic.mosaic_shared.types import HexMesh, MeshBC, MeshDirichletBC, MeshNeumannBC

_GYM_DIR = Path(__file__).parent.parent.parent
_TESSERACT_DIR = _GYM_DIR / "tesseracts" / "thermal-mesh"

# SIMP material parameters
_K_MAX = 1.0  # solid thermal conductivity
_P_EXP = 3.0  # SIMP penalisation exponent
_K_MIN_RATIO = 1e-3  # k_min / k_max


# ── Solver registry ──────────────────────────────────────────────────────────
# Solvers and per-solver metadata come from each tesseract's YAML; styling is
# applied from mosaic.benchmarks.plots.solver_styles. Only per-(solver, problem)
# material-parameter overrides are set here.

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_DIR)

# Preserve historical solver key.
_SOLVERS["torch_fem_thermal"] = _SOLVERS.pop("torch_fem")

apply_styles(_SOLVERS)


for _key in ("fenics_heat", "dealii_heat", "firedrake_heat", "torch_fem_thermal"):
    _SOLVERS[_key].input_overrides = {"k_max": _K_MAX, "p_exp": _P_EXP}


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


# ── IC generators ─────────────────────────────────────────────────────────────


def _zero_source(
    nx: int = 8,
    ny: int | None = None,
    nz: int = 1,
    N: int | None = None,
    **_,
) -> np.ndarray:
    """Zero volumetric source field, shape (nx·ny·nz,). Starting point for source recovery."""
    if N is not None:
        nx = N
    if ny is None:
        ny = max(1, nx // 2)
    return np.zeros((nx * ny * nz,), dtype=np.float32)


def _uniform(
    rho_0: float = 0.5,
    nx: int = 8,
    ny: int | None = None,
    nz: int = 1,
    N: int | None = None,
    **_,
) -> np.ndarray:
    """Uniform density ρ₀ over the mesh, shape (nx·ny·nz,).

    N, if provided, overrides nx (resolution_sweep convention: N = nx).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = max(1, nx // 2)
    return np.full((nx * ny * nz,), float(rho_0), dtype=np.float32)


def _random(
    rho_0: float = 0.5,
    noise: float = 0.3,
    nx: int = 8,
    ny: int | None = None,
    nz: int = 1,
    seed: int = 0,
    N: int | None = None,
    **_,
) -> np.ndarray:
    """Random density field centred at ρ₀ with Gaussian noise, clipped to [0.05, 0.95].

    Breaks spatial symmetry so that gradient visualisations show non-trivial
    per-cell sensitivity rather than a flat field.

    N, if provided, overrides nx (resolution_sweep convention: N = nx).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = max(1, nx // 2)
    rng = np.random.default_rng(seed)
    rho = rho_0 + noise * rng.standard_normal(nx * ny * nz).astype(np.float32)
    return np.clip(rho, 0.05, 0.95).astype(np.float32)


# ── Input factory ─────────────────────────────────────────────────────────────


# Canonical flat layout for per-cell fields (source, density, cell temperature):
#   shape (nz, ny, nx) with x innermost — i.e. ravel order is iz, iy, ix.
# Per-node fields (nodal temperature) follow (nz+1, ny+1, nx+1) with x innermost.
# Plot helpers (`benchmarks/plots/recovery.py::_reshape_canonical_2d`) reshape
# directly under this convention and imshow rows→y, cols→x with NO transpose.
def _gaussian_source(
    nx: int = 8,
    ny: int | None = None,
    nz: int = 1,
    N: int | None = None,
    amplitude: float = 1.0,
    cx: float = 0.5,
    cy: float = 0.5,
    sigma: float = 0.2,
    Lx: float = 2.0,
    Ly: float = 1.0,
    **_,
) -> np.ndarray:
    """Gaussian heat source centred at (cx*Lx, cy*Ly) with width sigma*min(Lx,Ly).

    Returns per-element source field, shape (nx*ny*nz,).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = max(1, nx // 2)

    dx = Lx / nx
    dy = Ly / ny
    # Element centres
    xs = (np.arange(nx) + 0.5) * dx  # (nx,)
    ys = (np.arange(ny) + 0.5) * dy  # (ny,)
    # Meshgrid: (nz, ny, nx) → ravel to (nx*ny*nz,) in z-y-x order
    # (matches _hex_mesh_arrays loop order: iz, iy, ix)
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # (ny, nx)
    x0 = cx * Lx
    y0 = cy * Ly
    width = sigma * min(Lx, Ly)
    G = amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * width**2))
    # Tile across z-layers
    G3d = np.tile(G[np.newaxis, :, :], (nz, 1, 1))  # (nz, ny, nx)
    return G3d.ravel().astype(np.float32)  # (nx*ny*nz,)


def _two_gaussians(
    nx: int = 8,
    ny: int | None = None,
    nz: int = 1,
    N: int | None = None,
    amplitude: float = 1.0,
    sigma: float = 0.15,
    Lx: float = 2.0,
    Ly: float = 1.0,
    **_,
) -> np.ndarray:
    """Two Gaussian sources at (0.3*Lx, 0.5*Ly) and (0.7*Lx, 0.5*Ly).

    Returns per-element source field, shape (nx*ny*nz,).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = max(1, nx // 2)

    s1 = _gaussian_source(
        nx=nx,
        ny=ny,
        nz=nz,
        amplitude=amplitude,
        cx=0.3,
        cy=0.5,
        sigma=sigma,
        Lx=Lx,
        Ly=Ly,
    )
    s2 = _gaussian_source(
        nx=nx,
        ny=ny,
        nz=nz,
        amplitude=amplitude,
        cx=0.7,
        cy=0.5,
        sigma=sigma,
        Lx=Lx,
        Ly=Ly,
    )
    return (s1 + s2).astype(np.float32)


def _make_inputs(
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
        n_points=int(len(points)),
        n_faces=int(len(cells)),
    )

    base = dict(
        rho=rho_data,
        source=source_data,
        target_temperature=target_temperature,
        hex_mesh=hex_mesh.model_dump(),
        boundary_conditions=bc.model_dump(),
    )
    overrides = dict(_SOLVERS[solver_name].input_overrides)
    return {**base, **overrides}


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
    for wi, xi in zip(gw, gp):
        for wj, eta in zip(gw, gp):
            for wk, zeta in zip(gw, gp):
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


# ── Config instance ───────────────────────────────────────────────────────────

CONFIG = ProblemConfig(
    name="thermal-mesh",
    category_label="Heat Conduction",
    description=(
        "Quasi-2D steady heat-conduction compliance minimisation on a heated slab with SIMP "
        "material penalisation (p=3). The effective conductivity k_eff(ρ) = k_min + (k_max − k_min)·ρ³ "
        "controls heat routing; the compliance C = ∮_Γ q_n·T dΓ is the work done by the heat flux "
        "on the temperature field. The hot-spot boundary condition (central 1/3 stripe) breaks "
        "y-symmetry and drives topology optimisation toward non-trivial branching structures. "
        "Also supports source-identification experiments: recover the volumetric heat source f(x) "
        "from temperature observations via the identification_error = ||T − T_target||² objective."
    ),
    bc_description=(
        "Quasi-2D heated slab on domain [0,2]×[0,1] (nz=1 HEX8 layer). "
        "Dirichlet: all nodes at x=0 held at T=0 (fixed temperature). "
        "Neumann (uniform): uniform heat flux Q_total over the right face (x=2). "
        "Neumann (hot-spot): flux concentrated on the central 1/3 stripe in y "
        "(Ly/3 ≤ y ≤ 2Ly/3) at the right face, driving non-trivial topology."
    ),
    tesseract_dir=_TESSERACT_DIR,
    output_key="thermal_compliance",
    ic_key="rho",
    solvers=_SOLVERS,
    make_ic={
        "uniform": IcSpec(
            fn=_uniform,
            description=(
                "Uniform SIMP thermal conductivity density ρ₀ over all hex mesh elements; "
                "standard homogeneous starting point for heat-conduction topology optimisation."
            ),
            plot_params={"rho_0": 0.5, "nx": 16, "ny": 8, "nz": 1},
        ),
        "random": IcSpec(
            fn=_random,
            description=(
                "Gaussian-noise density field centred at ρ₀=0.5 (σ=0.3, clipped to [0.05, 0.95]); "
                "breaks spatial symmetry to produce non-trivial per-cell gradient sensitivity maps."
            ),
            plot_params={
                "rho_0": 0.5,
                "noise": 0.3,
                "nx": 16,
                "ny": 8,
                "nz": 1,
                "seed": 0,
            },
        ),
        "gaussian_source": IcSpec(
            fn=_gaussian_source,
            description=(
                "Gaussian heat source centred at (cx·Lx, cy·Ly) = (0.5·Lx, 0.5·Ly) with "
                "width σ·min(Lx,Ly). Used as the control field for source-identification experiments "
                "(ic_field='source' in physics dict)."
            ),
            plot_params={
                "nx": 16,
                "ny": 8,
                "nz": 1,
                "amplitude": 1.0,
                "cx": 0.5,
                "cy": 0.5,
                "sigma": 0.2,
            },
        ),
        "zero_source": IcSpec(
            fn=_zero_source,
            description=(
                "Zero volumetric heat source; standard zero-initialisation for source-recovery experiments."
            ),
            plot_params={"nx": 16, "ny": 8, "nz": 1},
        ),
        "two_gaussians": IcSpec(
            fn=_two_gaussians,
            description=(
                "Two-Gaussian volumetric heat source at (0.3·Lx, 0.5·Ly) and (0.7·Lx, 0.5·Ly). "
                "Ground-truth source for source-recovery experiments."
            ),
            plot_params={"nx": 16, "ny": 8, "nz": 1},
        ),
    },
    make_inputs=_make_inputs,
    error_fn=l2_error_rel,
    diagnostics={
        "thermal_compliance": _get_thermal_compliance,
    },
    extra_output_keys=[],
    analytic=None,
    domain_extent=2.0,
    field_to_2d=None,
    ic_to_2d=_density_to_2d,
    field_cmap="hot",
    field_symmetric=False,
    diagnostic_fields=False,  # temperature shapes differ between solvers (per-node vs per-cell)
    resolution_key="nx",
    n_to_cells=lambda N: N * max(1, N // 2),  # nx=N, ny=N//2, nz=1
    units={"rho_0": "–"},
    forward_defaults={
        "baseline": dict(
            description=(
                "FV vs FEM discretisation divergence: thermal compliance vs mesh resolution N "
                "with heterogeneous random density. Non-uniform ρ stresses FV harmonic-mean vs "
                "FEM Galerkin conductivity interpolation; phase transition visible at coarse N."
            ),
            plot_description=(
                "Thermal compliance C vs mesh resolution N (nx=2,3,4,6,8,12,16,24; ny=nx//2; nz=1) "
                "with random density ρ~N(0.5,0.3) clipped to [0.05,0.95]. FV solvers diverge "
                "from FEM at coarse N due to harmonic-mean vs Galerkin conductivity interpolation; "
                "gap closes as O(h) with refinement. N=2–4 is the phase-transition regime."
            ),
            runs=[
                dict(
                    ic=dict(name="random", seed=0),
                    physics=dict(nz=1, Lx=2.0, Ly=1.0, Lz=1.0, Q_total=1.0),
                    sweep=dict(key="N", values=[2, 3, 4, 6, 8, 12, 16, 24]),
                )
            ],
        ),
        "agreement": dict(
            description=(
                "Near-void contrast test: thermal compliance vs element density ρ₀ sweeping "
                "from near-void (ρ=0.01) to near-solid (ρ=0.95). SIMP p=3 creates orders-of-magnitude "
                "conductivity contrast; FV and FEM handle near-void channels differently."
            ),
            plot_description=(
                "Thermal compliance C vs uniform element density ρ₀ ∈ [0.01, 0.95] at N=16. "
                "C ∝ ρ⁻³ due to SIMP (p=3); near-void (ρ→0) divergence between FV harmonic-mean "
                "and FEM Galerkin conductivity is the key discriminator. Log scale recommended."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=16, ny=8, nz=1, Lx=2.0, Ly=1.0, Lz=1.0, Q_total=1.0
                    ),
                    sweep=dict(
                        key="rho_0", values=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
                    ),
                )
            ],
        ),
        "physical_laws": dict(
            description=(
                "C ∝ Q² scaling law with hot-spot BC. Uniform flux would be trivially symmetric; "
                "hot-spot (central 1/3 stripe) breaks y-symmetry and tests flux-area handling."
            ),
            plot_description=(
                "Thermal compliance C vs total heat flux Q_total at fixed N=16, ρ₀=0.5, hot_spot=True. "
                "For a linear system C ∝ Q² (log-log slope 2.0). Hot-spot BC concentrates flux on "
                "central 1/3 stripe in y, breaking symmetry; deviations across solvers reveal "
                "errors in Neumann mask handling or compliance integral."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        N=16, nz=1, Lx=2.0, Ly=1.0, Lz=1.0, rho_0=0.5, hot_spot=True
                    ),
                    sweep=dict(key="Q_total", values=[0.25, 0.5, 1.0, 2.0, 4.0]),
                )
            ],
        ),
        "source_baseline": dict(
            description=(
                "Resolution convergence for source identification: temperature field L2 error vs N "
                "with a Gaussian source. Uses ic_field='source' so the IC is the heat-source field "
                "rather than the density."
            ),
            runs=[
                dict(
                    ic=dict(name="gaussian_source"),
                    physics=dict(
                        nz=1, Lx=2.0, Ly=1.0, Lz=1.0, rho_0=0.5, ic_field="source"
                    ),
                    sweep=dict(key="N", values=[4, 6, 8, 12, 16, 24]),
                )
            ],
        ),
        "source_linearity": dict(
            description=(
                "Linearity check: T(α·f) = α·T(f). Sweep source amplitude α. "
                "For a linear PDE the temperature scales linearly with the source amplitude."
            ),
            runs=[
                dict(
                    ic=dict(name="gaussian_source"),
                    physics=dict(
                        nx=16,
                        ny=8,
                        nz=1,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        rho_0=0.5,
                        ic_field="source",
                    ),
                    sweep=dict(key="amplitude", values=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0]),
                )
            ],
        ),
    },
    cost_defaults=dict(
        description="Wall-clock and memory profiling vs mesh size for all solvers.",
        plot_descriptions={
            "spatial_cost": "Forward-pass wall-clock time vs mesh size (nx) for all solvers.",
            "vjp_cost": "VJP wall-clock time vs mesh size (nx) for differentiable solvers.",
        },
        runs=[
            dict(
                physics=dict(Lx=2.0, Ly=1.0, Lz=1.0, Q_total=1.0, rho_0=0.5),
                cost=dict(
                    N_values=[16, 32, 64, 128, 256, 512, 1024, 2048, 4500], n_trials=3
                ),
            )
        ],
    ),
    gradient_defaults={
        "fd_check": dict(
            description="FD gradient check vs analytic VJP at nominal mesh and heat flux.",
            plot_description="U-curves (FD gradient error vs ε), direction cosine between AD and FD gradient vectors, and gradient magnitude field panels.",
            runs=[
                dict(
                    ic=dict(name="random", seed=0),
                    physics=dict(nx=8, ny=4, nz=1, Lx=2.0, Ly=1.0, Lz=1.0, Q_total=1.0),
                    fd=dict(eps_values=[1e0, 1e-1, 1e-2, 1e-3, 1e-4], n_dirs=6),
                )
            ],
        ),
        "param_sweep": dict(
            description="Gradient quality vs element density ρ₀ at fixed mesh.",
            plot_description="Gradient norm, best-ε FD error, direction cosine, and U-curves vs element density ρ₀.",
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(nx=8, ny=4, nz=1, Lx=2.0, Ly=1.0, Lz=1.0, Q_total=1.0),
                    fd=dict(eps_values=[1e0, 1e-1, 1e-2, 1e-3, 1e-4], n_dirs=6),
                    sweep=dict(key="rho_0", values=[0.1, 0.2, 0.4, 0.6, 0.8]),
                )
            ],
        ),
        "jacobian_svd": dict(
            description="Jacobian SVD and gradient subspace analysis at nominal mesh.",
            plot_description=(
                "Singular-value spectrum of the stacked per-solver gradient matrix and "
                "pairwise cosine similarity between JAX-FEM and FEniCS gradient directions "
                "for the thermal compliance objective. Near-unity cosine confirms consistent "
                "adjoint implementations; spectrum reveals dominant sensitivity modes of the "
                "density field."
            ),
            runs=[
                dict(
                    ic=dict(name="random", seed=0),
                    physics=dict(nx=8, ny=4, nz=1, Lx=2.0, Ly=1.0, Lz=1.0, Q_total=1.0),
                    jacobian=dict(n_alphas=21, alpha_range=0.2),
                )
            ],
        ),
        # Source-identification gradient experiments.
        # NOTE: these experiments use source as the differentiable input and
        # identification_error as the objective.  The global ic_key="rho" means
        # these experiments must be run with per-experiment ic_key override once
        # the harness supports it.  For now, physics dict carries ic_field="source"
        # to signal _make_inputs to set the source field from the IC, and the
        # caller must use output_key="identification_error" manually.
        "source_fd_check": dict(
            description=(
                "FD gradient check of d(identification_error)/d(source) at nominal mesh. "
                "Uses ic_field='source' and output_key='identification_error'."
            ),
            runs=[
                dict(
                    ic=dict(name="gaussian_source"),
                    ic_key="source",
                    output_key="identification_error",
                    physics=dict(
                        nx=8,
                        ny=4,
                        nz=1,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        rho_0=0.5,
                        target_from_two_gaussians=True,
                        ic_field="source",
                    ),
                    fd=dict(eps_values=[1e0, 1e-1, 1e-2, 1e-3, 1e-4], n_dirs=6),
                )
            ],
        ),
        "source_width_sweep": dict(
            description=(
                "Gradient quality vs source localisation σ. "
                "Phase transition: FEM/FD disagree as source narrows below element size."
            ),
            runs=[
                dict(
                    ic=dict(name="gaussian_source"),
                    ic_key="source",
                    output_key="identification_error",
                    physics=dict(
                        nx=16,
                        ny=8,
                        nz=1,
                        rho_0=0.5,
                        target_from_two_gaussians=True,
                        ic_field="source",
                    ),
                    fd=dict(eps_values=[1e-1, 1e-2, 1e-3, 1e-4], n_dirs=4),
                    sweep=dict(
                        key="sigma", values=[0.05, 0.1, 0.2, 0.3, 0.5], ic_sweep=True
                    ),
                )
            ],
        ),
    },
    inverse_defaults={
        "conductivity_recovery": dict(
            description=(
                "Recover a two-Gaussian conductivity field from temperature observations. "
                "Optimises rho (SIMP density, clipped to [x_min, 1]) to minimise "
                "identification_error = ||T(rho) - T_target||². Target temperature is "
                "produced by forward-solving with a two-Gaussian ground-truth conductivity "
                "at uniform zero volumetric source (driven by Neumann BC only)."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=16,
                        ny=8,
                        nz=1,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        rho_0=0.5,
                        Q_total=1.0,
                        compliance_key="identification_error",
                        penalty_weight=0.0,
                        x_min=1e-3,
                        snap_interval=20,
                        target_rho_from_two_gaussians=True,
                    ),
                    optim=dict(lr=1e-2, max_iters=2000, patience=200),
                )
            ],
        ),
        "conductivity_recovery_bfgs": dict(
            description=(
                "Recover a two-Gaussian conductivity field with L-BFGS. Same setup as "
                "conductivity_recovery but using L-BFGS with zoom line search."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=16,
                        ny=8,
                        nz=1,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        rho_0=0.5,
                        Q_total=1.0,
                        compliance_key="identification_error",
                        penalty_weight=0.0,
                        x_min=1e-3,
                        snap_interval=10,
                        target_rho_from_two_gaussians=True,
                    ),
                    optim=dict(max_iters=200, patience=30),
                )
            ],
        ),
    },
    status_checks={
        # Recovery / optimisation experiments must actually reduce loss,
        # not just complete. Same 50% floor as the other problems — solvers
        # landing at final/initial > 0.5 show up as anom so the status
        # accurately reflects "hasn't converged".
        "optimization": {"max_final_ratio": 0.5},
    },
)
