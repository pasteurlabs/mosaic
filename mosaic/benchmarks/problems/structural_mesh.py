from __future__ import annotations

from pathlib import Path

import numpy as np

from mosaic.benchmarks.core.config import IcSpec, ProblemConfig, SolverSpec
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.mosaic_shared.types import HexMesh, MeshBC, MeshDirichletBC, MeshNeumannBC

_GYM_DIR = Path(__file__).parent.parent.parent
_TESSERACT_DIR = _GYM_DIR / "tesseracts" / "structural-mesh"

# SIMP material parameters — matched between solvers
_E_MAX = 70_000.0  # Young's modulus of solid [MPa]
_NU = 0.3  # Poisson's ratio
_XMIN = 1e-3  # Void stiffness ratio (E_min / E_max)


_SOLVERS: dict[str, SolverSpec] = {
    "jax_fem": SolverSpec(
        name="JAX-FEM",
        backend="jax",
        family="fem",
        differentiable=True,
        ad_strategy="hybrid",
        uses_gpu=True,
        internal_dtype="float32",
        dir="jax-fem",
        color="#4477AA",
        scheme="FEM HEX8 linear elasticity (SIMP, adjoint AD)",
        image_tag="jax_fem_structural_mesh:latest",
        description="JAX-FEM HEX8 linear-elasticity solver; automatic differentiation adjoint via JAX.",
    ),
    "topopt_jl": SolverSpec(
        name="TopOpt.jl",
        backend="julia",
        family="fem",
        differentiable=True,
        ad_strategy="adjoint",
        uses_gpu=False,
        internal_dtype="float64",
        dir="topopt-jl",
        color="#228833",
        scheme="FEM HEX8 linear elasticity (SIMP, analytical adjoint)",
        image_tag="topopt_jl_structural_grid:latest",
        description="TopOpt.jl HEX8 FEM solver; analytical SIMP adjoint implemented in Julia.",
        input_overrides={
            "E": _E_MAX,
            "nu": _NU,
            "xmin": _XMIN,
        },
    ),
    "dealii_structural": SolverSpec(
        name="deal.II",
        backend="dealii",
        family="fem",
        differentiable=False,
        uses_gpu=False,
        internal_dtype="float64",
        dir="dealii",
        color="#CCBB44",
        scheme="FEM Q1 linear elasticity (SIMP, forward-only)",
        image_tag="dealii_structural_mesh:latest",
        description=(
            "deal.II Q1 (trilinear hexahedral) FEM linear-elasticity solver via C++ subprocess. "
            "SIMP stiffness E(ρ) = xmin·E_max + (1−xmin)·E_max·ρ^penal. "
            "UMFPACK direct solver. Forward-only — reference C++ implementation for "
            "cross-framework validation."
        ),
        input_overrides={
            "E_max": _E_MAX,
            "nu": _NU,
            "xmin": _XMIN,
        },
    ),
    "fenics_structural": SolverSpec(
        name="FEniCS",
        backend="fenics",
        family="fem",
        differentiable=True,
        ad_strategy="adjoint",
        uses_gpu=False,
        internal_dtype="float64",
        dir="fenics",
        color="#AA3377",
        scheme="FEM CG1 linear elasticity (SIMP, dolfin-adjoint)",
        image_tag="fenics_structural_mesh:latest",
        description=(
            "FEniCS/DOLFIN 2019.1 CG1 FEM linear-elasticity solver with SIMP penalisation. "
            "Gradient ∂C/∂ρ via dolfin-adjoint ReducedFunctional. "
            "DG0 density field; compliance assembled as action(L, u_sol). "
            "Von Mises stress via DG0 projection of deviatoric stress norm."
        ),
        input_overrides={
            "E_max": _E_MAX,
            "nu": _NU,
            "xmin": _XMIN,
        },
    ),
    "firedrake_structural": SolverSpec(
        name="Firedrake",
        backend="firedrake",
        family="fem",
        differentiable=True,
        ad_strategy="adjoint",
        uses_gpu=False,
        internal_dtype="float64",
        dir="firedrake",
        color="#EE3377",
        scheme="FEM CG1 linear elasticity (SIMP, pyadjoint ReducedFunctional)",
        image_tag="firedrake_structural_mesh:latest",
        description=(
            "Firedrake CG1 FEM linear-elasticity solver with SIMP penalisation. "
            "Gradient ∂C/∂ρ via firedrake-adjoint (pyadjoint ReducedFunctional). "
            "Uses meshio for GMSH conversion of the HexMesh input. "
            "Modern Firedrake complement to the FEniCS fenics-structural solver."
        ),
        input_overrides={
            "E_max": _E_MAX,
            "nu": _NU,
            "xmin": _XMIN,
        },
    ),
}


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


def _cantilever_bcs(
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


# ── IC generator ─────────────────────────────────────────────────────────────


def _uniform(
    rho_0: float = 0.5,
    nx: int = 8,
    ny: int | None = None,
    nz: int | None = None,
    N: int | None = None,
    **_,
) -> np.ndarray:
    """Uniform density ρ₀ over the mesh, shape (nx·ny·nz,).

    Default geometry: ny=2 (thin slab), nz=nx//2.  The thin-y default gives an
    almost-2D cantilever — better for plotting and topology visualisation.

    N, if provided, overrides nx (resolution_sweep convention: N = nx).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = 2
    if nz is None:
        nz = max(1, nx // 2)
    return np.full((nx * ny * nz,), float(rho_0), dtype=np.float32)


def _random(
    rho_0: float = 0.5,
    noise: float = 0.3,
    nx: int = 8,
    ny: int | None = None,
    nz: int | None = None,
    N: int | None = None,
    seed: int = 0,
    **_,
) -> np.ndarray:
    """Gaussian-noise density field centred at ρ₀, clipped to [0.05, 0.95].

    Breaks spatial symmetry so fd_check and jacobian_svd see non-trivial
    per-cell gradients rather than a flat field identical across solvers.
    N, if provided, overrides nx (resolution_sweep convention).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = 2
    if nz is None:
        nz = max(1, nx // 2)
    rng = np.random.default_rng(seed)
    rho = rho_0 + noise * rng.standard_normal(nx * ny * nz).astype(np.float32)
    return np.clip(rho, 0.05, 0.95).astype(np.float32)


def _two_density_bumps(
    nx: int = 16,
    ny: int | None = None,
    nz: int | None = None,
    N: int | None = None,
    rho_bg: float = 0.1,
    rho_peak: float = 0.95,
    sigma: float = 0.12,
    Lx: float = 2.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    **_,
) -> np.ndarray:
    """Ground-truth density with two stiff Gaussian bumps on a soft background.

    Two load-bearing ``rho_peak`` pillars of width σ·min(Lx,Lz) centred at
    (0.35·Lx, 0.5·Ly, 0.5·Lz) and (0.75·Lx, 0.5·Ly, 0.5·Lz); the background is
    ``rho_bg`` everywhere else. Clipped to [0.05, 1.0].

    Direct analog of thermal-mesh ``_two_gaussians`` — a spatially concentrated
    ground-truth field whose effect on the observed output (displacement here,
    temperature there) can be probed via gradient-based recovery. Recovering
    this density pattern from boundary-load displacement observations is the
    structural version of source-field recovery.

    Returned shape: (nx·ny·nz,), canonical z-y-x ravel (same as ``_hex_mesh_arrays``).
    """
    if N is not None:
        nx = N
    if ny is None:
        ny = 2
    if nz is None:
        nz = max(1, nx // 2)

    dx = Lx / nx
    dz = Lz / nz
    xs = (np.arange(nx) + 0.5) * dx  # (nx,)
    zs = (np.arange(nz) + 0.5) * dz  # (nz,)

    # The canonical structural-mesh geometry is a thin slab (ny=2 by default),
    # so the two bumps are y-invariant (extruded across y): the Gaussian lives
    # on the (x, z) cross-section, tiled along y.  This mirrors thermal-mesh
    # ``_two_gaussians`` which is z-invariant on the quasi-2D nz=1 slab.
    Z2, X2 = np.meshgrid(zs, xs, indexing="ij")  # (nz, nx)
    width = sigma * min(Lx, Lz)
    inv2w2 = 1.0 / (2.0 * width * width)

    x1, z1 = 0.35 * Lx, 0.5 * Lz
    x2, z2 = 0.75 * Lx, 0.5 * Lz
    g1 = np.exp(-((X2 - x1) ** 2 + (Z2 - z1) ** 2) * inv2w2)
    g2 = np.exp(-((X2 - x2) ** 2 + (Z2 - z2) ** 2) * inv2w2)
    peak2d = (rho_peak - rho_bg) * np.maximum(g1, g2) + rho_bg  # (nz, nx)
    # Tile across y: (nz, ny, nx)
    peak_field = np.broadcast_to(peak2d[:, None, :], (nz, ny, nx)).copy()
    return np.clip(peak_field.ravel(), 0.05, 1.0).astype(np.float32)


# ── Input factory ─────────────────────────────────────────────────────────────


def _make_inputs(
    solver_name: str,
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
        solver_name=solver_name,
        F_total=F_total,
        corner_load=corner_load,
        corner_y_high=corner_y_high,
        corner_z_high=corner_z_high,
        load_axis=load_axis,
    )

    hex_mesh = HexMesh(
        points=points.astype(np.float32),
        faces=cells.astype(np.int32),
        n_points=int(len(points)),
        n_faces=int(len(cells)),
    )

    base = dict(
        rho=rho_data,
        hex_mesh=hex_mesh.model_dump(),
        boundary_conditions=bc.model_dump(),
    )
    return {**base, **_SOLVERS[solver_name].input_overrides}


# ── Field projection ──────────────────────────────────────────────────────────


def _infer_mesh_dims(n_cells: int) -> tuple[int, int, int]:
    """Infer (nx, ny, nz) from a flat cell count for the thin-slab canonical geometry.

    Canonical geometry: ny=2, nz=nx//2 → n_cells = nx * 2 * (nx//2) = nx²
    so nx = round(sqrt(n_cells)).
    """
    nx = round(float(n_cells) ** 0.5)
    ny = 2
    nz = max(1, nx // 2)
    return nx, ny, nz


def _density_to_2d(rho: np.ndarray, **_) -> np.ndarray:
    """Mid-y cross-section of per-cell density field → (nz, nx) image."""
    nx, ny, nz = _infer_mesh_dims(len(rho))
    return rho.reshape(nz, ny, nx)[:, ny // 2, :]  # (nz, nx)


# ── Diagnostics ───────────────────────────────────────────────────────────────


def _get_compliance(compliance: np.ndarray, **_) -> float:
    """Structural compliance C = F^T U (scalar)."""
    return float(compliance)


# ── Config instance ───────────────────────────────────────────────────────────

CONFIG = ProblemConfig(
    name="structural-mesh",
    description=(
        "3D linear-elasticity compliance minimisation on a cantilever beam with SIMP "
        "material penalisation (p=3, E_max=70 000 MPa). The stiffness matrix K(ρ) couples "
        "every density element to the global displacement field via the constitutive "
        "relation E_eff(ρ) = E_min + (E_max − E_min)·ρ³; the compliance objective "
        "C = F^T K(ρ)⁻¹ F is smooth but non-convex in ρ, driving gradient-based "
        "topology optimisation toward sparse binary 0/1 layouts."
    ),
    bc_description=(
        "3-D cantilever beam on domain [0,2]×[0,1]×[0,1] (HEX8 elements, 2:1:1 aspect). "
        "Dirichlet: all nodes at x=0 have zero displacement (clamped). "
        "Neumann: a prescribed total force is applied to the right face (x=2) — "
        "either a uniform downward traction or a concentrated upward corner load "
        "depending on the experiment (controlled by the corner_load flag)."
    ),
    tesseract_dir=_TESSERACT_DIR,
    output_key="compliance",
    ic_key="rho",
    solvers=_SOLVERS,
    make_ic={
        "uniform": IcSpec(
            fn=_uniform,
            description=(
                "Uniform SIMP material density ρ₀ over all hex mesh elements; standard "
                "homogeneous starting point for topology optimisation of the cantilever beam."
            ),
            plot_params={"rho_0": 0.5, "nx": 16},
        ),
        "random": IcSpec(
            fn=_random,
            description=(
                "Gaussian-noise density field centred at ρ₀=0.5 (σ=0.3, clipped to [0.05, 0.95]); "
                "breaks spatial symmetry so gradient experiments see non-trivial per-cell sensitivity."
            ),
            plot_params={},
        ),
        "two_density_bumps": IcSpec(
            fn=_two_density_bumps,
            description=(
                "Ground-truth density with two stiff Gaussian pillars (ρ_peak=0.95, σ=0.12·min(Lx,Lz)) "
                "at (0.35·Lx, 0.5·Ly, 0.5·Lz) and (0.75·Lx, 0.5·Ly, 0.5·Lz) on a soft ρ_bg=0.1 "
                "background; analog of thermal-mesh ``two_gaussians`` for the load-recovery inverse "
                "experiment (recover density from displacement observations)."
            ),
            plot_params={"nx": 16, "ny": 2, "nz": 8},
        ),
    },
    make_inputs=_make_inputs,
    error_fn=l2_error_rel,
    diagnostics={
        "compliance": _get_compliance,
        # Note: von_mises_stress is also returned by both solvers but computed via
        # different approximations (jax_fem: quadrature-averaged; topopt_jl: centroid
        # B-matrix), giving systematically different magnitudes. Compliance is the
        # primary physically-comparable scalar for the agreement benchmark.
    },
    extra_output_keys=[],
    analytic=None,
    domain_extent=2.0,
    field_to_2d=None,  # compliance is scalar; no 2D field projection
    ic_to_2d=_density_to_2d,  # mid-y cross-section of density field ρ
    field_cmap="hot",
    field_symmetric=False,
    diagnostic_fields=False,  # compliance is scalar; stress fields not directly comparable
    resolution_key="nx",
    n_to_cells=lambda N: N * 2 * max(1, N // 2),  # nx=N, ny=2, nz=N//2
    units={"rho_0": "–"},
    forward_defaults={
        "baseline": dict(
            description="Inter-solver compliance agreement sweep over mesh resolution N.",
            plot_description=(
                "Structural compliance C = F^T U vs mesh resolution N for each solver "
                "(uniform density ρ₀=0.5, full-face downward load). "
                "Both solvers implement HEX8 FEM; compliance should agree to <1% at all resolutions."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=8,
                        ny=2,
                        nz=4,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        F_total=1.0,
                        corner_load=False,
                    ),
                    sweep=dict(key="N", values=[4, 6, 8, 12, 16]),
                )
            ],
        ),
        "agreement": dict(
            description="Forward accuracy sweep across element density ρ₀ for each solver.",
            plot_description=(
                "Structural compliance C = F^T U vs element density ρ₀ for each solver "
                "(log-scale; full-face downward load). "
                "jax_fem uses surface traction; topopt_jl distributes force uniformly across "
                "right-face nodes — non-uniform shape-function weighting in jax_fem causes "
                "a small but consistent compliance difference (~0.5–3%) across all densities."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=8,
                        ny=2,
                        nz=4,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        F_total=1.0,
                        corner_load=False,
                    ),
                    sweep=dict(key="rho_0", values=[0.2, 0.4, 0.5, 0.7, 0.9]),
                )
            ],
        ),
        "physical_laws": dict(
            description="Structural compliance vs applied load F_total to verify quadratic scaling law C ∝ F².",
            plot_description=(
                "Structural compliance C = F^T U vs total load F_total at fixed N=8 (nx=8, ny=2, nz=4), ρ₀=0.5. "
                "For linear elasticity C = F^T K⁻¹ F ∝ F², so log-log slope must be 2.0. "
                "Deviations across solvers reveal errors in the stiffness assembly or force application."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=8,
                        ny=2,
                        nz=4,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        corner_load=False,
                        rho_0=0.5,
                    ),
                    sweep=dict(key="F_total", values=[0.25, 0.5, 1.0, 2.0, 4.0]),
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
                physics=dict(
                    Lx=2.0, Ly=1.0, Lz=1.0, F_total=1.0, rho_0=0.5, corner_load=False
                ),
                cost=dict(
                    N_values=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3200],
                    n_trials=3,
                ),
            )
        ],
    ),
    gradient_defaults={
        "fd_check": dict(
            description="FD gradient check vs analytic VJP at nominal mesh and load.",
            plot_description="U-curves (FD gradient error vs ε), direction cosine between AD and FD gradient vectors, and gradient magnitude field panels.",
            runs=[
                dict(
                    ic=dict(name="random", seed=0),
                    physics=dict(
                        nx=8,
                        ny=2,
                        nz=4,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        F_total=1.0,
                        corner_load=True,
                    ),
                    fd=dict(
                        eps_values=[
                            2e0,
                            5e-1,
                            1e-1,
                            3e-2,
                            1e-2,
                            3e-3,
                            1e-3,
                            3e-4,
                            1e-4,
                        ],
                        n_dirs=6,
                    ),
                )
            ],
        ),
        "param_sweep": dict(
            description="Gradient quality vs element density ρ₀ at fixed mesh.",
            plot_description="Gradient norm, best-ε FD error, direction cosine, and U-curves vs element density ρ₀.",
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=8,
                        ny=2,
                        nz=4,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        F_total=1.0,
                        corner_load=True,
                    ),
                    fd=dict(
                        eps_values=[5e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4], n_dirs=6
                    ),
                    sweep=dict(key="rho_0", values=[0.2, 0.4, 0.6, 0.8]),
                )
            ],
        ),
        "jacobian_svd": dict(
            description="Jacobian SVD and gradient subspace analysis at nominal mesh.",
            plot_description=(
                "Singular-value spectrum of the stacked per-solver gradient matrix and "
                "pairwise cosine similarity between JAX-FEM and TopOpt.jl gradient directions. "
                "Both solvers implement the same SIMP adjoint so cosine similarity should be "
                "near 1; deviations indicate differing adjoint formulations or numerical precision."
            ),
            runs=[
                dict(
                    ic=dict(name="random", seed=0),
                    physics=dict(
                        nx=8,
                        ny=2,
                        nz=4,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        F_total=1.0,
                        corner_load=True,
                    ),
                    jacobian=dict(n_alphas=21, alpha_range=0.2),
                )
            ],
        ),
        "differentiability_table": dict(
            description="FD differentiability check for all array inputs and outputs.",
            plot_description=(
                "Differentiability table: status (ok/fail/not_differentiable) and relative FD error "
                "for each (input field, output field) pair. rho is the primary differentiable input; "
                "hex_mesh and boundary_conditions are non-differentiable structs."
            ),
            runs=[
                dict(
                    ic=dict(name="random", seed=0),
                    physics=dict(
                        nx=4,
                        ny=2,
                        nz=2,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        F_total=1.0,
                        corner_load=True,
                    ),
                    fd=dict(eps=1e-3, n_dirs=1),
                )
            ],
        ),
    },
    inverse_defaults={
        "topopt": dict(
            description="SIMP topology optimisation: minimise structural compliance under volume fraction constraint.",
            plot_description=(
                "SIMP topology optimisation on a 16×8×8 cantilever beam: minimise compliance "
                "C = F^T U subject to a 50% volume fraction constraint (Adam, lr=0.02). "
                "Density field evolves from uniform ρ=0.5 toward a binary 0/1 layout; "
                "both solvers converge to the same topology confirming consistent adjoint gradients."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=16,
                        ny=2,
                        nz=8,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        F_total=1.0,
                        corner_load=True,
                        v_frac=0.5,
                        compliance_key="compliance",
                        penalty_weight=50.0,
                        x_min=1e-3,
                        snap_interval=10,
                    ),
                    optim=dict(lr=5e-2, max_iters=2500, patience=100),
                )
            ],
        ),
        "topopt_bfgs": dict(
            description="SIMP topology optimisation with L-BFGS: minimise structural compliance under volume fraction constraint.",
            plot_description=(
                "SIMP topology optimisation on a 16×8×8 cantilever beam with L-BFGS: minimise compliance "
                "C = F^T U subject to a 50% volume fraction constraint."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=16,
                        ny=2,
                        nz=8,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        F_total=1.0,
                        corner_load=True,
                        v_frac=0.5,
                        compliance_key="compliance",
                        penalty_weight=50.0,
                        x_min=1e-3,
                        snap_interval=5,
                    ),
                    optim=dict(max_iters=100, patience=20),
                )
            ],
        ),
        "topopt_mma": dict(
            description="SIMP topology optimisation with MMA (nlopt LD_MMA): hard volume fraction constraint.",
            plot_description=(
                "SIMP topology optimisation on a 16×2×8 cantilever beam with MMA: minimise compliance "
                "C = F^T U subject to a hard 50% volume fraction inequality constraint."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=16,
                        ny=2,
                        nz=8,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        F_total=1.0,
                        corner_load=True,
                        v_frac=0.5,
                        compliance_key="compliance",
                        x_min=1e-3,
                        snap_interval=5,
                    ),
                    optim=dict(max_iters=200, patience=30),
                )
            ],
        ),
        "topopt_mma_fine": dict(
            description=(
                "SIMP topology optimisation with MMA on a fine 32×4×16 mesh (8× more elements than topopt_mma). "
                "Tests whether solver cross-solver agreement holds at higher resolution."
            ),
            plot_description=(
                "SIMP topology optimisation on a fine 32×4×16 cantilever beam with MMA: minimise compliance "
                "C = F^T U subject to a hard 50% volume fraction inequality constraint."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=32,
                        ny=4,
                        nz=16,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        F_total=1.0,
                        corner_load=True,
                        v_frac=0.5,
                        compliance_key="compliance",
                        x_min=1e-3,
                        snap_interval=10,
                    ),
                    optim=dict(max_iters=300, patience=40),
                )
            ],
        ),
        "topopt_mma_finest": dict(
            description=(
                "SIMP topology optimisation with MMA on a finest 64×8×32 mesh (64× more elements than topopt_mma, 8× more than topopt_mma_fine). "
                "Tests cross-solver agreement at near-production resolution."
            ),
            plot_description=(
                "SIMP topology optimisation on a finest 64×8×32 cantilever beam with MMA: minimise compliance "
                "C = F^T U subject to a hard 50% volume fraction inequality constraint."
            ),
            runs=[
                dict(
                    ic=dict(name="uniform", seed=0),
                    physics=dict(
                        nx=64,
                        ny=8,
                        nz=32,
                        Lx=2.0,
                        Ly=1.0,
                        Lz=1.0,
                        F_total=1.0,
                        corner_load=True,
                        v_frac=0.5,
                        compliance_key="compliance",
                        x_min=1e-3,
                        snap_interval=20,
                    ),
                    optim=dict(max_iters=400, patience=50),
                )
            ],
        ),
    },
)
