"""3D linear-elasticity SIMP topology optimisation on a cantilever beam.

The problem definition is split across three modules:

- :mod:`.ics`         — IC generators (``_uniform``, ``_random``,
                        ``_two_density_bumps``).
- :mod:`.physics`     — mesh and BC builders, input factory
                        (``make_inputs``), the ``_infer_mesh_dims``
                        helper, and the ``DIAGNOSTICS`` registry.
- :mod:`.optimization` — SIMP topology-optimisation runner.

This module performs solver discovery, the canonical :class:`Problem`
assembly, and the per-suite ``problem.add(...)`` calls with inline plot
descriptions.
"""

from __future__ import annotations

from mosaic.benchmarks.core.config import (
    Problem,
    SolverSpec,
    discover_solvers,
)
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.problems.shared.cost import (
    run_spatial_cost,
    run_temporal_cost,
    run_vjp_cost,
)
from mosaic.benchmarks.problems.shared.forward import run_agreement, run_physical_laws
from mosaic.benchmarks.problems.shared.gradient import (
    run_fd_check,
    run_jacobian_svd,
    run_param_sweep,
)
from mosaic.benchmarks.problems.shared.plots.cost import plot_cost
from mosaic.benchmarks.problems.shared.plots.forward import (
    plot_agreement,
    plot_physical_laws,
)
from mosaic.benchmarks.problems.shared.plots.gradient import (
    plot_fd_check,
    plot_jacobian_svd,
    plot_param_sweep,
)
from mosaic.benchmarks.problems.shared.plots.ics import plot_ic
from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles

from .ics import _random, _two_density_bumps, _uniform
from .optimization import run_topopt
from .physics import DIAGNOSTICS, make_inputs
from .plots import plot_topopt

_TESSERACT_SLUG = "structural-mesh"

# SIMP material parameters — matched between solvers
_E_MAX = 70_000.0  # Young's modulus of solid [MPa]
_NU = 0.3  # Poisson's ratio
_XMIN = 1e-3  # Void stiffness ratio (E_min / E_max)


# ── Solver registry ──────────────────────────────────────────────────────────

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_SLUG)

apply_styles(_SOLVERS)


# Material parameters: ``nu``/``xmin`` are shared; only the modulus kwarg name
# differs (TopOpt.jl uses ``E``, FEM backends use ``E_max``).
_MAT_SHARED = {"nu": _NU, "xmin": _XMIN}
_SOLVERS["topopt_jl"].input_overrides = {"E": _E_MAX, **_MAT_SHARED}
for _key in ("dealii_structural", "fenics_structural", "firedrake_structural"):
    _SOLVERS[_key].input_overrides = {"E_max": _E_MAX, **_MAT_SHARED}


# ── Problem assembly ─────────────────────────────────────────────────────────

problem = Problem(
    name="structural-mesh",
    category_label="Structural Mechanics",
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
    tesseract_dir=_TESSERACT_SLUG,
    output_key="compliance",
    ic_key="rho",
    solvers=list(_SOLVERS.values()),
    make_inputs=make_inputs,
    error_fn=l2_error_rel,
    domain_extent=2.0,
    resolution_key="nx",
)


# ── IC registrations ─────────────────────────────────────────────────────────

problem.add_ic(
    "uniform",
    _uniform,
    description=(
        "Uniform SIMP material density ρ₀ over all hex mesh elements; standard "
        "homogeneous starting point for topology optimisation of the cantilever beam."
    ),
    plot_params={"rho_0": 0.5, "nx": 16},
    plot=plot_ic,
)
problem.add_ic(
    "random",
    _random,
    description=(
        "Gaussian-noise density field centred at ρ₀=0.5 (σ=0.3, clipped to [0.05, 0.95]); "
        "breaks spatial symmetry so gradient experiments see non-trivial per-cell sensitivity."
    ),
    plot_params={},
    plot=plot_ic,
)
problem.add_ic(
    "two_density_bumps",
    _two_density_bumps,
    description=(
        "Ground-truth density with two stiff Gaussian pillars (ρ_peak=0.95, σ=0.12·min(Lx,Lz)) "
        "at (0.35·Lx, 0.5·Ly, 0.5·Lz) and (0.75·Lx, 0.5·Ly, 0.5·Lz) on a soft ρ_bg=0.1 "
        "background; analog of thermal-mesh ``two_gaussians`` for the load-recovery inverse "
        "experiment (recover density from displacement observations)."
    ),
    plot_params={"nx": 16, "ny": 2, "nz": 8},
    plot=plot_ic,
)


# ── Experiment registrations ─────────────────────────────────────────────────

# Forward
problem.add(
    "forward/baseline",
    run_agreement,
    plot_description="Structural compliance C = F^T U vs mesh resolution N for each solver, uniform density ρ₀=0.5, full-face downward load.",
    ic={"name": "uniform", "seed": 0},
    physics={
        "N": [4, 6, 8, 12, 16],
        "ny": 2,
        "nz": 4,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "corner_load": False,
    },
    plot=plot_agreement,
)
problem.add(
    "forward/agreement",
    run_agreement,
    plot_description="Structural compliance C = F^T U vs density ρ₀ at fixed mesh, sweeping uniform density to span the SIMP stiffness regime.",
    ic={"name": "uniform", "seed": 0},
    physics={
        "nx": 8,
        "ny": 2,
        "nz": 4,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "corner_load": False,
        "rho_0": [0.2, 0.4, 0.5, 0.7, 0.9],
    },
    plot=plot_agreement,
)
problem.add(
    "forward/physical_laws",
    run_physical_laws,
    plot_description="Diagnostic functionals (compliance, total displacement) vs total load F_total, validating linearity of the SIMP response.",
    diagnostics=DIAGNOSTICS,
    ic={"name": "uniform", "seed": 0},
    physics={
        "nx": 8,
        "ny": 2,
        "nz": 4,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "corner_load": False,
        "rho_0": 0.5,
        "F_total": [0.25, 0.5, 1.0, 2.0, 4.0],
    },
    plot=plot_physical_laws,
)

# Cost
problem.add(
    "cost/spatial_cost",
    run_spatial_cost,
    plot_description="Forward-pass wall-clock time vs mesh resolution N at one assembly step.",
    physics={
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "rho_0": 0.5,
        "corner_load": False,
    },
    cost={
        "N_values": [4, 6, 8, 12, 16],
        "steps_values": [1],
        "n_trials": 3,
    },
    plot=plot_cost,
)
problem.add(
    "cost/temporal_cost",
    run_temporal_cost,
    plot_description="Forward-pass wall-clock time vs solve count at fixed mesh (single-step assembly is the dominant cost — temporal axis collapses to one point).",
    physics={
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "rho_0": 0.5,
        "corner_load": False,
    },
    cost={
        "N_values": [4, 6, 8, 12, 16],
        "steps_values": [1],
        "n_trials": 3,
    },
    plot=plot_cost,
)
problem.add(
    "cost/vjp_cost",
    run_vjp_cost,
    plot_description="VJP wall-clock time vs mesh resolution N for differentiable solvers.",
    physics={
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "rho_0": 0.5,
        "corner_load": False,
    },
    cost={
        "N_values": [4, 6, 8, 12, 16],
        "steps_values": [1],
        "n_trials": 3,
    },
    plot=plot_cost,
)

# Gradient
problem.add(
    "gradient/fd_check",
    run_fd_check,
    plot_description="U-curves of finite-difference gradient error vs perturbation size ε with subspace cosine, validating VJP correctness on a random density.",
    ic={"name": "random", "seed": 0},
    physics={
        "nx": 8,
        "ny": 2,
        "nz": 4,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "corner_load": True,
    },
    fd={
        "eps_values": [
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
        "n_dirs": 6,
    },
    plot=plot_fd_check,
)
problem.add(
    "gradient/param_sweep",
    run_param_sweep,
    plot_description="Gradient norm, best-ε FD error, direction cosine, and U-curves vs uniform density ρ₀.",
    ic={"name": "uniform", "seed": 0},
    physics={
        "nx": 8,
        "ny": 2,
        "nz": 4,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "corner_load": True,
        "rho_0": [0.2, 0.4, 0.6, 0.8],
    },
    fd={
        "eps_values": [5e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4],
        "n_dirs": 6,
    },
    plot=plot_param_sweep,
)
problem.add(
    "gradient/jacobian_svd",
    run_jacobian_svd,
    plot_description="Singular-value spectrum of the stacked per-solver gradient matrix and pairwise cosine similarity between solver gradient directions.",
    ic={"name": "random", "seed": 0},
    physics={
        "nx": 8,
        "ny": 2,
        "nz": 4,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "corner_load": True,
    },
    jacobian={"n_alphas": 21, "alpha_range": 0.2},
    plot=plot_jacobian_svd,
)

# Optimization
problem.add(
    "optimization/topopt",
    run_topopt,
    plot_description="SIMP topology optimisation on a 16×8×8 cantilever beam with Adam (lr=0.05): compliance C = F^T U and density field evolution under a 50% volume-fraction constraint.",
    ic={"name": "uniform", "seed": 0},
    physics={
        "nx": 16,
        "ny": 2,
        "nz": 8,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "corner_load": True,
        "v_frac": 0.5,
        "compliance_key": "compliance",
        "penalty_weight": 50.0,
        "x_min": 1e-3,
        "snap_interval": 10,
    },
    optim={"lr": 5e-2, "max_iters": 2500, "patience": 100},
    plot=plot_topopt,
)
problem.add(
    "optimization/topopt_bfgs",
    run_topopt,
    optimizer="bfgs",
    plot_description="SIMP topology optimisation on a 16×8×8 cantilever beam with L-BFGS: compliance C = F^T U and density field evolution under a 50% volume-fraction constraint.",
    ic={"name": "uniform", "seed": 0},
    physics={
        "nx": 16,
        "ny": 2,
        "nz": 8,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "corner_load": True,
        "v_frac": 0.5,
        "compliance_key": "compliance",
        "penalty_weight": 50.0,
        "x_min": 1e-3,
        "snap_interval": 5,
    },
    optim={"max_iters": 100, "patience": 20},
    plot=plot_topopt,
)

__all__ = ["problem"]
