"""Assembled ``EXPERIMENTS`` / ``PLOT_FNS`` registries for structural-mesh.

The :class:`Problem` builder captures the problem's closure deps once
(``make_ic``, ``error_fn``, ``output_key``, …) and registers each
experiment + plot in a single ``.add(...)`` call.
"""

from __future__ import annotations

from mosaic.benchmarks.core.config import Problem
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
from mosaic.benchmarks.problems.shared.optimization import run_topopt
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
from mosaic.benchmarks.problems.shared.plots.optimization import plot_topopt

from .ics import MAKE_IC
from .physics import DIAGNOSTICS

# ── Shared run-lists (multi-use) ─────────────────────────────────────────────

_COST_RUNS = [
    {
        "physics": {
            "Lx": 2.0,
            "Ly": 1.0,
            "Lz": 1.0,
            "F_total": 1.0,
            "rho_0": 0.5,
            "corner_load": False,
        },
        "cost": {
            "N_values": [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3200],
            "n_trials": 3,
        },
    }
]


# ── Plot descriptions (keyed by full experiment path) ────────────────────────

_DESCRIPTIONS = {
    "forward/baseline": (
        "Structural compliance C = F^T U vs mesh resolution N for each solver, "
        "uniform density ρ₀=0.5, full-face downward load."
    ),
    "forward/agreement": (
        "Structural compliance C = F^T U vs element density ρ₀ for each solver, "
        "log-scale, full-face downward load."
    ),
    "forward/physical_laws": (
        "Structural compliance C = F^T U vs total load F_total at fixed mesh (nx=8, ny=2, nz=4), ρ₀=0.5, log-log."
    ),
    "cost/spatial_cost": "Forward-pass wall-clock time vs mesh size (nx) for each solver.",
    "cost/temporal_cost": "Forward-pass wall-clock time vs step count at fixed mesh size for each solver.",
    "cost/vjp_cost": "VJP wall-clock time vs mesh size (nx) for each differentiable solver.",
    "gradient/fd_check": (
        "FD gradient error vs step size ε (U-curves), AD–FD direction cosine, and "
        "gradient magnitude field panels for each solver."
    ),
    "gradient/param_sweep": (
        "Gradient norm, best-ε FD error, AD–FD direction cosine, and U-curves vs element density ρ₀."
    ),
    "gradient/jacobian_svd": (
        "Singular-value spectrum of the stacked per-solver gradient matrix and "
        "pairwise cosine similarity between solver gradient directions."
    ),
    "optimization/topopt": (
        "SIMP topology optimisation on a 16×8×8 cantilever beam with Adam (lr=0.05): "
        "compliance C = F^T U and density field evolution under a 50% volume-fraction constraint."
    ),
    "optimization/topopt_bfgs": (
        "SIMP topology optimisation on a 16×8×8 cantilever beam with L-BFGS: "
        "compliance C = F^T U and density field evolution under a 50% volume-fraction constraint."
    ),
}


# ── Problem + registrations ──────────────────────────────────────────────────

problem = Problem(
    make_ic=MAKE_IC,
    error_fn=l2_error_rel,
    output_key="compliance",
    ic_key="rho",
    domain_extent=2.0,
    resolution_key="nx",
    diagnostics=DIAGNOSTICS,
    descriptions=_DESCRIPTIONS,
)

# Forward
problem.add(
    "forward/baseline",
    run_agreement,
    runs=[
        {
            "ic": {"name": "uniform", "seed": 0},
            "physics": {
                "nx": 8,
                "ny": 2,
                "nz": 4,
                "Lx": 2.0,
                "Ly": 1.0,
                "Lz": 1.0,
                "F_total": 1.0,
                "corner_load": False,
            },
            "sweep": {"key": "N", "values": [4, 6, 8, 12, 16]},
        }
    ],
    plot=plot_agreement,
)
problem.add(
    "forward/agreement",
    run_agreement,
    runs=[
        {
            "ic": {"name": "uniform", "seed": 0},
            "physics": {
                "nx": 8,
                "ny": 2,
                "nz": 4,
                "Lx": 2.0,
                "Ly": 1.0,
                "Lz": 1.0,
                "F_total": 1.0,
                "corner_load": False,
            },
            "sweep": {"key": "rho_0", "values": [0.2, 0.4, 0.5, 0.7, 0.9]},
        }
    ],
    plot=plot_agreement,
)
problem.add(
    "forward/physical_laws",
    run_physical_laws,
    runs=[
        {
            "ic": {"name": "uniform", "seed": 0},
            "physics": {
                "nx": 8,
                "ny": 2,
                "nz": 4,
                "Lx": 2.0,
                "Ly": 1.0,
                "Lz": 1.0,
                "corner_load": False,
                "rho_0": 0.5,
            },
            "sweep": {"key": "F_total", "values": [0.25, 0.5, 1.0, 2.0, 4.0]},
        }
    ],
    plot=plot_physical_laws,
)

# Cost
problem.add("cost/spatial_cost", run_spatial_cost, runs=_COST_RUNS, plot=plot_cost)
problem.add("cost/temporal_cost", run_temporal_cost, runs=_COST_RUNS, plot=plot_cost)
problem.add("cost/vjp_cost", run_vjp_cost, runs=_COST_RUNS, plot=plot_cost)

# Gradient
problem.add(
    "gradient/fd_check",
    run_fd_check,
    runs=[
        {
            "ic": {"name": "random", "seed": 0},
            "physics": {
                "nx": 8,
                "ny": 2,
                "nz": 4,
                "Lx": 2.0,
                "Ly": 1.0,
                "Lz": 1.0,
                "F_total": 1.0,
                "corner_load": True,
            },
            "fd": {
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
        }
    ],
    plot=plot_fd_check,
)
problem.add(
    "gradient/param_sweep",
    run_param_sweep,
    runs=[
        {
            "ic": {"name": "uniform", "seed": 0},
            "physics": {
                "nx": 8,
                "ny": 2,
                "nz": 4,
                "Lx": 2.0,
                "Ly": 1.0,
                "Lz": 1.0,
                "F_total": 1.0,
                "corner_load": True,
            },
            "fd": {
                "eps_values": [5e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4],
                "n_dirs": 6,
            },
            "sweep": {"key": "rho_0", "values": [0.2, 0.4, 0.6, 0.8]},
        }
    ],
    plot=plot_param_sweep,
)
problem.add(
    "gradient/jacobian_svd",
    run_jacobian_svd,
    runs=[
        {
            "ic": {"name": "random", "seed": 0},
            "physics": {
                "nx": 8,
                "ny": 2,
                "nz": 4,
                "Lx": 2.0,
                "Ly": 1.0,
                "Lz": 1.0,
                "F_total": 1.0,
                "corner_load": True,
            },
            "jacobian": {"n_alphas": 21, "alpha_range": 0.2},
        }
    ],
    plot=plot_jacobian_svd,
)

# Optimization
problem.add(
    "optimization/topopt",
    run_topopt,
    runs=[
        {
            "ic": {"name": "uniform", "seed": 0},
            "physics": {
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
            "optim": {"lr": 5e-2, "max_iters": 2500, "patience": 100},
        }
    ],
    plot=plot_topopt,
)
problem.add(
    "optimization/topopt_bfgs",
    run_topopt,
    optimizer="bfgs",
    runs=[
        {
            "ic": {"name": "uniform", "seed": 0},
            "physics": {
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
            "optim": {"max_iters": 100, "patience": 20},
        }
    ],
    plot=plot_topopt,
)

# ICs (one entry per registered IC)
problem.add_ic("uniform", {"rho_0": 0.5, "nx": 16}, plot=plot_ic)
problem.add_ic("random", {}, plot=plot_ic)
problem.add_ic("two_density_bumps", {"nx": 16, "ny": 2, "nz": 8}, plot=plot_ic)


EXPERIMENTS = problem.experiments
PLOT_FNS = problem.plot_fns
