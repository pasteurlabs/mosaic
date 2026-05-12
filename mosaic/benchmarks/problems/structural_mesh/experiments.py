"""Experiment + plot registrations for structural-mesh.

Exposes :func:`register(problem)` which the package ``__init__`` calls
after building the canonical :class:`Problem` instance — so closure deps
(``make_ic``, ``error_fn``, ``output_key``, …) and the experiment/plot
registries live on a single ``Problem``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from mosaic.benchmarks.problems.shared.plots.optimization import plot_topopt

from .optimization import run_topopt

if TYPE_CHECKING:
    from mosaic.benchmarks.core.config import Problem

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
            "N_values": [4, 6, 8, 12, 16],
            "steps_values": [1],
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
        "Structural compliance C = F^T U vs density ρ₀ at fixed mesh, sweeping "
        "uniform density to span the SIMP stiffness regime."
    ),
    "forward/physical_laws": (
        "Diagnostic functionals (compliance, total displacement) vs total load "
        "F_total, validating linearity of the SIMP response."
    ),
    "cost/spatial_cost": (
        "Forward-pass wall-clock time vs mesh resolution N at one assembly step."
    ),
    "cost/temporal_cost": (
        "Forward-pass wall-clock time vs solve count at fixed mesh (single-step "
        "assembly is the dominant cost — temporal axis collapses to one point)."
    ),
    "cost/vjp_cost": "VJP wall-clock time vs mesh resolution N for differentiable solvers.",
    "gradient/fd_check": (
        "U-curves of finite-difference gradient error vs perturbation size ε "
        "with subspace cosine, validating VJP correctness on a random density."
    ),
    "gradient/param_sweep": (
        "Gradient norm, best-ε FD error, direction cosine, and U-curves vs uniform "
        "density ρ₀."
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


# ── Registrations ────────────────────────────────────────────────────────────


def register(problem: Problem) -> None:
    """Populate ``problem.experiments`` / ``problem.plot_fns`` / ``problem.descriptions``."""
    problem.descriptions.update(_DESCRIPTIONS)

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
    problem.add(
        "cost/temporal_cost", run_temporal_cost, runs=_COST_RUNS, plot=plot_cost
    )
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
