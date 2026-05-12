"""Experiment + plot registrations for thermal-mesh.

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
from mosaic.benchmarks.problems.shared.plots.optimization import (
    plot_conductivity_recovery,
)

from .optimization import run_conductivity_recovery

if TYPE_CHECKING:
    from mosaic.benchmarks.core.config import Problem

# ── Cost run-list (shared by spatial/temporal/vjp) ───────────────────────────

_COST_RUNS = [
    {
        "physics": {
            "Lx": 2.0,
            "Ly": 1.0,
            "Lz": 1.0,
            "Q_total": 1.0,
            "rho_0": 0.5,
        },
        "cost": {
            "N_values": [16, 32, 64, 128, 256, 512, 1024, 2048, 4500],
            "n_trials": 3,
        },
    }
]


# ── Plot descriptions (keyed by full experiment path) ────────────────────────

_DESCRIPTIONS = {
    "forward/baseline": (
        "Thermal compliance C vs mesh resolution N with random density; "
        "compares FV and FEM solvers across refinements."
    ),
    "forward/agreement": (
        "Thermal compliance C vs uniform element density ρ₀ at fixed N; "
        "compares solvers on a log scale."
    ),
    "forward/physical_laws": (
        "Thermal compliance C vs total heat flux Q_total at fixed N and ρ₀ "
        "with a hot-spot BC; shown on log-log axes."
    ),
    "forward/source_baseline": (
        "Thermal compliance C vs mesh resolution N with a Gaussian source field; "
        "compares solvers across refinements."
    ),
    "forward/source_linearity": (
        "Thermal compliance C vs source amplitude at fixed mesh; "
        "compares solvers on log-log axes."
    ),
    "cost/spatial_cost": "Forward-pass wall-clock time vs mesh size (nx) for all solvers.",
    "cost/temporal_cost": "Forward-pass wall-clock time vs time-axis size for all solvers.",
    "cost/vjp_cost": "VJP wall-clock time vs mesh size (nx) for differentiable solvers.",
    "gradient/fd_check": (
        "FD gradient error vs step size ε (U-curves), AD/FD direction cosine, "
        "and gradient magnitude field panels."
    ),
    "gradient/param_sweep": (
        "Gradient norm, best-ε FD error, AD/FD direction cosine, and U-curves "
        "vs element density ρ₀."
    ),
    "gradient/jacobian_svd": (
        "Singular-value spectrum of stacked per-solver gradients and pairwise "
        "cosine similarity between solver gradient directions."
    ),
    "gradient/source_fd_check": (
        "FD gradient error vs ε, AD/FD direction cosine, and gradient field panels "
        "for d(identification_error)/d(source)."
    ),
    "gradient/source_width_sweep": (
        "Gradient norm, best-ε FD error, AD/FD direction cosine, and U-curves "
        "vs source width σ."
    ),
    "optimization/conductivity_recovery": (
        "Optimisation traces (loss vs iteration) and recovered conductivity fields "
        "vs the two-Gaussian ground truth, using gradient descent."
    ),
    "optimization/conductivity_recovery_bfgs": (
        "Optimisation traces (loss vs iteration) and recovered conductivity fields "
        "vs the two-Gaussian ground truth, using L-BFGS."
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
                "ic": {"name": "random", "seed": 0},
                "physics": {"nz": 1, "Lx": 2.0, "Ly": 1.0, "Lz": 1.0, "Q_total": 1.0},
                "sweep": {"key": "N", "values": [2, 3, 4, 6, 8, 12, 16, 24]},
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
                    "nx": 16,
                    "ny": 8,
                    "nz": 1,
                    "Lx": 2.0,
                    "Ly": 1.0,
                    "Lz": 1.0,
                    "Q_total": 1.0,
                },
                "sweep": {
                    "key": "rho_0",
                    "values": [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
                },
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
                    "N": 16,
                    "nz": 1,
                    "Lx": 2.0,
                    "Ly": 1.0,
                    "Lz": 1.0,
                    "rho_0": 0.5,
                    "hot_spot": True,
                },
                "sweep": {"key": "Q_total", "values": [0.25, 0.5, 1.0, 2.0, 4.0]},
            }
        ],
        plot=plot_physical_laws,
    )
    problem.add(
        "forward/source_baseline",
        run_agreement,
        runs=[
            {
                "ic": {"name": "gaussian_source"},
                "physics": {
                    "nz": 1,
                    "Lx": 2.0,
                    "Ly": 1.0,
                    "Lz": 1.0,
                    "rho_0": 0.5,
                    "ic_field": "source",
                },
                "sweep": {"key": "N", "values": [4, 6, 8, 12, 16, 24]},
            }
        ],
        plot=plot_agreement,
    )
    problem.add(
        "forward/source_linearity",
        run_agreement,
        runs=[
            {
                "ic": {"name": "gaussian_source"},
                "physics": {
                    "nx": 16,
                    "ny": 8,
                    "nz": 1,
                    "Lx": 2.0,
                    "Ly": 1.0,
                    "Lz": 1.0,
                    "rho_0": 0.5,
                    "ic_field": "source",
                },
                "sweep": {
                    "key": "amplitude",
                    "values": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
                },
            }
        ],
        plot=plot_agreement,
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
                    "ny": 4,
                    "nz": 1,
                    "Lx": 2.0,
                    "Ly": 1.0,
                    "Lz": 1.0,
                    "Q_total": 1.0,
                },
                "fd": {"eps_values": [1e0, 1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 6},
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
                    "ny": 4,
                    "nz": 1,
                    "Lx": 2.0,
                    "Ly": 1.0,
                    "Lz": 1.0,
                    "Q_total": 1.0,
                },
                "fd": {"eps_values": [1e0, 1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 6},
                "sweep": {"key": "rho_0", "values": [0.1, 0.2, 0.4, 0.6, 0.8]},
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
                    "ny": 4,
                    "nz": 1,
                    "Lx": 2.0,
                    "Ly": 1.0,
                    "Lz": 1.0,
                    "Q_total": 1.0,
                },
                "jacobian": {"n_alphas": 21, "alpha_range": 0.2},
            }
        ],
        plot=plot_jacobian_svd,
    )
    # Source-identification gradient experiments.
    # These use source as the differentiable input and identification_error as the
    # objective; per-run ``ic_key`` / ``output_key`` overrides the global defaults.
    problem.add(
        "gradient/source_fd_check",
        run_fd_check,
        runs=[
            {
                "ic": {"name": "gaussian_source"},
                "ic_key": "source",
                "output_key": "identification_error",
                "physics": {
                    "nx": 8,
                    "ny": 4,
                    "nz": 1,
                    "Lx": 2.0,
                    "Ly": 1.0,
                    "Lz": 1.0,
                    "rho_0": 0.5,
                    "target_from_two_gaussians": True,
                    "ic_field": "source",
                },
                "fd": {"eps_values": [1e0, 1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 6},
            }
        ],
        plot=plot_fd_check,
    )
    problem.add(
        "gradient/source_width_sweep",
        run_param_sweep,
        runs=[
            {
                "ic": {"name": "gaussian_source"},
                "ic_key": "source",
                "output_key": "identification_error",
                "physics": {
                    "nx": 16,
                    "ny": 8,
                    "nz": 1,
                    "rho_0": 0.5,
                    "target_from_two_gaussians": True,
                    "ic_field": "source",
                },
                "fd": {"eps_values": [1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 4},
                "sweep": {
                    "key": "sigma",
                    "values": [0.05, 0.1, 0.2, 0.3, 0.5],
                    "ic_sweep": True,
                },
            }
        ],
        plot=plot_param_sweep,
    )

    # Optimization
    problem.add(
        "optimization/conductivity_recovery",
        run_conductivity_recovery,
        runs=[
            {
                "ic": {"name": "uniform", "seed": 0},
                "physics": {
                    "nx": 16,
                    "ny": 8,
                    "nz": 1,
                    "Lx": 2.0,
                    "Ly": 1.0,
                    "Lz": 1.0,
                    "rho_0": 0.5,
                    "Q_total": 1.0,
                    "compliance_key": "identification_error",
                    "penalty_weight": 0.0,
                    "x_min": 1e-3,
                    "snap_interval": 20,
                    "target_rho_from_two_gaussians": True,
                },
                "optim": {"lr": 1e-2, "max_iters": 2000, "patience": 200},
            }
        ],
        plot=plot_conductivity_recovery,
    )
    problem.add(
        "optimization/conductivity_recovery_bfgs",
        run_conductivity_recovery,
        optimizer="bfgs",
        runs=[
            {
                "ic": {"name": "uniform", "seed": 0},
                "physics": {
                    "nx": 16,
                    "ny": 8,
                    "nz": 1,
                    "Lx": 2.0,
                    "Ly": 1.0,
                    "Lz": 1.0,
                    "rho_0": 0.5,
                    "Q_total": 1.0,
                    "compliance_key": "identification_error",
                    "penalty_weight": 0.0,
                    "x_min": 1e-3,
                    "snap_interval": 10,
                    "target_rho_from_two_gaussians": True,
                },
                "optim": {"max_iters": 200, "patience": 30},
            }
        ],
        plot=plot_conductivity_recovery,
    )

    # ICs (one entry per registered IC)
    problem.add_ic("zero_source", {"nx": 16, "ny": 8, "nz": 1}, plot=plot_ic)
    problem.add_ic("uniform", {"rho_0": 0.5, "nx": 16, "ny": 8, "nz": 1}, plot=plot_ic)
    problem.add_ic(
        "random",
        {"rho_0": 0.5, "noise": 0.3, "nx": 16, "ny": 8, "nz": 1, "seed": 0},
        plot=plot_ic,
    )
    problem.add_ic(
        "gaussian_source",
        {
            "nx": 16,
            "ny": 8,
            "nz": 1,
            "amplitude": 1.0,
            "cx": 0.5,
            "cy": 0.5,
            "sigma": 0.2,
        },
        plot=plot_ic,
    )
    problem.add_ic("two_gaussians", {"nx": 16, "ny": 8, "nz": 1}, plot=plot_ic)
