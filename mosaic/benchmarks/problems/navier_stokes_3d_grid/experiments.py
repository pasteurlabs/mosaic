"""Experiment + plot registrations for ns-3d-grid.

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
    run_horizon_sweep,
    run_horizon_sweep_limits,
    run_jacobian_svd,
)
from mosaic.benchmarks.problems.shared.optimization import run_recovery
from mosaic.benchmarks.problems.shared.plots.cost import plot_cost
from mosaic.benchmarks.problems.shared.plots.forward import (
    plot_agreement,
    plot_physical_laws,
)
from mosaic.benchmarks.problems.shared.plots.gradient import (
    plot_fd_check,
    plot_horizon_sweep,
    plot_jacobian_svd,
    plot_jacobian_svd_comparison,
)
from mosaic.benchmarks.problems.shared.plots.ics import plot_ic
from mosaic.benchmarks.problems.shared.plots.optimization import plot_recovery

if TYPE_CHECKING:
    from mosaic.benchmarks.core.config import Problem

# ── Cost run-list (shared by spatial/temporal/vjp) ───────────────────────────

_COST_RUNS = [
    {
        "physics": {"nu": 0.01, "dt": 0.01, "lbm_N_base": 16},
        "cost": {
            "N_values": [16, 32, 48, 64],
            "steps_values": [10, 50, 100],
            "n_trials": 3,
        },
    }
]


# ── Plot descriptions (keyed by full experiment path) ────────────────────────

_DESCRIPTIONS = {
    "forward/baseline": (
        "Relative error vs grid resolution N at steps=1; validates single-step forward accuracy across 3D solvers."
    ),
    "forward/agreement": (
        "3D velocity magnitude fields and kinetic energy spectra per solver, swept over viscosity ν, "
        "compared against a fine-grid consensus reference."
    ),
    "forward/physical_laws": (
        "Divergence RMS and kinetic energy vs grid resolution N, step count, and viscosity ν for each solver; "
        "diagnoses incompressibility and energy decay in 3D."
    ),
    "cost/spatial_cost": (
        "Forward-pass wall-clock time vs grid resolution N for each solver."
    ),
    "cost/temporal_cost": (
        "Forward-pass wall-clock time vs number of integration steps for each solver."
    ),
    "cost/vjp_cost": (
        "VJP (gradient) wall-clock time vs grid resolution N for each differentiable solver."
    ),
    "gradient/fd_check": (
        "Finite-difference gradient error U-curves and direction cosine vs perturbation ε for each solver "
        "on the 3D Taylor-Green vortex IC."
    ),
    "gradient/horizon_sweep": (
        "Gradient norm, finite-difference error, and direction cosine vs rollout horizon T = steps × dt "
        "for each solver on the 3D TGV."
    ),
    "gradient/horizon_sweep_limits": (
        "Per-solver rollout-limit table reporting step count at first failure, failure type, "
        "and wall time per successful step."
    ),
    "gradient/jacobian_svd": (
        "Per-solver singular value spectra and cross-solver cosine similarity of the Jacobian "
        "for the 3D TGV IC."
    ),
    "gradient/jacobian_svd_steps20": (
        "Per-solver singular value spectra and cross-solver cosine similarity of the Jacobian "
        "at an extended rollout horizon (steps=20)."
    ),
    "gradient/jacobian_svd_steps40": (
        "Per-solver singular value spectra and cross-solver cosine similarity of the Jacobian "
        "at a long rollout horizon (steps=40)."
    ),
    "gradient/jacobian_svd_nu01": (
        "Per-solver singular value spectra and cross-solver cosine similarity of the Jacobian "
        "at higher viscosity (ν=0.01)."
    ),
    "optimization/recovery_constant_ic": (
        "Final IC recovery error per solver from zero-initialised gradient-descent optimisation."
    ),
    "optimization/recovery_constant_ic_bfgs": (
        "Final IC recovery error per solver from zero-initialised L-BFGS optimisation."
    ),
    "optimization/recovery_constant_ic_bfgs_proj": (
        "Final IC recovery error per solver from zero-initialised L-BFGS optimisation with divergence-free projection."
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
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"N": 16, "nu": 0.05, "dt": 0.01, "steps": 1},
                "sweep": {"key": "N", "values": [8, 16, 32]},
            }
        ],
        plot=plot_agreement,
    )
    problem.add(
        "forward/agreement",
        run_agreement,
        runs=[
            {
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"N": 16, "dt": 0.01, "steps": 50, "lbm_N_base": 16},
                "sweep": {"key": "nu", "values": [0.001, 0.01, 0.05]},
                # ins_jl excluded from the fine-grid reference set: the ins_jl
                # tesseract container crashes (ContainerDied) when running the
                # fine-grid reference (steps=250, dt=0.002) on a 16³ grid —
                # Julia OOM or resource exhaustion mid-computation. Short runs
                # (steps≤50) work fine; 3D is fully supported. Using only
                # exponax as the fine-grid reference avoids the crash and
                # provides a reliable single-solver consensus anchor.
                "reference": {"solvers": {"exponax"}, "dt": 0.002, "steps": 250},
            }
        ],
        plot=plot_agreement,
    )
    problem.add(
        "forward/physical_laws",
        run_physical_laws,
        runs=[
            {
                "name": "vs_N",
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"nu": 0.05, "dt": 0.01, "steps": 20, "lbm_N_base": 16},
                "sweep": {"key": "N", "values": [8, 16, 32]},
            },
            {
                "name": "vs_steps",
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"nu": 0.05, "dt": 0.01, "N": 16, "lbm_N_base": 16},
                "sweep": {"key": "steps", "values": [5, 10, 20, 50]},
            },
            {
                "name": "vs_nu",
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"dt": 0.01, "steps": 20, "N": 16, "lbm_N_base": 16},
                "sweep": {"key": "nu", "values": [0.001, 0.01, 0.05, 0.1]},
            },
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
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"N": 16, "nu": 0.001, "dt": 0.05, "steps": 10},
                "fd": {
                    "eps_values": [5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4],
                    "n_dirs": 10,
                },
            }
        ],
        plot=plot_fd_check,
    )
    problem.add(
        "gradient/horizon_sweep",
        run_horizon_sweep,
        runs=[
            {
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"N": 16, "nu": 0.001, "dt": 0.05},
                "fd": {"eps_values": [1e0, 1e-1, 1e-2, 1e-3], "n_dirs": 8},
                "sweep": {"key": "steps", "values": [10, 20, 40, 80, 160]},
            }
        ],
        plot=plot_horizon_sweep,
    )
    problem.add(
        "gradient/horizon_sweep_limits",
        run_horizon_sweep_limits,
        runs=[
            {
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"N": 20, "nu": 0.001, "dt": 0.05},
                "sweep": {
                    "key": "steps",
                    "values": [40, 80, 160, 320, 640, 1280, 2560, 5120, 10240],
                },
            }
        ],
        plot=plot_horizon_sweep,
    )
    problem.add(
        "gradient/jacobian_svd",
        run_jacobian_svd,
        runs=[
            {
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 10},
                "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
            }
        ],
        plot=plot_jacobian_svd,
    )
    problem.add(
        "gradient/jacobian_svd_steps20",
        run_jacobian_svd,
        runs=[
            {
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 20},
                "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
            }
        ],
        plot=plot_jacobian_svd,
    )
    problem.add(
        "gradient/jacobian_svd_steps40",
        run_jacobian_svd,
        runs=[
            {
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 40},
                "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
            }
        ],
        plot=plot_jacobian_svd,
    )
    problem.add(
        "gradient/jacobian_svd_nu01",
        run_jacobian_svd,
        runs=[
            {
                "ic": {"name": "tgv3d", "seed": 0},
                "physics": {"N": 8, "nu": 0.01, "dt": 0.05, "steps": 10},
                "jacobian": {"n_alphas": 41, "alpha_range": 0.3},
            }
        ],
        plot=plot_jacobian_svd,
    )

    # Optimization — single runner, optimizer choice as a config kwarg.
    problem.add(
        "optimization/recovery_constant_ic",
        run_recovery,
        optimizer="adam",
        runs=[
            {
                "ic": {"name": "rand_div_free", "seed": 0},
                "physics": {"N": 16, "nu": 0.01, "dt": 0.02, "steps": 100},
                "sweep": {"key": "steps", "values": [100]},
                "optim": {
                    "ic_init_type": "zeros",
                    "lr": 1e-3,
                    "max_iters": 500,
                    "patience": 50,
                    "failure_threshold": 2.0,
                    "snap_interval": 20,
                    "ic_seeds": [0, 1, 2],
                    "record_diagnostics": True,
                },
            }
        ],
        plot=plot_recovery,
    )
    problem.add(
        "optimization/recovery_constant_ic_bfgs",
        run_recovery,
        optimizer="bfgs",
        runs=[
            {
                "ic": {"name": "rand_div_free", "seed": 0},
                "physics": {"N": 16, "nu": 0.01, "dt": 0.02, "steps": 100},
                "sweep": {"key": "steps", "values": [100]},
                "optim": {
                    "ic_init_type": "zeros",
                    "max_iters": 100,
                    "patience": 20,
                    "failure_threshold": 2.0,
                    "snap_interval": 5,
                    "ic_seeds": [0, 1, 2],
                    "record_diagnostics": True,
                },
            }
        ],
        plot=plot_recovery,
    )
    problem.add(
        "optimization/recovery_constant_ic_bfgs_proj",
        run_recovery,
        optimizer="bfgs_proj",
        runs=[
            {
                "ic": {"name": "rand_div_free", "seed": 0},
                "physics": {"N": 16, "nu": 0.01, "dt": 0.02, "steps": 100},
                "sweep": {"key": "steps", "values": [100]},
                "optim": {
                    "ic_init_type": "zeros",
                    "max_iters": 100,
                    "patience": 20,
                    "failure_threshold": 2.0,
                    "snap_interval": 5,
                    "ic_seeds": [0, 1, 2],
                    "record_diagnostics": True,
                },
            }
        ],
        plot=plot_recovery,
    )

    # ICs (one entry per registered IC)
    problem.add_ic("tgv3d", {"N": 32}, plot=plot_ic)
    problem.add_ic("abc", {"N": 32}, plot=plot_ic)
    problem.add_ic("rand_div_free", {"N": 32}, plot=plot_ic)

    # Per-IC sub-plot key (single-IC agreement run keeps its dedicated plot
    # entry so re-plot flows targeting ``forward/agreement/tgv3d`` resolve).
    problem.plot_fns["forward/agreement/tgv3d"] = plot_agreement

    # Bonus plots (not paired with an experiment).
    problem.add_extra_plot(
        "_extra/gradient/jacobian_svd_comparison",
        lambda cfg, **_kw: plot_jacobian_svd_comparison(cfg),
    )
