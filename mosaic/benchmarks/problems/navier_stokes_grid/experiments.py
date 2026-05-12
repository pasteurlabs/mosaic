"""Assembled ``EXPERIMENTS`` / ``PLOT_FNS`` registries for ns-grid.

The :class:`Problem` builder captures the problem's closure deps once
(``make_ic``, ``error_fn``, ``output_key``, …) and registers each
experiment + plot in a single ``.add(...)`` call.
"""

from __future__ import annotations

import math

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
    run_horizon_sweep,
    run_jacobian_svd,
    run_param_sweep,
)
from mosaic.benchmarks.problems.shared.optimization import (
    run_drag_opt,
)
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
    plot_param_sweep,
)
from mosaic.benchmarks.problems.shared.plots.ics import plot_ic
from mosaic.benchmarks.problems.shared.plots.optimization import plot_drag_opt

from .ics import MAKE_IC, _tgv_analytic
from .physics import DIAGNOSTICS

# ── Run-lists (one per experiment) ───────────────────────────────────────────

_BASELINE_RUNS = [
    {
        "ic": {"name": "tgv", "seed": 0},
        "physics": {"N": 64, "nu": 0.05, "dt": 0.01, "steps": 1, "lbm_N_base": 64},
        "sweep": {"key": "N", "values": [16, 32, 64, 128]},
    }
]
_AGREEMENT_RUNS = [
    {
        "ic": {"name": "tgv", "seed": 42},
        "physics": {"N": 64, "dt": 0.05, "steps": 20},
        "sweep": {"key": "nu", "values": [0.001, 0.005, 0.01, 0.02, 0.05]},
        "fine": {"solvers": {"jax_cfd"}, "dt": 0.01, "steps": 100},
    },
    {
        "ic": {"name": "multimode", "seed": 42},
        "physics": {"N": 64, "dt": 0.05, "steps": 20},
        "sweep": {"key": "nu", "values": [0.001, 0.005, 0.01, 0.02, 0.05]},
        "fine": {"solvers": {"jax_cfd"}, "dt": 0.01, "steps": 100},
    },
]
_TGV_NU_SWEEP_RUNS = [
    {
        "ic": {"name": "tgv", "seed": 42},
        "physics": {"N": 64, "dt": 0.05, "steps": 20},
        "sweep": {"key": "nu", "values": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]},
        "fine": {"solvers": {"jax_cfd"}, "dt": 0.01, "steps": 100},
    },
]
_PHYSICAL_LAWS_RUNS = [
    {
        "name": "vs_N",
        "ic": {"name": "tgv", "seed": 0},
        "physics": {"nu": 0.05, "dt": 0.01, "steps": 20},
        "sweep": {"key": "N", "values": [16, 32, 64, 128]},
    },
    {
        "name": "vs_steps",
        "ic": {"name": "tgv", "seed": 0},
        "physics": {"nu": 0.05, "dt": 0.01, "N": 64},
        "sweep": {"key": "steps", "values": [5, 10, 20, 50, 100]},
    },
    {
        "name": "vs_nu",
        "ic": {"name": "tgv", "seed": 0},
        "physics": {"dt": 0.01, "steps": 20, "N": 64},
        "sweep": {"key": "nu", "values": [0.001, 0.005, 0.01, 0.05, 0.1]},
    },
]
_CYLINDER_RUNS = [
    {
        "ic": {"name": "uniform", "seed": 0},
        "physics": {
            "N": 64,
            "dt": 0.01,
            "steps": 500,
            "obstacle": {"shape": "cylinder", "center": [0.5, 0.5], "radius": 0.1},
        },
        "sweep": {"key": "nu", "values": [0.05, 0.02, 0.01, 0.005]},
    }
]

_COST_RUNS = [
    {
        "physics": {"nu": 0.01, "dt": 0.01},
        "cost": {
            "N_values": [64, 128, 192, 256],
            "steps_values": [10, 50, 100, 500, 1000],
            "n_trials": 3,
        },
    }
]

_FD_CHECK_RUNS = [
    {
        "ic": {"name": "multimode", "seed": 42},
        "physics": {"N": 16, "nu": 0.001, "dt": 0.05, "steps": 20},
        "fd": {"eps_values": [5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 20},
    },
]
_PARAM_SWEEP_RUNS = [
    {
        "ic": {"name": "multimode", "seed": 42},
        "physics": {"N": 16, "dt": 0.05, "steps": 200},
        "fd": {"eps_values": [5e0, 1e0, 1e-1, 1e-2, 1e-3], "n_dirs": 15},
        "sweep": {"key": "nu", "values": [0.05, 0.01, 0.005, 0.001]},
    },
]
_HORIZON_SWEEP_RUNS = [
    {
        "ic": {"name": "multimode", "seed": 42},
        "physics": {"N": 16, "nu": 0.001, "dt": 0.05},
        "fd": {"eps_values": [1e0, 1e-1, 1e-2, 1e-3], "n_dirs": 8},
        "sweep": {"key": "steps", "values": [5, 10, 20, 40, 80, 160, 320]},
    },
]
_JSVD_BASE_RUNS = [
    {
        "ic": {"name": "multimode", "seed": 42},
        "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 10},
        "jacobian": {},
    }
]
_JSVD_STEPS20_RUNS = [
    {
        "ic": {"name": "multimode", "seed": 42},
        "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 20},
        "jacobian": {},
    }
]
_JSVD_STEPS40_RUNS = [
    {
        "ic": {"name": "multimode", "seed": 42},
        "physics": {"N": 8, "nu": 0.001, "dt": 0.05, "steps": 40},
        "jacobian": {},
    }
]
_JSVD_NU01_RUNS = [
    {
        "ic": {"name": "multimode", "seed": 42},
        "physics": {"N": 8, "nu": 0.01, "dt": 0.05, "steps": 10},
        "jacobian": {},
    }
]

_DRAG_OPT_RUNS = [
    {
        "name": "re20",
        "ic": {"name": "flat_inflow", "seed": 0},
        "physics": {
            # Re = U·D/ν = 0.5·0.1/0.0025 = 20
            # N=32 used (not 64): IBM hard-masking in jax-cfd causes pressure
            # projection divergence at N=64 (discontinuity at obstacle boundary
            # overwhelms the periodic Poisson solve at step ~20).
            "N": 32,
            "nu": 0.0025,
            "dt": 0.02,
            "steps": 400,
            "domain_extent": 1.0,
            "U_mean": 0.5,
            "obstacle": {"shape": "cylinder", "center": [0.5, 0.5], "radius": 0.05},
        },
        "optim": {
            "lr": 5e-4,
            "max_iters": 500,
            "patience": 100,
            "flow_penalty_weight": 50.0,
            "snap_interval": 20,
        },
    },
]
_DRAG_OPT_BFGS_RUNS = [
    {
        "name": "re20",
        "ic": {"name": "flat_inflow", "seed": 0},
        "physics": {
            "N": 32,
            "nu": 0.0025,
            "dt": 0.02,
            "steps": 400,
            "domain_extent": 1.0,
            "U_mean": 0.5,
            "obstacle": {"shape": "cylinder", "center": [0.5, 0.5], "radius": 0.05},
        },
        "optim": {
            "max_iters": 50,
            "patience": 15,
            "flow_penalty_weight": 50.0,
            "snap_interval": 5,
        },
    },
]


# ── Plot descriptions (keyed by full experiment path) ────────────────────────

_DESCRIPTIONS = {
    "forward/baseline": "Relative error vs grid resolution N at steps=1; validates single-step forward accuracy across solvers.",
    "forward/agreement": "Relative error vs viscosity ν for each IC, with vorticity field snapshots compared against a fine-solver reference.",
    "forward/tgv_nu_sweep": "Relative error vs viscosity ν for each solver at a fixed TGV initial condition.",
    "forward/physical_laws": "Divergence RMS, kinetic energy, and analytic error vs N, steps, and ν for each solver.",
    "forward/cylinder": "Vorticity snapshots and kinetic-energy evolution vs time for each solver across a sweep of viscosities.",
    "cost/spatial_cost": "Forward-pass wall-clock time vs grid resolution N at fixed step count for all solvers.",
    "cost/temporal_cost": "Forward-pass wall-clock time vs step count at fixed N for all solvers.",
    "cost/vjp_cost": "VJP wall-clock time vs N and step count for differentiable solvers.",
    "gradient/fd_check": "U-curves of finite-difference gradient error vs perturbation size ε together with subspace cosine; validates VJP correctness.",
    "gradient/param_sweep": "Gradient norm, best-ε FD error, direction cosine, and U-curves vs the sweep parameter.",
    "gradient/horizon_sweep": "Gradient norm, FD error, and direction cosine vs rollout horizon T = steps*dt.",
    "gradient/jacobian_svd": "Singular-value spectrum and pairwise cross-solver cosine similarity of gradient subspaces.",
    "gradient/jacobian_svd_steps20": "Singular-value spectrum and pairwise cross-solver cosine similarity of gradient subspaces at an extended rollout horizon.",
    "gradient/jacobian_svd_steps40": "Singular-value spectrum and pairwise cross-solver cosine similarity of gradient subspaces at a long rollout horizon.",
    "gradient/jacobian_svd_nu01": "Singular-value spectrum and pairwise cross-solver cosine similarity of gradient subspaces at higher viscosity.",
    "optimization/drag_opt": "Drag convergence curves per solver, optimised vs initial inflow profiles, and final drag coefficient comparison.",
    "optimization/drag_opt_bfgs": "L-BFGS drag convergence curves per solver, optimised vs initial inflow profiles, and final drag coefficient comparison.",
}


# ── Problem + registrations ──────────────────────────────────────────────────

problem = Problem(
    make_ic=MAKE_IC,
    error_fn=l2_error_rel,
    output_key="result",
    ic_key="v0",
    domain_extent=2 * math.pi,
    resolution_key="N",
    analytic=_tgv_analytic,
    diagnostics=DIAGNOSTICS,
    descriptions=_DESCRIPTIONS,
)

# Forward
problem.add("forward/baseline", run_agreement, runs=_BASELINE_RUNS, plot=plot_agreement)
problem.add(
    "forward/agreement", run_agreement, runs=_AGREEMENT_RUNS, plot=plot_agreement
)
problem.add(
    "forward/tgv_nu_sweep", run_agreement, runs=_TGV_NU_SWEEP_RUNS, plot=plot_agreement
)
problem.add(
    "forward/physical_laws",
    run_physical_laws,
    runs=_PHYSICAL_LAWS_RUNS,
    plot=plot_physical_laws,
)
problem.add("forward/cylinder", run_agreement, runs=_CYLINDER_RUNS, plot=plot_agreement)

# Cost
problem.add("cost/spatial_cost", run_spatial_cost, runs=_COST_RUNS, plot=plot_cost)
problem.add("cost/temporal_cost", run_temporal_cost, runs=_COST_RUNS, plot=plot_cost)
problem.add("cost/vjp_cost", run_vjp_cost, runs=_COST_RUNS, plot=plot_cost)

# Gradient
problem.add("gradient/fd_check", run_fd_check, runs=_FD_CHECK_RUNS, plot=plot_fd_check)
problem.add(
    "gradient/param_sweep",
    run_param_sweep,
    runs=_PARAM_SWEEP_RUNS,
    plot=plot_param_sweep,
)
problem.add(
    "gradient/horizon_sweep",
    run_horizon_sweep,
    runs=_HORIZON_SWEEP_RUNS,
    plot=plot_horizon_sweep,
)
problem.add(
    "gradient/jacobian_svd",
    run_jacobian_svd,
    runs=_JSVD_BASE_RUNS,
    plot=plot_jacobian_svd,
)
problem.add(
    "gradient/jacobian_svd_steps20",
    run_jacobian_svd,
    runs=_JSVD_STEPS20_RUNS,
    plot=plot_jacobian_svd,
)
problem.add(
    "gradient/jacobian_svd_steps40",
    run_jacobian_svd,
    runs=_JSVD_STEPS40_RUNS,
    plot=plot_jacobian_svd,
)
problem.add(
    "gradient/jacobian_svd_nu01",
    run_jacobian_svd,
    runs=_JSVD_NU01_RUNS,
    plot=plot_jacobian_svd,
)

# Optimization
problem.add(
    "optimization/drag_opt", run_drag_opt, runs=_DRAG_OPT_RUNS, plot=plot_drag_opt
)
problem.add(
    "optimization/drag_opt_bfgs",
    run_drag_opt,
    optimizer="bfgs",
    runs=_DRAG_OPT_BFGS_RUNS,
    plot=plot_drag_opt,
)

# ICs (one entry per registered IC)
problem.add_ic("multimode", {"N": 64}, plot=plot_ic)
problem.add_ic("tgv", {"N": 64}, plot=plot_ic)
problem.add_ic("uniform", {"N": 64, "U": 1.0}, plot=plot_ic)
problem.add_ic("flat_inflow", {"N": 64, "U": 0.5}, plot=plot_ic)

# Bonus plots (not paired with an experiment).
problem.add_extra_plot(
    "_extra/gradient/jacobian_svd_comparison",
    lambda cfg, **_kw: plot_jacobian_svd_comparison(cfg),
)


EXPERIMENTS = problem.experiments
PLOT_FNS = problem.plot_fns
