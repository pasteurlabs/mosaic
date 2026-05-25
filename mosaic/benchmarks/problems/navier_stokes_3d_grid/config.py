"""3D incompressible Navier-Stokes on a triply-periodic grid.

The problem definition is split across three modules:

- :mod:`.ics`         — IC generators (``_tgv3d``, ``_abc_flow``,
                        ``_rand_div_free_3d``) and the ``_tgv3d_analytic``
                        reference solution.
- :mod:`.physics`     — input factory (``make_inputs``) and diagnostic
                        functions (``_divergence_rms``, ``_kinetic_energy``,
                        ``_energy_spectrum``).
- :mod:`.optimization` — IC-recovery runner.

This module performs solver discovery, the canonical :class:`Problem`
assembly, and the per-suite ``problem.add_experiment(...)`` calls with inline plot
descriptions, status checks, and per-experiment exclusions.
"""

from __future__ import annotations

import jax.numpy as jnp

from mosaic.benchmarks.core.config import (
    Problem,
    SolverSpec,
    discover_solvers,
)
from mosaic.benchmarks.core.status_checks import (
    max_final_ratio,
    max_rel_err,
    min_cosine,
    rel_err_peer_outlier,
)
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.problems.shared.cost import (
    spatial_cost,
    temporal_cost,
    vjp_cost,
)
from mosaic.benchmarks.problems.shared.forward import agreement, physical_laws
from mosaic.benchmarks.problems.shared.gradient import (
    fd_check,
    horizon_sweep_limits,
    jacobian_svd,
    param_sweep,
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
)
from mosaic.benchmarks.problems.shared.plots.ics import plot_ic
from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles

from .exclusions import register as _register_exclusions
from .extras import register as _register_extras
from .ics import _abc_flow, _rand_div_free_3d, _tgv3d, _tgv3d_analytic
from .optimization import recovery
from .physics import DIAGNOSTICS, make_inputs
from .plots import plot_recovery

_TESSERACT_SLUG = "navier-stokes-grid"


# ── Solver registry ──────────────────────────────────────────────────────────

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_SLUG)

# JAX-CFD is a 2D-only solver (spectral pressure solve doesn't generalise to
# the 3D periodic-box benchmark configuration); drop it from the 3D suite.
_SOLVERS.pop("jax_cfd", None)

apply_styles(_SOLVERS)

# ── Per-(solver, problem) overrides ──────────────────────────────────────────

_SOLVERS["exponax"].input_overrides = {
    "drag": jnp.array([0.0], dtype=jnp.float32),
    "order": 2,
    "kolmogorov_forcing": False,
    "injection_mode": 4,
    "injection_scale": jnp.array([1.0], dtype=jnp.float32),
}


# ── Shared run-lists ─────────────────────────────────────────────────────────

_COST_BASE_PHYS = {"nu": 0.01, "dt": 0.01, "lbm_N_base": 16}
_COST_N_VALUES = [16, 32, 48, 64]
_COST_STEPS_VALUES = [10, 50, 100]
_COST_REF_N = 48
_COST_REF_STEPS = 50
_COST_TRIALS = {"n_trials": 3}


# ── Problem assembly ─────────────────────────────────────────────────────────

problem = Problem(
    name="ns-3d-grid",
    category_label="Navier–Stokes (Grid)",
    description=(
        "3D incompressible Navier–Stokes on a triply-periodic domain with viscosity ν as "
        "the primary control parameter. The 3D extension admits helical structures, vortex "
        "stretching, and faster chaos onset than 2D: chaos horizon T* ≈ 8–16 s vs T* > 64 s "
        "in 2D (at ν=0.001, N=16). Gradient norms grow (vortex stretching) rather than "
        "decaying as in 2D."
    ),
    bc_description=(
        "Triply-periodic cubic domain [0, 2π]³; incompressibility enforced via "
        "pressure projection at each time step. No walls or inflow/outflow boundaries."
    ),
    tesseract_dir=_TESSERACT_SLUG,
    output_key="result",
    ic_key="v0",
    solvers=list(_SOLVERS.values()),
    make_inputs=make_inputs,
    error_fn=l2_error_rel,
    reference=_tgv3d_analytic,
    domain_extent=2 * float(jnp.pi),
    status_checks={
        # Suite-level default — applies to every `optimization/*` experiment.
        "optimization": [max_final_ratio(0.5)],
    },
)


# ── IC registrations ─────────────────────────────────────────────────────────

problem.add_ic(
    "tgv3d",
    fn=_tgv3d,
    description=(
        "3D Taylor–Green vortex u=sin(x)cos(y)cos(z), v=−cos(x)sin(y)cos(z), w=0; "
        "divergence-free IC that develops turbulent vortex structures and a "
        "peak dissipation rate around t≈9/ν. Shape (N,N,N,3)."
    ),
    plot_params={"N": 32},
    plot=plot_ic,
)
problem.add_ic(
    "abc",
    fn=_abc_flow,
    description=(
        "Arnold–Beltrami–Childress flow — a 3D Beltrami field that is a steady "
        "Euler solution with non-zero helicity. Particle trajectories are chaotic "
        "for A≈B≈C≈1, making it a demanding test for gradient signal at long horizons. "
        "Shape (N,N,N,3)."
    ),
    plot_params={"N": 32},
    plot=plot_ic,
)
problem.add_ic(
    "rand_div_free",
    fn=_rand_div_free_3d,
    description=(
        "Random divergence-free 3D velocity field generated via curl of a spectral "
        "vector potential (energy ring at |k|=2, width 1). Seed-controlled, "
        "evolves non-trivially under NS dynamics — unlike ABC flow it has no "
        "near-steady-state structure. Shape (N,N,N,3)."
    ),
    plot_params={"N": 32},
    plot=plot_ic,
)


# ── Experiment registrations ─────────────────────────────────────────────────

# Forward
problem.add_experiment(
    "forward/baseline",
    agreement,
    plot_description="Relative error vs grid resolution N at steps=1; validates single-step forward accuracy across 3D solvers.",
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": [8, 16, 32], "nu": 0.05, "dt": 0.01, "steps": 1},
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/agreement",
    agreement,
    plot_description="3D velocity magnitude fields and kinetic energy spectra per solver, swept over viscosity ν, compared against a fine-grid consensus reference.",
    ic={"name": "tgv3d", "seed": 0},
    physics={
        "N": 16,
        "nu": [0.001, 0.01, 0.05],
        "dt": 0.01,
        "steps": 50,
        "lbm_N_base": 16,
    },
    reference={"solvers": {"exponax"}, "dt": 0.002, "steps": 250},
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/physical_laws",
    physical_laws,
    plot_description="Divergence RMS and kinetic energy vs grid resolution N, step count, and viscosity ν for each solver; diagnoses incompressibility and energy decay in 3D.",
    diagnostics=DIAGNOSTICS,
    runs=[
        {
            "name": "vs_N",
            "ic": {"name": "tgv3d", "seed": 0},
            "physics": {
                "N": [8, 16, 32],
                "nu": 0.05,
                "dt": 0.01,
                "steps": 20,
                "lbm_N_base": 16,
            },
        },
        {
            "name": "vs_steps",
            "ic": {"name": "tgv3d", "seed": 0},
            "physics": {
                "nu": 0.05,
                "dt": 0.01,
                "steps": [5, 10, 20, 50],
                "N": 16,
                "lbm_N_base": 16,
            },
        },
        {
            "name": "vs_nu",
            "ic": {"name": "tgv3d", "seed": 0},
            "physics": {
                "nu": [0.001, 0.01, 0.05, 0.1],
                "dt": 0.01,
                "steps": 20,
                "N": 16,
                "lbm_N_base": 16,
            },
        },
    ],
    plot=plot_physical_laws,
)

# Cost
problem.add_experiment(
    "cost/spatial_cost",
    spatial_cost,
    plot_description="Forward-pass wall-clock time vs grid resolution N for each solver.",
    physics={**_COST_BASE_PHYS, "steps": _COST_REF_STEPS, "N": _COST_N_VALUES},
    cost=_COST_TRIALS,
    plot=plot_cost,
)
problem.add_experiment(
    "cost/temporal_cost",
    temporal_cost,
    plot_description="Forward-pass wall-clock time vs number of integration steps for each solver.",
    physics={**_COST_BASE_PHYS, "N": _COST_REF_N, "steps": _COST_STEPS_VALUES},
    cost=_COST_TRIALS,
    plot=plot_cost,
)
problem.add_experiment(
    "cost/vjp_cost",
    vjp_cost,
    plot_description="VJP (gradient) wall-clock time vs grid resolution N for each differentiable solver.",
    runs=[
        {
            "name": "by_N",
            "physics": {
                **_COST_BASE_PHYS,
                "steps": _COST_REF_STEPS,
                "N": _COST_N_VALUES,
            },
            "cost": _COST_TRIALS,
        },
        {
            "name": "by_steps",
            "physics": {
                **_COST_BASE_PHYS,
                "N": _COST_REF_N,
                "steps": _COST_STEPS_VALUES,
            },
            "cost": _COST_TRIALS,
        },
    ],
    plot=plot_cost,
)
# Gradient
problem.add_experiment(
    "gradient/fd_check",
    fd_check,
    plot_description="Finite-difference gradient error U-curves and direction cosine vs perturbation ε for each solver on the 3D Taylor-Green vortex IC.",
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": 16, "nu": 0.001, "dt": 0.05, "steps": 10},
    fd={
        "eps_values": [5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4],
        "n_dirs": 10,
    },
    plot=plot_fd_check,
    status_check=[
        min_cosine(0.99),
        # Best-ε median rel_error across FD directions. Catches the
        # warp_ns and phiflow 3D systematic backward-magnitude bias
        # (median rel_err ≈ 1.7e-2 / 1.6e-2) while leaving xlb/ins_jl/
        # pict/exponax (5e-6 to 1e-4) unflagged.
        max_rel_err(1e-3),
        # Peer-median outlier; ≥3 valid peers required.
        rel_err_peer_outlier(50.0),
    ],
)
problem.add_experiment(
    "gradient/horizon_sweep",
    param_sweep,
    plot_description="Gradient norm, finite-difference error, and direction cosine vs rollout horizon T = steps × dt for each solver on the 3D TGV.",
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": 16, "nu": 0.001, "dt": 0.05, "steps": [10, 20, 40, 80, 160]},
    fd={"eps_values": [1e0, 1e-1, 1e-2, 1e-3], "n_dirs": 8},
    plot=plot_horizon_sweep,
)
problem.add_experiment(
    "gradient/horizon_sweep_limits",
    horizon_sweep_limits,
    plot_description="Per-solver rollout-limit table reporting step count at first failure, failure type, and wall time per successful step.",
    ic={"name": "tgv3d", "seed": 0},
    physics={
        "N": 20,
        "nu": 0.001,
        "dt": 0.05,
        "steps": [40, 80, 160, 320, 640, 1280, 2560, 5120, 10240],
    },
    plot=plot_horizon_sweep,
)
problem.add_experiment(
    "gradient/jacobian_svd",
    jacobian_svd,
    plot_description="Per-solver singular value spectra and cross-solver cosine similarity of the Jacobian for the 3D TGV IC.",
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": 8, "nu": 0.001, "dt": 0.05, "steps": 10},
    jacobian={"n_alphas": 41, "alpha_range": 0.3},
    plot=plot_jacobian_svd,
)
problem.add_experiment(
    "gradient/jacobian_svd_steps20",
    jacobian_svd,
    plot_description="Per-solver singular value spectra and cross-solver cosine similarity of the Jacobian at an extended rollout horizon (steps=20).",
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": 8, "nu": 0.001, "dt": 0.05, "steps": 20},
    jacobian={"n_alphas": 41, "alpha_range": 0.3},
    plot=plot_jacobian_svd,
)
problem.add_experiment(
    "gradient/jacobian_svd_steps40",
    jacobian_svd,
    plot_description="Per-solver singular value spectra and cross-solver cosine similarity of the Jacobian at a long rollout horizon (steps=40).",
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": 8, "nu": 0.001, "dt": 0.05, "steps": 40},
    jacobian={"n_alphas": 41, "alpha_range": 0.3},
    plot=plot_jacobian_svd,
)
problem.add_experiment(
    "gradient/jacobian_svd_nu01",
    jacobian_svd,
    plot_description="Per-solver singular value spectra and cross-solver cosine similarity of the Jacobian at higher viscosity (ν=0.01).",
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": 8, "nu": 0.01, "dt": 0.05, "steps": 10},
    jacobian={"n_alphas": 41, "alpha_range": 0.3},
    plot=plot_jacobian_svd,
)

problem.add_experiment(
    "optimization/recovery_constant_ic_bfgs_proj",
    recovery,
    optimizer="bfgs_proj",
    _exp_key="recovery_constant_ic_bfgs_proj",
    plot_description="Final IC recovery error per solver from zero-initialised L-BFGS optimisation with divergence-free projection.",
    ic={"name": "rand_div_free", "seed": 0},
    physics={"N": 16, "nu": 0.01, "dt": 0.02, "steps": [100]},
    optim={
        "ic_init_type": "zeros",
        "max_iters": 100,
        "patience": 20,
        "failure_threshold": 2.0,
        "snap_interval": 5,
        "ic_seeds": [0, 1, 2],
        "record_diagnostics": True,
    },
    plot=plot_recovery,
)

# Per-IC sub-plot key (single-IC agreement run keeps its dedicated plot
# entry so re-plot flows targeting ``forward/agreement/tgv3d`` resolve).
problem.plot_fns["forward/agreement/tgv3d"] = plot_agreement

# Bonus plot (not paired with an experiment).
problem.add_extra_plot(
    "_extra/gradient/jacobian_svd_comparison",
    plot_jacobian_svd_comparison,
)

# All per-solver exclusions live in :mod:`.exclusions`.
_register_exclusions(problem)

# Cross-domain / cross-experiment paper figures (registered as _extra/<key>).
_register_extras(problem)

__all__ = ["problem"]
