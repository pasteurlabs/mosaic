# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""3D incompressible Navier-Stokes on a triply-periodic grid.

The problem definition is split across these modules:

- :mod:`.ics`          — IC generators and the ``_tgv3d_analytic`` reference.
- :mod:`.physics`      — input factory (``make_inputs``); ``DIAGNOSTICS`` is
                         re-exported from
                         :mod:`mosaic.benchmarks.problems.shared.diagnostics`.
- :mod:`.optimization` — IC-recovery runner.
- :mod:`.plots`        — per-experiment plot fns wired in below.
- :mod:`.exclusions`   — per-(solver, experiment) opt-outs.
- :mod:`.extras`       — cross-experiment aggregator plots.

This module performs solver discovery, the canonical :class:`Problem`
assembly, and the per-suite ``problem.add_experiment(...)`` calls with
inline plot descriptions, status checks, and per-experiment exclusions.
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
)
from mosaic.benchmarks.problems.shared.plots.cost import plot_cost
from mosaic.benchmarks.problems.shared.plots.forward import (
    plot_agreement,
    plot_physical_laws,
)
from mosaic.benchmarks.problems.shared.plots.gradient import (
    plot_fd_check,
    plot_jacobian_svd_comparison,
)
from mosaic.benchmarks.problems.shared.plots.ics import plot_ic
from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles

from .exclusions import register as _register_exclusions
from .extras import _plot_horizon_sweep_limits
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
        "**Fluid flow in 3D.** The 3D counterpart of the 2D fluid problem, and the "
        "harder stress test for differentiable simulation: real turbulence lives in 3D, "
        "and the gradients are correspondingly more delicate.\n\n"
        "We solve the 3D incompressible Navier–Stokes equations "
        "$\\partial_t \\mathbf{u} + (\\mathbf{u}\\cdot\\nabla)\\mathbf{u} "
        "= -\\nabla p + \\nu\\,\\nabla^2\\mathbf{u}$, $\\nabla\\cdot\\mathbf{u}=0$, "
        "with viscosity $\\nu$ as the primary control parameter. Unlike 2D, the 3D "
        "equations admit *vortex stretching*, which amplifies vorticity and brings on "
        "chaos far sooner: the chaos horizon is $T^\\ast \\approx 8\\text{–}16$ s versus "
        "$T^\\ast > 64$ s in 2D (at $\\nu=10^{-3}$, $N=16$). As a result gradient norms "
        "*grow* along the rollout here rather than decaying as they do in 2D."
    ),
    bc_description=(
        "Triply-periodic cubic domain $[0, 2\\pi]^3$ (the flow wraps around on all "
        "three axes). Incompressibility $\\nabla\\cdot\\mathbf{u}=0$ is enforced by a "
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
    plot_description=(
        "Relative error vs grid resolution N at steps=1; validates single-step forward"
        " accuracy across 3D solvers."
    ),
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": [8, 16, 32], "nu": 0.05, "dt": 0.01, "steps": 1},
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/agreement",
    agreement,
    plot_description=(
        "3D velocity magnitude fields and kinetic energy spectra per solver, swept over"
        " viscosity \u03bd, compared against the analytic TGV reference."
    ),
    ic={"name": "tgv3d", "seed": 0},
    physics={
        "N": 16,
        "nu": [0.001, 0.01, 0.05],
        "dt": 0.01,
        "steps": 50,
        "lbm_N_base": 16,
    },
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/physical_laws",
    physical_laws,
    plot_description=(
        "Divergence RMS and kinetic energy vs grid resolution N, step count, and"
        " viscosity \u03bd for each solver; diagnoses incompressibility and energy decay in 3D."
    ),
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
    plot_description=(
        "Finite-difference gradient error U-curves and direction cosine vs perturbation"
        " \u03b5 for each solver on the 3D Taylor-Green vortex IC."
    ),
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
    "gradient/horizon_sweep_limits",
    horizon_sweep_limits,
    plot_description=(
        "Per-solver rollout-limit table reporting step count at first failure,"
        " failure type, and wall time per successful step."
    ),
    ic={"name": "tgv3d", "seed": 0},
    physics={
        "N": 20,
        "nu": 0.001,
        "dt": 0.05,
        "steps": [40, 80, 160, 320, 640, 1280, 2560, 5120, 10240],
    },
    plot=_plot_horizon_sweep_limits,
)
problem.add_experiment(
    "gradient/jacobian_svd",
    jacobian_svd,
    plot_description=(
        "Per-solver singular value spectra and cross-solver cosine similarity of the"
        " Jacobian for the 3D TGV IC."
    ),
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": 8, "nu": 0.001, "dt": 0.05, "steps": 10},
    jacobian={"n_alphas": 41, "alpha_range": 0.3},
)
problem.add_experiment(
    "gradient/jacobian_svd_steps20",
    jacobian_svd,
    plot_description=(
        "Per-solver singular value spectra and cross-solver cosine similarity of the"
        " Jacobian at an extended rollout horizon (steps=20)."
    ),
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": 8, "nu": 0.001, "dt": 0.05, "steps": 20},
    jacobian={"n_alphas": 41, "alpha_range": 0.3},
)
problem.add_experiment(
    "gradient/jacobian_svd_steps40",
    jacobian_svd,
    plot_description=(
        "Per-solver singular value spectra and cross-solver cosine similarity of the"
        " Jacobian at a long rollout horizon (steps=40)."
    ),
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": 8, "nu": 0.001, "dt": 0.05, "steps": 40},
    jacobian={"n_alphas": 41, "alpha_range": 0.3},
)
problem.add_experiment(
    "gradient/jacobian_svd_nu01",
    jacobian_svd,
    plot_description=(
        "Per-solver singular value spectra and cross-solver cosine similarity of the"
        " Jacobian at higher viscosity (\u03bd=0.01)."
    ),
    ic={"name": "tgv3d", "seed": 0},
    physics={"N": 8, "nu": 0.01, "dt": 0.05, "steps": 10},
    jacobian={"n_alphas": 41, "alpha_range": 0.3},
)

problem.add_experiment(
    "optimization/recovery_constant_ic_bfgs_proj",
    recovery,
    optimizer="bfgs_proj",
    _exp_key="recovery_constant_ic_bfgs_proj",
    plot_description=(
        "Final IC recovery error per solver from zero-initialised L-BFGS optimisation"
        " with divergence-free projection."
    ),
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

# Single-IC agreement run gets its own plot entry so re-plot flows targeting
# ``forward/agreement/tgv3d`` resolve.
problem.add_extra_plot("forward/agreement/tgv3d", plot_agreement)

# Bonus plot (not paired with an experiment).
problem.add_extra_plot(
    "_extra/gradient/jacobian_svd_comparison",
    plot_jacobian_svd_comparison,
)

# All per-solver exclusions live in :mod:`.exclusions`.
_register_exclusions(problem)

__all__ = ["problem"]
