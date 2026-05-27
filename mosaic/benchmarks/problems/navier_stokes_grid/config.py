# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D incompressible Navier-Stokes on a periodic grid.

The problem definition is split across these modules:

- :mod:`.ics`          — IC generators and the ``_tgv_analytic`` reference.
- :mod:`.physics`      — input factory (``make_inputs``); ``DIAGNOSTICS`` is
                         re-exported from
                         :mod:`mosaic.benchmarks.problems.shared.diagnostics`.
- :mod:`.optimization` — drag-minimisation runner.
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
    max_error,
    max_peer_k,
    max_rel_err,
    median_k,
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
    jacobian_svd,
    param_sweep,
)
from mosaic.benchmarks.problems.shared.plots.cost import plot_cost
from mosaic.benchmarks.problems.shared.plots.forward import (
    plot_agreement,
    plot_forward_fields,
    plot_physical_laws,
)
from mosaic.benchmarks.problems.shared.plots.gradient import (
    plot_fd_check,
    plot_horizon_sweep,
    plot_jacobian_svd_comparison,
    plot_param_sweep,
)
from mosaic.benchmarks.problems.shared.plots.ics import plot_ic
from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles

from .exclusions import register as _register_exclusions
from .extras import register as _register_extras
from .ics import _flat_inflow, _multimode, _tgv, _tgv_analytic, _uniform_flow
from .optimization import drag_opt
from .physics import DIAGNOSTICS, make_inputs
from .plots import plot_drag_opt

_TESSERACT_SLUG = "navier-stokes-grid"


# ── Solver registry ──────────────────────────────────────────────────────────
# Per-solver fields (name, scheme, color, AD strategy, …) live in each
# tesseract's ``tesseract_config.yaml`` under ``metadata.mosaic``.
# Only per-(solver, problem) overrides (input_overrides / exclusions) are
# applied here.

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_SLUG)

# Exponax is a 3D-only spectral solver; exclude it from the 2D suite.
_SOLVERS.pop("exponax", None)

# Plot styling lives in mosaic.benchmarks.problems.shared.plots.solver_styles, not in YAML.
apply_styles(_SOLVERS)


_SOLVERS["jax_cfd"].input_overrides = {
    "density": jnp.array([1.0], dtype=jnp.float32),
    "inner_steps": 1,
}


# ── Problem assembly ─────────────────────────────────────────────────────────

problem = Problem(
    name="ns-grid",
    category_label="Navier–Stokes (Grid)",
    description=(
        "2D incompressible Navier–Stokes on a doubly-periodic domain with viscosity ν as "
        "the primary control parameter. The nonlinear advection term ∇·(u⊗u) transfers "
        "energy across scales; at low ν the flow develops turbulent cascades and the "
        "Lyapunov exponent grows, making long-horizon gradients exponentially sensitive "
        "to perturbations."
    ),
    bc_description=(
        "Doubly-periodic square domain [0, 2π]²; incompressibility enforced via "
        "pressure projection at each time step. No walls or inflow/outflow boundaries."
    ),
    tesseract_dir=_TESSERACT_SLUG,
    output_key="result",
    ic_key="v0",
    solvers=list(_SOLVERS.values()),
    make_inputs=make_inputs,
    error_fn=l2_error_rel,
    reference=_tgv_analytic,
    domain_extent=2 * float(jnp.pi),
    status_checks={
        # Suite-level defaults — apply to every `<suite>/*` experiment unless
        # overridden by an inline `status_check=` on the `.add_experiment()` call.
        "forward": [median_k(3.0), max_error(0.5)],
        "cost": [max_peer_k(20.0)],
        # Per-IC override (the `forward/agreement` parent run-list contains
        # both tgv and multimode; the multimode variant has a looser bound).
        "forward/agreement/multimode": [median_k(3.0), max_error(1.5)],
    },
)


# ── Initial conditions ───────────────────────────────────────────────────────
problem.add_ic(
    "multimode",
    fn=_multimode,
    description=(
        "Incompressible velocity field with energy concentrated in a ring at "
        "wavenumber k=2 (σ_k=0.5); supports multi-scale turbulent development."
    ),
    plot_params={"N": 64},
    plot=plot_ic,
)
problem.add_ic(
    "tgv",
    fn=_tgv,
    description=(
        "Taylor–Green vortex u=sin(x)cos(y), v=−cos(x)sin(y); has a closed-form "
        "analytic solution for viscous decay, enabling solver verification."
    ),
    plot_params={"N": 64},
    plot=plot_ic,
)
problem.add_ic(
    "uniform",
    fn=_uniform_flow,
    description=(
        "Uniform rightward flow u=(U, 0) — background flow for cylinder-wake "
        "(Kármán vortex street) experiments. Obstacle specified separately via "
        "the physics.obstacle field."
    ),
    plot_params={"N": 64, "U": 1.0},
    plot=plot_ic,
)
problem.add_ic(
    "flat_inflow",
    fn=_flat_inflow,
    description=(
        "Flat 1-D inlet velocity profile u_x(y) = U, shape (N,). "
        "Starting point for inflow-profile drag-optimisation experiments."
    ),
    plot_params={"N": 64, "U": 0.5},
    plot=plot_ic,
)


# ── Experiment registrations ─────────────────────────────────────────────────

# Forward
problem.add_experiment(
    "forward/baseline",
    agreement,
    plot_description=(
        "Relative error vs grid resolution N at steps=1; validates single-step forward"
        " accuracy across solvers."
    ),
    ic={"name": "tgv", "seed": 0},
    physics={
        "N": [16, 32, 64, 128],
        "nu": 0.05,
        "dt": 0.01,
        "steps": 1,
        "lbm_N_base": 64,
    },
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/agreement",
    agreement,
    plot_description=(
        "Relative error vs viscosity ν for each IC, with vorticity field snapshots"
        " compared against a fine-solver reference."
    ),
    ic=[{"name": "tgv", "seed": 42}, {"name": "multimode", "seed": 42}],
    physics={
        "N": 64,
        "dt": 0.05,
        "steps": 20,
        "nu": [0.001, 0.005, 0.01, 0.02, 0.05],
    },
    reference={"solvers": {"jax_cfd"}, "dt": 0.01, "steps": 100},
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/tgv_nu_sweep",
    agreement,
    plot_description="Relative error vs viscosity ν for each solver at a fixed TGV initial condition.",
    ic={"name": "tgv", "seed": 42},
    physics={
        "N": 64,
        "dt": 0.05,
        "steps": 20,
        "nu": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    },
    reference={"solvers": {"jax_cfd"}, "dt": 0.01, "steps": 100},
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/physical_laws",
    physical_laws,
    plot_description="Divergence RMS, kinetic energy, and analytic error vs N, steps, and ν for each solver.",
    diagnostics=DIAGNOSTICS,
    runs=[
        {
            "name": "vs_N",
            "ic": {"name": "tgv", "seed": 0},
            "physics": {"nu": 0.05, "dt": 0.01, "steps": 20, "N": [16, 32, 64, 128]},
        },
        {
            "name": "vs_steps",
            "ic": {"name": "tgv", "seed": 0},
            "physics": {"nu": 0.05, "dt": 0.01, "N": 64, "steps": [5, 10, 20, 50, 100]},
        },
        {
            "name": "vs_nu",
            "ic": {"name": "tgv", "seed": 0},
            "physics": {
                "dt": 0.01,
                "steps": 20,
                "N": 64,
                "nu": [0.001, 0.005, 0.01, 0.05, 0.1],
            },
        },
    ],
    plot=plot_physical_laws,
)
problem.add_experiment(
    "forward/cylinder",
    agreement,
    plot_description=(
        "Vorticity snapshots and kinetic-energy evolution vs time for each solver"
        " across a sweep of viscosities."
    ),
    ic={"name": "uniform", "seed": 0},
    physics={
        "N": 64,
        "dt": 0.01,
        "steps": 500,
        "obstacle": {
            "shape": "cylinder",
            "center": [0.5, 0.5],
            "radius": 0.1,
        },
        "nu": [0.05, 0.02, 0.01, 0.005],
    },
    reference_solver="openfoam",
    plot=plot_forward_fields,
    status_check=[median_k(50.0), max_error(0.5)],
)
# Cost
problem.add_experiment(
    "cost/spatial_cost",
    spatial_cost,
    plot_description="Forward-pass wall-clock time vs grid resolution N at fixed step count for all solvers.",
    physics={"nu": 0.01, "dt": 0.01, "steps": 100, "N": [64, 128, 192, 256]},
    cost={"n_trials": 3},
    plot=plot_cost,
)
problem.add_experiment(
    "cost/temporal_cost",
    temporal_cost,
    plot_description="Forward-pass wall-clock time vs step count at fixed N for all solvers.",
    physics={"nu": 0.01, "dt": 0.01, "N": 128, "steps": [10, 50, 100, 500, 1000]},
    cost={"n_trials": 3},
    plot=plot_cost,
)
problem.add_experiment(
    "cost/vjp_cost",
    vjp_cost,
    plot_description="VJP wall-clock time vs N and step count for differentiable solvers.",
    runs=[
        {
            "name": "by_N",
            "physics": {"nu": 0.01, "dt": 0.01, "steps": 100, "N": [64, 128, 192, 256]},
            "cost": {"n_trials": 3},
        },
        {
            "name": "by_steps",
            "physics": {
                "nu": 0.01,
                "dt": 0.01,
                "N": 128,
                "steps": [10, 50, 100, 500, 1000],
            },
            "cost": {"n_trials": 3},
        },
    ],
    plot=plot_cost,
)
# Gradient
problem.add_experiment(
    "gradient/fd_check",
    fd_check,
    plot_description=(
        "U-curves of finite-difference gradient error vs perturbation size ε together"
        " with subspace cosine; validates VJP correctness."
    ),
    ic={"name": "multimode", "seed": 42},
    physics={"N": 16, "nu": 0.001, "dt": 0.05, "steps": 20},
    fd={"eps_values": [5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 20},
    plot=plot_fd_check,
    status_check=[
        min_cosine(0.99),
        max_rel_err(1e-3),
        rel_err_peer_outlier(50.0),
    ],
)
problem.add_experiment(
    "gradient/param_sweep",
    param_sweep,
    plot_description="Gradient norm, best-ε FD error, direction cosine, and U-curves vs the sweep parameter.",
    ic={"name": "multimode", "seed": 42},
    physics={"N": 16, "dt": 0.05, "steps": 200, "nu": [0.05, 0.01, 0.005, 0.001]},
    fd={"eps_values": [5e0, 1e0, 1e-1, 1e-2, 1e-3], "n_dirs": 15},
    plot=plot_param_sweep,
)
problem.add_experiment(
    "gradient/horizon_sweep",
    param_sweep,
    plot_description="Gradient norm, FD error, and direction cosine vs rollout horizon T = steps*dt.",
    ic={"name": "multimode", "seed": 42},
    physics={"N": 16, "nu": 0.001, "dt": 0.05, "steps": [5, 10, 20, 40, 80, 160, 320]},
    fd={"eps_values": [1e0, 1e-1, 1e-2, 1e-3], "n_dirs": 8},
    plot=plot_horizon_sweep,
)
problem.add_experiment(
    "gradient/jacobian_svd",
    jacobian_svd,
    plot_description="Singular-value spectrum and pairwise cross-solver cosine similarity of gradient subspaces.",
    ic={"name": "multimode", "seed": 42},
    physics={"N": 8, "nu": 0.001, "dt": 0.05, "steps": 10},
)
problem.add_experiment(
    "gradient/jacobian_svd_steps20",
    jacobian_svd,
    plot_description=(
        "Singular-value spectrum and pairwise cross-solver cosine similarity of"
        " gradient subspaces at an extended rollout horizon."
    ),
    ic={"name": "multimode", "seed": 42},
    physics={"N": 8, "nu": 0.001, "dt": 0.05, "steps": 20},
)
problem.add_experiment(
    "gradient/jacobian_svd_steps40",
    jacobian_svd,
    plot_description=(
        "Singular-value spectrum and pairwise cross-solver cosine similarity of"
        " gradient subspaces at a long rollout horizon."
    ),
    ic={"name": "multimode", "seed": 42},
    physics={"N": 8, "nu": 0.001, "dt": 0.05, "steps": 40},
)
problem.add_experiment(
    "gradient/jacobian_svd_nu01",
    jacobian_svd,
    plot_description=(
        "Singular-value spectrum and pairwise cross-solver cosine similarity of"
        " gradient subspaces at higher viscosity."
    ),
    ic={"name": "multimode", "seed": 42},
    physics={"N": 8, "nu": 0.01, "dt": 0.05, "steps": 10},
)

# Optimization
problem.add_experiment(
    "optimization/drag_opt",
    drag_opt,
    plot_description=(
        "Drag convergence curves per solver, optimised vs initial inflow profiles,"
        " and final drag coefficient comparison."
    ),
    ic={"name": "flat_inflow", "seed": 0},
    physics={
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
        "obstacle": {
            "shape": "cylinder",
            "center": [0.5, 0.5],
            "radius": 0.05,
        },
    },
    optim={
        "lr": 5e-4,
        "max_iters": 500,
        "patience": 100,
        "flow_penalty_weight": 50.0,
        "snap_interval": 20,
    },
    plot=plot_drag_opt,
)
# Bonus plot (not paired with an experiment).
problem.add_extra_plot(
    "_extra/gradient/jacobian_svd_comparison",
    plot_jacobian_svd_comparison,
)

# All per-solver exclusions live in :mod:`.exclusions`; one call attaches the
# whole table to the canonical :class:`Problem` instance.
_register_exclusions(problem)

# Per-domain ``_extra/`` aggregator plots (cost overview, scaling, ucurves)
# live in :mod:`.extras`; the runner picks them up by scanning ``plot_fns``
# for ``_extra/`` keys.
_register_extras(problem)

__all__ = ["problem"]
