# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D incompressible Navier-Stokes on a periodic grid.

The problem definition is split across these modules:

- :mod:`.ics`          — IC generators and the ``_tgv_analytic`` reference.
- :mod:`.physics`      — input factory (``make_inputs``); ``DIAGNOSTICS`` is
                         re-exported from
                         :mod:`mosaic.benchmarks.problems.shared.diagnostics`.
- :mod:`.cylinder`     — the DFG 2D-1 channel: geometry, ``forward/cylinder``
                         kernel and its status checks.
- :mod:`.optimization` — drag-minimisation runner.
- :mod:`.plots`        — per-experiment plot fns wired in below.
- :mod:`.exclusions`   — per-(solver, experiment) opt-outs.

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
    max_final_peer_dev,
    max_final_ratio,
    max_peer_k,
    max_rel_err,
    median_k,
    min_cosine,
    min_order,
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

from . import cylinder
from .exclusions import register as _register_exclusions
from .ics import _multimode, _tgv, _tgv_analytic
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
        "**Fluid flow in 2D.** We simulate an incompressible fluid (think of stirred "
        "water) and differentiate through the simulation. The headline question is "
        "whether each solver returns a trustworthy gradient as the flow becomes chaotic.\n\n"
        "Formally, we solve the 2D incompressible Navier–Stokes equations "
        "$\\partial_t \\mathbf{u} + (\\mathbf{u}\\cdot\\nabla)\\mathbf{u} "
        "= -\\nabla p + \\nu\\,\\nabla^2\\mathbf{u}$, $\\nabla\\cdot\\mathbf{u}=0$, "
        "with the kinematic viscosity $\\nu$ as the primary control parameter. The "
        "nonlinear advection term $\\nabla\\cdot(\\mathbf{u}\\otimes\\mathbf{u})$ "
        "transfers energy across scales; at low $\\nu$ the flow develops turbulent "
        "cascades and a positive Lyapunov exponent, so long-horizon gradients grow "
        "exponentially sensitive to perturbations. This is the regime where "
        "backprop-through-simulation is hardest."
    ),
    bc_description=(
        "Doubly-periodic square domain $[0, 2\\pi]^2$ (the flow wraps around at every "
        "edge), with incompressibility $\\nabla\\cdot\\mathbf{u}=0$ enforced by a "
        "pressure projection at each time step. The cylinder experiments use the "
        "Schäfer–Turek DFG 2D-1 channel instead: $[0, 2.2] \\times [0, 0.41]$ with a "
        "cylinder of diameter 0.1 at $(0.2, 0.2)$, a parabolic inflow of mean speed "
        "0.2 on the left, a zero-gradient outflow at $p=0$ on the right, and no-slip "
        "walls; $\\nu = 10^{-3}$, so $Re = 20$."
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
        "optimization": [max_final_ratio(0.5)],
        # multimode has no closed form, so it scores against the consensus;
        # an absolute bound on the distance to that says nothing about accuracy.
        "forward/agreement/multimode": [median_k(3.0)],
    },
)


# ── Initial conditions ───────────────────────────────────────────────────────
problem.add_ic(
    "multimode",
    fn=_multimode,
    description=(
        "Incompressible velocity field with energy concentrated in a ring at "
        "wavenumber $k=2$ ($\\sigma_k=0.5$); supports multi-scale turbulent development."
    ),
    plot_params={"N": 64},
    plot=plot_ic,
)
problem.add_ic(
    "tgv",
    fn=_tgv,
    description=(
        "Taylor–Green vortex $u=\\sin(x)\\cos(y)$, $v=-\\cos(x)\\sin(y)$; has a closed-form "
        "analytic solution for viscous decay, enabling solver verification."
    ),
    plot_params={"N": 64},
    plot=plot_ic,
)
problem.add_ic(
    "dfg_inflow",
    fn=cylinder.inflow_profile,
    description=(
        "Parabolic inlet profile $u_x(y) = 4 U_m y (H - y) / H^2$ of the DFG 2D-1 "
        "channel ($U_m = 0.3$, $H = 0.41$), shape (ny,). The inflow of the cylinder "
        "experiments, and the starting point of the drag optimization."
    ),
    plot_params={"resolution": 20},
    plot=plot_ic,
)


# ── Experiment registrations ─────────────────────────────────────────────────

# Forward
problem.add_experiment(
    "forward/baseline",
    agreement,
    plot_description=(
        "Relative error vs grid resolution $N$ against the analytic TGV solution at"
        " $t=1.28$, refining space and time together at a fixed CFL number. The"
        " legend gives each solver's observed order of accuracy $p$."
    ),
    ic={"name": "tgv", "seed": 0},
    physics={
        "N": [16, 32, 64, 128],
        "nu": 0.05,
        # dt and steps apply at N=128. Coarser grids take fewer, larger steps
        # to the same end time (CFL ≈ 0.2).
        "dt": 0.01,
        "steps": 128,
        "cfl_N_base": 128,
    },
    plot=plot_agreement,
    # Space and time are refined together, so a scheme that is first order in
    # time (implicit Euler: OpenFOAM, PICT) converges at order 1. The check
    # catches error floors and non-convergence, not first-order schemes.
    status_check=[median_k(3.0), max_error(0.5), min_order(0.8)],
)
problem.add_experiment(
    "forward/agreement",
    agreement,
    plot_description=(
        "Relative error vs viscosity $\\nu$ for each IC, with vorticity field"
        " snapshots. The tgv variant is scored against the analytic TGV"
        " solution; multimode has no closed form and is scored against the"
        " cross-solver consensus."
    ),
    ic=[{"name": "tgv", "seed": 42}, {"name": "multimode", "seed": 42}],
    physics={
        "N": 64,
        "dt": 0.05,
        "steps": 20,
        "nu": [0.001, 0.005, 0.01, 0.02, 0.05],
    },
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/tgv_nu_sweep",
    agreement,
    plot_description="Relative error vs viscosity $\\nu$ for each solver at a fixed TGV initial condition.",
    ic={"name": "tgv", "seed": 42},
    physics={
        "N": 64,
        "dt": 0.05,
        "steps": 20,
        "nu": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    },
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/physical_laws",
    physical_laws,
    plot_description="Divergence RMS, kinetic energy, and analytic error vs $N$, steps, and $\\nu$ for each solver.",
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
    cylinder.cylinder,
    plot_description=(
        "Steady flow past a cylinder in a channel (Schäfer–Turek DFG 2D-1, Re = 20)"
        " at 10 and 20 cells per diameter. Each solver's drag coefficient at 20 is"
        " scored against the reference Cd = 5.5795. The pressure difference across"
        " the cylinder (reference 0.1175) and the lift coefficient are also reported."
        " The vorticity fields are for visual comparison."
    ),
    ic={"name": "dfg_inflow", "seed": 0},
    # 40 cells per diameter is left out of the routine run because it is not
    # scored and costs PhiFlow ~1000 s on an A100, past the per-call timeout on a
    # T4.
    # Checked locally, it gives Cd = 5.5791 (OpenFOAM), 5.647 (XLB) and
    # 5.471 (PhiFlow).
    physics={"resolution": [10, 20]},
    plot=plot_forward_fields,
    # Scored against the literature rather than against peers, since a
    # body-fitted and a staircase solver legitimately differ by a few percent on
    # the same grid. A first-order staircase finite-volume solver is 4.3%, 2.2%
    # and 1.0% below the reference at 10, 20 and 40 cells per diameter, and
    # body-fitted methods are expected within about 1% at 20, so 5% at 20
    # accepts both. A Cd still drifting by more than 0.1% between the last two
    # averaging windows is not at steady state.
    status_check=[
        cylinder.max_cd_err(0.05, resolution=20),
        cylinder.max_cd_drift(1e-3),
    ],
)
# Cost
problem.add_experiment(
    "cost/spatial_cost",
    spatial_cost,
    plot_description="Forward-pass wall-clock time vs grid resolution $N$ at fixed step count for all solvers.",
    physics={"nu": 0.01, "dt": 0.01, "steps": 100, "N": [64, 128, 192, 256]},
    cost={"n_trials": 3},
    plot=plot_cost,
)
problem.add_experiment(
    "cost/temporal_cost",
    temporal_cost,
    plot_description="Forward-pass wall-clock time vs step count at fixed $N$ for all solvers.",
    physics={"nu": 0.01, "dt": 0.01, "N": 128, "steps": [10, 50, 100, 500, 1000]},
    cost={"n_trials": 3},
    plot=plot_cost,
)
problem.add_experiment(
    "cost/vjp_cost",
    vjp_cost,
    plot_description="VJP wall-clock time vs $N$ and step count for differentiable solvers.",
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
        "U-curves of finite-difference gradient error vs perturbation size $\\varepsilon$ together"
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
    plot_description=(
        "Gradient norm, best-$\\varepsilon$ FD error, direction cosine, and "
        "U-curves vs the sweep parameter."
    ),
    ic={"name": "multimode", "seed": 42},
    physics={"N": 16, "dt": 0.05, "steps": 200, "nu": [0.05, 0.01, 0.005, 0.001]},
    fd={"eps_values": [5e0, 1e0, 1e-1, 1e-2, 1e-3], "n_dirs": 15},
    plot=plot_param_sweep,
)
problem.add_experiment(
    "gradient/horizon_sweep",
    param_sweep,
    plot_description="Gradient norm, FD error, and direction cosine vs rollout horizon $T = \\mathrm{steps}\\cdot dt$.",
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
        "Drag minimization over the inlet profile of the DFG 2D-1 channel at a"
        " fixed flow rate: drag coefficient per iteration, optimized vs initial"
        " inflow profiles, and the flow fields before and after."
    ),
    ic={"name": "dfg_inflow", "seed": 0},
    # Every iteration backpropagates through the whole run, so the run is kept
    # small, at 10 cells per diameter and t_end = 4. By then Cd is within 0.1% of
    # its steady value at this resolution.
    physics={"resolution": 10, "t_end": 4.0},
    # Stop once Cd has not dropped by 0.1% in 10 iterations. Adam keeps
    # nudging a plateaued Cd by a few 1e-4, which would otherwise count as
    # progress until max_iters. XLB's descent gains 1% or more per 10
    # iterations until it flattens out near iteration 45.
    optim={
        "lr": 1e-2,
        "max_iters": 100,
        "patience": 10,
        "min_rel_improvement": 1e-3,
        "snap_interval": 10,
    },
    plot=plot_drag_opt,
    # Provisional thresholds, to be confirmed on the full solver set. XLB
    # plateaus at Cd / Cd₀ = 0.53 after ~45 iterations, so 0.7 asks for at
    # least a 30% reduction. The spread between solvers' optima is not yet
    # known.
    status_check=[max_final_ratio(0.7), max_final_peer_dev(0.3)],
)
# Bonus plot (not paired with an experiment).
problem.add_extra_plot(
    "_extra/gradient/jacobian_svd_comparison",
    plot_jacobian_svd_comparison,
)

# All per-solver exclusions live in :mod:`.exclusions`; one call attaches the
# whole table to the canonical :class:`Problem` instance.
_register_exclusions(problem)

__all__ = ["problem"]
