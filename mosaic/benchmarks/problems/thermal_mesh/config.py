"""Quasi-2D steady heat-conduction compliance / source-identification benchmark.

The problem definition is split across three modules:

- :mod:`.ics`         — IC generators (``_zero_source``, ``_uniform``,
                        ``_random``, ``_gaussian_source``, ``_two_gaussians``).
- :mod:`.physics`     — mesh / BC builders, the reference FEM solve for
                        inverse-recovery ground truth, the
                        ``make_inputs`` factory, and ``DIAGNOSTICS``.
- :mod:`.optimization` — conductivity-recovery runner.

This module performs solver discovery, the canonical :class:`Problem`
assembly, and the per-suite ``problem.add_experiment(...)`` calls with inline plot
descriptions.
"""

from __future__ import annotations

from mosaic.benchmarks.core.config import (
    Problem,
    SolverSpec,
    discover_solvers,
)
from mosaic.benchmarks.core.status_checks import max_final_ratio
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
    plot_physical_laws,
)
from mosaic.benchmarks.problems.shared.plots.gradient import (
    plot_fd_check,
    plot_jacobian_svd,
    plot_param_sweep,
)
from mosaic.benchmarks.problems.shared.plots.ics import plot_ic
from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles

from .exclusions import register as _register_exclusions
from .extras import register as _register_extras
from .ics import _gaussian_source, _random, _two_gaussians, _uniform, _zero_source
from .optimization import conductivity_recovery
from .physics import DIAGNOSTICS, make_inputs
from .plots import plot_conductivity_recovery

_TESSERACT_SLUG = "thermal-mesh"

# SIMP material parameters
_K_MAX = 1.0  # solid thermal conductivity
_P_EXP = 3.0  # SIMP penalisation exponent


# ── Solver registry ──────────────────────────────────────────────────────────

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_SLUG)

apply_styles(_SOLVERS)


for _key in ("fenics_heat", "dealii_heat", "firedrake_heat", "torch_fem_thermal"):
    _SOLVERS[_key].input_overrides = {"k_max": _K_MAX, "p_exp": _P_EXP}


# ── Problem assembly ─────────────────────────────────────────────────────────

problem = Problem(
    name="thermal-mesh",
    category_label="Heat Conduction",
    description=(
        "Quasi-2D steady heat-conduction compliance minimisation on a heated slab with SIMP "
        "material penalisation (p=3). The effective conductivity k_eff(ρ) = k_min + (k_max − k_min)·ρ³ "
        "controls heat routing; the compliance C = ∮_Γ q_n·T dΓ is the work done by the heat flux "
        "on the temperature field. The hot-spot boundary condition (central 1/3 stripe) breaks "
        "y-symmetry and drives topology optimisation toward non-trivial branching structures. "
        "Also supports source-identification experiments: recover the volumetric heat source f(x) "
        "from temperature observations via the identification_error = ||T − T_target||² objective."
    ),
    bc_description=(
        "Quasi-2D heated slab on domain [0,2]×[0,1] (nz=1 HEX8 layer). "
        "Dirichlet: all nodes at x=0 held at T=0 (fixed temperature). "
        "Neumann (uniform): uniform heat flux Q_total over the right face (x=2). "
        "Neumann (hot-spot): flux concentrated on the central 1/3 stripe in y "
        "(Ly/3 ≤ y ≤ 2Ly/3) at the right face, driving non-trivial topology."
    ),
    tesseract_dir=_TESSERACT_SLUG,
    output_key="thermal_compliance",
    ic_key="rho",
    solvers=list(_SOLVERS.values()),
    make_inputs=make_inputs,
    error_fn=l2_error_rel,
    domain_extent=2.0,
    resolution_key="nx",
    status_checks={
        # Recovery / optimisation experiments must actually reduce loss,
        # not just complete. Same 50% floor as the other problems — solvers
        # landing at final/initial > 0.5 show up as anom so the status
        # accurately reflects "hasn't converged".
        "optimization": [max_final_ratio(0.5)],
    },
)


# ── Initial conditions ───────────────────────────────────────────────────────
problem.add_ic(
    "uniform",
    fn=_uniform,
    description=(
        "Uniform SIMP thermal conductivity density ρ₀ over all hex mesh elements; "
        "standard homogeneous starting point for heat-conduction topology optimisation."
    ),
    plot_params={"rho_0": 0.5, "nx": 16, "ny": 8, "nz": 1},
    plot=plot_ic,
)
problem.add_ic(
    "random",
    fn=_random,
    description=(
        "Gaussian-noise density field centred at ρ₀=0.5 (σ=0.3, clipped to [0.05, 0.95]); "
        "breaks spatial symmetry to produce non-trivial per-cell gradient sensitivity maps."
    ),
    plot_params={
        "rho_0": 0.5,
        "noise": 0.3,
        "nx": 16,
        "ny": 8,
        "nz": 1,
        "seed": 0,
    },
    plot=plot_ic,
)
problem.add_ic(
    "gaussian_source",
    fn=_gaussian_source,
    description=(
        "Gaussian heat source centred at (cx·Lx, cy·Ly) = (0.5·Lx, 0.5·Ly) with "
        "width σ·min(Lx,Ly). Used as the control field for source-identification experiments "
        "(ic_field='source' in physics dict)."
    ),
    plot_params={
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
problem.add_ic(
    "zero_source",
    fn=_zero_source,
    description=(
        "Zero volumetric heat source; standard zero-initialisation for source-recovery experiments."
    ),
    plot_params={"nx": 16, "ny": 8, "nz": 1},
    plot=plot_ic,
)
problem.add_ic(
    "two_gaussians",
    fn=_two_gaussians,
    description=(
        "Two-Gaussian volumetric heat source at (0.3·Lx, 0.5·Ly) and (0.7·Lx, 0.5·Ly). "
        "Ground-truth source for source-recovery experiments."
    ),
    plot_params={"nx": 16, "ny": 8, "nz": 1},
    plot=plot_ic,
)


# ── Experiment registrations ─────────────────────────────────────────────────

# Forward
problem.add_experiment(
    "forward/baseline",
    agreement,
    plot_description="Thermal compliance C vs mesh resolution N with random density; compares FV and FEM solvers across refinements.",
    ic={"name": "random", "seed": 0},
    physics={
        "N": [2, 3, 4, 6, 8, 12, 16, 24],
        "nz": 1,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "Q_total": 1.0,
    },
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/agreement",
    agreement,
    plot_description="Thermal compliance C vs uniform element density ρ₀ at fixed N; compares solvers on a log scale.",
    ic={"name": "uniform", "seed": 0},
    physics={
        "nx": 16,
        "ny": 8,
        "nz": 1,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "Q_total": 1.0,
        "rho_0": [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
    },
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/physical_laws",
    physical_laws,
    plot_description="Thermal compliance C vs total heat flux Q_total at fixed N and ρ₀ with a hot-spot BC; shown on log-log axes.",
    diagnostics=DIAGNOSTICS,
    ic={"name": "uniform", "seed": 0},
    physics={
        "N": 16,
        "nz": 1,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "rho_0": 0.5,
        "hot_spot": True,
        "Q_total": [0.25, 0.5, 1.0, 2.0, 4.0],
    },
    plot=plot_physical_laws,
)
problem.add_experiment(
    "forward/source_baseline",
    agreement,
    plot_description="Thermal compliance C vs mesh resolution N with a Gaussian source field; compares solvers across refinements.",
    ic={"name": "gaussian_source"},
    physics={
        "N": [4, 6, 8, 12, 16, 24],
        "nz": 1,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "rho_0": 0.5,
        "ic_field": "source",
    },
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/source_linearity",
    agreement,
    plot_description="Thermal compliance C vs source amplitude at fixed mesh; compares solvers on log-log axes.",
    ic={"name": "gaussian_source"},
    physics={
        "nx": 16,
        "ny": 8,
        "nz": 1,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "rho_0": 0.5,
        "ic_field": "source",
        "amplitude": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
    },
    plot=plot_agreement,
)

# Cost
_THERMAL_PHYS = {
    "Lx": 2.0,
    "Ly": 1.0,
    "Lz": 1.0,
    "Q_total": 1.0,
    "rho_0": 0.5,
}
_THERMAL_NX = [16, 32, 64, 128, 256, 512, 1024]
problem.add_experiment(
    "cost/spatial_cost",
    spatial_cost,
    plot_description="Forward-pass wall-clock time vs mesh size (nx) for all solvers.",
    physics={**_THERMAL_PHYS, "steps": 1, "nx": _THERMAL_NX},
    cost={"n_trials": 3},
    plot=plot_cost,
)
problem.add_experiment(
    "cost/temporal_cost",
    temporal_cost,
    plot_description="Forward-pass wall-clock time vs time-axis size for all solvers.",
    physics={**_THERMAL_PHYS, "nx": 64, "steps": [1]},
    cost={"n_trials": 3},
    plot=plot_cost,
)
problem.add_experiment(
    "cost/vjp_cost",
    vjp_cost,
    plot_description="VJP wall-clock time vs mesh size (nx) for differentiable solvers.",
    runs=[
        {
            "name": "by_N",
            "physics": {**_THERMAL_PHYS, "steps": 1, "nx": _THERMAL_NX},
            "cost": {"n_trials": 3},
        },
        {
            "name": "by_steps",
            "physics": {**_THERMAL_PHYS, "nx": 64, "steps": [1]},
            "cost": {"n_trials": 3},
        },
    ],
    plot=plot_cost,
)

# Gradient
problem.add_experiment(
    "gradient/fd_check",
    fd_check,
    plot_description="FD gradient error vs step size ε (U-curves), AD/FD direction cosine, and gradient magnitude field panels.",
    ic={"name": "random", "seed": 0},
    physics={
        "nx": 8,
        "ny": 4,
        "nz": 1,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "Q_total": 1.0,
    },
    fd={"eps_values": [1e0, 1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 6},
    plot=plot_fd_check,
)
problem.add_experiment(
    "gradient/param_sweep",
    param_sweep,
    plot_description="Gradient norm, best-ε FD error, AD/FD direction cosine, and U-curves vs element density ρ₀.",
    ic={"name": "uniform", "seed": 0},
    physics={
        "nx": 8,
        "ny": 4,
        "nz": 1,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "Q_total": 1.0,
        "rho_0": [0.1, 0.2, 0.4, 0.6, 0.8],
    },
    fd={"eps_values": [1e0, 1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 6},
    plot=plot_param_sweep,
)
problem.add_experiment(
    "gradient/jacobian_svd",
    jacobian_svd,
    plot_description="Singular-value spectrum of stacked per-solver gradients and pairwise cosine similarity between solver gradient directions.",
    ic={"name": "random", "seed": 0},
    physics={
        "nx": 8,
        "ny": 4,
        "nz": 1,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "Q_total": 1.0,
    },
    jacobian={"n_alphas": 21, "alpha_range": 0.2},
    plot=plot_jacobian_svd,
)

# Source-identification gradient experiments.
# Use source as the differentiable input and identification_error as the
# objective; per-run ``ic_key`` / ``output_key`` overrides the global defaults.
problem.add_experiment(
    "gradient/source_fd_check",
    fd_check,
    plot_description="FD gradient error vs ε, AD/FD direction cosine, and gradient field panels for d(identification_error)/d(source).",
    ic={"name": "gaussian_source"},
    ic_key="source",
    output_key="identification_error",
    physics={
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
    fd={"eps_values": [1e0, 1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 6},
    plot=plot_fd_check,
)
problem.add_experiment(
    "gradient/source_width_sweep",
    param_sweep,
    plot_description="Gradient norm, best-ε FD error, AD/FD direction cosine, and U-curves vs source width σ.",
    ic={"name": "gaussian_source"},
    ic_key="source",
    output_key="identification_error",
    physics={
        "nx": 16,
        "ny": 8,
        "nz": 1,
        "rho_0": 0.5,
        "target_from_two_gaussians": True,
        "ic_field": "source",
    },
    fd={"eps_values": [1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 4},
    sweep={
        "key": "sigma",
        "values": [0.05, 0.1, 0.2, 0.3, 0.5],
        "ic_sweep": True,
    },
    plot=plot_param_sweep,
)

# Optimization
# PR #22 (CI): keep only the LBFGS variant to bound suite wall time.
if False:
    problem.add_experiment(
        "optimization/conductivity_recovery",
        conductivity_recovery,
        plot_description="Optimisation traces (loss vs iteration) and recovered conductivity fields vs the two-Gaussian ground truth, using gradient descent.",
        ic={"name": "uniform", "seed": 0},
        physics={
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
        optim={"lr": 1e-2, "max_iters": 2000, "patience": 200},
        plot=plot_conductivity_recovery,
    )
problem.add_experiment(
    "optimization/conductivity_recovery_bfgs",
    conductivity_recovery,
    optimizer="bfgs",
    plot_description="Optimisation traces (loss vs iteration) and recovered conductivity fields vs the two-Gaussian ground truth, using L-BFGS.",
    ic={"name": "uniform", "seed": 0},
    physics={
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
    optim={"max_iters": 200, "patience": 30},
    plot=plot_conductivity_recovery,
)


# ── Cross-experiment extras ──────────────────────────────────────────────────
# Aggregator plots that span multiple experiments live in ``extras.py``; the
# register call wires them into ``problem.plot_fns`` under ``_extra/...``.
_register_extras(problem)


# ── Exclusions ───────────────────────────────────────────────────────────────
# Per-solver exclusions live in ``exclusions.py``; the register call here
# wires them into the same longest-prefix lookup ``mosaic status`` uses.
_register_exclusions(problem)


__all__ = ["problem"]
