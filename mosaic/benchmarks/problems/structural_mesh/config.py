# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""3D linear-elasticity SIMP topology optimisation on a cantilever beam.

The problem definition is split across these modules:

- :mod:`.ics`          — IC generators.
- :mod:`.physics`      — cantilever BC builder, input factory
                         (``make_inputs``), ``_infer_mesh_dims`` helper, and
                         ``DIAGNOSTICS``. The HEX8 mesh builder is shared
                         via :mod:`mosaic.benchmarks.problems.shared.mesh`.
- :mod:`.optimization` — SIMP topology-optimisation runner.
- :mod:`.plots`        — per-experiment plot fns wired in below.
- :mod:`.exclusions`   — per-(solver, experiment) opt-outs.
- :mod:`.extras`       — cross-experiment aggregator plots.

This module performs solver discovery, the canonical :class:`Problem`
assembly, and the per-suite ``problem.add_experiment(...)`` calls with
inline plot descriptions.
"""

from __future__ import annotations

from mosaic.benchmarks.core.config import (
    Problem,
    SolverSpec,
    discover_solvers,
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
    param_sweep,
)
from mosaic.benchmarks.problems.shared.plots.cost import plot_cost
from mosaic.benchmarks.problems.shared.plots.forward import (
    plot_agreement,
    plot_physical_laws,
)
from mosaic.benchmarks.problems.shared.plots.gradient import (
    plot_fd_check,
    plot_param_sweep,
)
from mosaic.benchmarks.problems.shared.plots.ics import plot_ic
from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles

from .exclusions import register as _register_exclusions
from .ics import _random, _two_density_bumps, _uniform
from .optimization import topopt
from .physics import DIAGNOSTICS, make_inputs
from .plots import plot_topopt

_TESSERACT_SLUG = "structural-mesh"

# SIMP material parameters — matched between solvers
_E_MAX = 70_000.0  # Young's modulus of solid [MPa]
_NU = 0.3  # Poisson's ratio
_XMIN = 1e-3  # Void stiffness ratio (E_min / E_max)


# ── Solver registry ──────────────────────────────────────────────────────────

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_SLUG)

apply_styles(_SOLVERS)


# Material parameters: ``nu``/``xmin`` are shared; only the modulus kwarg name
# differs (TopOpt.jl uses ``E``, FEM backends use ``E_max``).
_MAT_SHARED = {"nu": _NU, "xmin": _XMIN}
_SOLVERS["topopt_jl"].input_overrides = {"E": _E_MAX, **_MAT_SHARED}
for _key in ("dealii_structural", "fenics_structural", "firedrake_structural"):
    _SOLVERS[_key].input_overrides = {"E_max": _E_MAX, **_MAT_SHARED}


# ── Problem assembly ─────────────────────────────────────────────────────────

problem = Problem(
    name="structural-mesh",
    category_label="Structural mechanics",
    description=(
        "**Designing a stiff structure.** Given a fixed amount of material, where "
        "should it go to make a loaded beam as rigid as possible? This is *topology "
        "optimization*, solved by differentiating a finite-element "
        "stress analysis with respect to a per-element material density field $\\rho$.\n\n"
        "We minimize the compliance (inverse stiffness) of a 3D linear-elastic "
        "cantilever beam under the SIMP density-penalization scheme "
        "($p=3$, $E_\\max = 70{,}000$ MPa). Each element's stiffness follows the "
        "constitutive relation $E_\\text{eff}(\\rho) = E_\\min + (E_\\max - E_\\min)\\,\\rho^3$, "
        "and the global stiffness matrix $K(\\rho)$ couples every element to the "
        "displacement field. The objective $C = \\mathbf{F}^\\top K(\\rho)^{-1}\\mathbf{F}$ "
        "(external work under load $\\mathbf{F}$) is smooth but non-convex in $\\rho$, so "
        "gradient-based optimization drives the design toward sparse, near-binary "
        "$0/1$ material layouts, and the gradient must stay reliable throughout."
    ),
    bc_description=(
        "3D cantilever beam on domain $[0,2]\\times[0,1]\\times[0,1]$ "
        "(HEX8 elements, 2:1:1 aspect). "
        "Dirichlet: all nodes at $x=0$ have zero displacement (clamped wall). "
        "Neumann: a prescribed total force is applied to the right face ($x=2$), "
        "either a uniform downward traction or a concentrated upward corner load "
        "depending on the experiment (controlled by the corner_load flag)."
    ),
    tesseract_dir=_TESSERACT_SLUG,
    output_key="compliance",
    ic_key="rho",
    solvers=list(_SOLVERS.values()),
    make_inputs=make_inputs,
    error_fn=l2_error_rel,
    domain_extent=2.0,
    resolution_key="nx",
)


# ── IC registrations ─────────────────────────────────────────────────────────

problem.add_ic(
    "uniform",
    _uniform,
    description=(
        "Uniform SIMP material density $\\rho_0$ over all hex mesh elements; standard "
        "homogeneous starting point for topology optimisation of the cantilever beam."
    ),
    plot_params={"rho_0": 0.5, "nx": 16},
    plot=plot_ic,
)
problem.add_ic(
    "random",
    _random,
    description=(
        "Gaussian-noise density field centred at $\\rho_0=0.5$ ($\\sigma=0.3$, clipped to $[0.05, 0.95]$); "
        "breaks spatial symmetry so gradient experiments see non-trivial per-cell sensitivity."
    ),
    plot_params={},
    plot=plot_ic,
)
problem.add_ic(
    "two_density_bumps",
    _two_density_bumps,
    description=(
        "Ground-truth density with two stiff Gaussian pillars "
        "($\\rho_\\mathrm{peak}=0.95$, $\\sigma=0.12\\min(L_x, L_z)$) "
        "at $(0.35 L_x, 0.5 L_y, 0.5 L_z)$ and $(0.75 L_x, 0.5 L_y, 0.5 L_z)$ "
        "on a soft $\\rho_\\mathrm{bg}=0.1$ "
        "background; analog of thermal-mesh ``two_gaussians`` for the load-recovery inverse "
        "experiment (recover density from displacement observations)."
    ),
    plot_params={"nx": 16, "ny": 2, "nz": 8},
    plot=plot_ic,
)


# ── Experiment registrations ─────────────────────────────────────────────────

# Forward
problem.add_experiment(
    "forward/baseline",
    agreement,
    plot_description=(
        "Structural compliance $C = \\mathbf{F}^\\top \\mathbf{u}$ vs mesh resolution $N$ for each solver, "
        "uniform density $\\rho_0=0.5$, full-face downward load."
    ),
    ic={"name": "uniform", "seed": 0},
    physics={
        "N": [4, 6, 8, 12, 16],
        "ny": 2,
        "nz": 4,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "corner_load": False,
    },
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/agreement",
    agreement,
    plot_description=(
        "Structural compliance $C = \\mathbf{F}^\\top \\mathbf{u}$ vs density $\\rho_0$ at fixed mesh, "
        "sweeping uniform density to span the SIMP stiffness regime."
    ),
    ic={"name": "uniform", "seed": 0},
    physics={
        "nx": 8,
        "ny": 2,
        "nz": 4,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "corner_load": False,
        "rho_0": [0.2, 0.4, 0.5, 0.7, 0.9],
    },
    plot=plot_agreement,
)
problem.add_experiment(
    "forward/physical_laws",
    physical_laws,
    plot_description=(
        "Diagnostic functionals (compliance, total displacement) vs total load $F_\\mathrm{total}$, "
        "validating linearity of the SIMP response."
    ),
    diagnostics=DIAGNOSTICS,
    ic={"name": "uniform", "seed": 0},
    physics={
        "nx": 8,
        "ny": 2,
        "nz": 4,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "corner_load": False,
        "rho_0": 0.5,
        "F_total": [0.25, 0.5, 1.0, 2.0, 4.0],
    },
    plot=plot_physical_laws,
)

# Cost
_MESH_PHYS = {
    "Lx": 2.0,
    "Ly": 1.0,
    "Lz": 1.0,
    "F_total": 1.0,
    "rho_0": 0.5,
    "corner_load": False,
}
problem.add_experiment(
    "cost/spatial_cost",
    spatial_cost,
    plot_description="Forward-pass wall-clock time vs mesh resolution $N$ at one assembly step.",
    physics={**_MESH_PHYS, "steps": 1, "nx": [4, 6, 8, 12, 16]},
    cost={"n_trials": 3},
    plot=plot_cost,
)
problem.add_experiment(
    "cost/temporal_cost",
    temporal_cost,
    plot_description=(
        "Forward-pass wall-clock time vs solve count at fixed mesh "
        "(single-step assembly is the dominant cost — temporal axis "
        "collapses to one point)."
    ),
    physics={**_MESH_PHYS, "nx": 8, "steps": [1]},
    cost={"n_trials": 3},
    plot=plot_cost,
)
problem.add_experiment(
    "cost/vjp_cost",
    vjp_cost,
    plot_description="VJP wall-clock time vs mesh resolution $N$ for differentiable solvers.",
    runs=[
        {
            "name": "by_N",
            "physics": {**_MESH_PHYS, "steps": 1, "nx": [4, 6, 8, 12, 16]},
            "cost": {"n_trials": 3},
        },
        {
            "name": "by_steps",
            "physics": {**_MESH_PHYS, "nx": 8, "steps": [1]},
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
        "U-curves of finite-difference gradient error vs perturbation size $\\varepsilon$ "
        "with subspace cosine, validating VJP correctness on a random density."
    ),
    ic={"name": "random", "seed": 0},
    physics={
        "nx": 8,
        "ny": 2,
        "nz": 4,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "corner_load": True,
    },
    fd={
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
    plot=plot_fd_check,
)
problem.add_experiment(
    "gradient/param_sweep",
    param_sweep,
    plot_description=(
        "Gradient norm, best-$\\varepsilon$ FD error, direction cosine, and "
        "U-curves vs uniform density $\\rho_0$."
    ),
    ic={"name": "uniform", "seed": 0},
    physics={
        "nx": 8,
        "ny": 2,
        "nz": 4,
        "Lx": 2.0,
        "Ly": 1.0,
        "Lz": 1.0,
        "F_total": 1.0,
        "corner_load": True,
        "rho_0": [0.2, 0.4, 0.6, 0.8],
    },
    fd={
        "eps_values": [5e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4],
        "n_dirs": 6,
    },
    plot=plot_param_sweep,
)
problem.add_experiment(
    "optimization/topopt",
    topopt,
    plot_description=(
        "SIMP topology optimisation on a $16\\times8\\times8$ cantilever beam with Adam (lr=0.05): "
        "compliance $C = \\mathbf{F}^\\top \\mathbf{u}$ and density field evolution under a "
        "50% volume-fraction constraint."
    ),
    ic={"name": "uniform", "seed": 0},
    physics={
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
    optim={"lr": 5e-2, "max_iters": 2500, "patience": 100},
    plot=plot_topopt,
)


# ── Exclusions ───────────────────────────────────────────────────────────────
# Per-solver exclusions live in ``exclusions.py``; the register call here
# wires them into the same longest-prefix lookup ``mosaic status`` uses.
_register_exclusions(problem)


__all__ = ["problem"]
