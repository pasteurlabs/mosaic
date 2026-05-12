"""Assembled ``EXPERIMENTS`` registry for structural-mesh.

Every entry is a fully-explicit ``Experiment(fn=lambda ..., params=...)``
literal: the runner, the runs list, every closure-captured dependency, and
the introspection params are all visible at the call site. No helpers, no
dispatch tables — adding/changing an experiment is a local edit on the
entry itself.
"""

from __future__ import annotations

from mosaic.benchmarks.core.config import Experiment
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.shared.cost import (
    run_spatial_cost,
    run_temporal_cost,
    run_vjp_cost,
)
from mosaic.benchmarks.shared.forward import run_agreement, run_physical_laws
from mosaic.benchmarks.shared.gradient import (
    run_fd_check,
    run_jacobian_svd,
    run_param_sweep,
)
from mosaic.benchmarks.shared.ics import run_ic
from mosaic.benchmarks.shared.optimization import run_topopt, run_topopt_bfgs

from .ics import MAKE_IC
from .physics import DIAGNOSTICS

# ── Forward run-lists ────────────────────────────────────────────────────────

_BASELINE_RUNS = [
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
]
_AGREEMENT_RUNS = [
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
]
_PHYSICAL_LAWS_RUNS = [
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
]

# ── Cost run-list (shared by spatial/temporal/vjp) ───────────────────────────

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
            "N_values": [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3200],
            "n_trials": 3,
        },
    }
]

# ── Gradient run-lists ───────────────────────────────────────────────────────

_FD_CHECK_RUNS = [
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
]
_PARAM_SWEEP_RUNS = [
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
]
_JACOBIAN_SVD_RUNS = [
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
]

# ── Optimization run-lists ───────────────────────────────────────────────────

_TOPOPT_RUNS = [
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
]
_TOPOPT_BFGS_RUNS = [
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
]


# ── Assembled experiment registry ────────────────────────────────────────────

EXPERIMENTS = {
    # ─ Forward ─
    "forward/baseline": Experiment(
        fn=lambda cfg, tags, **kw: run_agreement(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="compliance",
            domain_extent=2.0,
            analytic=None,
            runs=_BASELINE_RUNS,
            exp_key="baseline",
            **kw,
        ),
        params={
            "runs": _BASELINE_RUNS,
            "plot_description": (
                "Structural compliance C = F^T U vs mesh resolution N for each solver "
                "(uniform density ρ₀=0.5, full-face downward load). "
                "Both solvers implement HEX8 FEM; compliance should agree to <1% at all resolutions."
            ),
        },
    ),
    "forward/agreement": Experiment(
        fn=lambda cfg, tags, **kw: run_agreement(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="compliance",
            domain_extent=2.0,
            analytic=None,
            runs=_AGREEMENT_RUNS,
            exp_key="agreement",
            **kw,
        ),
        params={
            "runs": _AGREEMENT_RUNS,
            "plot_description": (
                "Structural compliance C = F^T U vs element density ρ₀ for each solver "
                "(log-scale; full-face downward load). "
                "jax_fem uses surface traction; topopt_jl distributes force uniformly across "
                "right-face nodes — non-uniform shape-function weighting in jax_fem causes "
                "a small but consistent compliance difference (~0.5–3%) across all densities."
            ),
        },
    ),
    "forward/physical_laws": Experiment(
        fn=lambda cfg, tags, **kw: run_physical_laws(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="compliance",
            domain_extent=2.0,
            analytic=None,
            diagnostics=DIAGNOSTICS,
            runs=_PHYSICAL_LAWS_RUNS,
            **kw,
        ),
        params={
            "runs": _PHYSICAL_LAWS_RUNS,
            "plot_description": (
                "Structural compliance C = F^T U vs total load F_total at fixed N=8 (nx=8, ny=2, nz=4), ρ₀=0.5. "
                "For linear elasticity C = F^T K⁻¹ F ∝ F², so log-log slope must be 2.0. "
                "Deviations across solvers reveal errors in the stiffness assembly or force application."
            ),
        },
    ),
    # ─ Cost ─
    "cost/spatial_cost": Experiment(
        fn=lambda cfg, tags, **kw: run_spatial_cost(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            domain_extent=2.0,
            resolution_key="nx",
            runs=_COST_RUNS,
            **kw,
        ),
        params={
            "runs": _COST_RUNS,
            "plot_description": "Forward-pass wall-clock time vs mesh size (nx) for all solvers.",
        },
    ),
    "cost/temporal_cost": Experiment(
        fn=lambda cfg, tags, **kw: run_temporal_cost(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            domain_extent=2.0,
            resolution_key="nx",
            runs=_COST_RUNS,
            **kw,
        ),
        params={
            "runs": _COST_RUNS,
            "plot_description": "Forward-pass wall-clock time vs step count at fixed mesh size for all solvers.",
        },
    ),
    "cost/vjp_cost": Experiment(
        fn=lambda cfg, tags, **kw: run_vjp_cost(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            domain_extent=2.0,
            resolution_key="nx",
            output_key="compliance",
            ic_key="rho",
            runs=_COST_RUNS,
            **kw,
        ),
        params={
            "runs": _COST_RUNS,
            "plot_description": "VJP wall-clock time vs mesh size (nx) for differentiable solvers.",
        },
    ),
    # ─ Gradient ─
    "gradient/fd_check": Experiment(
        fn=lambda cfg, tags, **kw: run_fd_check(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="compliance",
            ic_key="rho",
            domain_extent=2.0,
            runs=_FD_CHECK_RUNS,
            exp_key="fd_check",
            **kw,
        ),
        params={
            "runs": _FD_CHECK_RUNS,
            "plot_description": "U-curves (FD gradient error vs ε), direction cosine between AD and FD gradient vectors, and gradient magnitude field panels.",
        },
    ),
    "gradient/param_sweep": Experiment(
        fn=lambda cfg, tags, **kw: run_param_sweep(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="compliance",
            ic_key="rho",
            domain_extent=2.0,
            runs=_PARAM_SWEEP_RUNS,
            exp_key="param_sweep",
            **kw,
        ),
        params={
            "runs": _PARAM_SWEEP_RUNS,
            "plot_description": "Gradient norm, best-ε FD error, direction cosine, and U-curves vs element density ρ₀.",
        },
    ),
    "gradient/jacobian_svd": Experiment(
        fn=lambda cfg, tags, **kw: run_jacobian_svd(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="compliance",
            ic_key="rho",
            domain_extent=2.0,
            runs=_JACOBIAN_SVD_RUNS,
            exp_key="jacobian_svd",
            **kw,
        ),
        params={
            "runs": _JACOBIAN_SVD_RUNS,
            "plot_description": (
                "Singular-value spectrum of the stacked per-solver gradient matrix and "
                "pairwise cosine similarity between JAX-FEM and TopOpt.jl gradient directions. "
                "Both solvers implement the same SIMP adjoint so cosine similarity should be "
                "near 1; deviations indicate differing adjoint formulations or numerical precision."
            ),
        },
    ),
    # ─ Optimization ─
    "optimization/topopt": Experiment(
        fn=lambda cfg, tags, **kw: run_topopt(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="compliance",
            domain_extent=2.0,
            runs=_TOPOPT_RUNS,
            **kw,
        ),
        params={
            "runs": _TOPOPT_RUNS,
            "plot_description": (
                "SIMP topology optimisation on a 16×8×8 cantilever beam: minimise compliance "
                "C = F^T U subject to a 50% volume fraction constraint (Adam, lr=0.02). "
                "Density field evolves from uniform ρ=0.5 toward a binary 0/1 layout; "
                "both solvers converge to the same topology confirming consistent adjoint gradients."
            ),
        },
    ),
    "optimization/topopt_bfgs": Experiment(
        fn=lambda cfg, tags, **kw: run_topopt_bfgs(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="compliance",
            domain_extent=2.0,
            runs=_TOPOPT_BFGS_RUNS,
            **kw,
        ),
        params={
            "runs": _TOPOPT_BFGS_RUNS,
            "plot_description": (
                "SIMP topology optimisation on a 16×8×8 cantilever beam with L-BFGS: minimise compliance "
                "C = F^T U subject to a 50% volume fraction constraint."
            ),
        },
    ),
    # ─ ICs ─
    "ics/uniform": Experiment(
        fn=lambda cfg, tags, **kw: run_ic(
            cfg,
            "uniform",
            make_ic=MAKE_IC,
            params={"rho_0": 0.5, "nx": 16},
        ),
        params={"rho_0": 0.5, "nx": 16},
    ),
    "ics/random": Experiment(
        fn=lambda cfg, tags, **kw: run_ic(
            cfg,
            "random",
            make_ic=MAKE_IC,
            params={},
        ),
        params={},
    ),
    "ics/two_density_bumps": Experiment(
        fn=lambda cfg, tags, **kw: run_ic(
            cfg,
            "two_density_bumps",
            make_ic=MAKE_IC,
            params={"nx": 16, "ny": 2, "nz": 8},
        ),
        params={"nx": 16, "ny": 2, "nz": 8},
    ),
}
