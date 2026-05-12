"""Assembled ``EXPERIMENTS`` registry for thermal-mesh.

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
from mosaic.benchmarks.shared.optimization import (
    run_conductivity_recovery,
    run_conductivity_recovery_bfgs,
)

from .ics import MAKE_IC
from .physics import DIAGNOSTICS

# ── Forward run-lists ────────────────────────────────────────────────────────

_BASELINE_RUNS = [
    {
        "ic": {"name": "random", "seed": 0},
        "physics": {"nz": 1, "Lx": 2.0, "Ly": 1.0, "Lz": 1.0, "Q_total": 1.0},
        "sweep": {"key": "N", "values": [2, 3, 4, 6, 8, 12, 16, 24]},
    }
]
_AGREEMENT_RUNS = [
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
        "sweep": {"key": "rho_0", "values": [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]},
    }
]
_PHYSICAL_LAWS_RUNS = [
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
]
_SOURCE_BASELINE_RUNS = [
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
]
_SOURCE_LINEARITY_RUNS = [
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
        "sweep": {"key": "amplitude", "values": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]},
    }
]

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

# ── Gradient run-lists ───────────────────────────────────────────────────────

_FD_CHECK_RUNS = [
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
]
_PARAM_SWEEP_RUNS = [
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
]
_JACOBIAN_SVD_RUNS = [
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
]
# Source-identification gradient experiments.
# These use source as the differentiable input and identification_error as the
# objective; per-run ``ic_key`` / ``output_key`` overrides the global defaults.
_SOURCE_FD_CHECK_RUNS = [
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
]
_SOURCE_WIDTH_SWEEP_RUNS = [
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
]

# ── Optimization run-lists ───────────────────────────────────────────────────

_CONDUCTIVITY_RECOVERY_RUNS = [
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
]
_CONDUCTIVITY_RECOVERY_BFGS_RUNS = [
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
            output_key="thermal_compliance",
            domain_extent=2.0,
            analytic=None,
            runs=_BASELINE_RUNS,
            exp_key="baseline",
            **kw,
        ),
        params={
            "runs": _BASELINE_RUNS,
            "plot_description": (
                "Thermal compliance C vs mesh resolution N (nx=2,3,4,6,8,12,16,24; ny=nx//2; nz=1) "
                "with random density ρ~N(0.5,0.3) clipped to [0.05,0.95]. FV solvers diverge "
                "from FEM at coarse N due to harmonic-mean vs Galerkin conductivity interpolation; "
                "gap closes as O(h) with refinement. N=2–4 is the phase-transition regime."
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
            output_key="thermal_compliance",
            domain_extent=2.0,
            analytic=None,
            runs=_AGREEMENT_RUNS,
            exp_key="agreement",
            **kw,
        ),
        params={
            "runs": _AGREEMENT_RUNS,
            "plot_description": (
                "Thermal compliance C vs uniform element density ρ₀ ∈ [0.01, 0.95] at N=16. "
                "C ∝ ρ⁻³ due to SIMP (p=3); near-void (ρ→0) divergence between FV harmonic-mean "
                "and FEM Galerkin conductivity is the key discriminator. Log scale recommended."
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
            output_key="thermal_compliance",
            domain_extent=2.0,
            analytic=None,
            diagnostics=DIAGNOSTICS,
            runs=_PHYSICAL_LAWS_RUNS,
            **kw,
        ),
        params={
            "runs": _PHYSICAL_LAWS_RUNS,
            "plot_description": (
                "Thermal compliance C vs total heat flux Q_total at fixed N=16, ρ₀=0.5, hot_spot=True. "
                "For a linear system C ∝ Q² (log-log slope 2.0). Hot-spot BC concentrates flux on "
                "central 1/3 stripe in y, breaking symmetry; deviations across solvers reveal "
                "errors in Neumann mask handling or compliance integral."
            ),
        },
    ),
    "forward/source_baseline": Experiment(
        fn=lambda cfg, tags, **kw: run_agreement(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="thermal_compliance",
            domain_extent=2.0,
            analytic=None,
            runs=_SOURCE_BASELINE_RUNS,
            exp_key="source_baseline",
            **kw,
        ),
        params={
            "runs": _SOURCE_BASELINE_RUNS,
            "plot_description": "",
        },
    ),
    "forward/source_linearity": Experiment(
        fn=lambda cfg, tags, **kw: run_agreement(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="thermal_compliance",
            domain_extent=2.0,
            analytic=None,
            runs=_SOURCE_LINEARITY_RUNS,
            exp_key="source_linearity",
            **kw,
        ),
        params={
            "runs": _SOURCE_LINEARITY_RUNS,
            "plot_description": "",
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
            "plot_description": "",
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
            output_key="thermal_compliance",
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
            output_key="thermal_compliance",
            ic_key="rho",
            domain_extent=2.0,
            runs=_FD_CHECK_RUNS,
            exp_key="fd_check",
            **kw,
        ),
        params={
            "runs": _FD_CHECK_RUNS,
            "plot_description": (
                "U-curves (FD gradient error vs ε), direction cosine between AD and FD "
                "gradient vectors, and gradient magnitude field panels."
            ),
        },
    ),
    "gradient/param_sweep": Experiment(
        fn=lambda cfg, tags, **kw: run_param_sweep(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="thermal_compliance",
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
            output_key="thermal_compliance",
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
                "pairwise cosine similarity between JAX-FEM and FEniCS gradient directions "
                "for the thermal compliance objective. Near-unity cosine confirms consistent "
                "adjoint implementations; spectrum reveals dominant sensitivity modes of the "
                "density field."
            ),
        },
    ),
    "gradient/source_fd_check": Experiment(
        fn=lambda cfg, tags, **kw: run_fd_check(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="thermal_compliance",
            ic_key="rho",
            domain_extent=2.0,
            runs=_SOURCE_FD_CHECK_RUNS,
            exp_key="source_fd_check",
            **kw,
        ),
        params={
            "runs": _SOURCE_FD_CHECK_RUNS,
            "plot_description": (
                "FD gradient check of d(identification_error)/d(source) at nominal mesh. "
                "Uses ic_field='source' and output_key='identification_error'."
            ),
        },
    ),
    "gradient/source_width_sweep": Experiment(
        fn=lambda cfg, tags, **kw: run_param_sweep(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            output_key="thermal_compliance",
            ic_key="rho",
            domain_extent=2.0,
            runs=_SOURCE_WIDTH_SWEEP_RUNS,
            exp_key="source_width_sweep",
            **kw,
        ),
        params={
            "runs": _SOURCE_WIDTH_SWEEP_RUNS,
            "plot_description": (
                "Gradient quality vs source localisation σ. "
                "Phase transition: FEM/FD disagree as source narrows below element size."
            ),
        },
    ),
    # ─ Optimization ─
    "optimization/conductivity_recovery": Experiment(
        fn=lambda cfg, tags, **kw: run_conductivity_recovery(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="thermal_compliance",
            domain_extent=2.0,
            runs=_CONDUCTIVITY_RECOVERY_RUNS,
            **kw,
        ),
        params={
            "runs": _CONDUCTIVITY_RECOVERY_RUNS,
            "plot_description": (
                "Recover a two-Gaussian conductivity field from temperature observations. "
                "Optimises rho (SIMP density, clipped to [x_min, 1]) to minimise "
                "identification_error = ||T(rho) - T_target||². Target temperature is "
                "produced by forward-solving with a two-Gaussian ground-truth conductivity "
                "at uniform zero volumetric source (driven by Neumann BC only)."
            ),
        },
    ),
    "optimization/conductivity_recovery_bfgs": Experiment(
        fn=lambda cfg, tags, **kw: run_conductivity_recovery_bfgs(
            cfg,
            tags,
            make_ic=MAKE_IC,
            make_inputs=cfg.make_inputs,
            error_fn=l2_error_rel,
            output_key="thermal_compliance",
            domain_extent=2.0,
            runs=_CONDUCTIVITY_RECOVERY_BFGS_RUNS,
            **kw,
        ),
        params={
            "runs": _CONDUCTIVITY_RECOVERY_BFGS_RUNS,
            "plot_description": (
                "Recover a two-Gaussian conductivity field with L-BFGS. Same setup as "
                "conductivity_recovery but using L-BFGS with zoom line search."
            ),
        },
    ),
    # ─ ICs ─
    "ics/zero_source": Experiment(
        fn=lambda cfg, tags, **kw: run_ic(
            cfg,
            "zero_source",
            make_ic=MAKE_IC,
            params={"nx": 16, "ny": 8, "nz": 1},
        ),
        params={"nx": 16, "ny": 8, "nz": 1},
    ),
    "ics/uniform": Experiment(
        fn=lambda cfg, tags, **kw: run_ic(
            cfg,
            "uniform",
            make_ic=MAKE_IC,
            params={"rho_0": 0.5, "nx": 16, "ny": 8, "nz": 1},
        ),
        params={"rho_0": 0.5, "nx": 16, "ny": 8, "nz": 1},
    ),
    "ics/random": Experiment(
        fn=lambda cfg, tags, **kw: run_ic(
            cfg,
            "random",
            make_ic=MAKE_IC,
            params={"rho_0": 0.5, "noise": 0.3, "nx": 16, "ny": 8, "nz": 1, "seed": 0},
        ),
        params={"rho_0": 0.5, "noise": 0.3, "nx": 16, "ny": 8, "nz": 1, "seed": 0},
    ),
    "ics/gaussian_source": Experiment(
        fn=lambda cfg, tags, **kw: run_ic(
            cfg,
            "gaussian_source",
            make_ic=MAKE_IC,
            params={
                "nx": 16,
                "ny": 8,
                "nz": 1,
                "amplitude": 1.0,
                "cx": 0.5,
                "cy": 0.5,
                "sigma": 0.2,
            },
        ),
        params={
            "nx": 16,
            "ny": 8,
            "nz": 1,
            "amplitude": 1.0,
            "cx": 0.5,
            "cy": 0.5,
            "sigma": 0.2,
        },
    ),
    "ics/two_gaussians": Experiment(
        fn=lambda cfg, tags, **kw: run_ic(
            cfg,
            "two_gaussians",
            make_ic=MAKE_IC,
            params={"nx": 16, "ny": 8, "nz": 1},
        ),
        params={"nx": 16, "ny": 8, "nz": 1},
    ),
}
