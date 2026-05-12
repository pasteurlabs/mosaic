"""Solver discovery, exclusions, and the final ``Problem`` instance."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from mosaic.benchmarks.core.config import (
    Exclusion,
    ExclusionCategory,
    Problem,
    SolverSpec,
    discover_solvers,
)
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.shared.plots.solver_styles import apply_styles

from .experiments import EXPERIMENTS, PLOT_FNS
from .ics import MAKE_IC, _tgv_analytic
from .physics import DIAGNOSTICS, build_make_inputs

_GYM_DIR = Path(__file__).parent.parent.parent.parent
_TESSERACT_DIR = _GYM_DIR / "tesseracts" / "navier-stokes-grid"


# ── Solver registry ──────────────────────────────────────────────────────────
# Per-solver fields (name, scheme, color, AD strategy, …) live in each
# tesseract's ``tesseract_config.yaml`` under ``metadata.mosaic``.
# Only per-(solver, problem) overrides (input_overrides / exclusions) are
# applied here.

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_DIR)

# Exponax is a 3D-only spectral solver; exclude it from the 2D suite.
_SOLVERS.pop("exponax", None)

# Preserve historical solver key used across paper plots and CLI references.
_SOLVERS["ins_jl"] = _SOLVERS.pop("incompressible_navier_stokes_jl")

# Plot styling lives in mosaic.benchmarks.shared.plots.solver_styles, not in YAML.
apply_styles(_SOLVERS)


_SOLVERS["jax_cfd"].input_overrides = {
    "density": jnp.array([1.0], dtype=jnp.float32),
    "inner_steps": 1,
}

# ── Per-solver exclusions / explained-anomalies ──────────────────────────────
# Each entry maps a solver name to a dict of experiment-keys → Exclusion. The
# key matching is most-specific-first (`suite/experiment` > `experiment` >
# `suite`), shared with the status display via `exclusion_lookup`.
_EXCLUSIONS: dict[str, dict[str, Exclusion]] = {
    "jax_cfd": {
        "forward/cylinder": Exclusion(
            ExclusionCategory.CATEGORICAL,
            "tesseract uses periodic FFT pressure solve + IBM volume penalization; "
            "channel-BC cylinder flow requires a non-periodic pressure solve that "
            "is not wired in this benchmark",
        ),
        "optimization/drag_opt": Exclusion(
            ExclusionCategory.CATEGORICAL,
            "periodic FFT pressure solve + IBM volume penalization is incompatible "
            "with cylinder obstacle channel BCs (same root cause as forward/cylinder)",
        ),
        "forward/baseline": Exclusion(
            ExclusionCategory.ANOMALY_EXPLAINED,
            "staggered MAC grid double-interpolation: collocated TGV IC -> "
            "staggered faces -> collocated output gives sin^2(pi/N) round-trip "
            "error at all N; 35-40x above collocated peers",
        ),
    },
    "phiflow": {
        "forward/agreement/tgv": Exclusion(
            ExclusionCategory.ANOMALY_EXPLAINED,
            "phiflow's double CenteredGrid↔StaggeredGrid resampling gives 4.18% amplitude "
            "damping (ratio=0.9582); cosine=0.9999924 (pattern correct); arithmetic-average "
            "output conversion fix worsened error 9×; upstream library change required",
        ),
    },
    "ins_jl": {
        "forward/cylinder": Exclusion(
            ExclusionCategory.CATEGORICAL,
            "no IBM or volume penalization — the cylinder obstacle cannot be "
            "represented in INS.jl; spectral/LU pressure projection is also "
            "periodic-only and incompatible with obstacle channel BCs",
        ),
        "optimization/drag_opt": Exclusion(
            ExclusionCategory.CATEGORICAL,
            "no IBM or volume penalization — the cylinder obstacle cannot be "
            "represented; inflow_profile VJP works only in periodic/channel mode "
            "without obstacles",
        ),
        "forward/baseline": Exclusion(
            ExclusionCategory.ANOMALY_EXPLAINED,
            "staggered MAC grid double-interpolation: collocated TGV IC -> "
            "staggered faces -> collocated output gives sin^2(pi/N) round-trip "
            "error at all N; 35-40x above collocated peers",
        ),
    },
    "openfoam": {
        "gradient": Exclusion(
            ExclusionCategory.CATEGORICAL,
            "standard icoFoam is non-differentiable (C++, no AD path); "
            "DAFoam/OpenFOAM-AD exist but are not deployed in this tesseract",
        ),
        "optimization": Exclusion(
            ExclusionCategory.CATEGORICAL,
            "standard icoFoam is non-differentiable forward-only solver",
        ),
        "cost/vjp_cost": Exclusion(
            ExclusionCategory.CATEGORICAL,
            "standard icoFoam has no VJP to benchmark",
        ),
    },
    "xlb": {
        "forward/baseline": Exclusion(
            ExclusionCategory.ANOMALY_EXPLAINED,
            "irreducible O(Ma²) LBM compressibility error floor: at fixed "
            "dt=0.01, Ma=u·dt/dx grows with N; at N=128 Ma~0.2 giving ~0.007 "
            "error floor (230× peers); anomalous at all N",
        ),
        "forward/agreement/tgv": Exclusion(
            ExclusionCategory.ANOMALY_EXPLAINED,
            "automatic k=9 sub-steps reduce Ma 0.88→0.098 (81× Ma² reduction); "
            "errors drop from 0.216-0.278 → 0.026-0.031 (11-24× peers); "
            "remaining floor is O(dx²) LBM spatial discretization at N=64, not reducible "
            "by further sub-stepping (tested k=9..27); valid=True",
        ),
        "forward/tgv_nu_sweep": Exclusion(
            ExclusionCategory.ANOMALY_EXPLAINED,
            "same root cause as forward/agreement/tgv — automatic k=9 sub-stepping reduces Ma 0.88→0.098 "
            "but residual O(dx²) LBM spatial discretization gives 11-24× peer errors "
            "at all nu values (0.0001–0.05); 0.0309 at nu=0.05 is 12.0× peer median; "
            "not reducible by further sub-stepping (tested k=9..27); valid=True",
        ),
    },
    "warp_ns": {
        "forward/cylinder": Exclusion(
            ExclusionCategory.CATEGORICAL,
            "warp-ns is periodic-only; obstacle flows are not supported",
        ),
        "optimization/drag_opt": Exclusion(
            ExclusionCategory.CATEGORICAL,
            "warp-ns is periodic-only; obstacle/inflow flows are not supported",
        ),
    },
}


_SOLVERS_LIST = list(_SOLVERS.values())

CONFIG = Problem(
    name="ns-grid",
    exclusions=_EXCLUSIONS,
    experiments=EXPERIMENTS,
    plot_fns=PLOT_FNS,
    category_label="Navier–Stokes (Grid)",
    n_to_cells=lambda n: n**2,
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
    tesseract_dir=_TESSERACT_DIR,
    output_key="result",
    ic_key="v0",
    solvers=_SOLVERS_LIST,
    make_ic=MAKE_IC,
    make_inputs=build_make_inputs(_SOLVERS_LIST),
    error_fn=l2_error_rel,
    diagnostics=DIAGNOSTICS,
    analytic=_tgv_analytic,
    domain_extent=2 * float(jnp.pi),
    units={"nu": "–"},
    status_checks={
        "forward": {"median_k": 3.0, "max_error": 0.5},
        "gradient/fd_check": {
            "min_cosine": 0.99,
            "max_rel_err": 1e-3,
            "rel_err_peer_k": 50.0,
        },
        "cost": {"max_peer_k": 20.0},
        "forward/cylinder": {"median_k": 50.0, "max_error": 0.5},
        "forward/agreement/multimode": {"median_k": 3.0, "max_error": 1.5},
    },
)
