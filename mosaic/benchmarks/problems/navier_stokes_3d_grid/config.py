"""Solver discovery, exclusions, and the final ``Problem`` instance."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from mosaic.benchmarks.core.config import (
    Exclusion,
    ExclusionCategory,
    Problem,
    SolverSpec,
    discover_solvers,
)
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles

from .experiments import EXPERIMENTS, PLOT_FNS
from .ics import MAKE_IC, _tgv3d_analytic
from .physics import DIAGNOSTICS, build_make_inputs


def _field_to_2d(v: np.ndarray) -> np.ndarray:
    """Extract a 2-D scalar from a 3-D velocity field (N,N,N,3).

    Returns the z-component of vorticity on the middle-z slice,
    shape (N, N).  Used as the primary visualisation slice for 3D field plots.
    """
    N = v.shape[0]
    zmid = N // 2
    vx = np.array(v[:, :, zmid, 0])
    vy = np.array(v[:, :, zmid, 1])
    dvydx = (np.roll(vy, -1, 0) - np.roll(vy, 1, 0)) * 0.5
    dvxdy = (np.roll(vx, -1, 1) - np.roll(vx, 1, 1)) * 0.5
    return (dvydx - dvxdy).astype(np.float32)


_GYM_DIR = Path(__file__).parent.parent.parent.parent
_TESSERACT_DIR = _GYM_DIR / "tesseracts" / "navier-stokes-grid"


# ── Solver registry ──────────────────────────────────────────────────────────
# Solvers and per-solver metadata come from each tesseract's YAML; styling is
# applied from mosaic.benchmarks.problems.shared.plots.solver_styles; only per-(solver, problem)
# overrides — exclusions, input_overrides, explained_anomalies, plus a few
# 3D-specific scheme/description tweaks — are set here.

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_DIR)

# JAX-CFD is a 2D-only solver (spectral pressure solve doesn't generalise to
# the 3D periodic-box benchmark configuration); drop it from the 3D suite.
_SOLVERS.pop("jax_cfd", None)

# Preserve historical solver key.
_SOLVERS["ins_jl"] = _SOLVERS.pop("incompressible_navier_stokes_jl")

apply_styles(_SOLVERS)

# ── Per-(solver, problem) overrides ──────────────────────────────────────────

_SOLVERS["exponax"].input_overrides = {
    "drag": jnp.array([0.0], dtype=jnp.float32),
    "order": 2,
    "kolmogorov_forcing": False,
    "injection_mode": 4,
    "injection_scale": jnp.array([1.0], dtype=jnp.float32),
}

_EXCLUSIONS: dict[str, dict[str, Exclusion]] = {
    "phiflow": {
        "cost/temporal_cost": Exclusion(
            ExclusionCategory.INFEASIBLE,
            "CUDA OOM during JAX CUDA graph profiling in 3D cost benchmark "
            "(allocate 16.09MiB failed); xla_gpu_autotune_level=0 fix deployed "
            "— pending re-run to confirm resolved",
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
}


_SOLVERS_LIST = list(_SOLVERS.values())

CONFIG = Problem(
    name="ns-3d-grid",
    exclusions=_EXCLUSIONS,
    experiments=EXPERIMENTS,
    plot_fns=PLOT_FNS,
    category_label="Navier–Stokes (Grid)",
    n_to_cells=lambda n: n**3,
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
    tesseract_dir=_TESSERACT_DIR,
    output_key="result",
    ic_key="v0",
    field_to_2d=_field_to_2d,
    solvers=_SOLVERS_LIST,
    make_ic=MAKE_IC,
    make_inputs=build_make_inputs(_SOLVERS_LIST),
    error_fn=l2_error_rel,
    analytic=_tgv3d_analytic,
    diagnostics=DIAGNOSTICS,
    domain_extent=2 * float(jnp.pi),
    units={"nu": "–"},
    status_checks={
        "gradient/fd_check": {
            "min_cosine": 0.99,
            # Best-ε median rel_error across FD directions. Catches the
            # warp_ns and phiflow 3D systematic backward-magnitude bias
            # (median rel_err ≈ 1.7e-2 / 1.6e-2) while leaving xlb/ins_jl/
            # pict/exponax (5e-6 to 1e-4) unflagged.
            "max_rel_err": 1e-3,
            # Peer-median outlier; ≥3 valid peers required.
            "rel_err_peer_k": 50.0,
        },
        "optimization": {"max_final_ratio": 0.5},
    },
)
