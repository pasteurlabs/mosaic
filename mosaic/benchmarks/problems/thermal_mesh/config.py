"""Solver discovery and the final ``Problem`` instance."""

from __future__ import annotations

from pathlib import Path

from mosaic.benchmarks.core.config import (
    Problem,
    SolverSpec,
    discover_solvers,
)
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.problems.shared.plots.solver_styles import apply_styles

from .experiments import EXPERIMENTS, PLOT_FNS
from .ics import MAKE_IC
from .physics import DIAGNOSTICS, _density_to_2d, build_make_inputs

_GYM_DIR = Path(__file__).parent.parent.parent.parent
_TESSERACT_DIR = _GYM_DIR / "tesseracts" / "thermal-mesh"

# SIMP material parameters
_K_MAX = 1.0  # solid thermal conductivity
_P_EXP = 3.0  # SIMP penalisation exponent


# ── Solver registry ──────────────────────────────────────────────────────────
# Solvers and per-solver metadata come from each tesseract's YAML; styling is
# applied from mosaic.benchmarks.problems.shared.plots.solver_styles. Only per-(solver, problem)
# material-parameter overrides are set here.

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_DIR)

# Preserve historical solver key.
_SOLVERS["torch_fem_thermal"] = _SOLVERS.pop("torch_fem")

apply_styles(_SOLVERS)


for _key in ("fenics_heat", "dealii_heat", "firedrake_heat", "torch_fem_thermal"):
    _SOLVERS[_key].input_overrides = {"k_max": _K_MAX, "p_exp": _P_EXP}


_SOLVERS_LIST = list(_SOLVERS.values())

CONFIG = Problem(
    name="thermal-mesh",
    exclusions={},
    experiments=EXPERIMENTS,
    plot_fns=PLOT_FNS,
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
    tesseract_dir=_TESSERACT_DIR,
    output_key="thermal_compliance",
    ic_key="rho",
    solvers=_SOLVERS_LIST,
    make_ic=MAKE_IC,
    make_inputs=build_make_inputs(_SOLVERS_LIST),
    error_fn=l2_error_rel,
    diagnostics=DIAGNOSTICS,
    domain_extent=2.0,
    field_to_2d=None,
    ic_to_2d=_density_to_2d,
    field_cmap="hot",
    field_symmetric=False,
    diagnostic_fields=False,  # temperature shapes differ between solvers (per-node vs per-cell)
    resolution_key="nx",
    n_to_cells=lambda N: N * max(1, N // 2),  # nx=N, ny=N//2, nz=1
    units={"rho_0": "–"},
    status_checks={
        # Recovery / optimisation experiments must actually reduce loss,
        # not just complete. Same 50% floor as the other problems — solvers
        # landing at final/initial > 0.5 show up as anom so the status
        # accurately reflects "hasn't converged".
        "optimization": {"max_final_ratio": 0.5},
    },
)
