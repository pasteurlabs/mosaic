"""Solver discovery and the final ``Problem`` instance for structural-mesh."""

from __future__ import annotations

from pathlib import Path

from mosaic.benchmarks.core.config import (
    Problem,
    SolverSpec,
    discover_solvers,
)
from mosaic.benchmarks.core.utils import l2_error_rel
from mosaic.benchmarks.shared.plots.solver_styles import apply_styles

from .experiments import EXPERIMENTS, PLOT_FNS
from .ics import MAKE_IC
from .physics import DIAGNOSTICS, _density_to_2d, build_make_inputs

_GYM_DIR = Path(__file__).parent.parent.parent.parent
_TESSERACT_DIR = _GYM_DIR / "tesseracts" / "structural-mesh"

# SIMP material parameters — matched between solvers
_E_MAX = 70_000.0  # Young's modulus of solid [MPa]
_NU = 0.3  # Poisson's ratio
_XMIN = 1e-3  # Void stiffness ratio (E_min / E_max)


# ── Solver registry ──────────────────────────────────────────────────────────
# Solvers and per-solver metadata come from each tesseract's YAML; styling is
# applied from mosaic.benchmarks.shared.plots.solver_styles. Only per-(solver, problem)
# overrides (material parameters via input_overrides) are set here.

_SOLVERS: dict[str, SolverSpec] = discover_solvers(_TESSERACT_DIR)

# Preserve historical solver keys used by paper plots and CLI references.
_SOLVERS["dealii_structural"] = _SOLVERS.pop("dealii")
_SOLVERS["fenics_structural"] = _SOLVERS.pop("fenics")
_SOLVERS["firedrake_structural"] = _SOLVERS.pop("firedrake")

apply_styles(_SOLVERS)


# Material parameters are per-(solver, problem): TopOpt.jl uses ``E`` while the
# other backends use ``E_max``; ``nu``/``xmin`` are shared.
_SOLVERS["topopt_jl"].input_overrides = {"E": _E_MAX, "nu": _NU, "xmin": _XMIN}
for _key in ("dealii_structural", "fenics_structural", "firedrake_structural"):
    _SOLVERS[_key].input_overrides = {"E_max": _E_MAX, "nu": _NU, "xmin": _XMIN}


_SOLVERS_LIST = list(_SOLVERS.values())

CONFIG = Problem(
    name="structural-mesh",
    exclusions={},
    experiments=EXPERIMENTS,
    plot_fns=PLOT_FNS,
    category_label="Structural Mechanics",
    description=(
        "3D linear-elasticity compliance minimisation on a cantilever beam with SIMP "
        "material penalisation (p=3, E_max=70 000 MPa). The stiffness matrix K(ρ) couples "
        "every density element to the global displacement field via the constitutive "
        "relation E_eff(ρ) = E_min + (E_max − E_min)·ρ³; the compliance objective "
        "C = F^T K(ρ)⁻¹ F is smooth but non-convex in ρ, driving gradient-based "
        "topology optimisation toward sparse binary 0/1 layouts."
    ),
    bc_description=(
        "3-D cantilever beam on domain [0,2]×[0,1]×[0,1] (HEX8 elements, 2:1:1 aspect). "
        "Dirichlet: all nodes at x=0 have zero displacement (clamped). "
        "Neumann: a prescribed total force is applied to the right face (x=2) — "
        "either a uniform downward traction or a concentrated upward corner load "
        "depending on the experiment (controlled by the corner_load flag)."
    ),
    tesseract_dir=_TESSERACT_DIR,
    output_key="compliance",
    ic_key="rho",
    solvers=_SOLVERS_LIST,
    make_ic=MAKE_IC,
    make_inputs=build_make_inputs(_SOLVERS_LIST),
    error_fn=l2_error_rel,
    diagnostics=DIAGNOSTICS,
    analytic=None,
    domain_extent=2.0,
    field_to_2d=None,  # compliance is scalar; no 2D field projection
    ic_to_2d=_density_to_2d,  # mid-y cross-section of density field ρ
    field_cmap="hot",
    field_symmetric=False,
    diagnostic_fields=False,  # compliance is scalar; stress fields not directly comparable
    resolution_key="nx",
    n_to_cells=lambda N: N * 2 * max(1, N // 2),  # nx=N, ny=2, nz=N//2
    units={"rho_0": "–"},
)
