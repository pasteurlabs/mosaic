"""Per-solver plot styling (color, linestyle, marker).

These attributes are presentation-only — they don't affect solver capability,
schemes, or differentiability — so they live here rather than in each solver's
``tesseract_config.yaml`` or in the problem config's ``SolverSpec`` definitions.

Harness plots use this registry via :func:`apply_styles`, which is called from
each problem config after :func:`discover_solvers` populates the spec dict.

Keys match the solver-key aliases used across the codebase
(e.g. ``ins_jl``, ``dealii_structural``) — not the raw tesseract directory
names. Apply styles only after re-keying the discovered dict.
"""

from __future__ import annotations

from typing import Any

# solver-key → {color, linestyle, marker}
SOLVER_STYLES: dict[str, dict[str, Any]] = {
    # ── Navier–Stokes (grid) ─────────────────────────────────────────────
    "jax_cfd": {"color": "#4477AA", "linestyle": "-", "marker": "o"},
    "phiflow": {"color": "#EE3333", "linestyle": "--", "marker": "s"},
    "ins_jl": {"color": "#228833", "linestyle": "-.", "marker": "^"},
    "openfoam": {"color": "#DDAA33", "linestyle": ":", "marker": "D"},
    "pict": {"color": "#AA44AA", "linestyle": (0, (5, 1)), "marker": "v"},
    "warp_ns": {"color": "#EE7733", "linestyle": (0, (1, 1)), "marker": "X"},
    "xlb": {"color": "#66CCEE", "linestyle": (0, (3, 1, 1, 1)), "marker": "P"},
    "exponax": {"color": "#33AA99", "linestyle": "-", "marker": "o"},
    # ── Structural mechanics ─────────────────────────────────────────────
    "jax_fem": {"color": "#4477AA", "linestyle": "-.", "marker": "o"},
    "topopt_jl": {"color": "#228833", "linestyle": ":", "marker": "s"},
    "dealii_structural": {"color": "#CCBB44", "linestyle": "-", "marker": "X"},
    "fenics_structural": {"color": "#AA3377", "linestyle": "--", "marker": "v"},
    "firedrake_structural": {
        "color": "#EE3377",
        "linestyle": (0, (5, 2)),
        "marker": "^",
    },
    # ── Heat conduction ──────────────────────────────────────────────────
    "dealii_heat": {"color": "#228833", "linestyle": "-", "marker": "X"},
    "fenics_heat": {"color": "#AA3377", "linestyle": "--", "marker": "v"},
    "firedrake_heat": {
        "color": "#CCBB44",
        "linestyle": (0, (5, 2)),
        "marker": "^",
    },
    "torch_fem_thermal": {
        "color": "#EE6677",
        "linestyle": (0, (3, 1, 1, 1)),
        "marker": "h",
    },
}

_DEFAULT: dict[str, Any] = {"color": "#999999", "linestyle": "-", "marker": "o"}


def get_style(solver_key: str) -> dict[str, Any]:
    """Return the ``{color, linestyle, marker}`` triple for *solver_key*."""
    return SOLVER_STYLES.get(solver_key, _DEFAULT)


def apply_styles(solvers: dict) -> dict:
    """Mutate each spec in *solvers* with the registered style and return it.

    Looks up the style by the dict key (after any re-keying), so call this
    after :func:`discover_solvers` and any aliasing. Specs with no matching
    entry get the neutral grey default.
    """
    for key, spec in solvers.items():
        style = get_style(key)
        spec.color = style["color"]
        spec.linestyle = style["linestyle"]
        spec.marker = style["marker"]
    return solvers
