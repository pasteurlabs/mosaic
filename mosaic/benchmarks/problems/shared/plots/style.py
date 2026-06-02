# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared matplotlib style and utilities for all benchmark plots.

Single :data:`RCPARAMS` profile — publication-quality, sized for a
standard single-column layout (``TEXTWIDTH`` inches wide), 8-pt
sans-serif, ``pdf.fonttype = 42`` so glyphs embed as Type 42 (TrueType)
for editable / searchable PDFs.

Every plot in the suite (per-experiment plots registered via
``plot=...`` on :meth:`Problem.add_experiment`, plus the cross-domain
``_extra/*`` aggregators registered via :meth:`Problem.add_extra_plot`)
uses this single profile, so figures match whether they're produced
during ``mosaic run --plots-only`` or saved off-line.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mosaic.benchmarks.core.console import print_saved

if TYPE_CHECKING:
    from mosaic.benchmarks.core.config import Problem

# ── Layout constant ───────────────────────────────────────────────────────────

# Target figure width in inches (5.5" = single column at standard venues).
# Imported by per-experiment plots so PDFs embed at print width without scaling.
TEXTWIDTH = 5.5


# ── rcParams ──────────────────────────────────────────────────────────────────

RCPARAMS: dict = {
    "font.family": "sans-serif",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.framealpha": 0.7,
    "legend.edgecolor": "0.8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
}


def apply_style() -> None:
    """Apply the shared benchmark style globally (call once at module import)."""
    plt.rcParams.update(RCPARAMS)


def rc_context() -> object:
    """Return ``plt.rc_context(RCPARAMS)`` — context manager for scoped use."""
    return plt.rc_context(RCPARAMS)


# ── Solver palette (per-solver display label + colour cycle) ──────────────────
#
# Paul Tol Vibrant (distinguishable under all forms of colour-blindness +
# greyscale) extended with a couple of muted tones for reference / excluded
# solvers.

SOLVER_STYLES: dict[str, tuple] = {
    # ── Fluid / NS ──────────────────────────────────────────────────────────
    "jax_cfd": ("JAX-CFD", "#0077BB", "-", "o"),
    "phiflow": ("PhiFlow", "#CC3311", "--", "s"),
    "ins_jl": ("INS.jl", "#33BBEE", "-.", "^"),
    "pict": ("PICT", "#EE3377", ":", "D"),
    "xlb": ("XLB", "#CCBB44", (0, (4, 1)), "v"),
    "warp_ns": ("Warp-NS", "#EE7733", (0, (1, 1)), "P"),
    "exponax": ("Exponax", "#009988", (0, (5, 1)), "<"),
    "openfoam": ("OpenFOAM", "#DDCC77", "--", "h"),
    # ── FEM / Structural ────────────────────────────────────────────────────
    "jax_fem": ("JAX-FEM", "#0077BB", "-", "o"),
    "topopt_jl": ("TopOpt.jl", "#009988", "--", "s"),
    "dealii_structural": ("deal.II", "#DDCC77", "-.", "^"),
    "fenics_structural": ("FEniCS", "#CC3311", ":", "D"),
    "firedrake_structural": ("Firedrake", "#EE7733", (0, (3, 1)), "v"),
    # ── FEM / Thermal ───────────────────────────────────────────────────────
    "dealii_heat": ("deal.II", "#DDCC77", "-.", "^"),
    "fenics_heat": ("FEniCS", "#CC3311", ":", "D"),
    "firedrake_heat": ("Firedrake", "#EE7733", (0, (3, 1)), "v"),
    "torch_fem_thermal": ("TorchFEM", "#009988", (0, (5, 1, 1, 1)), "<"),
}

# Canonical solver ordering — drives legend construction so multi-panel figures
# show solvers in the same order even when individual panels are subsets.
NS_ORDER: list[str] = [
    "jax_cfd",
    "phiflow",
    "ins_jl",
    "xlb",
    "pict",
    "warp_ns",
    "exponax",
    "openfoam",
]
FEM_ORDER: list[str] = [
    "jax_fem",
    "topopt_jl",
    "dealii_structural",
    "fenics_structural",
    "firedrake_structural",
    "dealii_heat",
    "fenics_heat",
    "firedrake_heat",
    "torch_fem_thermal",
]
STRUCTURAL_ORDER: list[str] = [
    "jax_fem",
    "topopt_jl",
    "dealii_structural",
    "fenics_structural",
    "firedrake_structural",
]
THERMAL_ORDER: list[str] = [
    "firedrake_heat",
    "jax_fem",
    "fenics_heat",
    "dealii_heat",
    "torch_fem_thermal",
]


def _norm_solver_name(s: str) -> str:
    """Case-insensitive, separator-insensitive normaliser for solver names."""
    return s.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")


def resolve_solver_alias(name: str) -> str | None:
    """Map a solver string to its canonical alias in :data:`SOLVER_STYLES`.

    Handles three forms:
      * the alias key itself (``"openfoam"``)        → returned as-is
      * the display label  (``"OpenFOAM"``)          → resolved via styles
      * the spec.name from YAML (``"jax-cfd"``)      → resolved via normalisation

    Returns ``None`` if no entry matches. Used to bridge ``result.json``
    keys (which are ``spec.name``) and ``NS_ORDER`` / ``FEM_ORDER`` /
    ``SOLVER_STYLES`` (which are alias-keyed).
    """
    if name in SOLVER_STYLES:
        return name
    target = _norm_solver_name(name)
    for k, v in SOLVER_STYLES.items():
        if _norm_solver_name(k) == target or _norm_solver_name(v[0]) == target:
            return k
    return None


def solver_props(name: str) -> tuple:
    """``(label, color, linestyle, marker)`` for *name*, with grey fallback.

    Accepts either an alias key (``"openfoam"``), a display label
    (``"OpenFOAM"``), or a spec.name (``"jax-cfd"``). Falls back to a
    neutral grey when no match is found.
    """
    alias = resolve_solver_alias(name)
    if alias is not None:
        return SOLVER_STYLES[alias]
    return (name, "#888888", "-", "o")


def make_handle(solver: str) -> mlines.Line2D:
    """Legend ``Line2D`` proxy for *solver* — used for shared cross-panel legends."""
    label, color, ls, mk = solver_props(solver)
    return mlines.Line2D(
        [],
        [],
        color=color,
        linestyle=ls,
        marker=mk,
        markersize=5,
        markeredgewidth=0,
        linewidth=1.6,
        label=label,
    )


def dedup_handles(handles: list) -> list:
    """Drop duplicate-label legend handles (first occurrence wins)."""
    seen: set[str] = set()
    out = []
    for h in handles:
        lbl = h.get_label()
        if lbl not in seen:
            out.append(h)
            seen.add(lbl)
    return out


# ── Save helper ───────────────────────────────────────────────────────────────


def unit_label(name: str, units: dict[str, str] | None) -> str:
    """Return 'name  [unit]' when a unit is registered, else 'name'.

    Tolerates ``units=None`` so problems that don't define a units dict can
    pass it through directly without each call site having to guard.
    """
    unit = (units or {}).get(name, "")
    return f"{name}  [{unit}]" if unit else name


def save_fig(fig: Figure, stem: str, out_dir: Path) -> None:
    """Save *fig* as PNG under *out_dir*, then close it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")
    print_saved(f"{out_dir}/{stem}.png")
    plt.close(fig)


# ── Shared figure legend ──────────────────────────────────────────────────────


def _is_constrained(fig: Figure) -> bool:
    """True when *fig* is driven by the constrained-layout engine.

    Bottom legends must use ``loc="outside ..."`` (which the engine reserves
    space for) rather than a manual ``bbox_to_anchor`` + ``tight_layout`` that
    would conflict with it.
    """
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    return isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)


def fig_shared_legend(
    fig: Figure,
    axes: Any,
    *,
    ncol: int | None = None,
    bottom: float = 0.12,
) -> None:
    """Place a single shared legend centred below all subplots.

    Collects labelled handles from *axes* (an Axes, flat list, or 2-D array),
    deduplicates by label, then attaches a figure-level legend below the plot
    area.  Also calls ``tight_layout`` with a bottom rect that reserves space
    for the legend — callers must **not** call ``tight_layout`` separately.
    """
    # Flatten 2-D numpy arrays; wrap a single Axes in a list.
    if hasattr(axes, "flat"):
        axes_flat = list(axes.flat)
    elif hasattr(axes, "get_legend_handles_labels"):
        axes_flat = [axes]
    else:
        axes_flat = list(axes)

    seen: set[str] = set()
    handles: list = []
    labels: list = []
    for ax in axes_flat:
        if not hasattr(ax, "get_legend_handles_labels"):
            continue
        for h, lbl in zip(*ax.get_legend_handles_labels(), strict=False):
            if lbl not in seen:
                seen.add(lbl)
                handles.append(h)
                labels.append(lbl)

    constrained = _is_constrained(fig)
    if not handles:
        if not constrained:
            fig.tight_layout()
        return

    _ncol = ncol if ncol is not None else len(handles)
    if constrained:
        # Constrained layout reserves space for an "outside" legend automatically;
        # a manual tight_layout would conflict with the layout engine.
        fig.legend(handles, labels, loc="outside lower center", ncol=_ncol)
        return
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=_ncol,
        bbox_to_anchor=(0.5, 0),
        bbox_transform=fig.transFigure,
        framealpha=0.7,
    )
    fig.tight_layout(rect=[0, bottom, 1, 1])


# ── Paper figure layouts ──────────────────────────────────────────────────────
#
# Canonical house layout for every line/curve and image figure in the suite,
# factored out of the original horizon_sweep / fd_check figures so all plots
# read identically: a TEXTWIDTH-wide figure (so PDFs embed at single-column
# print width without scaling), RCPARAMS active, and — for solver curves — a
# single shared legend centred below the panels in canonical solver order.
#
# Use:
#   paper_row(3)             → 1×3 row of line panels      (sweep summaries)
#   paper_grid(nrows, ncols) → multi-row grid of line panels (per-solver curves)
#   paper_image_grid(r, c)   → grid of square image panels  (field renders)
#   solver_legend(fig, seen) → shared bottom legend, canonical order
#   cosine_defect(values)    → 1 − cosine, for a log "direction accuracy" axis

# Aspect (height / TEXTWIDTH) of a single row of line panels.
ROW_ASPECT = 0.42
# Per-row height (inches) for multi-row line grids.
GRID_ROW_H = 1.7
# Min per-column width (inches) for line grids — grids wider than TEXTWIDTH/this
# many columns grow past TEXTWIDTH so panels don't collapse to zero size.
GRID_COL_W = 1.35
# Per-panel size (inches) for image / field grids (≈ TEXTWIDTH at 3–4 cols).
IMG_PANEL = 1.45
# Floor for ``1 − cosine`` so a perfect cosine still plots on a log axis.
COSINE_DEFECT_FLOOR = 1e-12
# Canonical y-label for the log "direction accuracy" panel.
COSINE_DEFECT_YLABEL = "$1 -$ cosine"


def paper_row(
    ncols: int = 1,
    *,
    aspect: float = ROW_ASPECT,
    dpi: int = 300,
    squeeze: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """One TEXTWIDTH-wide row of *ncols* line-plot panels at the house aspect.

    The canonical layout for sweep / curve summaries (cf. ``horizon_sweep``).
    Pair with :func:`solver_legend` for the shared bottom legend. ``RCPARAMS``
    must already be active (modules call :func:`apply_style` at import).

    Uses constrained layout so multi-panel rows never overlap their y-labels /
    titles; the shared legend helpers reserve their own space via an "outside"
    placement.
    """
    return plt.subplots(
        1,
        ncols,
        figsize=(TEXTWIDTH, TEXTWIDTH * aspect),
        dpi=dpi,
        squeeze=squeeze,
        layout="constrained",
        **kwargs,
    )


def paper_grid(
    nrows: int,
    ncols: int,
    *,
    row_h: float = GRID_ROW_H,
    dpi: int = 300,
    squeeze: bool = False,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Grid of line-plot panels; height scales with *nrows*.

    TEXTWIDTH wide for up to ~4 columns; wider beyond that so panels don't
    collapse to zero size (which disables constrained layout). For per-solver /
    per-sweep-value curve grids. Hide padding panels via ``ax.set_visible(False)``.
    """
    width = max(TEXTWIDTH, ncols * GRID_COL_W)
    return plt.subplots(
        nrows,
        ncols,
        figsize=(width, row_h * nrows + 0.4),
        dpi=dpi,
        squeeze=squeeze,
        layout="constrained",
        **kwargs,
    )


def paper_image_grid(
    nrows: int,
    ncols: int,
    *,
    panel: float = IMG_PANEL,
    dpi: int = 300,
    squeeze: bool = False,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Grid of square-ish image / field panels (≈ TEXTWIDTH wide at 3–4 cols).

    For field renders (IC, optimised density, recovered fields, …). Each panel
    is *panel* inches square; add colorbars with :func:`imshow_with_cbar`.

    Deliberately NOT constrained-layout: image grids commonly attach
    ``make_axes_locatable`` colorbars (via :func:`imshow_with_cbar`), which the
    constrained-layout engine can't manage. Callers space panels with
    ``tight_layout`` / ``subplots_adjust`` as needed.
    """
    return plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * panel, nrows * panel),
        dpi=dpi,
        squeeze=squeeze,
        **kwargs,
    )


def solver_legend(
    fig: Figure,
    seen: Any,
    *,
    order: list[str] | None = None,
    extra_handles: list | None = None,
    ncol: int | None = None,
    y: float = 0.01,
) -> None:
    """Shared solver legend centred below the panels, solvers in canonical order.

    *seen* is an iterable of solver aliases present in the figure; *order* is the
    canonical ordering (defaults to ``NS_ORDER + FEM_ORDER``). *extra_handles*
    appends custom ``Line2D`` proxies (e.g. a failure marker). No-op when nothing
    is present. Styling (font, frame) comes from ``RCPARAMS``.
    """
    order = order if order is not None else (NS_ORDER + FEM_ORDER)
    seen_set = set(seen)
    handles = dedup_handles([make_handle(s) for s in order if s in seen_set])
    if extra_handles:
        handles += list(extra_handles)
    if not handles:
        return
    _ncol = ncol if ncol is not None else min(len(handles), 6)
    if _is_constrained(fig):
        fig.legend(
            handles=handles, loc="outside lower center", ncol=_ncol, handlelength=2.0
        )
        return
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=_ncol,
        handlelength=2.0,
    )


def cosine_defect(values: Any, *, floor: float = COSINE_DEFECT_FLOOR) -> list[float]:
    """``1 − cosine`` (floored) for a log "direction accuracy" axis.

    Near-perfect cosine similarity reads on a log axis instead of being squashed
    against 1.0. Pair with ``ax.loglog`` / ``ax.set_yscale('log')`` and
    ``ax.set_ylabel(COSINE_DEFECT_YLABEL)``.
    """
    return [max(1.0 - float(c), floor) for c in values]


# ── Multi-panel grid helper ───────────────────────────────────────────────────


def subplots_grid(
    n: int,
    ncols: int = 4,
    *,
    panel_w: float = 4.0,
    panel_h: float = 4.0,
    **kwargs: Any,
) -> tuple[Figure, list]:
    """Create a balanced grid of *n* subplots, wrapping after *ncols* columns.

    Columns are balanced so no row is much shorter than the others
    (e.g. n=5, ncols=4 → 3+2 not 4+1).  Any padding panels are hidden.

    Returns ``(fig, axes)`` where *axes* is a flat list of exactly *n* Axes.
    Pass ``**kwargs`` directly to ``plt.subplots`` (e.g. ``sharey=True``).
    """
    _ncols = math.ceil(n / math.ceil(n / ncols)) if n > ncols else n
    _nrows = math.ceil(n / _ncols)
    fig, ax_grid = plt.subplots(
        _nrows,
        _ncols,
        figsize=(_ncols * panel_w, _nrows * panel_h),
        squeeze=False,
        **kwargs,
    )
    axes = list(ax_grid.flat)
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


# ── Line-style cycle ──────────────────────────────────────────────────────────

_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

# Colour shades per solver family (2-4 distinguishable tones per hue).
# Families: lbm, projection, spectral, fv, fem, ml, "".
_FAMILY_PALETTE: dict[str, list[str]] = {
    "lbm": ["#4477AA", "#88BBDD", "#2255AA", "#AACCEE"],
    "projection": ["#228833", "#44AA55", "#66CC77", "#99DDAA"],
    "spectral": ["#AA3377", "#DD77AA", "#882255", "#EEBBD4"],
    "fv": ["#EE7733", "#CC4411", "#FFAA66", "#994411"],
    "fem": ["#EE3333", "#AA1111", "#FF7777", "#882222"],
    "ml": ["#CCBB44", "#EECC66", "#AAAA22", "#FFEE99"],
    "sph": ["#33BBEE", "#66DDFF", "#0099CC", "#99EEFF"],
    "": ["#555555", "#888888", "#AAAAAA", "#CCCCCC"],
}


def solver_line_props(idx: int, color: str, *, marker: bool = True) -> dict:
    """Kwargs for ax.plot/loglog/semilogy that keep N solvers distinguishable.

    Legacy API: idx selects linestyle from the global cycle; color is fixed.
    Prefer solver_plot_props() when solver_styles() is already available.
    """
    props: dict = {
        "color": color,
        "linestyle": _LINESTYLES[idx % len(_LINESTYLES)],
    }
    if marker:
        props["marker"] = _MARKERS[idx % len(_MARKERS)]
    return props


def solver_plot_props(style: dict, *, marker: bool = True) -> dict:
    """Return ax.plot kwargs from a pre-computed solver style dict.

    *style* is one entry from the dict returned by solver_styles(), e.g.::

        styles = solver_styles(cfg)
        ax.plot(x, y, label=styles[name]["label"], **solver_plot_props(styles[name]))
    """
    props: dict = {
        "color": style.get("color", "#888888"),
        "linestyle": style.get("linestyle", "-"),
    }
    if marker:
        mk = style.get("marker")
        if mk:
            props["marker"] = mk
    return props


# ── Solver style registry ─────────────────────────────────────────────────────


def solver_styles(
    cfg: Problem, *, differentiable_only: bool = False
) -> dict[str, dict]:
    """Return {solver_name: {"color", "label", "linestyle", "marker"}} for plotting.

    Solvers in the same family share a hue; linestyle and marker cycle within
    each family so individual solvers remain distinguishable.
    """
    from collections import defaultdict

    specs = {
        spec.name: spec
        for spec in cfg.solvers
        if not differentiable_only or getattr(spec, "differentiable", True)
    }

    # Count per-family to assign shade/linestyle indices.
    family_counters: dict[str, int] = defaultdict(int)
    result: dict[str, dict] = {}

    # How many distinct families are present?  Family-based hues only help when
    # there are multiple families; if every solver belongs to the same family
    # the palette gives no cross-solver discrimination, so fall back to the
    # per-solver spec.color in that case.
    all_families = {getattr(s, "family", "") or "" for s in specs.values()}
    n_distinct_families = len(all_families - {""})

    for name, spec in specs.items():
        family = getattr(spec, "family", "") or ""
        idx = family_counters[family]
        family_counters[family] += 1

        palette = _FAMILY_PALETTE.get(family, _FAMILY_PALETTE[""])
        # Use family palette only when multiple families co-exist, no explicit
        # linestyle override is set, and the family has multiple members.
        n_in_family = sum(
            1 for s in specs.values() if (getattr(s, "family", "") or "") == family
        )
        has_explicit_style = getattr(spec, "linestyle", None) is not None
        if (
            family
            and n_in_family > 1
            and n_distinct_families > 1
            and not has_explicit_style
        ):
            color = palette[idx % len(palette)]
        else:
            color = spec.color

        result[name] = {
            "color": color,
            "label": spec.name,
            "linestyle": getattr(spec, "linestyle", None)
            or _LINESTYLES[idx % len(_LINESTYLES)],
            "marker": getattr(spec, "marker", None) or _MARKERS[idx % len(_MARKERS)],
        }
    return result


# ── Physics field transforms ──────────────────────────────────────────────────


def vorticity_2d(v: np.ndarray) -> np.ndarray:
    """Vorticity of a 2-D velocity field, shape (N, N, 1, 2) → (N, N)."""
    vx, vy = v[:, :, 0, 0], v[:, :, 0, 1]
    dvydx = (np.roll(vy, -1, 0) - np.roll(vy, 1, 0)) * 0.5
    dvxdy = (np.roll(vx, -1, 1) - np.roll(vx, 1, 1)) * 0.5
    return dvydx - dvxdy


def density_slice_2d(rho: np.ndarray) -> np.ndarray:
    """Middle-z overdensity slice of a 3-D density field, shape (N, N, N) → (N, N).

    Returns δ = ρ/ρ̄ − 1 so the field is zero-mean and can be displayed with a
    symmetric diverging colormap (overdense positive, underdense negative).
    """
    sl = np.array(rho[:, :, rho.shape[2] // 2])
    mean = sl.mean()
    if mean != 0:
        sl = sl / mean - 1.0
    return sl


def log_density_slice_2d(rho: np.ndarray) -> np.ndarray:
    """Middle-z log-density slice of a 3-D density field, shape (N, N, N) → (N, N).

    Returns log10(ρ/ρ̄), clipped at −2 (voids at 1% of mean density).
    Use with field_symmetric=False and field_cmap="inferno" to reveal the full
    dynamic range from underdense voids to overdense halos.
    """
    sl = np.array(rho[:, :, rho.shape[2] // 2], dtype=np.float64)
    mean = sl.mean()
    if mean > 0:
        sl = sl / mean
    return np.clip(np.log10(np.maximum(sl, 1e-6)), -2.0, None).astype(np.float32)


def density_contrast_slice_2d(delta: np.ndarray) -> np.ndarray:
    """Middle-z slice of a 3-D density contrast field δ₀, shape (N, N, N) → (N, N).

    The field is already mean-zero (no normalisation applied).
    Use with field_symmetric=True and cmap='RdBu_r'.
    Suitable for IC fields (linear density contrast) where mean ≈ 0.
    """
    return np.array(delta[:, :, delta.shape[2] // 2], dtype=np.float32)


def density_projection_2d(rho: np.ndarray) -> np.ndarray:
    """Log-projected density of a 3-D field, shape (N, N, N) → (N, N).

    Sums ρ along the z-axis and returns log10(Σρ / mean(Σρ)), clipped at -1.
    Use with field_symmetric=False and field_cmap="inferno" to show halo structure.
    """
    proj = np.asarray(rho, dtype=np.float64).sum(axis=2)
    mean = proj.mean()
    if mean > 0:
        proj = proj / mean
    return np.clip(np.log10(proj + 1e-6), -1.0, None).astype(np.float32)


def grad_magnitude_2d(g: np.ndarray) -> np.ndarray:
    """Gradient magnitude slice for multiple field shapes.

    Handles: 2-D velocity (N,N,1,2), 3-D velocity (N,N,N,3),
    3-D scalar (N,N,N), and 1-D flat field (N,).
    """
    if g.ndim == 4 and g.shape[-1] == 2:
        return np.sqrt(g[:, :, 0, 0] ** 2 + g[:, :, 0, 1] ** 2)
    if g.ndim == 4 and g.shape[-1] == 3:
        # 3-D velocity field: take absolute value of z=0 slice vorticity
        return np.abs(vorticity_2d(g)).astype(np.float32)
    if g.ndim == 1:
        # 1-D flat field: infer 3-D shape assuming canonical 2:1:1 cantilever geometry
        n_cells = len(g)
        nx = round(float(n_cells) ** 0.5)  # n_cells = nx² for ny=2, nz=nx//2
        ny = 2
        nz = max(1, nx // 2)
        g3d = g.reshape(nz, ny, nx)
        return np.abs(g3d[:, ny // 2, :]).astype(np.float32)
    # 3-D scalar field (e.g. n-body IC gradient): take middle-z slice magnitude
    return np.abs(g[:, :, g.shape[2] // 2]).astype(np.float32)


# ── imshow + colorbar ─────────────────────────────────────────────────────────


def imshow_with_cbar(
    ax: Any, fig: Figure, data: np.ndarray, **imshow_kwargs: Any
) -> Any:
    """Imshow with a locatable colorbar matched to the axes height."""
    im = ax.imshow(data, **imshow_kwargs)
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    return im


# ── Field grid ────────────────────────────────────────────────────────────────


def particle_overlay_grid(
    panels: list,
    title: str,
    *,
    ncols: int = 3,
    s: float = 6,
    alpha: float = 0.5,
) -> Figure:
    """3-D scatter grid with all solvers overlaid in each panel.

    *panels* is a list of ``(panel_label, entries)`` where *entries* is a list
    of ``(data_label, positions, color)`` — positions shape ``(N, 3)``.
    One subplot per panel; all entries drawn into the same axes so spatial
    distributions can be compared directly.
    """
    nrows = math.ceil(len(panels) / ncols)
    fig_w = max(8.0, ncols * 3.0)
    fig = plt.figure(figsize=(fig_w, nrows * 3.0))

    all_pos = np.concatenate(
        [pos for _, entries in panels for _, pos, _ in entries], axis=0
    )
    lo, hi = float(all_pos.min()), float(all_pos.max())
    pad = (hi - lo) * 0.05

    for idx, (panel_label, entries) in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        for data_label, pos, color in entries:
            ax.scatter(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                s=s,
                alpha=alpha,
                color=color,
                linewidths=0,
                depthshade=True,
                label=data_label,
            )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_zlim(lo - pad, hi + pad)
        ax.set_title(panel_label, pad=2)
        ax.set_xlabel("x", labelpad=1)
        ax.set_ylabel("y", labelpad=1)
        ax.set_zlabel("z", labelpad=1)
        ax.tick_params(labelsize=6)
        if entries:
            ax.legend(fontsize=6, loc="upper left")

    for idx in range(len(panels), nrows * ncols):
        ax_hide = fig.add_subplot(nrows, ncols, idx + 1)
        ax_hide.set_visible(False)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def particle_scatter_grid(
    panels: list,
    title: str,
    *,
    ncols: int = 4,
    s: float = 6,
    alpha: float = 0.65,
) -> Figure:
    """3-D scatter plot grid for particle positions.

    Each entry in *panels* is one of:
      (label, positions)           – uses default color (#4477AA)
      (label, positions, color)    – uses given hex color
    *positions* has shape (N, 3).
    """
    nrows = math.ceil(len(panels) / ncols)
    fig_w = max(8.0, ncols * 2.8)
    fig = plt.figure(figsize=(fig_w, nrows * 2.8))

    all_xyz = np.concatenate([p[1] for p in panels], axis=0)
    lo, hi = float(all_xyz.min()), float(all_xyz.max())
    pad = (hi - lo) * 0.05

    for idx, panel in enumerate(panels):
        label = panel[0]
        pos = panel[1]
        color = panel[2] if len(panel) > 2 else "#4477AA"
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            s=s,
            alpha=alpha,
            color=color,
            linewidths=0,
            depthshade=True,
        )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_zlim(lo - pad, hi + pad)
        ax.set_title(label, pad=2)
        ax.set_xlabel("x", labelpad=1)
        ax.set_ylabel("y", labelpad=1)
        ax.set_zlabel("z", labelpad=1)
        ax.tick_params(labelsize=6)

    for idx in range(len(panels), nrows * ncols):
        ax_hide = fig.add_subplot(nrows, ncols, idx + 1)
        ax_hide.set_visible(False)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def field_grid(
    panels: list,
    title: str,
    *,
    cmap: str = "RdBu_r",
    shared_scale: bool = False,
    symmetric: bool = True,
    ncols: int = 4,
) -> Figure:
    """Render a grid of 2-D field panels, each with a locatable colorbar.

    Each entry in *panels* is one of:
      (label, arr)                 – uses default cmap / scale
      (label, arr, extra_kwargs)   – extra_kwargs override any imshow kwarg
                                     (useful for per-panel cmap or vmin/vmax)
    """
    nrows = math.ceil(len(panels) / ncols)
    fig_w = max(8.0, ncols * 2.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, nrows * 2.6), squeeze=False)

    arrs = [p[1] for p in panels]
    vmax = max(np.abs(a).max() for a in arrs) if shared_scale else None

    for idx, panel in enumerate(panels):
        label, arr = panel[0], panel[1]
        extra_kw = panel[2] if len(panel) > 2 else {}
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        v = vmax if shared_scale else np.abs(arr).max()
        base_kw = {
            "origin": "lower",
            "cmap": cmap,
            "interpolation": "nearest",
            "vmin": -v if symmetric else 0,
            "vmax": v,
        }
        base_kw.update(extra_kw)
        imshow_with_cbar(ax, fig, arr.T, **base_kw)
        ax.set_title(label)
        ax.axis("off")

    for idx in range(len(panels), nrows * ncols):
        axes[divmod(idx, ncols)].set_visible(False)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig
