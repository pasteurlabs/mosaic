"""Shared matplotlib style and utilities for all benchmark plots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection

from mosaic.benchmarks.core.console import print_saved

if TYPE_CHECKING:
    from mosaic.benchmarks.core.config import ProblemConfig

# ── rcParams ──────────────────────────────────────────────────────────────────

RCPARAMS: dict = {
    "font.family": "sans-serif",
    "font.size": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.7,
    "legend.edgecolor": "0.8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
}


def apply_style() -> None:
    """Apply the shared benchmark style globally (call once at module import)."""
    plt.rcParams.update(RCPARAMS)


# ── Save helper ───────────────────────────────────────────────────────────────


def unit_label(name: str, units: dict[str, str]) -> str:
    """Return 'name  [unit]' when a unit is registered, else 'name'."""
    unit = units.get(name, "")
    return f"{name}  [{unit}]" if unit else name


def save_fig(fig: Figure, stem: str, out_dir: Path) -> None:
    """Save *fig* as both PNG and PDF under *out_dir*, then close it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    print_saved(f"{out_dir}/{stem}.{{png,pdf}}")
    plt.close(fig)


# ── Shared figure legend ──────────────────────────────────────────────────────


def fig_shared_legend(
    fig: Figure,
    axes,
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

    if not handles:
        fig.tight_layout()
        return

    _ncol = ncol if ncol is not None else len(handles)
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


# ── Multi-panel grid helper ───────────────────────────────────────────────────


def subplots_grid(
    n: int,
    ncols: int = 4,
    *,
    panel_w: float = 4.0,
    panel_h: float = 4.0,
    **kwargs,
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
    """kwargs for ax.plot/loglog/semilogy that keep N solvers distinguishable.

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
    cfg: ProblemConfig, *, differentiable_only: bool = False
) -> dict[str, dict]:
    """Return {solver_name: {"color", "label", "linestyle", "marker"}} for plotting.

    Solvers in the same family share a hue; linestyle and marker cycle within
    each family so individual solvers remain distinguishable.
    """
    from collections import defaultdict

    specs = {
        name: spec
        for name, spec in cfg.solvers.items()
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
    """Gradient magnitude slice. Handles multiple shapes:
    - 2-D velocity gradient  (N, N, 1, 2) → sqrt(gx² + gy²)
    - 3-D velocity gradient  (N, N, N, 3) → |curl(z=0 slice)| via vorticity_2d
    - 3-D scalar gradient    (N, N, N)    → |g| at the middle-z slice
    - 1-D flat field         (N,)         → reshape assuming nx²/4 cells, mid-y slice
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


def imshow_with_cbar(ax, fig: Figure, data: np.ndarray, **imshow_kwargs):
    """imshow with a locatable colorbar matched to the axes height."""
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
