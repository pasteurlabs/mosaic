"""Cross-domain ``_extra/`` aggregator plots for ns-grid.

These figures read result.json files from multiple problems' results
directories (ns-grid, ns-3d-grid, structural-mesh, thermal-mesh) and
synthesise them into a single paper-figure PDF. ns-grid is their natural
home because it is the first NS domain in the campaign — the runner picks
them up automatically when ``mosaic run --plots-only ns-grid`` fires.

Each ``_plot_<name>`` function has the signature ``(cfg, **kw) -> None``
expected by :meth:`Problem.add_extra_plot`. It resolves
``<results>/<cfg.name>/_extra/`` itself and delegates to a private
``_<name>_impl(out_dir)`` body that preserves the on-disk PDF filenames
the legacy paper-build pipeline expects.
"""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.style import (
    FEM_ORDER,
    NS_ORDER,
    PAPER_RCPARAMS,
    STRUCTURAL_ORDER,
    TEXTWIDTH,
    THERMAL_ORDER,
    dedup_handles,
    make_handle,
    paper_rc_context,
    solver_props,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extra_dir(cfg: Problem) -> Path:
    out_dir = results_dir() / cfg.name / "_extra"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _n_to_elements(N: int, subdir: str) -> int:
    if subdir == "ns-grid":
        return N**2
    if subdir == "ns-3d-grid":
        return N**3
    if subdir == "structural-mesh":
        return N * 2 * max(1, N // 2)
    if subdir == "thermal-mesh":
        return N * max(1, N // 2)
    return N


# ─────────────────────────────────────────────────────────────────────────────
# cost_overview
# ─────────────────────────────────────────────────────────────────────────────

_COST_DOMAINS = [
    ("2D NS", "ns-grid", "N"),
    ("3D NS", "ns-3d-grid", "N"),
    ("Structural", "structural-mesh", "mesh level"),
    ("Thermal", "thermal-mesh", "mesh level"),
]

_FAILURE_MARKER = {
    "OOM": "v",
    "nan": "X",
    "error": "D",
    "timeout": "s",
    "container_died": "D",
}
_FAILURE_LABEL = {
    "OOM": "OOM (VRAM)",
    "nan": "NaN gradient",
    "error": "error",
    "timeout": "timeout",
    "container_died": "error",
}


def _extract_by_n(by_n: dict) -> dict[int, float]:
    out = {}
    for k, v in by_n.items():
        if v is not None and isinstance(v, dict) and "mean" in v:
            out[int(k)] = float(v["mean"])
    return out


def _extract_mem_by_n(by_n: dict) -> dict[int, float]:
    """Extract peak (V)RAM per N — VRAM if available, else RAM."""
    out = {}
    for k, v in by_n.items():
        if v is None or not isinstance(v, dict):
            continue
        vram = v.get("vram_peak_mib") or 0.0
        ram = v.get("ram_peak_mib") or 0.0
        mem = vram if vram > 50 else ram
        if mem > 0:
            out[int(k)] = float(mem)
    return out


def _extract_fail(by_n: dict) -> tuple[int | None, str | None]:
    """Return (N, failure_type) for the first failed entry, or (None, None)."""
    for k, v in sorted(by_n.items(), key=lambda x: int(x[0])):
        if v is not None and isinstance(v, dict) and v.get("status") == "failed":
            return int(k), v.get("failure_type", "error")
    return None, None


def _add_failure(
    ax, last_el: float, fail_el: float, last_val: float, ft: str, color: str, ls: str
) -> None:
    """Draw horizontal connector + failure marker from last ok point to fail N."""
    fm = _FAILURE_MARKER.get(ft, "D")
    ax.loglog(
        [last_el, fail_el],
        [last_val, last_val],
        color=color,
        linestyle=ls,
        linewidth=1.0,
        marker="none",
        zorder=3,
    )
    ax.loglog(
        [fail_el],
        [last_val],
        marker=fm,
        color=color,
        markersize=9,
        markeredgewidth=1.2,
        markeredgecolor="white",
        linestyle="none",
        zorder=6,
    )


def _cost_overview_impl(out_dir: Path) -> None:
    plt.rcParams.update(PAPER_RCPARAMS)

    fig_w = TEXTWIDTH
    fig, axes = plt.subplots(4, 4, figsize=(fig_w, fig_w * 1.55), sharex="col")
    fig.subplots_adjust(
        left=0.07, right=0.98, bottom=0.18, top=0.97, wspace=0.30, hspace=0.18
    )

    ns_seen: set[str] = set()
    fem_seen: set[str] = set()
    failure_types_seen: set[str] = set()

    for col, (domain_label, subdir, _res_key) in enumerate(_COST_DOMAINS):
        cost_dir = results_dir() / subdir / "cost"

        fwd_path = cost_dir / "spatial_cost" / "result.json"
        vjp_path = cost_dir / "vjp_cost" / "result.json"
        fwd_data = load_json(fwd_path).get("by_N", {}) if fwd_path.exists() else {}
        vjp_data = load_json(vjp_path).get("by_N", {}) if vjp_path.exists() else {}

        ax_fwd = axes[0, col]
        ax_vjp = axes[1, col]
        ax_fmem = axes[2, col]
        ax_vmem = axes[3, col]

        all_solvers = sorted(set(fwd_data) | set(vjp_data))
        all_ns: set[int] = set()

        fmem_any = False
        vmem_any = False

        for solver in all_solvers:
            _label, color, ls, mk = solver_props(solver)
            kw = {
                "color": color,
                "linestyle": ls,
                "marker": mk,
                "markersize": 4,
                "markeredgewidth": 0,
                "linewidth": 1.6,
            }

            fwd_pts = _extract_by_n(fwd_data.get(solver, {}))
            vjp_pts = _extract_by_n(vjp_data.get(solver, {}))
            fmem_pts = _extract_mem_by_n(fwd_data.get(solver, {}))
            vmem_pts = _extract_mem_by_n(vjp_data.get(solver, {}))

            fwd_fail_N, fwd_ft = _extract_fail(fwd_data.get(solver, {}))
            vjp_fail_N, vjp_ft = _extract_fail(vjp_data.get(solver, {}))

            if fwd_pts:
                ns_f = sorted(fwd_pts)
                els_f = [_n_to_elements(n, subdir) for n in ns_f]
                ax_fwd.loglog(els_f, [fwd_pts[n] for n in ns_f], **kw)
                all_ns.update(els_f)
                if fwd_fail_N is not None:
                    fail_el = _n_to_elements(fwd_fail_N, subdir)
                    _add_failure(
                        ax_fwd, els_f[-1], fail_el, fwd_pts[ns_f[-1]], fwd_ft, color, ls
                    )
                    all_ns.add(fail_el)
                    failure_types_seen.add(fwd_ft)

            if vjp_pts:
                ns_v = sorted(vjp_pts)
                els_v = [_n_to_elements(n, subdir) for n in ns_v]
                ax_vjp.loglog(els_v, [vjp_pts[n] for n in ns_v], **kw)
                all_ns.update(els_v)
                if vjp_fail_N is not None:
                    fail_el = _n_to_elements(vjp_fail_N, subdir)
                    _add_failure(
                        ax_vjp, els_v[-1], fail_el, vjp_pts[ns_v[-1]], vjp_ft, color, ls
                    )
                    all_ns.add(fail_el)
                    failure_types_seen.add(vjp_ft)

            if fmem_pts:
                ns_fm = sorted(fmem_pts)
                els_fm = [_n_to_elements(n, subdir) for n in ns_fm]
                ax_fmem.loglog(els_fm, [fmem_pts[n] for n in ns_fm], **kw)
                all_ns.update(els_fm)
                if fwd_fail_N is not None:
                    fail_el = _n_to_elements(fwd_fail_N, subdir)
                    _add_failure(
                        ax_fmem,
                        els_fm[-1],
                        fail_el,
                        fmem_pts[ns_fm[-1]],
                        fwd_ft,
                        color,
                        ls,
                    )
                fmem_any = True

            if vmem_pts:
                ns_vm = sorted(vmem_pts)
                els_vm = [_n_to_elements(n, subdir) for n in ns_vm]
                ax_vmem.loglog(els_vm, [vmem_pts[n] for n in ns_vm], **kw)
                all_ns.update(els_vm)
                if vjp_fail_N is not None:
                    fail_el = _n_to_elements(vjp_fail_N, subdir)
                    _add_failure(
                        ax_vmem,
                        els_vm[-1],
                        fail_el,
                        vmem_pts[ns_vm[-1]],
                        vjp_ft,
                        color,
                        ls,
                    )
                vmem_any = True

            if solver in NS_ORDER:
                ns_seen.add(solver)
            if solver in FEM_ORDER:
                fem_seen.add(solver)

        ax_fwd.set_title(domain_label)
        ax_fwd.set_ylabel("Forward time (s)" if col == 0 else "")
        ax_vjp.set_ylabel(r"$t_{\mathrm{vjp}}$ (s)" if col == 0 else "")
        ax_fmem.set_ylabel("Fwd (V)RAM (MiB)" if col == 0 else "")
        ax_vmem.set_ylabel("VJP (V)RAM (MiB)" if col == 0 else "")
        ax_vmem.set_xlabel("Elements")

        if not fmem_any:
            ax_fmem.set_visible(False)
        if not vmem_any:
            ax_vmem.set_visible(False)

        tick_els = sorted(all_ns)
        if len(tick_els) > 4:
            idx = np.round(np.linspace(0, len(tick_els) - 1, 4)).astype(int)
            tick_els = [tick_els[i] for i in idx]
        for ax in (ax_fwd, ax_vjp, ax_fmem, ax_vmem):
            if not ax.get_visible():
                continue
            ax.set_xticks(tick_els)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, _: f"{round(x / 1000):.0f}k" if x >= 1000 else str(int(x))
                )
            )
            ax.tick_params(axis="x", labelsize=7.5, rotation=40)
            plt.setp(ax.get_xticklabels(), ha="right")
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
            ax.yaxis.set_minor_locator(mticker.NullLocator())

    ns_handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in ns_seen])
    fem_handles = dedup_handles([make_handle(s) for s in FEM_ORDER if s in fem_seen])

    failure_handles = [
        mlines.Line2D(
            [],
            [],
            marker=_FAILURE_MARKER[ft],
            color="0.4",
            linestyle="none",
            markersize=7,
            markeredgewidth=1.0,
            markeredgecolor="white",
            label=_FAILURE_LABEL[ft],
        )
        for ft in ["OOM", "nan", "error", "timeout"]
        if ft in failure_types_seen
    ]

    legend_kw = {"fontsize": 7.5, "framealpha": 0.7, "handlelength": 2.0}
    fig.legend(
        handles=ns_handles,
        loc="upper center",
        bbox_to_anchor=(0.20, 0.13),
        ncol=max(1, math.ceil(len(ns_handles) / 4)),
        **legend_kw,
    )
    fig.legend(
        handles=fem_handles,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.13),
        ncol=max(1, math.ceil(len(fem_handles) / 3)),
        **legend_kw,
    )
    if failure_handles:
        fig.legend(
            handles=failure_handles,
            loc="upper center",
            bbox_to_anchor=(0.90, 0.13),
            ncol=1,
            **legend_kw,
        )

    out = out_dir / "appendix_cost_overview.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def _plot_cost_overview(cfg: Problem, **_kw) -> None:
    _cost_overview_impl(_extra_dir(cfg))


# ─────────────────────────────────────────────────────────────────────────────
# scaling
# ─────────────────────────────────────────────────────────────────────────────

_GPU_SOLVERS: frozenset[str] = frozenset(
    {
        "jax_cfd",
        "phiflow",
        "pict",
        "xlb",
        "warp_ns",
        "exponax",
        "jax_fem",
        "torch_fem_thermal",
    }
)

_DEALII_SOLVERS: frozenset[str] = frozenset({"dealii_structural", "dealii_heat"})


def _device_suffix(solver: str) -> str:
    return " (G)" if solver in _GPU_SOLVERS else " (C)"


def _extract_scaling(by_n: dict) -> dict[int, float]:
    out = {}
    for k, v in by_n.items():
        if v is not None and isinstance(v, dict) and v.get("mean") is not None:
            out[int(k)] = float(v["mean"])
    return out


def _load_cost(subdir: str, experiment: str) -> dict[str, dict[int, float]]:
    p = results_dir() / subdir / "cost" / experiment / "result.json"
    if not p.exists():
        return {}
    data = load_json(p)
    return {s: _extract_scaling(nd) for s, nd in data.get("by_N", {}).items()}


def _make_ns_combined_fig() -> plt.Figure:
    """2D NS (top) and 3D NS (bottom) with a single shared legend."""
    plt.rcParams.update(PAPER_RCPARAMS)

    ns_domains = [
        ("2D NS", "ns-grid"),
        ("3D NS", "ns-3d-grid"),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(TEXTWIDTH, TEXTWIDTH * 0.60),
        sharey=False,
        dpi=300,
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, bottom=0.28, top=0.94, wspace=0.40, hspace=0.55
    )

    all_seen: set[str] = set()

    for row, (domain_label, subdir) in enumerate(ns_domains):
        ax_fwd, ax_vjp, ax_ratio = axes[row]

        fwd_data = _load_cost(subdir, "spatial_cost")
        vjp_data = _load_cost(subdir, "vjp_cost")

        all_els: set[int] = set()
        seen: set[str] = set()

        for solver in NS_ORDER:
            fwd_pts = fwd_data.get(solver, {})
            vjp_pts = vjp_data.get(solver, {})
            if not fwd_pts and not vjp_pts:
                continue

            _label, color, ls, mk = solver_props(solver)
            kw = {
                "color": color,
                "linestyle": ls,
                "marker": mk,
                "markersize": 4,
                "markeredgewidth": 0,
                "linewidth": 1.5,
            }

            if fwd_pts:
                ns_f = sorted(fwd_pts)
                els_f = [_n_to_elements(n, subdir) for n in ns_f]
                ax_fwd.loglog(els_f, [fwd_pts[n] for n in ns_f], **kw)
                all_els.update(els_f)

            if vjp_pts:
                ns_v = sorted(vjp_pts)
                els_v = [_n_to_elements(n, subdir) for n in ns_v]
                ax_vjp.loglog(els_v, [vjp_pts[n] for n in ns_v], **kw)
                all_els.update(els_v)

            common_ns = sorted(set(fwd_pts) & set(vjp_pts))
            if len(common_ns) >= 2:
                els_c = [_n_to_elements(n, subdir) for n in common_ns]
                ratios = [vjp_pts[n] / fwd_pts[n] for n in common_ns]
                ax_ratio.loglog(els_c, ratios, **kw)
                all_els.update(els_c)

            seen.add(solver)
            all_seen.add(solver)

        ax_ratio.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, zorder=0)

        if row == 0:
            ax_fwd.set_title("Forward time")
            ax_vjp.set_title("VJP time")
            ax_ratio.set_title("VJP / forward")

        ax_fwd.set_ylabel(f"{domain_label}\nTime (s)", fontsize=7.5)
        ax_vjp.set_xlabel("DOFs", fontsize=7.5)

        for ax in (ax_fwd, ax_vjp, ax_ratio):
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
            ax.yaxis.set_minor_locator(mticker.NullLocator())

        tick_els = sorted(all_els)
        if len(tick_els) > 4:
            idx = np.round(np.linspace(0, len(tick_els) - 1, 4)).astype(int)
            tick_els = [tick_els[i] for i in idx]

        fmt = mticker.FuncFormatter(
            lambda x, _: f"{round(x / 1000):.0f}k" if x >= 1000 else str(int(x))
        )
        for ax in (ax_fwd, ax_vjp, ax_ratio):
            ax.set_xticks(tick_els)
            ax.xaxis.set_major_formatter(fmt)
            ax.tick_params(axis="x", labelsize=7, rotation=35)
            plt.setp(ax.get_xticklabels(), ha="right")
            ax.tick_params(axis="y", labelsize=7)

    # Shared legend — union of all solvers shown in either row
    handles = []
    for s in NS_ORDER:
        if s not in all_seen:
            continue
        h = make_handle(s)
        h.set_label(h.get_label() + _device_suffix(s))
        handles.append(h)
    handles = dedup_handles(handles)

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=5,
        fontsize=6.5,
        framealpha=0.8,
        handlelength=2.0,
        borderpad=0.5,
        labelspacing=0.3,
    )

    return fig


def _make_fem_combined_fig() -> plt.Figure:
    """Structural (top) and Thermal (bottom) with a single shared legend.
    deal.II VJP data is suppressed (no native adjoint)."""
    plt.rcParams.update(PAPER_RCPARAMS)

    fem_domains = [
        ("Structural", "structural-mesh", STRUCTURAL_ORDER),
        ("Thermal", "thermal-mesh", THERMAL_ORDER),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(TEXTWIDTH, TEXTWIDTH * 0.60),
        sharey=False,
        dpi=300,
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, bottom=0.28, top=0.94, wspace=0.40, hspace=0.55
    )

    all_seen: set[str] = set()

    for row, (domain_label, subdir, order) in enumerate(fem_domains):
        ax_fwd, ax_vjp, ax_ratio = axes[row]

        fwd_data = _load_cost(subdir, "spatial_cost")
        vjp_data = _load_cost(subdir, "vjp_cost")

        all_els: set[int] = set()
        seen: set[str] = set()

        for solver in order:
            fwd_pts = fwd_data.get(solver, {})
            # suppress VJP for deal.II — no native adjoint
            vjp_pts = {} if solver in _DEALII_SOLVERS else vjp_data.get(solver, {})
            if not fwd_pts and not vjp_pts:
                continue

            _label, color, ls, mk = solver_props(solver)
            kw = {
                "color": color,
                "linestyle": ls,
                "marker": mk,
                "markersize": 4,
                "markeredgewidth": 0,
                "linewidth": 1.5,
            }

            if fwd_pts:
                ns_f = sorted(fwd_pts)
                els_f = [_n_to_elements(n, subdir) for n in ns_f]
                ax_fwd.loglog(els_f, [fwd_pts[n] for n in ns_f], **kw)
                all_els.update(els_f)

            if vjp_pts:
                ns_v = sorted(vjp_pts)
                els_v = [_n_to_elements(n, subdir) for n in ns_v]
                ax_vjp.loglog(els_v, [vjp_pts[n] for n in ns_v], **kw)
                all_els.update(els_v)

            common_ns = sorted(set(fwd_pts) & set(vjp_pts))
            if len(common_ns) >= 2:
                els_c = [_n_to_elements(n, subdir) for n in common_ns]
                ratios = [vjp_pts[n] / fwd_pts[n] for n in common_ns]
                ax_ratio.loglog(els_c, ratios, **kw)
                all_els.update(els_c)

            seen.add(solver)
            all_seen.add(solver)

        ax_ratio.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, zorder=0)

        if row == 0:
            ax_fwd.set_title("Forward time")
            ax_vjp.set_title("VJP time")
            ax_ratio.set_title("VJP / forward")

        ax_fwd.set_ylabel(f"{domain_label}\nTime (s)", fontsize=7.5)
        ax_vjp.set_xlabel("DOFs", fontsize=7.5)

        for ax in (ax_fwd, ax_vjp, ax_ratio):
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
            ax.yaxis.set_minor_locator(mticker.NullLocator())

        tick_els = sorted(all_els)
        if len(tick_els) > 4:
            idx = np.round(np.linspace(0, len(tick_els) - 1, 4)).astype(int)
            tick_els = [tick_els[i] for i in idx]

        fmt = mticker.FuncFormatter(
            lambda x, _: f"{round(x / 1000):.0f}k" if x >= 1000 else str(int(x))
        )
        for ax in (ax_fwd, ax_vjp, ax_ratio):
            ax.set_xticks(tick_els)
            ax.xaxis.set_major_formatter(fmt)
            ax.tick_params(axis="x", labelsize=7, rotation=35)
            plt.setp(ax.get_xticklabels(), ha="right")
            ax.tick_params(axis="y", labelsize=7)

    # Shared legend — union of all solvers shown in either row
    all_order = list(dict.fromkeys(STRUCTURAL_ORDER + THERMAL_ORDER))
    handles = []
    for s in all_order:
        if s not in all_seen:
            continue
        h = make_handle(s)
        h.set_label(h.get_label() + _device_suffix(s))
        handles.append(h)
    handles = dedup_handles(handles)

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=5,
        fontsize=6.5,
        framealpha=0.8,
        handlelength=2.0,
        borderpad=0.5,
        labelspacing=0.3,
    )

    return fig


def _scaling_impl(out_dir: Path) -> None:
    # Combined NS figure (2D + 3D, shared legend)
    fig = _make_ns_combined_fig()
    for ext in ("pdf", "png"):
        out = out_dir / f"scaling_ns_combined.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)

    # Combined FEM figure (Structural + Thermal, shared legend, no deal.II VJP)
    fig = _make_fem_combined_fig()
    for ext in ("pdf", "png"):
        out = out_dir / f"scaling_fem_combined.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


def _plot_scaling(cfg: Problem, **_kw) -> None:
    _scaling_impl(_extra_dir(cfg))


# ─────────────────────────────────────────────────────────────────────────────
# ucurves
# ─────────────────────────────────────────────────────────────────────────────


def _ucurves_configs() -> dict[str, dict]:
    R = results_dir()
    return {
        "f2": {
            "path": R / "ns-grid" / "gradient" / "horizon_sweep" / "result.json",
            "out": "appendix_ucurves_f2.pdf",
            "title": "F2 — FD U-curves (2D NS)",
            "ncols": 4,
        },
        "f3": {
            "path": R / "ns-3d-grid" / "gradient" / "horizon_sweep" / "result.json",
            "out": "appendix_ucurves_f3.pdf",
            "title": "F3 — FD U-curves (3D NS)",
            "ncols": 5,
        },
    }


def _plot_ucurve_domain(cfg: dict, out_dir: Path) -> None:
    path: Path = cfg["path"]
    if not path.exists():
        print(f"[ucurves] {path} not found — skipping")
        return

    data = load_json(path)
    by_solver: dict = data["by_solver"]

    # Collect all step values across solvers, sorted numerically
    all_steps: list[int] = sorted(
        {int(s) for sv in by_solver.values() for s in sv},
        key=int,
    )
    ncols: int = cfg["ncols"]
    nrows: int = int(np.ceil(len(all_steps) / ncols))

    panel_w = TEXTWIDTH / ncols
    panel_h = panel_w * 0.92
    fig_h = nrows * panel_h + 0.55  # extra for legend

    fig = plt.figure(figsize=(TEXTWIDTH, fig_h))
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        figure=fig,
        left=0.10,
        right=0.98,
        top=1.0 - 0.12 / fig_h,
        bottom=0.52 / fig_h,
        hspace=0.65,
        wspace=0.40,
    )

    seen: set[str] = set()

    for idx, steps in enumerate(all_steps):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        for solver in NS_ORDER:
            sv = by_solver.get(solver)
            if sv is None:
                continue
            entry = sv.get(str(steps))
            if entry is None:
                continue
            eps_sweep: dict = entry.get("eps_sweep", {})
            if not eps_sweep:
                continue

            eps_vals = sorted(eps_sweep.keys(), key=float)
            xs = [float(e) for e in eps_vals]
            ys = [eps_sweep[e]["rel_error_mean"] for e in eps_vals]

            if not all(np.isfinite(y) and y > 0 for y in ys):
                pairs = [
                    (x, y)
                    for x, y in zip(xs, ys, strict=False)
                    if np.isfinite(y) and y > 0
                ]
                if not pairs:
                    continue
                xs, ys = zip(*pairs, strict=False)

            _, color, ls, mk = solver_props(solver)
            ax.loglog(
                xs,
                ys,
                color=color,
                linestyle=ls,
                marker=mk,
                markersize=3.5,
                markeredgewidth=0,
                linewidth=1.4,
            )
            seen.add(solver)

        ax.set_title(f"$T={steps}$", fontsize=8)
        ax.set_xlabel(r"$\varepsilon$", fontsize=7.5)
        if col == 0:
            ax.set_ylabel("Rel. FD error", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    for idx in range(len(all_steps), nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in seen])
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(handles), 6),
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="0.8",
        handlelength=2.0,
    )

    out = out_dir / cfg["out"]
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def _ucurves_impl(out_dir: Path) -> None:
    with paper_rc_context():
        for cfg in _ucurves_configs().values():
            _plot_ucurve_domain(cfg, out_dir)


def _plot_ucurves(cfg: Problem, **_kw) -> None:
    _ucurves_impl(_extra_dir(cfg))


# ─────────────────────────────────────────────────────────────────────────────
# ics_figures
# ─────────────────────────────────────────────────────────────────────────────

_ICS_RCPARAMS = {
    "font.family": "sans-serif",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def _vorticity_np(u, v, N, L=2 * np.pi):
    kn = np.fft.fftfreq(N, d=L / (2 * np.pi * N))
    KX, KY = np.meshgrid(kn, kn, indexing="ij")
    return np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v) - 1j * KY * np.fft.fft2(u)))


def _ic_tgv(N=64, L=2 * np.pi):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return _vorticity_np(np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y), N, L)


def _ic_multimode(N=64, L=2 * np.pi, seed=42):
    rng = np.random.default_rng(seed)
    kn = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kn, kn, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)
    envelope = np.exp(-0.5 * ((K - 2.0) / 0.5) ** 2)
    phases = rng.uniform(0, 2 * np.pi, (N, N))
    psi_hat = envelope * np.exp(1j * phases)
    psi_hat = 0.5 * (psi_hat + np.conj(psi_hat[::-1, ::-1]))
    psi_hat[0, 0] = 0.0
    kf = 2 * np.pi / L
    u = np.real(np.fft.ifft2(1j * KY * kf * psi_hat))
    v = np.real(np.fft.ifft2(-1j * KX * kf * psi_hat))
    return _vorticity_np(u, v, N, L)


def _ic_tgv3d_slice(N=32, L=2 * np.pi):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    return _vorticity_np(u, v, N, L)


def _ic_abc_slice(N=32, L=2 * np.pi, A=1.0, B=1.0, C=1.0):
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = A * np.sin(0) + C * np.cos(Y)
    v = B * np.sin(X) + A * np.cos(0)
    return _vorticity_np(u, v, N, L)


def _ic_struct_uniform(nx=48, ny=24, rho_0=0.5):
    return np.full((nx, ny), rho_0)


def _ic_struct_random(nx=48, ny=24, seed=0):
    return np.clip(np.random.default_rng(seed).normal(0.5, 0.3, (nx, ny)), 0.05, 0.95)


def _ic_struct_two_bumps(nx=48, ny=24):
    Lx, Ly = 2.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    sigma = 0.12 * min(Lx, Ly)
    rho = np.full((nx, ny), 0.1)
    for cx in [0.35, 0.75]:
        rho += 0.85 * np.exp(
            -((X - cx * Lx) ** 2 + (Y - 0.5 * Ly) ** 2) / (2 * sigma**2)
        )
    return np.clip(rho, 0.05, 0.95)


def _ic_thermal_uniform(nx=48, ny=24, rho_0=0.5):
    return np.full((nx, ny), rho_0)


def _ic_thermal_random(nx=48, ny=24, seed=0):
    return np.clip(np.random.default_rng(seed).normal(0.5, 0.3, (nx, ny)), 0.05, 0.95)


def _ic_thermal_gaussian(nx=48, ny=24, sigma=0.2):
    Lx, Ly = 2.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    s = np.exp(
        -((X - 0.5 * Lx) ** 2 + (Y - 0.5 * Ly) ** 2) / (2 * (sigma * min(Lx, Ly)) ** 2)
    )
    return s / s.max()


def _imshow_sym_ic(ax, field, cmap="RdBu_r"):
    vmax = float(np.abs(field).max()) or 1.0
    ax.imshow(
        field.T,
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
    )
    ax.axis("off")


def _imshow_pos_ic(ax, field, cmap="viridis"):
    ax.imshow(
        field.T,
        origin="lower",
        cmap=cmap,
        vmin=field.min(),
        vmax=field.max(),
        aspect="equal",
        interpolation="bilinear",
    )
    ax.axis("off")


def _ics_figures_impl(out_dir: Path) -> None:
    plt.rcParams.update(_ICS_RCPARAMS)

    _PH3 = (TEXTWIDTH / 3) * 0.85
    # Combined NS (2D + 3D) — 4 columns
    _PH4 = (TEXTWIDTH / 4) * 0.85
    labels_ns = ["TGV (2D)", "Multimode (2D)", "TGV (3D)", "ABC (3D)"]
    fields_ns = [_ic_tgv(), _ic_multimode(), _ic_tgv3d_slice(), _ic_abc_slice()]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, _PH4))
    fig.subplots_adjust(wspace=0.05)
    for ax, field, lbl in zip(axes, fields_ns, labels_ns, strict=False):
        _imshow_sym_ic(ax, field)
        ax.set_title(lbl, fontsize=8, pad=3)
    fig.savefig(out_dir / "appendix_ics_ns_combined.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'appendix_ics_ns_combined.pdf'}")

    # Structural
    labels_struct = ["Uniform density", "Random density", "Two density bumps"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, _PH3))
    fig.subplots_adjust(wspace=0.05)
    _imshow_pos_ic(axes[0], _ic_struct_uniform())
    _imshow_pos_ic(axes[1], _ic_struct_random())
    _imshow_pos_ic(axes[2], _ic_struct_two_bumps())
    for ax, lbl in zip(axes, labels_struct, strict=False):
        ax.set_title(lbl, fontsize=8, pad=3)
    fig.savefig(out_dir / "appendix_ics_structural_mesh.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'appendix_ics_structural_mesh.pdf'}")

    # Thermal — 1 row × 3 cols
    labels_thermal = ["Uniform conductivity", "Random conductivity", "Gaussian source"]
    fields_thermal = [
        _ic_thermal_uniform(),
        _ic_thermal_random(),
        _ic_thermal_gaussian(),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, _PH3))
    fig.subplots_adjust(wspace=0.05)
    for ax, field, lbl in zip(axes, fields_thermal, labels_thermal, strict=False):
        _imshow_pos_ic(ax, field, cmap="hot")
        ax.set_title(lbl, fontsize=8, pad=3)
    fig.savefig(out_dir / "appendix_ics_thermal_mesh.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'appendix_ics_thermal_mesh.pdf'}")


def _plot_ics_figures(cfg: Problem, **_kw) -> None:
    _ics_figures_impl(_extra_dir(cfg))


# ─────────────────────────────────────────────────────────────────────────────
# domain_illustrations
# ─────────────────────────────────────────────────────────────────────────────

_CONTROL_COLOR = "#2471a3"
_OBJECTIVE_COLOR = "#c0392b"
_PHYS_COLOR = "#333333"
_OBJ_BOX_KW = {
    "boxstyle": "round,pad=0.3",
    "facecolor": "#F2F2F2",
    "edgecolor": "#AAAAAA",
    "lw": 0.6,
}
_OBJ_FONTSIZE = 5
_CTRL_FONTSIZE = 4.5
_LABEL_FONTSIZE = 4.5
_OFFSET_OBJECTIVE = 0.1


def _make_domain1(out_dir: Path) -> None:
    """Domain 1: 2D Cylinder Flow — Inflow Optimization."""
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.4, TEXTWIDTH * 0.22), dpi=300)
    ax.set_xlim(-0.5, 7.8)
    ax.set_ylim(-1.0, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")

    font_kw = {"fontfamily": "sans-serif"}

    chan_x0, chan_x1 = 0.0, 6.5
    chan_y0, chan_y1 = 0.0, 2.0
    chan_w = chan_x1 - chan_x0
    chan_h = chan_y1 - chan_y0

    fluid = mpatches.FancyBboxPatch(
        (chan_x0, chan_y0),
        chan_w,
        chan_h,
        boxstyle="square,pad=0",
        facecolor="#dceefb",
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(fluid)

    wall_thick = 0.10
    top_wall = mpatches.FancyBboxPatch(
        (chan_x0, chan_y1),
        chan_w,
        wall_thick,
        boxstyle="square,pad=0",
        facecolor="#888888",
        edgecolor="#555555",
        linewidth=0.5,
        zorder=2,
    )
    bot_wall = mpatches.FancyBboxPatch(
        (chan_x0, chan_y0 - wall_thick),
        chan_w,
        wall_thick,
        boxstyle="square,pad=0",
        facecolor="#888888",
        edgecolor="#555555",
        linewidth=0.5,
        zorder=2,
    )
    ax.add_patch(top_wall)
    ax.add_patch(bot_wall)

    cyl_cx = chan_x0 + chan_w / 3
    cyl_cy = chan_y0 + chan_h / 2
    cyl_r = 0.22
    cyl = plt.Circle(
        (cyl_cx, cyl_cy), cyl_r, fc="#3b6fa0", ec="#1a3a5c", linewidth=0.5, zorder=5
    )
    ax.add_patch(cyl)

    drag_arrow_x = cyl_cx - cyl_r - 0.08
    ax.annotate(
        "",
        xy=(drag_arrow_x - 0.7, cyl_cy),
        xytext=(drag_arrow_x, cyl_cy),
        arrowprops={
            "arrowstyle": "->,head_width=0.08,head_length=0.06",
            "color": _OBJECTIVE_COLOR,
            "lw": 0.8,
        },
        zorder=6,
    )
    ax.text(
        drag_arrow_x,
        cyl_cy - 0.5,
        r"Drag $F_D$",
        ha="right",
        va="center",
        fontsize=_LABEL_FONTSIZE,
        fontweight="bold",
        color=_OBJECTIVE_COLOR,
        **font_kw,
    )

    n_arrows = 9
    ys = np.linspace(chan_y0 + 0.12, chan_y1 - 0.12, n_arrows)
    y_mid = (chan_y0 + chan_y1) / 2
    max_len = 1.0
    for y in ys:
        frac = 1.0 - ((y - y_mid) / (chan_h / 2)) ** 2
        length = max_len * max(frac, 0.08)
        ax.annotate(
            "",
            xy=(chan_x0 + length, y),
            xytext=(chan_x0 - 0.05, y),
            arrowprops={
                "arrowstyle": "->,head_width=0.08,head_length=0.06",
                "color": _CONTROL_COLOR,
                "lw": 0.6,
            },
            zorder=4,
        )

    ax.text(
        chan_x0 - 0.15,
        chan_y1 + 0.45,
        r"Control: inflow profile $u(y)$",
        ha="left",
        va="bottom",
        fontsize=_CTRL_FONTSIZE,
        fontweight="bold",
        color=_CONTROL_COLOR,
        **font_kw,
    )

    for y in np.linspace(chan_y0 + 0.25, chan_y1 - 0.25, 5):
        ax.annotate(
            "",
            xy=(chan_x1 + 0.35, y),
            xytext=(chan_x1 - 0.15, y),
            arrowprops={
                "arrowstyle": "->,head_width=0.08,head_length=0.06",
                "color": _CONTROL_COLOR,
                "lw": 0.6,
            },
            zorder=4,
        )

    xs_stream = np.linspace(cyl_cx + cyl_r + 0.25, chan_x1 - 0.3, 200)
    offsets = [0.0, 0.35, -0.35, 0.65, -0.65]
    amplitudes = [0.18, 0.14, 0.14, 0.06, 0.06]
    phases = [0.0, np.pi, 0.0, np.pi / 2, -np.pi / 2]

    for off, amp, ph in zip(offsets, amplitudes, phases, strict=False):
        local_amp = amp * np.clip(
            (xs_stream - xs_stream[0]) / (xs_stream[-1] - xs_stream[0]), 0, 1
        )
        ys_stream = (
            cyl_cy
            + off
            + local_amp
            * np.sin(
                2.5
                * np.pi
                * (xs_stream - xs_stream[0])
                / (xs_stream[-1] - xs_stream[0])
                * 2
                + ph
            )
        )
        ys_stream = np.clip(ys_stream, chan_y0 + 0.05, chan_y1 - 0.05)
        ax.plot(xs_stream, ys_stream, color="#5dade2", lw=0.4, alpha=0.7, zorder=1)

    vortex_colors = ["#e74c3c", "#2471a3"]
    vortex_xs = np.linspace(cyl_cx + cyl_r + 0.8, chan_x1 - 1.0, 4)
    for i, vx in enumerate(vortex_xs):
        sign = 1 if i % 2 == 0 else -1
        vy = cyl_cy + sign * 0.22
        ell = mpatches.Ellipse(
            (vx, vy),
            0.36,
            0.23,
            angle=sign * 10,
            fc="none",
            ec=vortex_colors[i % 2],
            lw=0.6,
            ls="--",
            alpha=0.55,
            zorder=3,
        )
        ax.add_patch(ell)
        theta_start = 30 if sign > 0 else 210
        arc_angles = np.linspace(
            np.radians(theta_start), np.radians(theta_start + 260), 40
        )
        rx, ry = 0.12, 0.07
        arc_x = vx + rx * np.cos(arc_angles)
        arc_y = vy + ry * np.sin(arc_angles)
        ax.plot(arc_x, arc_y, color=vortex_colors[i % 2], lw=0.6, alpha=0.5, zorder=3)
        ax.annotate(
            "",
            xy=(arc_x[-1], arc_y[-1]),
            xytext=(arc_x[-3], arc_y[-3]),
            arrowprops={
                "arrowstyle": "->,head_width=0.05,head_length=0.04",
                "color": vortex_colors[i % 2],
                "lw": 0.6,
            },
            zorder=3,
        )

    for y_off in [-0.65, -0.35, 0.0, 0.35, 0.65]:
        xs_up = np.linspace(chan_x0 + 1.1, cyl_cx - cyl_r - 0.15, 60)
        ys_up = np.full_like(xs_up, cyl_cy + y_off)
        if abs(y_off) < 0.5:
            ys_up += 0.04 * y_off * np.linspace(0, 1, len(xs_up)) ** 2
        ys_up = np.clip(ys_up, chan_y0 + 0.05, chan_y1 - 0.05)
        ax.plot(xs_up, ys_up, color="#5dade2", lw=0.4, alpha=0.7, zorder=1)

    fig.text(
        0.5,
        _OFFSET_OBJECTIVE,
        r"Objective: $\min_{u(y)}\; F_D$",
        ha="center",
        va="bottom",
        fontsize=_OBJ_FONTSIZE,
        color=_PHYS_COLOR,
        bbox=_OBJ_BOX_KW,
    )

    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.10)
    fig.savefig(
        out_dir / "domain1_2d_fluids.png",
        dpi=300,
        facecolor="white",
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.10)
    fig.savefig(
        out_dir / "domain1_2d_fluids.pdf",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved {out_dir / 'domain1_2d_fluids.png'}")


def _make_domain2a_ic_recovery(out_dir: Path) -> None:
    """Domain 2A: 3D Initial Condition Recovery."""

    def taylor_green_vorticity(n=80):
        x = np.linspace(0, 2 * np.pi, n)
        y = np.linspace(0, 2 * np.pi, n)
        X, Y = np.meshgrid(x, y)
        return -2 * np.sin(X) * np.sin(Y)

    def decayed_vorticity(n=80):
        x = np.linspace(0, 2 * np.pi, n)
        y = np.linspace(0, 2 * np.pi, n)
        X, Y = np.meshgrid(x, y)
        angle = 0.4
        ca, sa = np.cos(angle), np.sin(angle)
        Xr = ca * X + sa * Y
        Yr = -sa * X + ca * Y
        return 0.55 * (-2) * np.sin(Xr) * np.sin(Yr)

    def paint_face(ax, vort, face, origin, size, cmap, vmax):
        n = vort.shape[0]
        o = np.array(origin, dtype=float)
        s = size
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        cm = plt.get_cmap(cmap)
        step = max(1, n // 80)
        coords = np.linspace(0, s, n)
        for i in range(0, n - step, step):
            for j in range(0, n - step, step):
                val = vort[i : i + step, j : j + step].mean()
                ci, cj = coords[i], coords[j]
                di = coords[min(i + step, n - 1)] - ci
                dj = coords[min(j + step, n - 1)] - cj
                if face == "xy_bottom":
                    verts = [
                        [*o, ci, cj, 0],
                        [*o, ci + di, cj, 0],
                        [*o, ci + di, cj + dj, 0],
                        [*o, ci, cj + dj, 0],
                    ]
                elif face == "xz_front":
                    verts = [
                        [*o, ci, 0, cj],
                        [*o, ci + di, 0, cj],
                        [*o, ci + di, 0, cj + dj],
                        [*o, ci, 0, cj + dj],
                    ]
                elif face == "yz_left":
                    verts = [
                        [*o, 0, ci, cj],
                        [*o, 0, ci + di, cj],
                        [*o, 0, ci + di, cj + dj],
                        [*o, 0, ci, cj + dj],
                    ]
                poly = Poly3DCollection([verts], alpha=0.75, zorder=2)
                poly.set_facecolor(cm(norm(val)))
                poly.set_edgecolor("none")
                ax.add_collection3d(poly)

    def draw_box_edges(ax, origin, size):
        o = np.array(origin)
        s = size
        corners = (
            np.array(
                [
                    [0, 0, 0],
                    [s, 0, 0],
                    [s, s, 0],
                    [0, s, 0],
                    [0, 0, s],
                    [s, 0, s],
                    [s, s, s],
                    [0, s, s],
                ]
            )
            + o
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for i, j in edges:
            ax.plot(
                *zip(corners[i], corners[j], strict=False),
                color="0.5",
                linewidth=0.5,
                linestyle="-",
                zorder=5,
            )

    BLUE = _CONTROL_COLOR
    PURPLE = _OBJECTIVE_COLOR
    GRAY = "0.4"

    box_size = 1.0
    gap = 1.3

    fig3d = plt.figure(figsize=(TEXTWIDTH * 0.4, TEXTWIDTH * 0.26), dpi=400)
    ax = fig3d.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
    ax.view_init(elev=20, azim=-55)
    ax.set_proj_type("persp", focal_length=0.35)

    omega_ic = taylor_green_vorticity(160)
    omega_tgt = decayed_vorticity(160)

    lo = (0, 0, 0)
    paint_face(ax, omega_ic, "xy_bottom", lo, box_size, "RdBu_r", 2.0)
    paint_face(ax, omega_ic.T, "xz_front", lo, box_size, "RdBu_r", 2.0)
    paint_face(ax, omega_ic, "yz_left", lo, box_size, "RdBu_r", 2.0)
    draw_box_edges(ax, lo, box_size)

    ro = (box_size + gap, 0, 0)
    paint_face(ax, omega_tgt, "xy_bottom", ro, box_size, "PiYG_r", 1.5)
    paint_face(ax, omega_tgt.T, "xz_front", ro, box_size, "PiYG_r", 1.5)
    paint_face(ax, omega_tgt, "yz_left", ro, box_size, "PiYG_r", 1.5)
    draw_box_edges(ax, ro, box_size)

    ax.plot(
        [box_size + 0.12, box_size + gap - 0.12],
        [0.5, 0.5],
        [0.5, 0.5],
        color=GRAY,
        linewidth=0.8,
        zorder=10,
        solid_capstyle="round",
    )
    ax.quiver(
        box_size + gap - 0.25,
        0.5,
        0.5,
        0.12,
        0,
        0,
        color=GRAY,
        arrow_length_ratio=0.55,
        linewidth=0.8,
        zorder=10,
    )
    ax.text(
        box_size + gap / 2,
        0.5,
        0.7,
        "Navier–Stokes\nevolution",
        fontsize=3.5,
        color=_PHYS_COLOR,
        ha="center",
        fontweight="bold",
        zorder=12,
    )

    total_w = 2 * box_size + gap
    xlim = (-0.1, total_w + 0.5)
    ylim = (-0.1, 1.15)
    zlim = (-0.05, 1.35)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
    ax.set_axis_off()

    buf = BytesIO()
    fig3d.savefig(buf, format="png", dpi=400, facecolor="white")
    buf.seek(0)
    plt.close(fig3d)

    img = plt.imread(buf)
    img = img[
        int(img.shape[0] * 0.30) : -int(img.shape[0] * 0.05),
        int(img.shape[1] * 0.20) : -int(img.shape[1] * 0.20),
        :,
    ]

    fig = plt.figure(figsize=(TEXTWIDTH * 0.4, TEXTWIDTH * 0.26), dpi=300)
    ax2d = fig.add_axes((0.0, 0.10, 1.0, 0.76))
    ax2d.imshow(img)
    ax2d.axis("off")

    fig.text(
        0.28,
        0.92,
        r"Control: $\mathbf{v}_0$",
        fontsize=_CTRL_FONTSIZE,
        color=BLUE,
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    fig.text(
        0.72,
        0.92,
        r"Target: $\mathbf{v}(T)$",
        fontsize=_CTRL_FONTSIZE,
        color=PURPLE,
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    fig.text(
        0.5,
        _OFFSET_OBJECTIVE,
        r"Objective: $\min_{\mathbf{v}_0}"
        r" \|\mathbf{v}(T;\,\mathbf{v}_0)"
        r" - \mathbf{v}_{\mathrm{target}}\|^2$",
        fontsize=_OBJ_FONTSIZE,
        color=_PHYS_COLOR,
        ha="center",
        va="bottom",
        bbox=_OBJ_BOX_KW,
    )

    fig.savefig(out_dir / "domain2a_3d_ic_recovery.png", dpi=300, facecolor="white")
    fig.savefig(
        out_dir / "domain2a_3d_ic_recovery.pdf",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved {out_dir / 'domain2a_3d_ic_recovery.png'}")


def _make_domain2a_cavity(out_dir: Path) -> None:
    """Domain 2A: 3D Lid-Driven Cavity — Lid Velocity Optimization."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=-50)

    edges = [
        ([0, 1], [0, 0], [0, 0]),
        ([0, 1], [1, 1], [0, 0]),
        ([0, 0], [0, 1], [0, 0]),
        ([1, 1], [0, 1], [0, 0]),
        ([0, 1], [0, 0], [1, 1]),
        ([0, 1], [1, 1], [1, 1]),
        ([0, 0], [0, 1], [1, 1]),
        ([1, 1], [0, 1], [1, 1]),
        ([0, 0], [0, 0], [0, 1]),
        ([1, 1], [0, 0], [0, 1]),
        ([0, 0], [1, 1], [0, 1]),
        ([1, 1], [1, 1], [0, 1]),
    ]
    for xs, ys, zs in edges:
        ax.plot(xs, ys, zs, color="0.3", linewidth=1.0, zorder=1)

    top_face = Poly3DCollection(
        [[(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]],
        alpha=0.25,
        facecolor="#4C9BE8",
        edgecolor="#2070C0",
        linewidths=1.5,
        zorder=5,
    )
    ax.add_collection3d(top_face)

    np.random.seed(42)
    n_arrows = 6
    gx = np.linspace(0.15, 0.85, n_arrows)
    gy = np.linspace(0.15, 0.85, n_arrows)
    GX, GY = np.meshgrid(gx, gy)
    GX = GX.ravel()
    GY = GY.ravel()

    base_angle = np.pi * 0.15
    U = (
        0.06
        * (1 + 0.5 * np.sin(2 * np.pi * GY))
        * np.cos(base_angle + 0.4 * np.sin(np.pi * GX))
    )
    V = 0.06 * 0.4 * np.sin(2 * np.pi * GX) * np.cos(np.pi * GY)

    for i in range(len(GX)):
        ax.quiver(
            GX[i],
            GY[i],
            1.0,
            U[i],
            V[i],
            0,
            color="#1A5FAF",
            arrow_length_ratio=0.35,
            linewidth=1.0,
            zorder=6,
        )

    meas_plane = Poly3DCollection(
        [[(0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0.5)]],
        alpha=0.15,
        facecolor="#F5A623",
        edgecolor="#D4850F",
        linewidths=1.0,
        linestyle="--",
        zorder=3,
    )
    ax.add_collection3d(meas_plane)

    def draw_curve(ax, pts, color="#888888", lw=1.0):
        pts = np.array(pts)
        t = np.linspace(0, 1, len(pts))
        t_fine = np.linspace(0, 1, 88)
        xs = np.interp(t_fine, t, pts[:, 0])
        ys = np.interp(t_fine, t, pts[:, 1])
        zs = np.interp(t_fine, t, pts[:, 2])
        ax.plot(xs, ys, zs, color=color, linewidth=lw, zorder=2)
        idx = 55
        ax.quiver(
            xs[idx],
            ys[idx],
            zs[idx],
            xs[idx + 1] - xs[idx],
            ys[idx + 1] - ys[idx],
            zs[idx + 1] - zs[idx],
            color=color,
            arrow_length_ratio=0.6,
            linewidth=lw,
            zorder=2,
        )

    draw_curve(
        ax,
        [
            (0.50, 0.50, 0.92),
            (0.80, 0.50, 0.75),
            (0.80, 0.50, 0.40),
            (0.50, 0.50, 0.15),
            (0.20, 0.50, 0.40),
            (0.20, 0.50, 0.75),
            (0.45, 0.50, 0.90),
        ],
        color="#666666",
        lw=1.2,
    )
    draw_curve(
        ax,
        [
            (0.50, 0.25, 0.88),
            (0.70, 0.25, 0.65),
            (0.65, 0.25, 0.30),
            (0.35, 0.25, 0.25),
            (0.25, 0.25, 0.55),
            (0.40, 0.25, 0.85),
        ],
        color="#999999",
        lw=0.9,
    )

    ax.text(
        0.50,
        -0.12,
        1.08,
        r"Control: lid velocity $\mathbf{v}_{\mathrm{lid}}(x,y)$",
        fontsize=10,
        color="#1A5FAF",
        ha="center",
        fontweight="bold",
        zorder=10,
    )
    ax.text(
        1.08,
        1.05,
        0.50,
        "Measurement plane\n$z = 0.5$",
        fontsize=9,
        color="#C06A00",
        ha="left",
        va="center",
        zorder=10,
    )
    ax.text(
        0.50,
        -0.18,
        -0.08,
        "No-slip walls (5 faces)",
        fontsize=9,
        color="0.35",
        ha="center",
        style="italic",
        zorder=10,
    )
    ax.text(
        -0.05,
        1.15,
        -0.05,
        "Re = 100–400",
        fontsize=10,
        color="0.25",
        ha="left",
        fontweight="bold",
        zorder=10,
    )

    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.set_zlim(-0.05, 1.15)
    ax.set_xlabel("x", fontsize=10, labelpad=2)
    ax.set_ylabel("y", fontsize=10, labelpad=2)
    ax.set_zlabel("z", fontsize=10, labelpad=2)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_zticks([0, 0.5, 1])
    ax.tick_params(labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("0.85")
    ax.yaxis.pane.set_edgecolor("0.85")
    ax.zaxis.pane.set_edgecolor("0.85")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    fig.suptitle(
        "Task 2A: 3D Lid-Driven Cavity — Lid Velocity Optimization",
        fontsize=13,
        fontweight="bold",
        y=0.95,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(
        out_dir / "domain2a_3d_cavity.png",
        dpi=150,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved {out_dir / 'domain2a_3d_cavity.png'}")


def _make_domain2b_topology(out_dir: Path) -> None:
    """Domain 2B: 3D Topology Optimization for Flow Devices."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    _Lx, Ly, Lz = 3.0, 1.0, 1.0
    x_inlet = 0.0
    x_design_start = 0.6
    x_design_end = 2.4
    x_outlet = 3.0

    def box_faces(x0, x1, y0, y1, z0, z1):
        return [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]

    ax.add_collection3d(
        Poly3DCollection(
            box_faces(x_inlet, x_design_start, 0, Ly, 0, Lz),
            alpha=0.15,
            facecolor="#a8d8ea",
            edgecolor="#4a90a4",
            linewidth=0.6,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            box_faces(x_design_start, x_design_end, 0, Ly, 0, Lz),
            alpha=0.12,
            facecolor="#ffe0a0",
            edgecolor="#c49030",
            linewidth=0.6,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            box_faces(x_design_end, x_outlet, 0, Ly, 0, Lz),
            alpha=0.15,
            facecolor="#a8d8ea",
            edgecolor="#4a90a4",
            linewidth=0.6,
        )
    )

    def draw_ellipsoid(ax, cx, cy, cz, rx, ry, rz, color="#555555", alpha=0.7):
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x = cx + rx * np.outer(np.cos(u), np.sin(v))
        y = cy + ry * np.outer(np.sin(u), np.sin(v))
        z = cz + rz * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True, linewidth=0)

    draw_ellipsoid(ax, 1.1, 0.35, 0.5, 0.15, 0.12, 0.30, color="#666666", alpha=0.75)
    draw_ellipsoid(ax, 1.5, 0.70, 0.35, 0.20, 0.10, 0.15, color="#555555", alpha=0.75)
    draw_ellipsoid(ax, 1.9, 0.30, 0.65, 0.12, 0.18, 0.20, color="#666666", alpha=0.75)
    draw_ellipsoid(ax, 1.6, 0.55, 0.75, 0.10, 0.15, 0.12, color="#777777", alpha=0.70)
    draw_ellipsoid(ax, 2.1, 0.65, 0.45, 0.14, 0.10, 0.18, color="#555555", alpha=0.75)

    for yy in [0.3, 0.7]:
        for zz in [0.3, 0.7]:
            ax.quiver(
                -0.15,
                yy,
                zz,
                0.55,
                0,
                0,
                arrow_length_ratio=0.25,
                color="#1565C0",
                linewidth=1.8,
                alpha=0.8,
            )
    for yy in [0.3, 0.7]:
        for zz in [0.3, 0.7]:
            ax.quiver(
                x_outlet - 0.1,
                yy,
                zz,
                0.45,
                0,
                0,
                arrow_length_ratio=0.25,
                color="#1565C0",
                linewidth=1.8,
                alpha=0.8,
            )

    arr_y = -0.15
    arr_z = -0.15
    ax.plot(
        [0.05, 2.95],
        [arr_y, arr_y],
        [arr_z, arr_z],
        color="#B71C1C",
        linewidth=2.0,
        linestyle="-",
    )
    ax.quiver(
        0.05,
        arr_y,
        arr_z,
        -0.001,
        0,
        0,
        arrow_length_ratio=100,
        color="#B71C1C",
        linewidth=2.0,
    )
    ax.quiver(
        2.95,
        arr_y,
        arr_z,
        0.001,
        0,
        0,
        arrow_length_ratio=100,
        color="#B71C1C",
        linewidth=2.0,
    )
    ax.text(
        1.5,
        arr_y - 0.08,
        arr_z - 0.12,
        r"$\Delta p = p_{\mathrm{in}} - p_{\mathrm{out}}$",
        fontsize=11,
        color="#B71C1C",
        ha="center",
        va="top",
        fontweight="bold",
    )

    ax.text(
        0.3,
        0.5,
        1.15,
        "Inlet",
        fontsize=11,
        ha="center",
        va="bottom",
        color="#0D47A1",
        fontweight="bold",
    )
    ax.text(
        2.7,
        0.5,
        1.15,
        "Outlet",
        fontsize=11,
        ha="center",
        va="bottom",
        color="#0D47A1",
        fontweight="bold",
    )
    ax.text(
        1.5,
        0.5,
        1.20,
        "Design region",
        fontsize=11,
        ha="center",
        va="bottom",
        color="#BF360C",
        fontweight="bold",
    )
    ax.text(
        1.5,
        0.5,
        -0.45,
        r"Control: density field $\rho(x,y,z) \in [0,1]$",
        fontsize=9.5,
        ha="center",
        va="top",
        color="#6D4C00",
        fontstyle="italic",
    )
    ax.text(
        1.5,
        1.25,
        -0.30,
        "Brinkman penalization on regular grid",
        fontsize=9,
        ha="center",
        va="top",
        color="#37474F",
        fontstyle="italic",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#90A4AE",
            "alpha": 0.85,
        },
    )
    ax.text(
        1.5, -0.05, 1.02, "wall", fontsize=7, ha="center", color="#607D8B", alpha=0.7
    )
    ax.text(
        1.5, 1.05, 1.02, "wall", fontsize=7, ha="center", color="#607D8B", alpha=0.7
    )

    fig.suptitle(
        "Task 2B: 3D Topology Optimization for Flow Devices",
        fontsize=14,
        fontweight="bold",
        y=0.94,
    )

    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_zlim(-0.3, 1.3)
    ax.set_xlabel("x", fontsize=9, labelpad=2)
    ax.set_ylabel("y", fontsize=9, labelpad=2)
    ax.set_zlabel("z", fontsize=9, labelpad=2)
    ax.set_box_aspect([3, 1, 1])
    ax.view_init(elev=22, azim=-55)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="both", which="both", length=0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(
        out_dir / "domain2b_3d_topology.png",
        dpi=150,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved {out_dir / 'domain2b_3d_topology.png'}")


def _make_domain3(out_dir: Path) -> None:
    """Domain 3: Cantilever Beam — Compliance Minimization."""
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.4, TEXTWIDTH * 0.22), dpi=300)

    x0, y0 = 1.5, 1.0
    W, H = 6.0, 2.0
    nx, ny = 90, 30
    rho = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            xn = i / (nx - 1)
            yn = j / (ny - 1)
            density = 0.0
            top_thickness = 0.12 + 0.06 * (1 - xn)
            if yn > (1.0 - top_thickness):
                density = max(density, 0.85 + 0.15 * (1 - xn))
            bot_thickness = 0.12 + 0.06 * (1 - xn)
            if yn < bot_thickness:
                density = max(density, 0.85 + 0.15 * (1 - xn))
            n_bays = 5
            for k in range(n_bays):
                bay_left = k / n_bays
                bay_right = (k + 1) / n_bays
                bay_mid = (bay_left + bay_right) / 2.0
                strut_half_w = 0.035 + 0.02 * (1 - bay_mid)
                if bay_left <= xn <= bay_right:
                    local_x = (xn - bay_left) / (bay_right - bay_left)
                    for target in [
                        1.0 - local_x * 0.5,
                        local_x * 0.5,
                        0.5 + local_x * 0.5,
                        0.5 - local_x * 0.5,
                    ]:
                        if abs(yn - target) < strut_half_w:
                            density = max(density, 0.7 + 0.2 * (1 - xn))
            if xn < 0.04:
                density = max(density, 1.0)
            dist_to_load = np.sqrt((xn - 1.0) ** 2 + (yn - 0.5) ** 2)
            if dist_to_load < 0.12:
                density = max(density, 0.9)
            rho[j, i] = density

    def smooth_2d(arr, n=3):
        out = arr.copy()
        for _ in range(n):
            padded = np.pad(out, 1, mode="edge")
            out = (
                padded[:-2, :-2]
                + padded[:-2, 1:-1]
                + padded[:-2, 2:]
                + padded[1:-1, :-2]
                + padded[1:-1, 1:-1]
                + padded[1:-1, 2:]
                + padded[2:, :-2]
                + padded[2:, 1:-1]
                + padded[2:, 2:]
            ) / 9.0
        return out

    rho = smooth_2d(rho, n=3)
    rho = np.clip(rho, 0, 1)

    cmap = LinearSegmentedColormap.from_list(
        "topo", ["#FFFFFF", "#BDD7EE", "#4472C4", "#1F3864"]
    )

    extent = [x0, x0 + W, y0, y0 + H]
    ax.imshow(
        rho,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="bilinear",
    )

    ax.add_patch(
        patches.FancyBboxPatch(
            (x0, y0),
            W,
            H,
            boxstyle="square,pad=0",
            linewidth=0.6,
            edgecolor="black",
            facecolor="none",
        )
    )

    wall_w = 0.3
    ax.add_patch(
        patches.Rectangle(
            (x0 - wall_w, y0 - 0.1),
            wall_w,
            H + 0.2,
            linewidth=0.4,
            edgecolor="#333333",
            facecolor="#DDDDDD",
            hatch="////",
        )
    )
    ax.plot(
        [x0 - wall_w, x0 - wall_w],
        [y0 - 0.1, y0 + H + 0.1],
        color="#333333",
        linewidth=0.7,
    )
    ax.text(
        x0 - wall_w / 2,
        y0 + H + 0.30,
        "Clamped",
        ha="center",
        va="bottom",
        fontsize=4,
        fontweight="bold",
        color="#333333",
    )

    load_x = x0 + W
    load_y = y0 + H / 2
    arrow_len = 0.8
    ax.annotate(
        "",
        xy=(load_x + 0.15, load_y - arrow_len),
        xytext=(load_x + 0.15, load_y + 0.05),
        arrowprops={
            "arrowstyle": "->",
            "color": _OBJECTIVE_COLOR,
            "lw": 1.0,
            "mutation_scale": 6,
        },
    )
    ax.text(
        load_x + 0.45,
        load_y - arrow_len / 2 + 0.05,
        "      $\\mathbf{F}$\n(tip load)",
        ha="left",
        va="center",
        fontsize=_LABEL_FONTSIZE,
        fontweight="bold",
        color=_OBJECTIVE_COLOR,
    )

    ax.text(
        x0 + W / 2,
        y0 + H + 0.45,
        r"Control: element densities $\rho_e$",
        ha="center",
        va="bottom",
        fontsize=_CTRL_FONTSIZE,
        fontweight="bold",
        color=_CONTROL_COLOR,
    )
    fig.text(
        0.5,
        _OFFSET_OBJECTIVE,
        r"Objective: $\min_\rho\; \mathbf{f}^\top \mathbf{u}(\rho)\quad"
        r"\mathrm{s.t.}\quad \sum_e\, v_e \rho_e \leq V_f$",
        ha="center",
        va="bottom",
        fontsize=_OBJ_FONTSIZE,
        color=_PHYS_COLOR,
        bbox=_OBJ_BOX_KW,
    )

    ax.set_xlim(x0 - 0.8, x0 + W + 1.6)
    ax.set_ylim(y0 - 0.9, y0 + H + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.18)
    fig.savefig(
        out_dir / "domain3_structures.png",
        dpi=300,
        facecolor="white",
    )
    fig.savefig(
        out_dir / "domain3_structures.pdf",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved {out_dir / 'domain3_structures.png'}")


def _make_domain4(out_dir: Path) -> None:
    """Domain 4: Steady-State Heat Conduction — Conductivity Inversion."""
    N = 200
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    def gaussian(X, Y, cx, cy, sx, sy, amp):
        return amp * np.exp(
            -((X - cx) ** 2) / (2 * sx**2) - (Y - cy) ** 2 / (2 * sy**2)
        )

    k_field = (
        1.0
        + gaussian(X, Y, 0.3, 0.7, 0.10, 0.10, 2.5)
        + gaussian(X, Y, 0.7, 0.3, 0.12, 0.09, 2.0)
        + gaussian(X, Y, 0.5, 0.55, 0.08, 0.11, 1.5)
    )
    T_obs = (
        0.15 * (1 - X)
        + gaussian(X, Y, 0.55, 0.50, 0.25, 0.25, 0.6)
        + gaussian(X, Y, 0.35, 0.65, 0.18, 0.18, 0.3)
        + 0.08 * np.sin(np.pi * Y)
    )

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH * 0.4, TEXTWIDTH * 0.26), dpi=300
    )
    fig.subplots_adjust(wspace=0.7, top=0.92, bottom=0.10, left=0.04, right=0.96)

    ax = ax_l
    ax.set_title(
        r"Control: $k(x,y)$",
        fontsize=_CTRL_FONTSIZE,
        fontweight="bold",
        color=_CONTROL_COLOR,
        pad=4,
    )
    ax.imshow(
        k_field[::-1],
        extent=[0, 1, 0, 1],
        cmap="viridis",
        aspect="auto",
        vmin=0.8,
        vmax=4.0,
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="square,pad=0",
            linewidth=0.5,
            edgecolor="k",
            facecolor="none",
        )
    )

    ax.plot([0, 0], [0, 1], color="0.45", linewidth=1.5, solid_capstyle="butt")
    ax.text(
        -0.05,
        0.5,
        r"$T = T_0$ (Dirichlet)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=3.5,
        color="0.45",
    )

    ax.plot([1, 1], [0, 1], color="darkorange", linewidth=1.5, solid_capstyle="butt")
    for yy in np.linspace(0.15, 0.85, 5):
        n_wave = 40
        t_arr = np.linspace(0, 0.14, n_wave + 6)
        fade_start = int(0.80 * n_wave)
        flat_start = int(0.95 * n_wave)
        envelope = np.ones(len(t_arr))
        envelope[fade_start:flat_start] = 0.5 * (
            1 + np.cos(np.linspace(0, np.pi, flat_start - fade_start))
        )
        envelope[flat_start:] = 0
        wave = yy + 0.030 * envelope * np.sin(2 * np.pi * t_arr / 0.04)
        ax.plot(1 + t_arr, wave, color="darkorange", linewidth=0.9)
        ax.annotate(
            "",
            xy=(1.17, yy),
            xytext=(1.12, yy),
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-|>,head_width=0.14,head_length=0.14",
                "color": "darkorange",
                "lw": 0.75,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    ax.text(
        0.47,
        0.82,
        r"$\nabla \cdot (k \nabla T) + q = 0$",
        ha="center",
        va="center",
        fontsize=_LABEL_FONTSIZE,
        color="white",
        path_effects=[pe.withStroke(linewidth=1.2, foreground="black", alpha=0.4)],
        transform=ax.transAxes,
    )

    ax.annotate(
        "",
        xy=(1.70, 0.62),
        xytext=(1.10, 0.62),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>,head_width=0.14,head_length=0.18",
            "color": "black",
            "lw": 1,
        },
    )
    ax.text(
        1.40,
        0.67,
        "Steady-state\nsolve",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=4.5,
        fontstyle="italic",
    )

    ax.set_xlim(-0.12, 1.14)
    ax.set_ylim(-0.12, 1.14)
    ax.set_box_aspect(1)
    ax.set_axis_off()

    ax = ax_r
    ax.set_title(
        "Observed\ntemperature field",
        fontsize=_LABEL_FONTSIZE,
        fontweight="bold",
        color=_PHYS_COLOR,
        pad=-10,
    )
    ax.imshow(
        T_obs[::-1],
        extent=[0, 1, 0, 1],
        cmap="coolwarm",
        aspect="auto",
        vmin=0,
        vmax=1.0,
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="square,pad=0",
            linewidth=0.5,
            edgecolor="k",
            facecolor="none",
        )
    )

    ax.text(
        0.50,
        0.75,
        r"$T_{\mathrm{obs}} = T(x,y;\, k^*)$",
        ha="center",
        va="bottom",
        fontsize=4.7,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.annotate(
        "",
        xy=(-0.60, 0.38),
        xytext=(0.00, 0.38),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>,head_width=0.14,head_length=0.18",
            "color": _OBJECTIVE_COLOR,
            "lw": 1,
        },
    )
    ax.text(
        -0.30,
        0.32,
        r"Invert: find $k(x,y)$" + "\n" + r"that produces $T_{\mathrm{obs}}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=4.2,
        color=_OBJECTIVE_COLOR,
        fontstyle="italic",
    )
    fig.text(
        0.5,
        _OFFSET_OBJECTIVE,
        r"Objective:  $\min_k \; \| T(k) - T_{\mathrm{obs}} \|^2$",
        ha="center",
        va="bottom",
        fontsize=_OBJ_FONTSIZE,
        color=_PHYS_COLOR,
        bbox=_OBJ_BOX_KW,
    )

    ax.set_xlim(-0.12, 1.14)
    ax.set_ylim(-0.12, 1.14)
    ax.set_box_aspect(1)
    ax.set_axis_off()

    fig.savefig(out_dir / "domain4_heat.png", dpi=300, facecolor="white")
    fig.savefig(out_dir / "domain4_heat.pdf", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_dir / 'domain4_heat.png'}")


def _domain_illustrations_impl(out_dir: Path) -> None:
    _make_domain1(out_dir)
    _make_domain2a_ic_recovery(out_dir)
    _make_domain2a_cavity(out_dir)
    _make_domain2b_topology(out_dir)
    _make_domain3(out_dir)
    _make_domain4(out_dir)


def _plot_domain_illustrations(cfg: Problem, **_kw) -> None:
    _domain_illustrations_impl(_extra_dir(cfg))


# ─────────────────────────────────────────────────────────────────────────────
# visual_abstract
# ─────────────────────────────────────────────────────────────────────────────

_VA_C = {
    "bg": "#FFFFFF",
    "jax": "#4285F4",
    "pytorch": "#EE4C2C",
    "julia": "#9558B2",
    "cpp": "#6C757D",
    "fenics": "#2CA02C",
    "arrow": "#495057",
    "tess": "#2C3E50",
    "text": "#212529",
    "muted": "#868E96",
    "api_bg": "#E9ECEF",
    "fwd": "#2980B9",
    "inv": "#C0392B",
    "placeholder": "#B0BEC5",
}

_VA_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial"],
    "text.color": _VA_C["text"],
    "figure.facecolor": _VA_C["bg"],
    "savefig.facecolor": _VA_C["bg"],
}


def _load_flow_data() -> dict | None:
    """Load flow fields and profiles from the drag_opt benchmark results."""
    base = results_dir() / "ns-grid" / "optimization" / "drag_opt" / "re20"
    flow_path = base / "flow_fields.npz"
    prof_path = base / "profiles.npz"
    if not flow_path.exists() or not prof_path.exists():
        print(f"[visual_abstract] result data not found at {base} — skipping")
        return None
    flows = try_load_npz(flow_path)
    profs = try_load_npz(prof_path)
    return {
        "flow_initial": flows["flow_initial"][:, :, 0, :],
        "flow_final": flows["flow_final_xlb"][:, :, 0, :],
        "profile_initial": profs["initial"],
        "profile_final": profs["final_xlb"],
    }


def _va_card(ax, cx, cy, w, h, name, fs, alpha=0.85) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012",
            facecolor="#4a5568",
            edgecolor="none",
            alpha=alpha,
            lw=0,
            zorder=2,
        )
    )
    ax.text(
        cx,
        cy,
        name,
        fontsize=fs,
        fontweight="medium",
        color="white",
        ha="center",
        va="center",
        zorder=3,
        fontfamily="monospace",
    )


def _va_section_title(ax, text: str) -> None:
    ax.text(
        0.50,
        1.05,
        text,
        fontsize=16,
        fontweight="bold",
        color=_VA_C["text"],
        ha="center",
        va="top",
        transform=ax.transAxes,
    )


def _va_flow_arrow(fig, x0, y0, x1, y1) -> None:
    fig.patches.append(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="->,head_width=6,head_length=4.5",
            color=_VA_C["arrow"],
            lw=2,
            mutation_scale=1,
            transform=fig.transFigure,
            zorder=10,
            clip_on=False,
        )
    )


def _va_colored_arrow(fig, x0, y0, x1, y1, color) -> None:
    rad = -0.15 if y1 > y0 else 0.15
    conn = f"arc3,rad={rad}"
    style = "->,head_width=5,head_length=4"
    fig.patches.append(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            color="white",
            lw=3,
            mutation_scale=1,
            connectionstyle=conn,
            transform=fig.transFigure,
            zorder=9,
            clip_on=False,
        )
    )
    fig.patches.append(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            color=color,
            lw=2,
            mutation_scale=1,
            connectionstyle=conn,
            transform=fig.transFigure,
            zorder=10,
            clip_on=False,
        )
    )


def _va_draw_flow_panel(ax, flow_field, profile, border_color) -> None:
    ux = flow_field[:, :, 0]
    uy = flow_field[:, :, 1]
    speed = np.sqrt(ux**2 + uy**2)
    extent = [0, 1, 0, 1]
    ax.imshow(
        speed.T,
        origin="lower",
        extent=extent,
        cmap="YlGnBu_r",
        vmin=0,
        vmax=0.85,
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )
    ax.add_patch(
        plt.Circle((0.33, 0.5), 0.06, fc="#2c3e50", ec="#1a252f", lw=1.0, zorder=3)
    )
    wall_h = 0.015
    ax.add_patch(
        Rectangle((0, 1 - wall_h), 1, wall_h, fc="#555555", ec="none", zorder=2)
    )
    ax.add_patch(Rectangle((0, 0), 1, wall_h, fc="#555555", ec="none", zorder=2))

    ny = len(profile)
    ys = np.linspace(0.03, 0.97, ny)
    max_arrow_len = 0.14
    p_min, p_max = profile.min(), profile.max()
    if p_max > p_min:
        p_norm = (profile - p_min) / (p_max - p_min)
    else:
        p_norm = np.ones_like(profile)

    for y, pn in list(zip(ys, p_norm, strict=False))[::2]:
        length = max_arrow_len * max(pn, 0.05)
        x_tip = length
        x_tail = -0.01
        ax.plot(
            [x_tail, x_tip],
            [y, y],
            color="black",
            lw=3.6,
            solid_capstyle="butt",
            zorder=4,
        )
        ax.plot(
            x_tip + 0.003,
            y,
            marker=">",
            color="black",
            ms=6.0,
            markeredgewidth=0,
            zorder=4,
        )
        ax.plot(
            [x_tail, x_tip],
            [y, y],
            color=border_color,
            lw=3.2,
            solid_capstyle="butt",
            zorder=5,
        )
        ax.plot(
            x_tip + 0.003,
            y,
            marker=">",
            color=border_color,
            ms=5.5,
            markeredgewidth=0,
            zorder=5,
        )

    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.01, 1.01)
    ax.axis("off")


def _va_draw_backends(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _va_section_title(ax, "Solver backends")
    solvers = [
        "PhiFlow",
        "XLB",
        "PICT",
        "FEniCS",
        "INS.jl",
        "OpenFOAM",
        "deal.II",
        "···",
    ]
    w, h = 0.72, 0.080
    n = len(solvers)
    total_h = n * h + (n - 1) * 0.022
    y_top = 0.5 + total_h / 2
    for i, name in enumerate(solvers):
        cy = y_top - i * (h + 0.022) - h / 2
        _va_card(ax, 0.50, cy, w, h, name, 13.0, alpha=0.85 if name != "···" else 0.35)


def _va_draw_interface(ax) -> tuple[tuple[float, float], tuple[float, float]]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _va_section_title(ax, "Standardized\ninterface")
    tcx, tcy = 0.50, 0.50
    tw, th = 0.84, 0.22
    ax.add_patch(
        FancyBboxPatch(
            (tcx - tw / 2, tcy - th / 2),
            tw,
            th,
            boxstyle="round,pad=0.018",
            facecolor=_VA_C["tess"],
            edgecolor=_VA_C["tess"],
            alpha=0.92,
            lw=2.0,
            zorder=2,
        )
    )
    ax.text(
        tcx,
        tcy + 0.06,
        "Tesseract",
        fontsize=16,
        color="white",
        ha="center",
        va="center",
        fontweight="bold",
        zorder=4,
        fontfamily="monospace",
    )
    ax.text(
        tcx,
        tcy - 0.01,
        "apply(x) → y",
        fontsize=15,
        color="#85C1E9",
        ha="center",
        va="center",
        zorder=4,
        fontfamily="monospace",
        fontweight="medium",
    )
    ax.text(
        tcx,
        tcy - 0.08,
        "vjp(x, v) → g",
        fontsize=15,
        color="#F1948A",
        ha="center",
        va="center",
        zorder=4,
        fontfamily="monospace",
        fontweight="medium",
    )
    fwd_label = (tcx + tw / 2 - 0.02, tcy - 0.01)
    vjp_label = (tcx + tw / 2 - 0.02, tcy - 0.08)
    return fwd_label, vjp_label


def _va_draw_tasks(ax, data) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _va_section_title(ax, "Benchmark tasks")
    if data is None:
        ax.text(
            0.5,
            0.5,
            "(result data not available)",
            ha="center",
            va="center",
            fontsize=12,
            color=_VA_C["muted"],
        )
        return
    iax_top = ax.inset_axes([0.02, 0.57, 0.96, 0.34])
    _va_draw_flow_panel(
        iax_top, data["flow_initial"], data["profile_initial"], _VA_C["fwd"]
    )
    for sp in iax_top.spines.values():
        sp.set_visible(True)
        sp.set_color(_VA_C["fwd"])
        sp.set_linewidth(2.0)
    ax.text(
        0.50,
        0.92,
        "forward solve",
        fontsize=13,
        color=_VA_C["fwd"],
        ha="center",
        va="bottom",
        style="italic",
        zorder=10,
    )
    iax_bot = ax.inset_axes([0.02, 0.08, 0.96, 0.34])
    _va_draw_flow_panel(
        iax_bot, data["flow_final"], data["profile_final"], _VA_C["inv"]
    )
    for sp in iax_bot.spines.values():
        sp.set_visible(True)
        sp.set_color(_VA_C["inv"])
        sp.set_linewidth(2.0)
    ax.text(
        0.50,
        0.43,
        "optimized inflow (via gradient)",
        fontsize=13,
        color=_VA_C["inv"],
        ha="center",
        va="bottom",
        style="italic",
        zorder=10,
    )


def _va_draw_results(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _va_section_title(ax, "Evaluation results")
    solvers = ["PhiFlow", "XLB", "PICT", "JAX-CFD", "INS.jl", "Warp-NS", "OpenFOAM"]
    col_headers = ["Fwd", "VJP", "Grad", "Opt."]
    n_rows = len(solvers)
    x0, y0 = 0.04, 0.08
    tw, th = 0.92, 0.82
    header_h = 0.07
    row_h = (th - header_h) / n_rows
    col_w = [0.26, 0.16, 0.16, 0.16, 0.16]
    total_cw = sum(col_w)
    col_w = [c / total_cw * tw for c in col_w]
    sym = {
        "ok": ("●", "#222222"),
        "pt": ("◑", "#555555"),
        "no": ("✗", "#555555"),
        "na": ("—", "#AAAAAA"),
    }
    cell_data = [
        ["ok", "ok", "ok", "pt"],
        ["ok", "ok", "ok", "ok"],
        ["ok", "ok", "ok", "ok"],
        ["ok", "ok", "ok", "no"],
        ["ok", "ok", "pt", "no"],
        ["ok", "ok", "pt", "pt"],
        ["ok", "na", "na", "na"],
    ]
    hy = y0 + th
    ax.plot([x0, x0 + tw], [hy, hy], color=_VA_C["text"], lw=1.6)
    hx = x0 + col_w[0]
    hdr_y = hy - header_h / 2
    for j, hdr in enumerate(col_headers):
        cx = hx + sum(col_w[1 : j + 1]) + col_w[j + 1] / 2
        ax.text(
            cx,
            hdr_y,
            hdr,
            fontsize=10,
            fontweight="bold",
            color=_VA_C["text"],
            ha="center",
            va="center",
        )
    rule_y = hy - header_h
    ax.plot([x0, x0 + tw], [rule_y, rule_y], color=_VA_C["text"], lw=0.8)
    for i, solver in enumerate(solvers):
        ry = rule_y - (i + 0.5) * row_h
        ax.text(
            x0 + 0.02,
            ry,
            solver,
            fontsize=10,
            color=_VA_C["text"],
            ha="left",
            va="center",
            fontfamily="monospace",
        )
        for j, glyph in enumerate(cell_data[i]):
            cx = hx + sum(col_w[1 : j + 1]) + col_w[j + 1] / 2
            symbol, color = sym[glyph]
            ax.text(
                cx,
                ry,
                symbol,
                fontsize=11,
                color=color,
                ha="center",
                va="center",
                zorder=2,
            )
    bot_y = rule_y - n_rows * row_h
    ax.plot([x0, x0 + tw], [bot_y, bot_y], color=_VA_C["text"], lw=1.6)


def _visual_abstract_impl(out_dir: Path) -> None:
    data = _load_flow_data()

    with plt.rc_context(_VA_RCPARAMS):
        fig = plt.figure(figsize=(14, 5.8), dpi=200)
        gap = 0.028
        widths = [0.14, 0.16, 0.22, 0.32]
        left = 0.008
        bottom, h = 0.015, 0.95

        positions = []
        x = left
        for w in widths:
            positions.append(x)
            x += w + gap

        axes = []
        for pos, w in zip(positions, widths, strict=False):
            axes.append(fig.add_axes([pos, bottom, w, h]))

        _va_draw_backends(axes[0])
        fwd_label, vjp_label = _va_draw_interface(axes[1])
        _va_draw_tasks(axes[2], data)
        _va_draw_results(axes[3])

        my = 0.48
        _va_flow_arrow(fig, positions[0] + widths[0], my, positions[1], my)
        _va_flow_arrow(fig, positions[2] + widths[2], my, positions[3], my)

        ax_iface = axes[1]
        ax_tasks = axes[2]

        fwd_src = ax_iface.transData.transform(fwd_label)
        fwd_src_fig = fig.transFigure.inverted().transform(fwd_src)
        fwd_dst = ax_tasks.transData.transform((0.0, 0.71))
        fwd_dst_fig = fig.transFigure.inverted().transform(fwd_dst)
        _va_colored_arrow(
            fig,
            fwd_src_fig[0],
            fwd_src_fig[1],
            fwd_dst_fig[0],
            fwd_dst_fig[1],
            _VA_C["fwd"],
        )

        vjp_src = ax_iface.transData.transform(vjp_label)
        vjp_src_fig = fig.transFigure.inverted().transform(vjp_src)
        vjp_dst = ax_tasks.transData.transform((0.0, 0.24))
        vjp_dst_fig = fig.transFigure.inverted().transform(vjp_dst)
        _va_colored_arrow(
            fig,
            vjp_src_fig[0],
            vjp_src_fig[1],
            vjp_dst_fig[0],
            vjp_dst_fig[1],
            _VA_C["inv"],
        )

        for ext in ("pdf", "png"):
            out = out_dir / f"visual_abstract.{ext}"
            fig.savefig(
                out,
                bbox_inches="tight",
                pad_inches=0.04,
                dpi=300 if ext == "png" else 200,
            )
            print(f"Saved {out}")
        plt.close(fig)


def _plot_visual_abstract(cfg: Problem, **_kw) -> None:
    _visual_abstract_impl(_extra_dir(cfg))


# ─────────────────────────────────────────────────────────────────────────────
# Registration entry point
# ─────────────────────────────────────────────────────────────────────────────


def register(problem: Problem) -> None:
    """Register the cross-domain ``_extra/`` plot fns on *problem*.

    The runner discovers ``"_extra/<name>"`` keys in ``problem.plot_fns`` and
    fires them unconditionally during ``mosaic run`` / ``mosaic run --plots-only``.
    """
    problem.add_extra_plot("_extra/cost_overview", _plot_cost_overview)
    problem.add_extra_plot("_extra/scaling", _plot_scaling)
    problem.add_extra_plot("_extra/ucurves", _plot_ucurves)
    problem.add_extra_plot("_extra/ics_figures", _plot_ics_figures)
    problem.add_extra_plot("_extra/domain_illustrations", _plot_domain_illustrations)
    problem.add_extra_plot("_extra/visual_abstract", _plot_visual_abstract)
