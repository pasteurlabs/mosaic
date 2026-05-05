"""Generate Figure: Solver scaling — forward time, VJP time, gradient overhead ratio vs DOFs.

One figure per domain (4 total):
  scaling_ns_grid.pdf      — 2D NS
  scaling_ns_3d_grid.pdf   — 3D NS
  scaling_structural.pdf   — Structural  (also used in main paper)
  scaling_thermal.pdf      — Thermal

Each figure: 3 panels in a single row — forward | VJP | ratio (log-log).
Solver labels carry (G) / (C) to indicate GPU vs. CPU execution.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    NS_ORDER,
    RCPARAMS,
    STRUCTURAL_ORDER,
    THERMAL_ORDER,
    dedup_handles,
    make_handle,
    solver_props,
)

RESULTS = Path(__file__).parent.parent.parent / "results"

DOMAINS = [
    ("2D NS", "ns-grid", "scaling_ns_grid", NS_ORDER),
    ("3D NS", "ns-3d-grid", "scaling_ns_3d_grid", NS_ORDER),
    ("Structural", "structural-mesh", "scaling_structural", STRUCTURAL_ORDER),
    ("Thermal", "thermal-mesh", "scaling_thermal", THERMAL_ORDER),
]

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


def _device_suffix(solver: str) -> str:
    return " (G)" if solver in _GPU_SOLVERS else " (C)"


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


def _extract(by_n: dict) -> dict[int, float]:
    out = {}
    for k, v in by_n.items():
        if v is not None and isinstance(v, dict) and v.get("mean") is not None:
            out[int(k)] = float(v["mean"])
    return out


def _load_cost(subdir: str, experiment: str) -> dict[str, dict[int, float]]:
    p = RESULTS / subdir / "cost" / experiment / "result.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {s: _extract(nd) for s, nd in data.get("by_N", {}).items()}


def _make_domain_fig(domain_label: str, subdir: str, order: list[str]):
    plt.rcParams.update(RCPARAMS)

    fig, (ax_fwd, ax_vjp, ax_ratio) = plt.subplots(
        1,
        3,
        figsize=(TEXTWIDTH, TEXTWIDTH * 0.3),
        sharey=False,
        dpi=300,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.42, top=0.91, wspace=0.40)

    fwd_data = _load_cost(subdir, "spatial_cost")
    vjp_data = _load_cost(subdir, "vjp_cost")

    all_els: set[int] = set()
    seen: set[str] = set()

    for solver in order:
        fwd_pts = fwd_data.get(solver, {})
        vjp_pts = vjp_data.get(solver, {})
        if not fwd_pts and not vjp_pts:
            continue

        label, color, ls, mk = solver_props(solver)
        kw = dict(
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=4,
            markeredgewidth=0,
            linewidth=1.5,
        )

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

    ax_ratio.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, zorder=0)

    ax_fwd.set_title("Forward time")
    ax_vjp.set_title("VJP time")
    ax_ratio.set_title("VJP / forward")

    ax_fwd.set_ylabel("Time (s)", fontsize=7.5)
    ax_vjp.set_xlabel("DOFs", fontsize=7.5)

    for ax in (ax_fwd, ax_vjp, ax_ratio):
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    # x-ticks: up to 4 evenly spaced DOF values
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

    # Legend — lower-left corner, solver labels annotated with (G)/(C)
    handles = []
    for s in order:
        if s not in seen:
            continue
        h = make_handle(s)
        h.set_label(h.get_label() + _device_suffix(s))
        handles.append(h)
    handles = dedup_handles(handles)

    ncol = 5  # min(2, max(1, math.ceil(len(handles) / 3)))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=ncol,
        fontsize=6.5,
        framealpha=0.8,
        handlelength=2.0,
        borderpad=0.5,
        labelspacing=0.3,
    )

    return fig


def generate(out_dir: Path) -> None:
    for domain_label, subdir, stem, order in DOMAINS:
        fig = _make_domain_fig(domain_label, subdir, order)
        for ext in ("pdf", "png"):
            out = out_dir / f"{stem}.{ext}"
            fig.savefig(out)
            print(f"Saved {out}")
        plt.close(fig)
    return fig


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
