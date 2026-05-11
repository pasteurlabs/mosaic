"""Generate Figure: Forward and VJP cost overview across all four benchmark domains.

4-row × 4-column grid:
  row 0 — forward time (log-log)
  row 1 — VJP time (log-log)
  row 2 — forward peak (V)RAM (log-log)
  row 3 — VJP peak (V)RAM (log-log)

GPU solvers plot VRAM; CPU-only solvers plot RAM.
Rows 2-3 are skipped per column when no memory data is available.
Failure markers (OOM ▼, error ◆, NaN ×) are shown at the failing N with a
short horizontal connector from the last successful point.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.utils import results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    FEM_ORDER,
    NS_ORDER,
    RCPARAMS,
    dedup_handles,
    make_handle,
    solver_props,
)

DOMAINS = [
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


def generate(out_dir: Path) -> None:
    plt.rcParams.update(RCPARAMS)

    fig_w = TEXTWIDTH
    fig, axes = plt.subplots(4, 4, figsize=(fig_w, fig_w * 1.55), sharex="col")
    fig.subplots_adjust(
        left=0.07, right=0.98, bottom=0.18, top=0.97, wspace=0.30, hspace=0.18
    )

    ns_seen: set[str] = set()
    fem_seen: set[str] = set()
    failure_types_seen: set[str] = set()

    for col, (domain_label, subdir, _res_key) in enumerate(DOMAINS):
        cost_dir = results_dir() / subdir / "cost"

        fwd_path = cost_dir / "spatial_cost" / "result.json"
        vjp_path = cost_dir / "vjp_cost" / "result.json"
        fwd_data = (
            json.loads(fwd_path.read_text()).get("by_N", {})
            if fwd_path.exists()
            else {}
        )
        vjp_data = (
            json.loads(vjp_path.read_text()).get("by_N", {})
            if vjp_path.exists()
            else {}
        )

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


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
