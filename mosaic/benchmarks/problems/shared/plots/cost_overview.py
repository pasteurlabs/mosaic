"""Per-domain ``_extra/cost_overview`` rendering.

A single-column figure showing per-N cost metrics (forward time, VJP time,
forward (V)RAM, VJP (V)RAM) for one problem subdirectory. Steady-state
problems (structural, thermal) drop the temporal cost row — only spatial
forward / VJP time + memory remain.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.io import load_json, results_dir
from mosaic.benchmarks.problems.shared.plots.style import (
    NS_ORDER,
    PAPER_RCPARAMS,
    STRUCTURAL_ORDER,
    TEXTWIDTH,
    THERMAL_ORDER,
    dedup_handles,
    make_handle,
    solver_props,
)

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


def plot_cost_overview(  # noqa: C901 — paper-fig assembler; refactor tracked separately
    out_dir: Path,
    *,
    subdir: str,
    domain_label: str,
    solver_order: list[str],
    steady_state: bool = False,
) -> None:
    """Render a single-column cost overview for one problem subdirectory.

    Writes ``cost_overview.pdf`` into *out_dir*. Rows:
      * forward time
      * VJP time
      * forward (V)RAM
      * VJP (V)RAM

    Steady-state problems drop the temporal-cost row (kept rows are
    spatial forward/VJP + their memory).
    """
    plt.rcParams.update(PAPER_RCPARAMS)

    cost_dir = results_dir() / subdir / "cost"
    fwd_path = cost_dir / "spatial_cost" / "result.json"
    vjp_path = cost_dir / "vjp_cost" / "result.json"
    fwd_data = load_json(fwd_path).get("by_N", {}) if fwd_path.exists() else {}
    vjp_data = load_json(vjp_path).get("by_N", {}) if vjp_path.exists() else {}

    all_solvers = sorted(set(fwd_data) | set(vjp_data))

    # Pre-detect which rows will have data so the figure can size to fit.
    def _any_pts(data, extractor):
        return any(extractor(data.get(s, {})) for s in all_solvers)

    want_fwd = _any_pts(fwd_data, _extract_by_n)
    want_vjp = _any_pts(vjp_data, _extract_by_n)
    want_fmem = _any_pts(fwd_data, _extract_mem_by_n)
    want_vmem = _any_pts(vjp_data, _extract_mem_by_n)

    row_specs = [
        (want_fwd, "fwd", "Forward time (s)"),
        (want_vjp, "vjp", r"$t_{\mathrm{vjp}}$ (s)"),
        (want_fmem, "fmem", "Fwd (V)RAM (MiB)"),
        (want_vmem, "vmem", "VJP (V)RAM (MiB)"),
    ]
    active_rows = [(k, ylab) for has, k, ylab in row_specs if has]
    if not active_rows:
        return  # nothing to plot

    n_rows = len(active_rows)
    fig_w = TEXTWIDTH * 0.55
    panel_h = fig_w * 0.55
    fig, axes_arr = plt.subplots(
        n_rows, 1, figsize=(fig_w, panel_h * n_rows + 0.7), sharex=True, squeeze=False
    )
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.18, top=0.95, hspace=0.22)
    ax_by_key = {k: axes_arr[i, 0] for i, (k, _) in enumerate(active_rows)}
    for k, ylab in active_rows:
        ax_by_key[k].set_ylabel(ylab)
    ax_fwd = ax_by_key.get("fwd")
    ax_vjp = ax_by_key.get("vjp")
    ax_fmem = ax_by_key.get("fmem")
    ax_vmem = ax_by_key.get("vmem")
    all_ns: set[int] = set()
    seen: set[str] = set()
    failure_types_seen: set[str] = set()

    for solver in all_solvers:
        alias = solver.lower().replace("-", "_").replace(".", "_")
        # Direct alias lookup: SOLVER_STYLES keys are aliases (jax_cfd, openfoam, ...)
        # and result.json keys are display names ("OpenFOAM", "jax-cfd"). Normalise.
        from mosaic.benchmarks.problems.shared.plots.style import SOLVER_STYLES

        if alias not in SOLVER_STYLES:
            # Try matching by stripping common label tweaks.
            for key, (label, *_rest) in SOLVER_STYLES.items():
                if label == solver:
                    alias = key
                    break
        _label, color, ls, mk = solver_props(alias)
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

        if alias in solver_order:
            seen.add(alias)

    visible_axes = [axes_arr[i, 0] for i in range(n_rows)]
    visible_axes[0].set_title(domain_label)
    visible_axes[-1].set_xlabel("Elements")

    tick_els = sorted(all_ns)
    if len(tick_els) > 4:
        idx = np.round(np.linspace(0, len(tick_els) - 1, 4)).astype(int)
        tick_els = [tick_els[i] for i in idx]
    for ax in visible_axes:
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

    solver_handles = dedup_handles([make_handle(s) for s in solver_order if s in seen])

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

    legend_kw = {"fontsize": 7.0, "framealpha": 0.7, "handlelength": 2.0}
    if solver_handles:
        fig.legend(
            handles=solver_handles,
            loc="upper center",
            bbox_to_anchor=(0.4, 0.13),
            ncol=max(1, math.ceil(len(solver_handles) / 3)),
            **legend_kw,
        )
    if failure_handles:
        fig.legend(
            handles=failure_handles,
            loc="upper center",
            bbox_to_anchor=(0.82, 0.13),
            ncol=1,
            **legend_kw,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cost_overview.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# Default solver-order lookup keyed by ``Problem.name``. Lets the helper
# below auto-pick the right ordering without each domain importing the
# style module to thread the constant through.
_DEFAULT_ORDER_BY_PROBLEM: dict[str, list[str]] = {
    "ns-grid": NS_ORDER,
    "ns-3d-grid": NS_ORDER,
    "structural-mesh": STRUCTURAL_ORDER,
    "thermal-mesh": THERMAL_ORDER,
}


def plot_cost_overview_for(
    cfg,
    *,
    steady_state: bool,
    solver_order: list[str] | None = None,
) -> None:
    """Render the cost-overview figure for ``cfg`` into its ``_extra`` dir.

    Reads ``cfg.tesseract_dir.name`` as the results subdirectory and
    ``cfg.category_label`` (with ``cfg.name`` fallback) as the panel
    label. ``solver_order`` defaults to the per-domain entry in
    ``_DEFAULT_ORDER_BY_PROBLEM`` keyed by ``cfg.name``.
    """
    out_dir = results_dir() / cfg.name / "_extra"
    out_dir.mkdir(parents=True, exist_ok=True)
    if solver_order is None:
        solver_order = _DEFAULT_ORDER_BY_PROBLEM.get(cfg.name, cfg.solver_names)
    plot_cost_overview(
        out_dir,
        subdir=cfg.tesseract_dir.name,
        domain_label=cfg.category_label or cfg.name,
        solver_order=solver_order,
        steady_state=steady_state,
    )
