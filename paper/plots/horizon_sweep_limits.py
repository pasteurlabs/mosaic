"""Generate Figure: VJP rollout-length limit sweep on ns-3d-grid.

Three panels against rollout steps (log-log): peak VRAM, wall time of the
VJP call, and gradient norm. Each solver's line ends at its last successful
horizon. Memory failures (out of memory, container died) are marked at the
device limit in the VRAM panel; non-finite gradients are marked at the top
edge of the gradient-norm panel.

Each horizon is a single VJP call; only the first horizon follows a warm-up
call, so later wall times include any recompilation for the new step count.

Output: horizon_sweep_limits.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory

from . import TEXTWIDTH, by_solver, is_ok, load_result
from .style import NS_ORDER, RCPARAMS, make_handle, solver_props

DEVICE_GB = 80  # A100 80 GB = 81920 MiB; GB here means MiB / 1024
# failure_type -> (marker, legend label)
_FAILURES = {
    "OOM": ("x", "out of memory"),
    "container_died": ("*", "container died"),
    "nan": ("X", "NaN gradient"),
}
_JITTER = 1.12  # x factor separating failure markers at the same horizon


def _jittered(fails: dict[str, int]) -> dict[str, float]:
    """Spread failure markers that share a horizon symmetrically in log x."""
    out = {}
    for step in set(fails.values()):
        group = [s for s in fails if fails[s] == step]
        for i, s in enumerate(group):
            out[s] = step * _JITTER ** (i - (len(group) - 1) / 2)
    return out


def _failure_marker(failure_type: str) -> dict:
    marker = _FAILURES[failure_type][0]
    if marker == "x":
        return {"marker": marker, "markersize": 6, "markeredgewidth": 1.4}
    return {"marker": marker, "markersize": 6.5, "markeredgewidth": 0}


def generate(out_dir: Path) -> None:
    data = by_solver(load_result("ns-3d-grid", "gradient/horizon_sweep_limits"))
    solvers = [s for s in NS_ORDER if s in data]

    ok: dict[str, list[tuple[int, dict]]] = {}
    fail: dict[str, tuple[int, str]] = {}
    for s in solvers:
        pts = sorted((int(k), m) for k, m in data[s].items())
        ok[s] = [(t, m) for t, m in pts if is_ok(m)]
        failed = [(t, m["failure_type"]) for t, m in pts if m.get("status") == "failed"]
        if failed:
            fail[s] = failed[0]

    mem_fails = {s: t for s, (t, ft) in fail.items() if ft != "nan"}
    nan_fails = {s: t for s, (t, ft) in fail.items() if ft == "nan"}
    mem_x, nan_x = _jittered(mem_fails), _jittered(nan_fails)

    with plt.rc_context(RCPARAMS):
        fig, (ax_vr, ax_wt, ax_gn) = plt.subplots(
            1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.36)
        )
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.38, top=0.92, wspace=0.42)

        for s in solvers:
            _, color, ls, mk = solver_props(s)
            kw = {
                "color": color,
                "linestyle": ls,
                "marker": mk,
                "markersize": 3,
                "markeredgewidth": 0,
                "linewidth": 1.3,
            }
            steps = [t for t, _ in ok[s]]
            ax_vr.plot(steps, [m["vram_peak_mib"] / 1024 for _, m in ok[s]], **kw)
            ax_wt.plot(steps, [m["wall_time_s"] for _, m in ok[s]], **kw)
            ax_gn.plot(steps, [m["grad_norm"] for _, m in ok[s]], **kw)

            fkw = {"color": color, "linestyle": "none", "zorder": 5, "clip_on": False}
            if s in mem_x:
                ax_vr.plot(mem_x[s], DEVICE_GB, **_failure_marker(fail[s][1]), **fkw)
            if s in nan_x:
                ax_gn.plot(
                    nan_x[s],
                    1.0,
                    **_failure_marker("nan"),
                    **fkw,
                    transform=blended_transform_factory(
                        ax_gn.transData, ax_gn.transAxes
                    ),
                )

        ax_vr.axhline(DEVICE_GB, color="0.4", linestyle="--", linewidth=0.8, zorder=1)
        ax_vr.text(
            40, DEVICE_GB * 0.85, "80 GB device", va="top", fontsize=6.5, color="0.4"
        )

        for ax, title, ylabel in (
            (ax_vr, "Peak VRAM", "GB"),
            (ax_wt, "VJP wall time", "Seconds"),
            (ax_gn, "Gradient norm", r"$\|\nabla_{u_0}\mathcal{L}\|$"),
        ):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Rollout steps")
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xlim(30, 1.1e5)

        failure_types = [ft for ft in _FAILURES if ft in {f for _, f in fail.values()}]
        handles = [make_handle(s) for s in solvers] + [
            mlines.Line2D(
                [],
                [],
                color="0.3",
                linestyle="none",
                label=_FAILURES[ft][1],
                **_failure_marker(ft),
            )
            for ft in failure_types
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=-(-len(handles) // 2),
            fontsize=6.5,
            handlelength=2.2,
            columnspacing=1.2,
            frameon=False,
        )

        out = out_dir / "horizon_sweep_limits.pdf"
        fig.savefig(out)
        print(f"Saved {out}")
        plt.close(fig)
