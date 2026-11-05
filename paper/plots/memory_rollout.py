"""Generate Figure: peak VJP memory against rollout length on ns-3d-grid.

Peak VRAM of one VJP against rollout steps at N=48 (``cost/vjp_cost/by_steps``),
one line per solver on log-log axes. Line style encodes the differentiation
strategy from the solver metadata: solid for autodiff through the stored
trajectory, dashed for hybrid adjoints, dash-dotted for autodiff with block
(sqrt-steps) checkpointing. A solver whose next horizon ran out of memory gets a
dotted connector to an x on the device limit. Grey guides show O(T) and O(sqrt T)
growth.

Output: memory_rollout.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from . import TEXTWIDTH, by_solver, is_ok, load_result, solver_metadata
from .style import NS_ORDER, RCPARAMS, solver_props

PROBLEM = "ns-3d-grid"
DEVICE_GB = 80  # A100 80 GB = 81920 MiB; GB here means MiB / 1024
_STRATEGY_LS = {"autodiff": "-", "hybrid": "--", "checkpointed": "-."}
_STRATEGY_SUFFIX = {
    "autodiff": "",
    "hybrid": " (hybrid)",
    "checkpointed": r" ($\sqrt{T}$ checkpoints)",
}
_JITTER = 1.1  # x factor separating failure markers at the same horizon


def _strategy(meta: dict) -> str:
    ids = {iv["id"] for iv in meta.get("interventions", [])}
    return "checkpointed" if "sqrt-checkpointing" in ids else meta["ad_strategy"]


def generate(out_dir: Path) -> None:
    data = by_solver(load_result(PROBLEM, "cost/vjp_cost/by_steps"))
    meta = solver_metadata(PROBLEM)
    solvers = [s for s in NS_ORDER if s in data]

    fails = {}
    for s in solvers:
        failed = [int(k) for k, m in data[s].items() if m.get("status") == "failed"]
        if failed:
            fails[s] = min(failed)
    fail_x = {}
    for step in set(fails.values()):
        group = [s for s in solvers if fails.get(s) == step]
        for i, s in enumerate(group):
            fail_x[s] = step * _JITTER ** (i - (len(group) - 1) / 2)

    with plt.rc_context(RCPARAMS):
        fig, ax = plt.subplots(figsize=(TEXTWIDTH / 2, TEXTWIDTH * 0.56))
        fig.subplots_adjust(left=0.17, right=0.97, top=0.98, bottom=0.33)

        for s in solvers:
            label, color, _, mk = solver_props(s)
            strategy = _strategy(meta[s])
            pts = sorted(
                (int(k), m["vram_peak_mib"] / 1024)
                for k, m in data[s].items()
                if is_ok(m)
            )
            steps, gib = zip(*pts, strict=True)
            ax.plot(
                steps,
                gib,
                color=color,
                linestyle=_STRATEGY_LS[strategy],
                marker=mk,
                markersize=3.5,
                markeredgewidth=0,
                linewidth=1.3,
                zorder=4,
                label=label + _STRATEGY_SUFFIX[strategy],
            )
            if s in fail_x:
                ax.plot(
                    [steps[-1], fail_x[s]],
                    [gib[-1], DEVICE_GB],
                    color=color,
                    linestyle=":",
                    linewidth=0.9,
                )
                ax.plot(
                    fail_x[s],
                    DEVICE_GB,
                    color=color,
                    marker="x",
                    markersize=6,
                    markeredgewidth=1.4,
                    clip_on=False,
                    zorder=3,
                )

        ax.axhline(DEVICE_GB, color="0.4", linestyle="--", linewidth=0.8, zorder=1)
        ax.text(10, DEVICE_GB * 1.14, "80 GB device", fontsize=6.5, color="0.4")

        # O(T) and O(sqrt T) guides fanning out from a common origin.
        t = np.array([11.0, 35.0])
        for power, text in ((1.0, r"$\propto T$"), (0.5, r"$\propto \sqrt{T}$")):
            g = 20.0 * (t / t[0]) ** power
            ax.plot(t, g, color="0.55", linewidth=0.8, zorder=1)
            ax.text(t[-1] * 1.08, g[-1], text, fontsize=6.5, color="0.4", va="center")

        ax.set_xscale("log")
        ax.set_yscale("log")
        sweep = sorted({int(k) for v in data.values() for k in v})
        ax.set_xticks(sweep)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_ylim(0.4, 120)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set_xlabel("Rollout steps $T$")
        ax.set_ylabel("Peak VRAM of one VJP (GB)")

        handles, _ = ax.get_legend_handles_labels()
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="0.3",
                marker="x",
                linestyle="none",
                markersize=6,
                markeredgewidth=1.4,
                label="out of memory",
            )
        )
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
            fontsize=6.5,
            handlelength=2.4,
            columnspacing=1.0,
            frameon=False,
        )

        out = out_dir / "memory_rollout.pdf"
        fig.savefig(out)
        print(f"Saved {out}")
        plt.close(fig)
