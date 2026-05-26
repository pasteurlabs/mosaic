# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Figure: Cross-solver agreement and baseline convergence for NS domains.

2 rows (F2 top, F3 bottom) × 3 columns:
  col 0 — agreement vs ν on periodic/TGV domain (analytic reference)
  col 1 — F2 only: cylinder flow vs ν (consensus reference); F3: empty
  col 2 — baseline convergence vs N (analytic reference, 1 step)

Output: appendix_agreement.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.utils import results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    NS_ORDER,
    RCPARAMS,
    dedup_handles,
    make_handle,
    solver_props,
)


def _configs():
    """(row, col, path, x_label, log_x, title, y_label)."""
    R = results_dir()
    return [
        (
            0,
            0,
            R / "ns-grid" / "forward" / "tgv_nu_sweep" / "result.json",
            r"$\nu$",
            True,
            "F2 — TGV agreement vs $\\nu$",
            "TGV analytic error",
        ),
        (
            0,
            1,
            R / "ns-grid" / "forward" / "cylinder" / "result.json",
            r"$\nu$",
            True,
            "F2 — cylinder flow vs $\\nu$",
            "Consensus error",
        ),
        (
            0,
            2,
            R / "ns-grid" / "forward" / "baseline" / "result.json",
            "$N$",
            True,
            "F2 — convergence vs $N$",
            "TGV analytic error",
        ),
        (
            1,
            0,
            R / "ns-3d-grid" / "forward" / "agreement" / "result.json",
            r"$\nu$",
            True,
            "F3 — TGV agreement vs $\\nu$",
            "TGV analytic error",
        ),
        (
            1,
            2,
            R / "ns-3d-grid" / "forward" / "baseline" / "result.json",
            "$N$",
            True,
            "F3 — convergence vs $N$",
            "TGV analytic error",
        ),
    ]


def generate(out_dir: Path) -> None:
    """Generate cross-solver agreement and baseline convergence figure."""
    with plt.rc_context(RCPARAMS):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(TEXTWIDTH, TEXTWIDTH * 0.72),
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.50, wspace=0.38, bottom=0.20)

        # hide unused panel
        axes[1][1].set_visible(False)

        seen: set[str] = set()

        for row, col, path, x_label, log_x, title, y_label in _configs():
            ax = axes[row][col]

            if not path.exists():
                ax.set_visible(False)
                continue

            data = json.loads(path.read_text())
            by_param = data["by_param"]
            params = sorted(by_param.keys(), key=float)

            for solver in NS_ORDER:
                _, color, ls, mk = solver_props(solver)
                xs, ys = [], []
                for p in params:
                    entry = by_param[p].get(solver)
                    if isinstance(entry, dict):
                        err = entry.get("error")
                        if (
                            err is not None
                            and isinstance(err, float)
                            and np.isfinite(err)
                            and err > 0
                        ):
                            xs.append(float(p))
                            ys.append(err)
                if xs:
                    ax.semilogy(
                        xs,
                        ys,
                        color=color,
                        linestyle=ls,
                        marker=mk,
                        markersize=3.5,
                        markeredgewidth=0,
                        linewidth=1.5,
                    )
                    seen.add(solver)

            ax.set_title(title, fontsize=8)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            if log_x:
                ax.set_xscale("log")
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.tick_params(axis="x", labelsize=7, rotation=30)

        handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in seen])
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 5),
            fontsize=7.5,
            framealpha=0.9,
            edgecolor="0.8",
            handlelength=2.0,
        )

        out = out_dir / "appendix_agreement.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
