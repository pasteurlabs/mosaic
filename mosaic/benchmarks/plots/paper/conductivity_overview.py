"""Generate Figure: Thermal conductivity recovery overview — Adam vs L-BFGS.

Layout:
  Row 0: identification error history — loglog, all solvers × both methods
  Row 1: Adam recovered conductivity profiles, all solvers + truth
  Row 2: L-BFGS recovered conductivity profiles, all solvers + truth

Output: conductivity_recovery_overview.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.utils import results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    RCPARAMS,
    THERMAL_ORDER,
    dedup_handles,
    make_handle,
    solver_props,
)


def _methods():
    base = results_dir() / "thermal-mesh" / "optimization"
    return {
        "adam": ("Adam", "-", base / "conductivity_recovery"),
        "bfgs": ("L-BFGS", "--", base / "conductivity_recovery_bfgs"),
    }


# See note in recovery_overview.py: optax L-BFGS averages ~3 grad evaluations
# per outer iteration (zoom line search), Adam exactly 1.
_GRAD_EVALS_PER_ITER: dict[str, int] = {"adam": 1, "bfgs": 3}
_GRAD_EVAL_LABEL = "Gradient evaluations"


def generate(out_dir: Path) -> None:
    loaded: dict[str, tuple] = {}
    for key, (*_, path) in _methods().items():
        rp = path / "result.json"
        fp = path / "rho_fields.npz"
        if not rp.exists():
            print(f"[conductivity_overview] {rp} not found — skipping {key}")
            continue
        npz = np.load(fp, allow_pickle=True) if fp.exists() else None
        loaded[key] = (json.loads(rp.read_text()), npz)

    if not loaded:
        return

    with plt.rc_context(RCPARAMS):
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 1.10))
        outer = gridspec.GridSpec(
            3,
            1,
            figure=fig,
            height_ratios=[1.0, 0.65, 0.65],
            left=0.10,
            right=0.98,
            top=0.96,
            bottom=0.10,
            hspace=0.52,
        )
        ax_conv = fig.add_subplot(outer[0])
        ax_adam = fig.add_subplot(outer[1])
        ax_bfgs = fig.add_subplot(outer[2])

        seen_solvers: set[str] = set()

        # ── Convergence — all solvers × both methods ──────────────────────
        for key, (m_label, m_ls, *_) in _methods().items():
            if key not in loaded:
                continue
            result, _ = loaded[key]
            for solver in THERMAL_ORDER:
                sdata = result["by_solver"].get(solver)
                if sdata is None:
                    continue
                errors = sdata.get("errors", [])
                if not errors:
                    continue
                _, s_color, _, _ = solver_props(solver)
                f = _GRAD_EVALS_PER_ITER.get(key, 1)
                xs = [(i + 1) * f for i in range(len(errors))]
                ax_conv.loglog(
                    xs,
                    errors,
                    color=s_color,
                    linestyle=m_ls,
                    linewidth=1.3,
                    alpha=0.9,
                )
                seen_solvers.add(solver)

        ax_conv.set_title("Thermal conductivity recovery")
        ax_conv.set_xlabel(_GRAD_EVAL_LABEL)
        ax_conv.set_ylabel("Identification error")

        # ── Profile panels ────────────────────────────────────────────────
        for ax, key, title in [
            (ax_adam, "adam", "Adam — recovered profiles"),
            (ax_bfgs, "bfgs", "L-BFGS — recovered profiles"),
        ]:
            if key not in loaded:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_title(title)
                continue

            _, npz = loaded[key]
            if npz is None:
                ax.set_title(title)
                continue

            xs = np.arange(npz["rho_truth"].shape[0])
            ax.plot(
                xs,
                npz["rho_truth"],
                color="0.2",
                linestyle="--",
                linewidth=1.4,
                label="Truth",
                zorder=3,
            )

            for solver in THERMAL_ORDER:
                rho_key = f"rho_final_{solver}"
                if rho_key not in npz.files:
                    continue
                _, s_color, _, _ = solver_props(solver)
                ax.plot(xs, npz[rho_key], color=s_color, linewidth=1.1, alpha=0.85)
                seen_solvers.add(solver)

            ax.set_title(title)
            ax.set_xlabel("Node index")
            ax.set_ylabel("Conductivity")

        # ── Legend ────────────────────────────────────────────────────────
        truth_handle = mlines.Line2D(
            [], [], color="0.2", linestyle="--", linewidth=1.4, label="Truth"
        )
        solver_handles = dedup_handles(
            [make_handle(s) for s in THERMAL_ORDER if s in seen_solvers]
        )
        fig.legend(
            handles=[truth_handle] + solver_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=5,
            fontsize=6.5,
            framealpha=0.8,
            edgecolor="0.8",
            handlelength=1.8,
        )

        out = out_dir / "conductivity_recovery_overview.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
