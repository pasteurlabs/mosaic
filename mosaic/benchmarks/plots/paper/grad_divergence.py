"""Generate Figure: gradient divergence over the 3D NS IC optimization.

For each optimizer (Adam, Adam+proj, L-BFGS, L-BFGS+proj) and each NS solver,
plot the per-iteration ``max|∇·g|`` (divergence of the gradient passed to the
optimizer) and ``max|∇·u|`` (divergence of the IC iterate).

Methods whose result.json does not contain the diagnostic series are silently
skipped, so the same script works whether or not the L-BFGS variants have been
rerun with diagnostics enabled.

Output: grad_divergence.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
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


def _methods() -> dict[str, tuple]:
    base = results_dir() / "ns-3d-grid" / "optimization"
    return {
        "adam": ("Adam", "-.", base / "recovery_constant_ic"),
        "adam_proj": ("Adam+proj", ":", base / "recovery_constant_ic_proj"),
        "bfgs": ("L-BFGS", "--", base / "recovery_constant_ic_bfgs"),
        "bfgs_proj": ("L-BFGS+proj", "-", base / "recovery_constant_ic_bfgs_proj"),
    }


# Adam: 1 grad eval / outer iter; optax L-BFGS averages ~3 (zoom line search).
_GRAD_EVALS_PER_ITER: dict[str, int] = {
    "adam": 1,
    "adam_proj": 1,
    "bfgs": 3,
    "bfgs_proj": 3,
}
_GRAD_EVAL_LABEL = "Gradient evaluations"

_STEP_KEY = "100"
_FLOOR = 1e-12


def _load_results() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, (_, _, path) in _methods().items():
        rp = path / "result.json"
        if not rp.exists():
            continue
        out[key] = json.loads(rp.read_text())
    return out


def _series(entry: dict, field: str) -> list[float] | None:
    vals = entry.get(field) if entry else None
    if not vals:
        return None
    return [max(float(v), _FLOOR) for v in vals]


def _plot_panel(
    ax,
    results: dict[str, dict],
    field: str,
    seen_solvers: set[str],
    seen_methods: set[str],
) -> None:
    methods = _methods()
    for key, result in results.items():
        m_label, m_ls, _ = methods[key]
        by_sweep = result.get("by_sweep", {})
        f = _GRAD_EVALS_PER_ITER.get(key, 1)
        for solver in NS_ORDER:
            entry = by_sweep.get(solver, {}).get(_STEP_KEY)
            vals = _series(entry, field)
            if vals is None:
                continue
            _, color, _, _ = solver_props(solver)
            xs = np.array([(i + 1) * f for i in range(len(vals))])
            ax.loglog(xs, vals, color=color, linestyle=m_ls, linewidth=1.3, alpha=0.9)
            seen_solvers.add(solver)
            seen_methods.add(key)
    ax.set_xlabel(_GRAD_EVAL_LABEL)


def generate(out_dir: Path) -> None:
    results = _load_results()
    if not results:
        print("[grad_divergence] no recovery results found — skipping")
        return

    with plt.rc_context(RCPARAMS):
        fig, (ax_g, ax_u) = plt.subplots(1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.42))
        fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.32, wspace=0.30)

        seen_solvers: set[str] = set()
        seen_methods: set[str] = set()
        _plot_panel(ax_g, results, "grad_divs", seen_solvers, seen_methods)
        _plot_panel(ax_u, results, "ic_divs", seen_solvers, seen_methods)

        ax_g.set_title(r"Gradient divergence  $\max\,|\nabla\!\cdot g|$")
        ax_g.set_ylabel("max divergence")
        ax_u.set_title(r"IC divergence  $\max\,|\nabla\!\cdot u|$")

        methods = _methods()
        method_handles = [
            mlines.Line2D(
                [],
                [],
                color="0.3",
                linestyle=methods[k][1],
                linewidth=1.3,
                label=methods[k][0],
            )
            for k in methods
            if k in seen_methods
        ]
        solver_handles = dedup_handles(
            [make_handle(s) for s in NS_ORDER if s in seen_solvers]
        )

        fig.legend(
            handles=method_handles + solver_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=5,
            fontsize=6.5,
            framealpha=0.8,
            edgecolor="0.8",
            handlelength=2.0,
        )

        out = out_dir / "grad_divergence.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
