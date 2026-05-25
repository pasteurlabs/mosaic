"""Generate Figure: Horizon sweep for ns-grid (2D NS).

1×3 panels, log x-axis, same visual language as horizon_sweep_limits:
  - gradient norm vs rollout steps  (log-log)
  - best FD relative error vs steps (log-log)
  - best FD cosine similarity vs steps (semilogx, linear y)

Failure modes (NaN grad_norm or non-finite FD error) are marked with a ×.
"""

from __future__ import annotations

import json
from collections import defaultdict
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

_FAILURE_MARKER = "X"
_FAILURE_LABEL = "NaN gradient"
_JITTER_LOG = 0.04


def generate(out_dir: Path) -> None:
    path = results_dir() / "ns-grid" / "gradient" / "horizon_sweep" / "result.json"
    if not path.exists():
        print(f"[horizon_sweep] {path} not found — skipping")
        return

    data = json.loads(path.read_text())
    by_solver = data["by_solver"]

    with plt.rc_context(RCPARAMS):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(TEXTWIDTH, TEXTWIDTH * 0.42),
        )
        fig.subplots_adjust(bottom=0.32, wspace=0.58, left=0.09, right=0.98, top=0.91)

        ax_gn, ax_err, ax_cos = axes

        _EXCLUDED = {"fenics_ns", "su2", "openfoam"}
        ordered = [s for s in NS_ORDER if s in by_solver and s not in _EXCLUDED]

        # collect failure steps for jitter
        fail_at_step: dict[int, list[str]] = defaultdict(list)
        for solver in ordered:
            sv = by_solver[solver]
            for k, v in sv.items():
                gn = v.get("grad_norm", 1.0)
                if not np.isfinite(gn) or gn <= 0:
                    fail_at_step[int(k)].append(solver)

        jitter_x: dict[tuple[str, int], float] = {}
        for step, solvers_here in fail_at_step.items():
            n = len(solvers_here)
            for i, sv in enumerate(solvers_here):
                if n == 1:
                    jitter_x[(sv, step)] = float(step)
                else:
                    log_off = (2 * i / (n - 1) - 1) * _JITTER_LOG
                    jitter_x[(sv, step)] = step * 10**log_off

        seen: set[str] = set()
        failure_seen = False

        for solver in ordered:
            sv = by_solver[solver]
            label, color, ls, mk = solver_props(solver)

            step_keys = sorted(sv.keys(), key=int)
            ok_steps, ok_gn, ok_err, ok_cos = [], [], [], []
            fail_steps = []

            for k in step_keys:
                v = sv[k]
                gn = v.get("grad_norm", float("nan"))
                eps_sweep = v.get("eps_sweep", {})
                if eps_sweep:
                    best_err = min(
                        float(e["rel_error_mean"]) for e in eps_sweep.values()
                    )
                    best_cos = max(float(e["cosine_mean"]) for e in eps_sweep.values())
                else:
                    best_err = float("nan")
                    best_cos = float("nan")

                if (
                    np.isfinite(gn)
                    and gn > 0
                    and np.isfinite(best_err)
                    and best_err > 0
                ):
                    ok_steps.append(int(k))
                    ok_gn.append(gn)
                    ok_err.append(best_err)
                    ok_cos.append(best_cos)
                else:
                    fail_steps.append(int(k))

            kw = dict(
                color=color,
                linestyle=ls,
                marker="o",
                markersize=4,
                markeredgewidth=0,
                linewidth=1.6,
                zorder=3,
            )

            ok_cos_defect = [max(1.0 - c, 1e-12) for c in ok_cos]

            if ok_steps:
                ax_gn.loglog(ok_steps, ok_gn, **kw)
                ax_err.loglog(ok_steps, ok_err, **kw)
                ax_cos.loglog(ok_steps, ok_cos_defect, **kw)
                seen.add(solver)

            for fs in fail_steps:
                jx = jitter_x.get((solver, fs), float(fs))
                mk_kw = dict(
                    marker=_FAILURE_MARKER,
                    color=color,
                    markersize=9,
                    markeredgewidth=1.2,
                    markeredgecolor="white",
                    linestyle="none",
                    zorder=6,
                )
                if ok_gn:
                    ax_gn.loglog([jx], [ok_gn[-1]], **mk_kw)
                    ax_err.loglog([jx], [ok_err[-1]], **mk_kw)
                    ax_cos.loglog([jx], [ok_cos_defect[-1]], **mk_kw)
                failure_seen = True

        ax_gn.set_title("Gradient norm")
        ax_gn.set_xlabel("Rollout steps $T$")
        ax_gn.set_ylabel(r"$\|\nabla\mathcal{L}\|$")

        ax_err.set_title("FD relative error (best $\\varepsilon$)")
        ax_err.set_xlabel("Rollout steps $T$")
        ax_err.set_ylabel("Relative FD error")

        ax_cos.set_title("Cosine similarity (best $\\varepsilon$)")
        ax_cos.set_xlabel("Rollout steps $T$")
        ax_cos.set_ylabel("$1 -$ cosine")

        # ── Legend ───────────────────────────────────────────────────────────
        handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in seen])
        if failure_seen:
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    marker=_FAILURE_MARKER,
                    color="0.4",
                    linestyle="none",
                    markersize=7,
                    markeredgewidth=1.0,
                    markeredgecolor="white",
                    label=_FAILURE_LABEL,
                )
            )

        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 6),
            fontsize=7.5,
            framealpha=0.7,
            edgecolor="0.8",
            handlelength=2.0,
        )

        out = out_dir / "horizon_sweep.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
