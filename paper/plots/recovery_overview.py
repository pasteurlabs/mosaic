"""Generate Figures: F3 IC recovery with Adam and L-BFGS (SciPy's L-BFGS-B), with
and without the divergence-free gradient projection.

recovery_overview.pdf: IC error, IC divergence and optimization loss against
gradient evaluations for Adam, L-BFGS and L-BFGS+proj on every solver
(top), and u_x at the middle z-slice for L-BFGS+proj / PhiFlow: recovered
and true IC, recovered and true final state (bottom).
recovery_main.pdf: the top row for PICT, INS.jl and XLB.

Curves follow the first of the three IC seeds. The x-axis counts the
value-and-gradient evaluations each optimizer made, including L-BFGS's
line-search trials.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, ticker

from . import TEXTWIDTH, by_solver, is_ok, load_result, results_dir
from .style import NS_ORDER, RCPARAMS, dedup_handles, make_handle, solver_props

_PROBLEM = "ns-3d-grid"
# optimizer key -> (label, linestyle, experiment)
_METHODS: dict[str, tuple[str, object, str]] = {
    "adam": ("Adam", "-.", "optimization/recovery_constant_ic"),
    "bfgs": ("L-BFGS", "--", "optimization/recovery_constant_ic_bfgs"),
    "bfgs_proj": ("L-BFGS+proj", "-", "optimization/recovery_constant_ic_bfgs_proj"),
}
_OVERVIEW_METHODS = ("adam", "bfgs", "bfgs_proj")
_MAIN_SOLVERS = ["PICT", "INS.jl", "XLB"]
_FIELD_METHOD, _FIELD_SOLVER = "bfgs_proj", "PhiFlow"
_GRAD_EVAL_LABEL = "Gradient evaluations"


class _Run:
    """One optimizer's recovery result: per-solver metrics and stored fields."""

    def __init__(self, key: str) -> None:
        _, _, experiment = _METHODS[key]
        result = load_result(_PROBLEM, experiment)
        (step,) = result["sweep"]["values"]
        self.key = key
        self.experiment = experiment
        self.snap = int(result["params"]["optim"]["snap_interval"])
        self.metrics = {
            s: m[str(step)] for s, m in by_solver(result).items() if is_ok(m[str(step)])
        }
        self.fields = np.load(
            results_dir() / _PROBLEM / experiment / "recovery_fields.npz"
        )
        self.field_solvers = [str(s) for s in self.fields["solver_names"]]

    def evals(self, solver: str) -> list[int]:
        """Evaluations up to and including the one that computed each loss."""
        history = self.metrics[solver].get("n_evals_history")
        if history is None:
            raise ValueError(
                f"{self.experiment}/{solver} has no n_evals_history; it was run"
                " before the harness recorded evaluation counts"
            )
        return history

    def snapshot_evals(self, solver: str, n: int) -> list[int]:
        """Evaluations behind each of the first *n* IC snapshots.

        Snapshot j is the iterate after (j + 1) * snap iterations. Adam's last
        iterate is never evaluated, so its count is one past the last loss.
        """
        history = self.evals(solver)
        iters = [(j + 1) * self.snap for j in range(n)]
        return [history[i] if i < len(history) else history[-1] + 1 for i in iters]

    def ic_history(self, solver: str) -> np.ndarray:
        return self.fields[f"ic_history_{self.field_solvers.index(solver)}"]


def _div_rms(field: np.ndarray) -> float:
    """RMS of the central-difference divergence of a (N, N, N, 3) field, in grid units."""
    div = sum(np.gradient(field[..., c], axis=c) for c in range(3))
    return float(np.sqrt(np.mean(div**2)))


def _plot_rows(
    axes, runs: dict[str, _Run], keys, solvers, *, linewidth: float
) -> set[str]:
    """Draw IC error, IC divergence and loss for *keys* x *solvers*; return the solvers drawn."""
    ax_err, ax_div, ax_loss = axes
    seen: set[str] = set()
    for key in keys:
        run = runs[key]
        ls = _METHODS[key][1]
        for solver in solvers:
            m = run.metrics.get(solver)
            if m is None:
                continue
            _, color, _, _ = solver_props(solver)
            kw = {"color": color, "linestyle": ls, "linewidth": linewidth, "alpha": 0.9}
            hist = m["ic_error_history"]
            ax_err.loglog(run.snapshot_evals(solver, len(hist)), hist, **kw)
            ax_loss.loglog(run.evals(solver), m["errors"], **kw)
            snaps = run.ic_history(solver)
            divs = [max(_div_rms(u), 1e-9) for u in snaps]
            ax_div.loglog(run.snapshot_evals(solver, len(snaps)), divs, **kw)
            seen.add(solver)

    ic_true_div = _div_rms(runs[keys[0]].fields["ic_true"])
    ax_div.axhline(ic_true_div, color="0.55", linestyle="--", linewidth=1.0, zorder=0)
    # The divergence spans about a decade: label 1-2-5 steps, not just powers of ten.
    ax_div.yaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax_div.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_div.yaxis.set_minor_formatter(ticker.NullFormatter())
    for ax, title in zip(
        axes, ("IC recovery error", "IC divergence", "Optimization loss"), strict=True
    ):
        ax.set_title(title)
        ax.set_xlabel(_GRAD_EVAL_LABEL)
    return seen


def _legend(fig, keys, solvers, seen, *, anchor_y: float, ncol: int, fontsize: float):
    method_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.3",
            linestyle=_METHODS[k][1],
            linewidth=1.3,
            label=_METHODS[k][0],
        )
        for k in keys
    ]
    solver_handles = dedup_handles([make_handle(s) for s in solvers if s in seen])
    fig.legend(
        handles=method_handles + solver_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=ncol,
        fontsize=fontsize,
        framealpha=0.8,
        edgecolor="0.8",
        handlelength=1.8,
    )


def _field_panels(fig, axes, run: _Run) -> None:
    """u_x at the middle z-slice: recovered and true IC, recovered and true final state."""
    i = run.field_solvers.index(_FIELD_SOLVER)
    f = run.fields
    z = f["ic_true"].shape[2] // 2
    panels = [
        (f[f"ic_rec_{i}"], "IC recovered"),
        (f["ic_true"], "IC true"),
        (f[f"final_rec_{i}"], "Final state"),
        (f[f"final_gt_{i}"], "Final true"),
    ]
    slices = [(u[:, :, z, 0], title) for u, title in panels]
    vlim = float(np.percentile(np.abs([s for s, _ in slices]), 99))
    for ax, (data, title) in zip(axes, slices, strict=True):
        im = ax.imshow(
            data.T,
            origin="lower",
            aspect="equal",
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=7.0)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(
        im, ax=axes, fraction=0.015, pad=0.02, ticks=np.linspace(-vlim, vlim, 5)
    )
    cb.ax.tick_params(labelsize=5.5)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))


def _overview(runs: dict[str, _Run], out_path: Path) -> None:
    fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.80))
    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 0.82],
        left=0.06,
        right=0.98,
        top=0.94,
        bottom=0.13,
        hspace=0.38,
    )
    top = outer[0].subgridspec(1, 3, wspace=0.30)
    bottom = outer[1].subgridspec(1, 4, wspace=0.22)
    axes = [fig.add_subplot(top[0, i]) for i in range(3)]
    seen = _plot_rows(axes, runs, _OVERVIEW_METHODS, NS_ORDER, linewidth=1.3)
    _legend(fig, _OVERVIEW_METHODS, NS_ORDER, seen, anchor_y=0.0, ncol=5, fontsize=6.5)
    _field_panels(
        fig, [fig.add_subplot(bottom[0, i]) for i in range(4)], runs[_FIELD_METHOD]
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def _main(runs: dict[str, _Run], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.36))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.36, wspace=0.30)
    seen = _plot_rows(axes, runs, _OVERVIEW_METHODS, _MAIN_SOLVERS, linewidth=1.4)
    _legend(
        fig, _OVERVIEW_METHODS, _MAIN_SOLVERS, seen, anchor_y=0.0, ncol=6, fontsize=7.0
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate(out_dir: Path) -> None:
    runs = {key: _Run(key) for key in _METHODS}
    with plt.rc_context(RCPARAMS):
        _overview(runs, out_dir / "recovery_overview.pdf")
        _main(runs, out_dir / "recovery_main.pdf")
