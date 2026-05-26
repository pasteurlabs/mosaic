# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Figures: IC recovery overview — Adam vs L-BFGS vs L-BFGS+proj.

Figure 1 (recovery_overview.pdf):
  Row 0: IC error | IC divergence | Optimization loss — all methods × all solvers
  Row 1: field panels (L-BFGS+proj / PhiFlow)

Figure 2 (recovery_adam_proj.pdf):
  Single panel: Adam vs Adam+proj across all solvers — shows projection
  offers no benefit for first-order optimisation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
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


def _methods():
    base = results_dir() / "ns-3d-grid" / "optimization"
    return {
        "adam": ("Adam", "-.", base / "recovery_constant_ic"),
        "adam_proj": ("Adam+proj", ":", base / "recovery_constant_ic_proj"),
        "bfgs": ("L-BFGS", "--", base / "recovery_constant_ic_bfgs"),
        "bfgs_proj": ("L-BFGS+proj", "-", base / "recovery_constant_ic_bfgs_proj"),
    }


_FIELD_SOLVER = "phiflow"
_FIELD_METHOD = "bfgs_proj"
_STEP_KEY = "100"
_Z_SLICE = 8
_VEL = 0

# Approximate gradient evaluations per outer optimizer iteration. Adam runs one
# value+grad call per step; optax's L-BFGS uses a zoom line search whose probes
# also evaluate value+grad, which empirically averages ~3× the per-iter cost
# (consistent with measured wall times across phiflow/ins_jl/pict).
_GRAD_EVALS_PER_ITER: dict[str, int] = {
    "adam": 1,
    "adam_proj": 1,
    "bfgs": 3,
    "bfgs_proj": 3,
}
_GRAD_EVAL_LABEL = "Gradient evaluations"


def _div_rms(field: np.ndarray) -> float:
    div = (
        np.gradient(field[..., 0], axis=0)
        + np.gradient(field[..., 1], axis=1)
        + np.gradient(field[..., 2], axis=2)
    )
    return float(np.sqrt(np.mean(div**2)))


def _solver_idx(npz: Any, name: str) -> int | None:
    names = list(npz["solver_names"])
    return names.index(name) if name in names else None


def _snap_xs(n: int) -> list[int]:
    return list(range(1, n + 1))


def _snap_interval(result: dict) -> int:
    return int(result.get("params", {}).get("optim", {}).get("snap_interval") or 1)


def _x_per_iter(key: str, n: int) -> list[float]:
    f = _GRAD_EVALS_PER_ITER.get(key, 1)
    return [(i + 1) * f for i in range(n)]


def _x_snapshot(key: str, n: int, snap_interval: int) -> list[float]:
    f = _GRAD_EVALS_PER_ITER.get(key, 1)
    return [(i + 1) * snap_interval * f for i in range(n)]


def _generate_overview(loaded: dict[str, Any], out_path: Path) -> None:
    ref_npz = (loaded.get(_FIELD_METHOD) or next(iter(loaded.values())))[1]
    ic_true_div = _div_rms(ref_npz["ic_true"]) if ref_npz is not None else None

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
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.30)
    bot_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.22)

    ax_conv = fig.add_subplot(top_gs[0, 0])
    ax_div = fig.add_subplot(top_gs[0, 1])
    ax_loss = fig.add_subplot(top_gs[0, 2])
    field_axes = [fig.add_subplot(bot_gs[0, i]) for i in range(4)]

    seen_solvers: set[str] = set()

    for key, (_m_label, m_ls, _) in _methods().items():
        if key == "adam_proj":
            continue  # shown separately in recovery_adam_proj.pdf
        if key not in loaded:
            continue
        result, npz = loaded[key]
        by_sweep = result["by_sweep"]
        snap = _snap_interval(result)

        for solver in NS_ORDER:
            entry = by_sweep.get(solver, {}).get(_STEP_KEY)
            if entry is None:
                continue
            hist = entry.get("ic_error_history")
            if not hist:
                continue
            _, s_color, _, _ = solver_props(solver)
            kw = {"color": s_color, "linestyle": m_ls, "linewidth": 1.3, "alpha": 0.9}

            ax_conv.loglog(_x_snapshot(key, len(hist), snap), list(hist), **kw)
            seen_solvers.add(solver)

            errors = entry.get("errors")
            if errors:
                ax_loss.loglog(_x_per_iter(key, len(errors)), list(errors), **kw)

            if npz is not None:
                idx = _solver_idx(npz, solver)
                if idx is not None:
                    h = npz[f"ic_history_{idx}"]
                    dys = [max(_div_rms(h[t]), 1e-9) for t in range(h.shape[0])]
                    ax_div.loglog(_x_snapshot(key, len(dys), snap), dys, **kw)

    ax_conv.set_title("IC recovery error")
    ax_conv.set_xlabel(_GRAD_EVAL_LABEL)

    if ic_true_div is not None:
        ax_div.axhline(
            ic_true_div, color="0.55", linestyle="--", linewidth=1.0, zorder=0
        )
    ax_div.set_title("IC divergence")
    ax_div.set_xlabel(_GRAD_EVAL_LABEL)

    ax_loss.set_title("Optimization loss")
    ax_loss.set_xlabel(_GRAD_EVAL_LABEL)

    method_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.3",
            linestyle=_methods()[k][1],
            linewidth=1.3,
            label=_methods()[k][0],
        )
        for k in _methods()
        if k != "adam_proj"
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
        handlelength=1.8,
    )

    # Field panels
    fkey = _FIELD_METHOD if _FIELD_METHOD in loaded else list(loaded.keys())[-1]
    _, npz_f = loaded[fkey]
    if npz_f is not None:
        idx_f = _solver_idx(npz_f, _FIELD_SOLVER)
        if idx_f is not None:

            def _sl(f: np.ndarray) -> np.ndarray:
                return f[:, :, _Z_SLICE, _VEL]

            ic_rec = _sl(npz_f[f"ic_rec_{idx_f}"])
            ic_true = _sl(npz_f["ic_true"])
            fin_rec = _sl(npz_f[f"final_rec_{idx_f}"])
            fin_gt = _sl(npz_f["final_gt_shared"])
            vlim = float(np.percentile(np.abs([ic_rec, ic_true, fin_rec, fin_gt]), 99))

            last_im = None
            for ax, (data, title) in zip(
                field_axes,
                [
                    (ic_rec, "IC recovered"),
                    (ic_true, "IC true"),
                    (fin_rec, "Final state"),
                    (fin_gt, "Final true"),
                ],
                strict=False,
            ):
                last_im = ax.imshow(
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
                last_im,
                ax=field_axes,
                fraction=0.015,
                pad=0.02,
                ticks=np.linspace(-vlim, vlim, 5),
            )
            cb.ax.tick_params(labelsize=5.5)
            cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def _generate_adam_proj(loaded: dict[str, Any], out_path: Path) -> None:
    """Focused Adam vs Adam+proj comparison across all solvers."""
    keys = ("adam", "adam_proj")
    if not all(k in loaded for k in keys):
        print(
            "[recovery_overview] adam or adam_proj not loaded — skipping adam_proj figure"
        )
        return

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.38))
    fig.subplots_adjust(bottom=0.42, wspace=0.25, left=0.05, right=0.98)
    ax_conv, ax_div, ax_loss = axes

    ref_npz = loaded.get("adam_proj", loaded.get("adam"))[1]
    ic_true_div = _div_rms(ref_npz["ic_true"]) if ref_npz is not None else None

    seen_solvers: set[str] = set()
    for key in keys:
        result, npz = loaded[key]
        _, m_ls, _ = _methods()[key]
        by_sweep = result["by_sweep"]
        snap = _snap_interval(result)
        for solver in NS_ORDER:
            entry = by_sweep.get(solver, {}).get(_STEP_KEY)
            if entry is None:
                continue
            hist = entry.get("ic_error_history")
            if not hist:
                continue
            _, s_color, _, _ = solver_props(solver)
            kw = {"color": s_color, "linestyle": m_ls, "linewidth": 1.3, "alpha": 0.9}

            ax_conv.plot(_x_snapshot(key, len(hist), snap), list(hist), **kw)
            seen_solvers.add(solver)

            errors = entry.get("errors")
            if errors:
                ax_loss.plot(_x_per_iter(key, len(errors)), list(errors), **kw)

            if npz is not None:
                idx = _solver_idx(npz, solver)
                if idx is not None:
                    h = npz[f"ic_history_{idx}"]
                    dys = [max(_div_rms(h[t]), 1e-9) for t in range(h.shape[0])]
                    ax_div.plot(_x_snapshot(key, len(dys), snap), dys, **kw)

    ax_conv.set_title("IC recovery error")
    ax_conv.set_xlabel(_GRAD_EVAL_LABEL)

    if ic_true_div is not None:
        ax_div.axhline(
            ic_true_div, color="0.55", linestyle="--", linewidth=1.0, zorder=0
        )
    ax_div.set_title("IC divergence")
    ax_div.set_xlabel(_GRAD_EVAL_LABEL)

    ax_loss.set_title("Optimization loss")
    ax_loss.set_xlabel(_GRAD_EVAL_LABEL)

    method_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.3",
            linestyle=_methods()[k][1],
            linewidth=1.3,
            label=_methods()[k][0],
        )
        for k in keys
    ]
    solver_handles = dedup_handles(
        [make_handle(s) for s in NS_ORDER if s in seen_solvers]
    )
    fig.legend(
        handles=method_handles + solver_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=6.5,
        framealpha=0.8,
        edgecolor="0.8",
        handlelength=1.8,
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


_MAIN_SUBSET = ["pict", "ins_jl", "xlb"]


def _generate_main_subset(loaded: dict[str, Any], out_path: Path) -> None:
    """Slim version for the main paper.

    Only the top row (IC error, IC divergence, optimisation loss) and only the
    ``_MAIN_SUBSET`` solvers are shown.
    """
    ref_npz = (loaded.get(_FIELD_METHOD) or next(iter(loaded.values())))[1]
    ic_true_div = _div_rms(ref_npz["ic_true"]) if ref_npz is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.36))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.36, wspace=0.30)
    ax_conv, ax_div, ax_loss = axes

    seen_solvers: set[str] = set()

    for key, (_m_label, m_ls, _) in _methods().items():
        if key == "adam_proj":
            continue
        if key not in loaded:
            continue
        result, npz = loaded[key]
        by_sweep = result["by_sweep"]
        snap = _snap_interval(result)

        for solver in _MAIN_SUBSET:
            entry = by_sweep.get(solver, {}).get(_STEP_KEY)
            if entry is None:
                continue
            hist = entry.get("ic_error_history")
            if not hist:
                continue
            _, s_color, _, _ = solver_props(solver)
            kw = {"color": s_color, "linestyle": m_ls, "linewidth": 1.4, "alpha": 0.9}

            ax_conv.loglog(_x_snapshot(key, len(hist), snap), list(hist), **kw)
            seen_solvers.add(solver)

            errors = entry.get("errors")
            if errors:
                ax_loss.loglog(_x_per_iter(key, len(errors)), list(errors), **kw)

            if npz is not None:
                idx = _solver_idx(npz, solver)
                if idx is not None:
                    h = npz[f"ic_history_{idx}"]
                    dys = [max(_div_rms(h[t]), 1e-9) for t in range(h.shape[0])]
                    ax_div.loglog(_x_snapshot(key, len(dys), snap), dys, **kw)

    ax_conv.set_title("IC recovery error")
    ax_conv.set_xlabel(_GRAD_EVAL_LABEL)

    if ic_true_div is not None:
        ax_div.axhline(
            ic_true_div, color="0.55", linestyle="--", linewidth=1.0, zorder=0
        )
    ax_div.set_title("IC divergence")
    ax_div.set_xlabel(_GRAD_EVAL_LABEL)

    ax_loss.set_title("Optimization loss")
    ax_loss.set_xlabel(_GRAD_EVAL_LABEL)

    method_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.3",
            linestyle=_methods()[k][1],
            linewidth=1.4,
            label=_methods()[k][0],
        )
        for k in _methods()
        if k != "adam_proj"
    ]
    solver_handles = dedup_handles(
        [make_handle(s) for s in _MAIN_SUBSET if s in seen_solvers]
    )
    fig.legend(
        handles=method_handles + solver_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=6,
        fontsize=7.0,
        framealpha=0.8,
        edgecolor="0.8",
        handlelength=1.8,
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate(out_dir: Path) -> None:
    """Generate IC recovery overview figures — Adam vs L-BFGS vs L-BFGS+proj."""
    loaded: dict[str, tuple] = {}
    for key, (*_, path) in _methods().items():
        rp = path / "result.json"
        fp = path / "recovery_fields.npz"
        if not rp.exists():
            print(f"[recovery_overview] {rp} not found — skipping {key}")
            continue
        npz = np.load(fp, allow_pickle=True) if fp.exists() else None
        loaded[key] = (json.loads(rp.read_text()), npz)

    if not loaded:
        return

    with plt.rc_context(RCPARAMS):
        _generate_overview(loaded, out_dir / "recovery_overview.pdf")
        _generate_adam_proj(loaded, out_dir / "recovery_adam_proj.pdf")
        _generate_main_subset(loaded, out_dir / "recovery_main.pdf")


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
