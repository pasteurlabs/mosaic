"""Generate Figures: IC recovery overview — Adam vs L-BFGS vs L-BFGS+proj.

Figure 1 (recovery_overview.pdf):
  Row 0: IC error | IC divergence | Optimisation loss — all methods × all solvers
  Row 1: field panels (L-BFGS+proj / PhiFlow)

Figure 2 (recovery_adam_proj.pdf):
  Single panel: Adam vs Adam+proj across all solvers — shows projection
  offers no benefit for first-order optimisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    NS_ORDER,
    RCPARAMS,
    dedup_handles,
    make_handle,
    solver_props,
)

RESULTS = Path(__file__).parent.parent.parent / "results"
BASE = RESULTS / "ns-3d-grid" / "optimization"

METHODS: dict[str, tuple] = {
    "adam": ("Adam", "-.", BASE / "recovery_constant_ic"),
    "adam_proj": ("Adam+proj", ":", BASE / "recovery_constant_ic_proj"),
    "bfgs": ("L-BFGS", "--", BASE / "recovery_constant_ic_bfgs"),
    "bfgs_proj": ("L-BFGS+proj", "-", BASE / "recovery_constant_ic_bfgs_proj"),
}

_FIELD_SOLVER = "phiflow"
_FIELD_METHOD = "bfgs_proj"
_STEP_KEY = "100"
_Z_SLICE = 8
_VEL = 0


def _div_rms(field: np.ndarray) -> float:
    div = (
        np.gradient(field[..., 0], axis=0)
        + np.gradient(field[..., 1], axis=1)
        + np.gradient(field[..., 2], axis=2)
    )
    return float(np.sqrt(np.mean(div**2)))


def _solver_idx(npz, name: str) -> int | None:
    names = list(npz["solver_names"])
    return names.index(name) if name in names else None


def _snap_xs(n: int) -> list[int]:
    return list(range(1, n + 1))


def _generate_overview(loaded, out_path: Path) -> None:
    ref_npz = (loaded.get(_FIELD_METHOD) or list(loaded.values())[0])[1]
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

    for key, (m_label, m_ls, _) in METHODS.items():
        if key == "adam_proj":
            continue  # shown separately in recovery_adam_proj.pdf
        if key not in loaded:
            continue
        result, npz = loaded[key]
        by_sweep = result["by_sweep"]

        for solver in NS_ORDER:
            entry = by_sweep.get(solver, {}).get(_STEP_KEY)
            if entry is None:
                continue
            hist = entry.get("ic_error_history")
            if not hist:
                continue
            _, s_color, _, _ = solver_props(solver)
            kw = dict(color=s_color, linestyle=m_ls, linewidth=1.3, alpha=0.9)

            ax_conv.loglog(_snap_xs(len(hist)), list(hist), **kw)
            seen_solvers.add(solver)

            errors = entry.get("errors")
            if errors:
                ax_loss.loglog(_snap_xs(len(errors)), list(errors), **kw)

            if npz is not None:
                idx = _solver_idx(npz, solver)
                if idx is not None:
                    h = npz[f"ic_history_{idx}"]
                    dys = [max(_div_rms(h[t]), 1e-9) for t in range(h.shape[0])]
                    ax_div.loglog(_snap_xs(len(dys)), dys, **kw)

    ax_conv.set_title("IC recovery error")
    ax_conv.set_xlabel("Checkpoint")

    if ic_true_div is not None:
        ax_div.axhline(
            ic_true_div, color="0.55", linestyle="--", linewidth=1.0, zorder=0
        )
    ax_div.set_title("IC divergence")
    ax_div.set_xlabel("Checkpoint")

    ax_loss.set_title("Optimisation loss")
    ax_loss.set_xlabel("Checkpoint")

    method_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.3",
            linestyle=METHODS[k][1],
            linewidth=1.3,
            label=METHODS[k][0],
        )
        for k in METHODS
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


def _generate_adam_proj(loaded, out_path: Path) -> None:
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
        _, m_ls, _ = METHODS[key]
        by_sweep = result["by_sweep"]
        for solver in NS_ORDER:
            entry = by_sweep.get(solver, {}).get(_STEP_KEY)
            if entry is None:
                continue
            hist = entry.get("ic_error_history")
            if not hist:
                continue
            _, s_color, _, _ = solver_props(solver)
            kw = dict(color=s_color, linestyle=m_ls, linewidth=1.3, alpha=0.9)

            ax_conv.plot(_snap_xs(len(hist)), list(hist), **kw)
            seen_solvers.add(solver)

            errors = entry.get("errors")
            if errors:
                ax_loss.plot(_snap_xs(len(errors)), list(errors), **kw)

            if npz is not None:
                idx = _solver_idx(npz, solver)
                if idx is not None:
                    h = npz[f"ic_history_{idx}"]
                    dys = [max(_div_rms(h[t]), 1e-9) for t in range(h.shape[0])]
                    ax_div.plot(_snap_xs(len(dys)), dys, **kw)

    ax_conv.set_title("IC recovery error")
    ax_conv.set_xlabel("Checkpoint")

    if ic_true_div is not None:
        ax_div.axhline(
            ic_true_div, color="0.55", linestyle="--", linewidth=1.0, zorder=0
        )
    ax_div.set_title("IC divergence")
    ax_div.set_xlabel("Checkpoint")

    ax_loss.set_title("Optimisation loss")
    ax_loss.set_xlabel("Checkpoint")

    method_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.3",
            linestyle=METHODS[k][1],
            linewidth=1.3,
            label=METHODS[k][0],
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


def generate(out_dir: Path) -> None:
    loaded: dict[str, tuple] = {}
    for key, (*_, path) in METHODS.items():
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


if __name__ == "__main__":
    generate(Path(__file__).parent.parent.parent.parent.parent / "paper" / "figures")
