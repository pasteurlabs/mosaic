"""Cross-domain / cross-experiment paper figures for ns-3d-grid.

Registered as ``_extra/<name>`` plots on the per-problem ``Problem`` instance
so ``mosaic run --plots-only`` invokes them automatically.

Each public ``_plot_*`` function takes the standard per-experiment plot
signature ``(cfg, **kw)`` and writes its figure(s) under
``<results>/<cfg.name>/_extra/``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir, try_load_npz
from mosaic.benchmarks.problems.shared.plots.cost_overview import (
    plot_cost_overview_for,
)
from mosaic.benchmarks.problems.shared.plots.style import (
    NS_ORDER,
    PAPER_RCPARAMS,
    SOLVER_STYLES,
    TEXTWIDTH,
    dedup_handles,
    make_handle,
    paper_rc_context,
    resolve_solver_alias,
    solver_props,
)


def _alias_to_display_from_by_sweep(by_sweep: dict) -> dict[str, str]:
    """Build alias→display-name map from a ``by_sweep`` dict's keys.

    ``by_sweep`` is keyed by ``spec.name`` (display form) but the canonical
    ordering lists (``NS_ORDER`` etc.) are alias-keyed; this map bridges the
    two so plot loops can iterate aliases in canonical order.
    """
    out: dict[str, str] = {}
    for display_name in by_sweep:
        a = resolve_solver_alias(display_name)
        if a is not None:
            out[a] = display_name
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extra_out_dir(cfg: Problem) -> Path:
    out = results_dir() / cfg.name / "_extra"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# recovery_overview — Adam vs L-BFGS vs L-BFGS+proj
# ─────────────────────────────────────────────────────────────────────────────


def _ro_methods() -> dict[str, tuple]:
    base = results_dir() / "ns-3d-grid" / "optimization"
    return {
        "adam": ("Adam", "-.", base / "recovery_constant_ic"),
        "adam_proj": ("Adam+proj", ":", base / "recovery_constant_ic_proj"),
        "bfgs": ("L-BFGS", "--", base / "recovery_constant_ic_bfgs"),
        "bfgs_proj": ("L-BFGS+proj", "-", base / "recovery_constant_ic_bfgs_proj"),
    }


_RO_FIELD_SOLVER = "phiflow"
_RO_FIELD_METHOD = "bfgs_proj"
_RO_STEP_KEY = "100"
_RO_Z_SLICE = 8
_RO_VEL = 0

# Approximate gradient evaluations per outer optimizer iteration.
_RO_GRAD_EVALS_PER_ITER: dict[str, int] = {
    "adam": 1,
    "adam_proj": 1,
    "bfgs": 3,
    "bfgs_proj": 3,
}
_RO_GRAD_EVAL_LABEL = "Gradient evaluations"


def _ro_div_rms(field: np.ndarray) -> float:
    div = (
        np.gradient(field[..., 0], axis=0)
        + np.gradient(field[..., 1], axis=1)
        + np.gradient(field[..., 2], axis=2)
    )
    return float(np.sqrt(np.mean(div**2)))


def _ro_solver_idx(npz, name: str) -> int | None:
    names = list(npz["solver_names"])
    return names.index(name) if name in names else None


def _ro_snap_interval(result: dict) -> int:
    return int(result.get("params", {}).get("optim", {}).get("snap_interval") or 1)


def _ro_x_per_iter(key: str, n: int) -> list[float]:
    f = _RO_GRAD_EVALS_PER_ITER.get(key, 1)
    return [(i + 1) * f for i in range(n)]


def _ro_x_snapshot(key: str, n: int, snap_interval: int) -> list[float]:
    f = _RO_GRAD_EVALS_PER_ITER.get(key, 1)
    return [(i + 1) * snap_interval * f for i in range(n)]


def _ro_generate_overview(loaded, out_path: Path) -> None:
    ref_npz = (loaded.get(_RO_FIELD_METHOD) or next(iter(loaded.values())))[1]
    ic_true_div = _ro_div_rms(ref_npz["ic_true"]) if ref_npz is not None else None

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

    for key, (_m_label, m_ls, _) in _ro_methods().items():
        if key == "adam_proj":
            continue  # shown separately in recovery_adam_proj.pdf
        if key not in loaded:
            continue
        result, npz = loaded[key]
        by_sweep = result["by_sweep"]
        snap = _ro_snap_interval(result)

        alias_to_display = _alias_to_display_from_by_sweep(by_sweep)
        for alias in NS_ORDER:
            display_name = alias_to_display.get(alias)
            if display_name is None:
                continue
            entry = by_sweep.get(display_name, {}).get(_RO_STEP_KEY)
            if entry is None:
                continue
            hist = entry.get("ic_error_history")
            if not hist:
                continue
            _, s_color, _, _ = solver_props(alias)
            kw = {"color": s_color, "linestyle": m_ls, "linewidth": 1.3, "alpha": 0.9}

            ax_conv.loglog(_ro_x_snapshot(key, len(hist), snap), list(hist), **kw)
            seen_solvers.add(alias)

            errors = entry.get("errors")
            if errors:
                ax_loss.loglog(_ro_x_per_iter(key, len(errors)), list(errors), **kw)

            if npz is not None:
                idx = _ro_solver_idx(npz, display_name)
                if idx is not None:
                    h = npz[f"ic_history_{idx}"]
                    dys = [max(_ro_div_rms(h[t]), 1e-9) for t in range(h.shape[0])]
                    ax_div.loglog(_ro_x_snapshot(key, len(dys), snap), dys, **kw)

    ax_conv.set_title("IC recovery error")
    ax_conv.set_xlabel(_RO_GRAD_EVAL_LABEL)

    if ic_true_div is not None:
        ax_div.axhline(
            ic_true_div, color="0.55", linestyle="--", linewidth=1.0, zorder=0
        )
    ax_div.set_title("IC divergence")
    ax_div.set_xlabel(_RO_GRAD_EVAL_LABEL)

    ax_loss.set_title("Optimization loss")
    ax_loss.set_xlabel(_RO_GRAD_EVAL_LABEL)

    method_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.3",
            linestyle=_ro_methods()[k][1],
            linewidth=1.3,
            label=_ro_methods()[k][0],
        )
        for k in _ro_methods()
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
    fkey = _RO_FIELD_METHOD if _RO_FIELD_METHOD in loaded else list(loaded.keys())[-1]
    _, npz_f = loaded[fkey]
    if npz_f is not None:
        # ``_RO_FIELD_SOLVER`` is an alias; npz["solver_names"] holds display
        # names. Resolve via the by_sweep map for the chosen method.
        _f_by_sweep = loaded[fkey][0].get("by_sweep", {})
        _f_a2d = _alias_to_display_from_by_sweep(_f_by_sweep)
        _f_display = _f_a2d.get(_RO_FIELD_SOLVER, _RO_FIELD_SOLVER)
        idx_f = _ro_solver_idx(npz_f, _f_display)
        if idx_f is not None:

            def _sl(f: np.ndarray) -> np.ndarray:
                return f[:, :, _RO_Z_SLICE, _RO_VEL]

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


def _ro_generate_adam_proj(loaded, out_path: Path) -> None:
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
    ic_true_div = _ro_div_rms(ref_npz["ic_true"]) if ref_npz is not None else None

    seen_solvers: set[str] = set()
    for key in keys:
        result, npz = loaded[key]
        _, m_ls, _ = _ro_methods()[key]
        by_sweep = result["by_sweep"]
        snap = _ro_snap_interval(result)
        alias_to_display = _alias_to_display_from_by_sweep(by_sweep)
        for alias in NS_ORDER:
            display_name = alias_to_display.get(alias)
            if display_name is None:
                continue
            entry = by_sweep.get(display_name, {}).get(_RO_STEP_KEY)
            if entry is None:
                continue
            hist = entry.get("ic_error_history")
            if not hist:
                continue
            _, s_color, _, _ = solver_props(alias)
            kw = {"color": s_color, "linestyle": m_ls, "linewidth": 1.3, "alpha": 0.9}

            ax_conv.plot(_ro_x_snapshot(key, len(hist), snap), list(hist), **kw)
            seen_solvers.add(alias)

            errors = entry.get("errors")
            if errors:
                ax_loss.plot(_ro_x_per_iter(key, len(errors)), list(errors), **kw)

            if npz is not None:
                idx = _ro_solver_idx(npz, display_name)
                if idx is not None:
                    h = npz[f"ic_history_{idx}"]
                    dys = [max(_ro_div_rms(h[t]), 1e-9) for t in range(h.shape[0])]
                    ax_div.plot(_ro_x_snapshot(key, len(dys), snap), dys, **kw)

    ax_conv.set_title("IC recovery error")
    ax_conv.set_xlabel(_RO_GRAD_EVAL_LABEL)

    if ic_true_div is not None:
        ax_div.axhline(
            ic_true_div, color="0.55", linestyle="--", linewidth=1.0, zorder=0
        )
    ax_div.set_title("IC divergence")
    ax_div.set_xlabel(_RO_GRAD_EVAL_LABEL)

    ax_loss.set_title("Optimization loss")
    ax_loss.set_xlabel(_RO_GRAD_EVAL_LABEL)

    method_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.3",
            linestyle=_ro_methods()[k][1],
            linewidth=1.3,
            label=_ro_methods()[k][0],
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


_RO_MAIN_SUBSET = ["pict", "ins_jl", "xlb"]


def _ro_generate_main_subset(loaded, out_path: Path) -> None:
    """Slim version for the main paper: only the top row and ``_RO_MAIN_SUBSET`` solvers."""
    ref_npz = (loaded.get(_RO_FIELD_METHOD) or next(iter(loaded.values())))[1]
    ic_true_div = _ro_div_rms(ref_npz["ic_true"]) if ref_npz is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.36))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.36, wspace=0.30)
    ax_conv, ax_div, ax_loss = axes

    seen_solvers: set[str] = set()

    for key, (_m_label, m_ls, _) in _ro_methods().items():
        if key == "adam_proj":
            continue
        if key not in loaded:
            continue
        result, npz = loaded[key]
        by_sweep = result["by_sweep"]
        snap = _ro_snap_interval(result)

        alias_to_display = _alias_to_display_from_by_sweep(by_sweep)
        for alias in _RO_MAIN_SUBSET:
            display_name = alias_to_display.get(alias)
            if display_name is None:
                continue
            entry = by_sweep.get(display_name, {}).get(_RO_STEP_KEY)
            if entry is None:
                continue
            hist = entry.get("ic_error_history")
            if not hist:
                continue
            _, s_color, _, _ = solver_props(alias)
            kw = {"color": s_color, "linestyle": m_ls, "linewidth": 1.4, "alpha": 0.9}

            ax_conv.loglog(_ro_x_snapshot(key, len(hist), snap), list(hist), **kw)
            seen_solvers.add(alias)

            errors = entry.get("errors")
            if errors:
                ax_loss.loglog(_ro_x_per_iter(key, len(errors)), list(errors), **kw)

            if npz is not None:
                idx = _ro_solver_idx(npz, display_name)
                if idx is not None:
                    h = npz[f"ic_history_{idx}"]
                    dys = [max(_ro_div_rms(h[t]), 1e-9) for t in range(h.shape[0])]
                    ax_div.loglog(_ro_x_snapshot(key, len(dys), snap), dys, **kw)

    ax_conv.set_title("IC recovery error")
    ax_conv.set_xlabel(_RO_GRAD_EVAL_LABEL)

    if ic_true_div is not None:
        ax_div.axhline(
            ic_true_div, color="0.55", linestyle="--", linewidth=1.0, zorder=0
        )
    ax_div.set_title("IC divergence")
    ax_div.set_xlabel(_RO_GRAD_EVAL_LABEL)

    ax_loss.set_title("Optimization loss")
    ax_loss.set_xlabel(_RO_GRAD_EVAL_LABEL)

    method_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.3",
            linestyle=_ro_methods()[k][1],
            linewidth=1.4,
            label=_ro_methods()[k][0],
        )
        for k in _ro_methods()
        if k != "adam_proj"
    ]
    solver_handles = dedup_handles(
        [make_handle(s) for s in _RO_MAIN_SUBSET if s in seen_solvers]
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


def _plot_recovery_overview(cfg: Problem, **_kw) -> None:
    """``_extra/recovery_overview`` — paper IC-recovery overview figures."""
    out_dir = _extra_out_dir(cfg)
    loaded: dict[str, tuple] = {}
    for key, (*_, path) in _ro_methods().items():
        rp = path / "result.json"
        fp = path / "recovery_fields.npz"
        if not rp.exists():
            print(f"[recovery_overview] {rp} not found — skipping {key}")
            continue
        npz = try_load_npz(fp) if fp.exists() else None
        loaded[key] = (load_json(rp), npz)

    if not loaded:
        return

    with plt.rc_context(PAPER_RCPARAMS):
        _ro_generate_overview(loaded, out_dir / "recovery_overview.pdf")
        _ro_generate_adam_proj(loaded, out_dir / "recovery_adam_proj.pdf")
        _ro_generate_main_subset(loaded, out_dir / "recovery_main.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# grad_divergence — gradient/IC divergence across optimizers
# ─────────────────────────────────────────────────────────────────────────────


def _gd_methods() -> dict[str, tuple]:
    base = results_dir() / "ns-3d-grid" / "optimization"
    return {
        "adam": ("Adam", "-.", base / "recovery_constant_ic"),
        "adam_proj": ("Adam+proj", ":", base / "recovery_constant_ic_proj"),
        "bfgs": ("L-BFGS", "--", base / "recovery_constant_ic_bfgs"),
        "bfgs_proj": ("L-BFGS+proj", "-", base / "recovery_constant_ic_bfgs_proj"),
    }


_GD_GRAD_EVALS_PER_ITER: dict[str, int] = {
    "adam": 1,
    "adam_proj": 1,
    "bfgs": 3,
    "bfgs_proj": 3,
}
_GD_GRAD_EVAL_LABEL = "Gradient evaluations"

_GD_STEP_KEY = "100"
_GD_FLOOR = 1e-12


def _gd_load_results() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, (_, _, path) in _gd_methods().items():
        rp = path / "result.json"
        if not rp.exists():
            continue
        out[key] = load_json(rp)
    return out


def _gd_series(entry: dict, field: str) -> list[float] | None:
    vals = entry.get(field) if entry else None
    if not vals:
        return None
    return [max(float(v), _GD_FLOOR) for v in vals]


def _gd_plot_panel(
    ax,
    results: dict[str, dict],
    field: str,
    seen_solvers: set[str],
    seen_methods: set[str],
) -> None:
    methods = _gd_methods()
    for key, result in results.items():
        _m_label, m_ls, _ = methods[key]
        by_sweep = result.get("by_sweep", {})
        f = _GD_GRAD_EVALS_PER_ITER.get(key, 1)
        alias_to_display = _alias_to_display_from_by_sweep(by_sweep)
        for alias in NS_ORDER:
            display_name = alias_to_display.get(alias)
            if display_name is None:
                continue
            entry = by_sweep.get(display_name, {}).get(_GD_STEP_KEY)
            vals = _gd_series(entry, field)
            if vals is None:
                continue
            _, color, _, _ = solver_props(alias)
            xs = np.array([(i + 1) * f for i in range(len(vals))])
            ax.loglog(xs, vals, color=color, linestyle=m_ls, linewidth=1.3, alpha=0.9)
            seen_solvers.add(alias)
            seen_methods.add(key)
    ax.set_xlabel(_GD_GRAD_EVAL_LABEL)


def _plot_grad_divergence(cfg: Problem, **_kw) -> None:
    """``_extra/grad_divergence`` — gradient/IC divergence over IC optimisation."""
    out_dir = _extra_out_dir(cfg)
    results = _gd_load_results()
    if not results:
        print("[grad_divergence] no recovery results found — skipping")
        return

    with plt.rc_context(PAPER_RCPARAMS):
        fig, (ax_g, ax_u) = plt.subplots(1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.42))
        fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.32, wspace=0.30)

        seen_solvers: set[str] = set()
        seen_methods: set[str] = set()
        _gd_plot_panel(ax_g, results, "grad_divs", seen_solvers, seen_methods)
        _gd_plot_panel(ax_u, results, "ic_divs", seen_solvers, seen_methods)

        ax_g.set_title(r"Gradient divergence  $\max\,|\nabla\!\cdot g|$")
        ax_g.set_ylabel("max divergence")
        ax_u.set_title(r"IC divergence  $\max\,|\nabla\!\cdot u|$")

        methods = _gd_methods()
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


# ─────────────────────────────────────────────────────────────────────────────
# horizon_sweep_limits — VJP rollout-length limit sweep
# ─────────────────────────────────────────────────────────────────────────────


_HSL_SOLVER_ORDER = [
    "phiflow",
    "xlb",
    "pict",
    "warp_ns",
    "exponax",
    "ins_jl",
]
_HSL_OF_COLOR = SOLVER_STYLES.get("openfoam", ("OpenFOAM", "#DDCC77", "--", "h"))[1]
_HSL_OF_LS = SOLVER_STYLES.get("openfoam", ("OpenFOAM", "#DDCC77", "--", "h"))[2]

_HSL_FAILURE_MARKER = {"OOM": "v", "nan": "X", "error": "D", "timeout": "s"}
_HSL_FAILURE_LABEL = {
    "OOM": "OOM (VRAM)",
    "nan": "NaN gradient",
    "error": "error",
    "timeout": "timeout",
}
_HSL_VRAM_LIMIT_MIB = 16_384


# Gradient norm panel: 3-section piecewise y-scale.
_HSL_GN_YMIN = 1.0
_HSL_GN_LOWER_BREAK = 1.6
_HSL_GN_LOWER_FACTOR = 0.15
_HSL_GN_MIDDLE_FACTOR = 4.0
_HSL_BREAK_LOG = 3.0

_HSL_WT_YMIN = 0.0
_HSL_WT_BREAK_LOG = 2.0

# Piecewise log x-axis.
_HSL_X_BREAK_LOG = 2.5
_HSL_X_UPPER_FACTOR = 0.4


def _hsl_x_log_forward(steps):
    steps = np.asarray(steps, dtype=float)
    log_x = np.log10(np.maximum(steps, 1e-10))
    return np.where(
        log_x <= _HSL_X_BREAK_LOG,
        log_x,
        _HSL_X_BREAK_LOG + (log_x - _HSL_X_BREAK_LOG) * _HSL_X_UPPER_FACTOR,
    )


def _hsl_x_log_inverse(disp):
    disp = np.asarray(disp, dtype=float)
    log_x = np.where(
        disp <= _HSL_X_BREAK_LOG,
        disp,
        _HSL_X_BREAK_LOG + (disp - _HSL_X_BREAK_LOG) / _HSL_X_UPPER_FACTOR,
    )
    return np.power(10.0, log_x)


_HSL_JITTER_LOG = 0.04


def _hsl_solver_style(name: str) -> tuple:
    return SOLVER_STYLES.get(name, (name, "#888888", "-", "o"))


def _hsl_openfoam_fd_vjp_estimate(
    sweep_steps: list[int], N_sweep: int = 20
) -> dict[int, float]:
    """Estimate FD VJP wall time for OpenFOAM: N_inputs × forward_cost(T)."""
    _temporal_cost_path = (
        results_dir() / "ns-3d-grid" / "cost" / "temporal_cost" / "result.json"
    )
    if not _temporal_cost_path.exists():
        return {}
    td = load_json(_temporal_cost_path)
    # ``by_steps`` is keyed by spec.name (display form); find the entry that
    # resolves to the ``openfoam`` alias.
    by_steps = td.get("by_steps", {}) or {}
    of_data: dict = {}
    for display_name, vals in by_steps.items():
        if resolve_solver_alias(display_name) == "openfoam":
            of_data = vals or {}
            break
    pts = sorted(
        [
            (int(s), v["mean"])
            for s, v in of_data.items()
            if isinstance(v, dict) and "mean" in v
        ],
    )
    if len(pts) < 2:
        return {}
    steps_arr = np.array([p[0] for p in pts], dtype=float)
    cost_arr = np.array([p[1] for p in pts], dtype=float)
    A = np.column_stack([np.ones_like(steps_arr), steps_arr])
    startup, per_step = np.linalg.lstsq(A, cost_arr, rcond=None)[0]
    per_step_scaled = per_step * (N_sweep / 16) ** 3
    N_inputs = N_sweep**3 * 3
    return {T: N_inputs * max(startup + per_step_scaled * T, 1e-3) for T in sweep_steps}


def _hsl_order_solvers(by_solver: dict) -> tuple[set[str], list[str]]:
    excluded = {"fenics_ns", "fenics_ns_3d", "su2"}
    present = set(by_solver.keys())
    ordered = [s for s in _HSL_SOLVER_ORDER if s in present] + [
        s for s in present if s not in _HSL_SOLVER_ORDER and s not in excluded
    ]
    return present, ordered


def _hsl_parse_one_solver(step_results: dict) -> dict:
    all_steps = sorted(step_results.keys(), key=int)

    ok_steps, ok_vram, ok_wall, ok_gnorm = [], [], [], []
    fail_step = fail_vram = fail_wall = fail_ram = fail_ft = None
    for k in all_steps:
        r = step_results[k]
        if r["status"] == "ok":
            ok_steps.append(int(k))
            ok_vram.append(r.get("vram_peak_mib") or 0.0)
            ok_wall.append(r["wall_time_s"])
            ok_gnorm.append(r.get("grad_norm") or 1.0)
        elif r["status"] == "failed" and fail_step is None:
            fail_step = int(k)
            fail_vram = r.get("vram_peak_mib") or 1.0
            fail_wall = r["wall_time_s"]
            fail_ram = r.get("ram_peak_mib") or 1.0
            fail_ft = r["failure_type"]

    ok_ram = [step_results[str(s)].get("ram_peak_mib") for s in ok_steps]
    cpu_only = all(v == 0.0 for v in ok_vram) and bool(ok_vram)
    has_ram = any(r is not None and r > 0 for r in ok_ram)

    return {
        "ok_steps": ok_steps,
        "ok_vram": ok_vram,
        "ok_wall": ok_wall,
        "ok_gnorm": ok_gnorm,
        "ok_ram": ok_ram,
        "cpu_only": cpu_only,
        "has_ram": has_ram,
        "fail_step": fail_step,
        "fail_vram": fail_vram,
        "fail_wall": fail_wall,
        "fail_ram": fail_ram,
        "fail_ft": fail_ft,
    }


def _hsl_parse_all_solvers(by_solver: dict, ordered: list[str]) -> dict[str, dict]:
    return {solver: _hsl_parse_one_solver(by_solver[solver]) for solver in ordered}


def _hsl_build_gn_scale(solver_data: dict[str, dict]) -> tuple:
    all_log10 = [
        np.log10(max(g, 1e-30)) for d in solver_data.values() for g in d["ok_gnorm"]
    ]
    max_log10 = max(all_log10) if all_log10 else _HSL_BREAK_LOG + 1.0
    upper_data = max(max_log10 - _HSL_BREAK_LOG, 0.1)
    middle_height = _HSL_BREAK_LOG - _HSL_GN_LOWER_BREAK
    middle_disp = middle_height * _HSL_GN_MIDDLE_FACTOR
    upper_factor = (middle_disp / 3.0) / upper_data
    lower_disp_height = (_HSL_GN_LOWER_BREAK - _HSL_GN_YMIN) * _HSL_GN_LOWER_FACTOR
    gn_lower_break_disp = _HSL_GN_YMIN + lower_disp_height
    gn_upper_break_disp = gn_lower_break_disp + middle_disp

    def gn_display(v: float) -> float:
        if v <= _HSL_GN_LOWER_BREAK:
            return _HSL_GN_YMIN + (v - _HSL_GN_YMIN) * _HSL_GN_LOWER_FACTOR
        elif v <= _HSL_BREAK_LOG:
            return (
                gn_lower_break_disp + (v - _HSL_GN_LOWER_BREAK) * _HSL_GN_MIDDLE_FACTOR
            )
        else:
            return gn_upper_break_disp + (v - _HSL_BREAK_LOG) * upper_factor

    return gn_display, max_log10


def _hsl_build_wt_scale(solver_data: dict[str, dict], of_fd: dict[int, float]) -> tuple:
    all_wt_log10 = [
        np.log10(max(t, 1e-10)) for d in solver_data.values() for t in d["ok_wall"]
    ]
    if of_fd:
        all_wt_log10 += [np.log10(max(c, 1e-10)) for c in of_fd.values()]
    max_wt_log10 = max(all_wt_log10) if all_wt_log10 else _HSL_WT_BREAK_LOG + 1.0
    wt_lower_height = _HSL_WT_BREAK_LOG - _HSL_WT_YMIN
    wt_upper_data = max(max_wt_log10 - _HSL_WT_BREAK_LOG, 0.1)
    wt_upper_factor = (wt_lower_height / 3.0) / wt_upper_data

    def wt_display(v: float) -> float:
        if v <= _HSL_WT_BREAK_LOG:
            return v
        return _HSL_WT_BREAK_LOG + (v - _HSL_WT_BREAK_LOG) * wt_upper_factor

    return wt_display, max_wt_log10, wt_upper_factor, wt_upper_data


def _hsl_compute_jitter(
    solver_data: dict[str, dict], ordered: list[str]
) -> dict[tuple[str, int], float]:
    fail_at_step: dict[int, list[str]] = defaultdict(list)
    for solver in ordered:
        fs = solver_data[solver]["fail_step"]
        if fs is not None:
            fail_at_step[fs].append(solver)

    jitter_x: dict[tuple[str, int], float] = {}
    for step, solvers_here in fail_at_step.items():
        n = len(solvers_here)
        for i, sv in enumerate(solvers_here):
            if n == 1:
                jitter_x[(sv, step)] = float(step)
            else:
                t = i / (n - 1)
                log_off = (2 * t - 1) * _HSL_JITTER_LOG
                jitter_x[(sv, step)] = step * 10**log_off
    return jitter_x


def _hsl_plot_vram_panel(ax_vr, d: dict, kw: dict, kw_line: dict) -> None:
    ok_steps = d["ok_steps"]
    fail_step = d["fail_step"]
    if not d["cpu_only"]:
        ax_vr.loglog(ok_steps, [max(v, 1) for v in d["ok_vram"]], **kw)
        if fail_step:
            ax_vr.loglog(
                [ok_steps[-1], fail_step],
                [max(d["ok_vram"][-1], 1), max(d["fail_vram"], 1)],
                **kw_line,
            )
    elif d["has_ram"]:
        ax_vr.loglog(ok_steps, [max(r, 1) for r in d["ok_ram"]], **kw)
        if fail_step and d["fail_ram"]:
            ax_vr.loglog(
                [ok_steps[-1], fail_step],
                [max(d["ok_ram"][-1], 1), max(d["fail_ram"], 1)],
                **kw_line,
            )
    else:
        ax_vr.loglog([], [], **kw)


def _hsl_plot_wt_panel(ax_wt, d: dict, kw: dict, kw_line: dict, wt_display) -> None:
    ok_steps = d["ok_steps"]
    ok_wall = d["ok_wall"]
    fail_step = d["fail_step"]
    log_wt = [np.log10(max(t, 1e-10)) for t in ok_wall]
    disp_wt = [wt_display(v) for v in log_wt]
    ax_wt.semilogx(ok_steps, disp_wt, **kw)
    if fail_step:
        last_wt_disp = wt_display(np.log10(max(ok_wall[-1], 1e-10)))
        fail_wt_disp = wt_display(np.log10(max(d["fail_wall"], 1e-10)))
        ax_wt.semilogx(
            [ok_steps[-1], fail_step],
            [last_wt_disp, fail_wt_disp],
            **kw_line,
        )


def _hsl_plot_gn_panel(
    ax_gn, d: dict, kw: dict, kw_line: dict, gn_display, jx: float | None
) -> None:
    ok_steps = d["ok_steps"]
    ok_gnorm = d["ok_gnorm"]
    fail_step = d["fail_step"]
    fail_ft = d["fail_ft"]
    log_gnorm = [np.log10(max(g, 1e-30)) for g in ok_gnorm]
    disp_gnorm = [gn_display(v) for v in log_gnorm]
    ax_gn.semilogx(ok_steps, disp_gnorm, **kw)
    if fail_step and ok_gnorm:
        last_disp = gn_display(np.log10(max(ok_gnorm[-1], 1e-30)))
        ax_gn.semilogx([ok_steps[-1], fail_step], [last_disp, last_disp], **kw_line)
        ax_gn.semilogx(
            [jx],
            [last_disp],
            **{
                **kw,
                "marker": _HSL_FAILURE_MARKER.get(fail_ft, "D"),
                "markersize": 9,
                "markeredgewidth": 1.2,
                "markeredgecolor": "white",
                "linestyle": "none",
                "zorder": 6,
            },
        )


def _hsl_plot_failure_markers(
    ax_vr, ax_wt, d: dict, color: str, jx: float | None, wt_display
) -> None:
    fail_ft = d["fail_ft"]
    fm = _HSL_FAILURE_MARKER.get(fail_ft, "D")
    mk_kw = {
        "marker": fm,
        "color": color,
        "markersize": 9,
        "markeredgewidth": 1.2,
        "markeredgecolor": "white",
        "linestyle": "none",
        "zorder": 6,
    }
    if not d["cpu_only"]:
        ax_vr.loglog([jx], [max(d["fail_vram"], 1)], **mk_kw)
    ax_wt.semilogx([jx], [wt_display(np.log10(max(d["fail_wall"], 1e-10)))], **mk_kw)


def _hsl_plot_solvers(
    axes: tuple,
    solver_data: dict[str, dict],
    ordered: list[str],
    jitter_x: dict[tuple[str, int], float],
    gn_display,
    wt_display,
) -> set[str]:
    ax_vr, ax_wt, ax_gn = axes
    failure_types_seen: set[str] = set()
    for solver in ordered:
        d = solver_data[solver]
        label, color, ls, _ = _hsl_solver_style(solver)
        fail_step = d["fail_step"]
        jx = jitter_x.get((solver, fail_step)) if fail_step is not None else None

        kw = {
            "color": color,
            "linestyle": ls,
            "marker": "o",
            "markersize": 4,
            "markeredgewidth": 0,
            "linewidth": 1.6,
            "label": label,
            "zorder": 3,
        }
        kw_line = {
            "color": color,
            "linestyle": ls,
            "marker": "none",
            "linewidth": 1.6,
            "zorder": 3,
        }

        if d["ok_steps"]:
            _hsl_plot_vram_panel(ax_vr, d, kw, kw_line)
            _hsl_plot_wt_panel(ax_wt, d, kw, kw_line, wt_display)
            _hsl_plot_gn_panel(ax_gn, d, kw, kw_line, gn_display, jx)
        else:
            for ax in (ax_vr, ax_wt, ax_gn):
                ax.loglog([], [], **kw)

        if fail_step is not None:
            _hsl_plot_failure_markers(ax_vr, ax_wt, d, color, jx, wt_display)
            failure_types_seen.add(d["fail_ft"])
    return failure_types_seen


def _hsl_plot_openfoam_fd(ax_wt, of_fd: dict[int, float], wt_display) -> None:
    if not of_fd:
        return
    of_steps = sorted(of_fd)
    of_disp = [wt_display(np.log10(max(of_fd[s], 1e-10))) for s in of_steps]
    ax_wt.semilogx(
        of_steps,
        of_disp,
        color=_HSL_OF_COLOR,
        linestyle=_HSL_OF_LS,
        marker="h",
        markersize=4,
        markeredgewidth=0,
        linewidth=1.6,
        label="OpenFOAM (FD est.)",
        zorder=3,
    )


def _hsl_decorate_vram_panel(ax_vr) -> None:
    ax_vr.axhline(
        _HSL_VRAM_LIMIT_MIB, color="0.35", linestyle="--", linewidth=1.0, zorder=2
    )
    trans = blended_transform_factory(ax_vr.transAxes, ax_vr.transData)
    ax_vr.text(
        0.28,
        _HSL_VRAM_LIMIT_MIB,
        "16 GiB",
        transform=trans,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="0.35",
        clip_on=True,
    )
    ax_vr.set_title("Peak (V)RAM")
    ax_vr.set_xlabel("Rollout steps $T$")
    ax_vr.set_ylabel("MiB")


def _hsl_set_panel_titles(ax_wt, ax_gn) -> None:
    ax_wt.set_title("Wall time")
    ax_wt.set_xlabel("Rollout steps $T$")
    ax_wt.set_ylabel("Seconds")

    ax_gn.set_title("Gradient norm")
    ax_gn.set_xlabel("Rollout steps $T$")
    ax_gn.set_ylabel(r"$\|\nabla\mathcal{L}\|$")


def _hsl_set_wt_yticks(
    ax_wt, max_wt_log10: float, wt_upper_factor: float, wt_upper_data: float
) -> None:
    wt_below_ticks = [v for v in [0, 1, 2] if v >= _HSL_WT_YMIN]
    wt_above_ticks = [v for v in [4, 6, 8] if v <= max_wt_log10 + 0.5]
    wt_above_disp = [
        _HSL_WT_BREAK_LOG + (v - _HSL_WT_BREAK_LOG) * wt_upper_factor
        for v in wt_above_ticks
    ]
    ax_wt.set_yticks(list(wt_below_ticks) + wt_above_disp)
    ax_wt.set_yticklabels([rf"$10^{{{v}}}$" for v in wt_below_ticks + wt_above_ticks])
    wt_ymax = _HSL_WT_BREAK_LOG + wt_upper_data * wt_upper_factor
    ax_wt.set_ylim(_HSL_WT_YMIN, wt_ymax + 0.1)


def _hsl_set_gn_yticks(ax_gn, gn_display, max_log10: float) -> None:
    lower_ticks = [v for v in [1] if _HSL_GN_YMIN <= v < _HSL_GN_LOWER_BREAK]
    middle_ticks = [2, 3]
    above_ticks = [v for v in [5, 7, 9] if v <= max_log10 + 0.5]
    all_gn_ticks = lower_ticks + middle_ticks + above_ticks
    all_gn_disp = [gn_display(v) for v in all_gn_ticks]
    ax_gn.set_yticks(all_gn_disp)
    ax_gn.set_yticklabels([rf"$10^{{{v}}}$" for v in all_gn_ticks])
    gn_ymax = gn_display(max_log10)
    ax_gn.set_ylim(_HSL_GN_YMIN, gn_ymax + 0.05)


def _hsl_set_piecewise_x_axis(ax_vr, ax_wt, ax_gn, all_sweep_steps: list[int]) -> None:
    x_min_data = min(all_sweep_steps) if all_sweep_steps else 1
    x_max_data = max(all_sweep_steps) if all_sweep_steps else 1e4
    x_pad = 0.06 * (np.log10(x_max_data) - np.log10(x_min_data))
    x_lim_lo = 10 ** (np.log10(x_min_data) - x_pad)
    x_lim_hi = 10 ** (np.log10(x_max_data) + x_pad)
    for ax in (ax_wt, ax_gn):
        ax.set_xscale(
            "function",
            functions=(_hsl_x_log_forward, _hsl_x_log_inverse),
        )
        ax.set_xticks([10, 100, 1000, 10000])
        ax.set_xticklabels([r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$"])
        ax.axvline(
            10**_HSL_X_BREAK_LOG,
            color="0.7",
            linestyle=":",
            linewidth=0.6,
            zorder=0,
        )
        ax.set_xlim(x_lim_lo, x_lim_hi)
    ax_vr.set_xlim(x_lim_lo, x_lim_hi)


def _hsl_build_solver_handles(present: set[str], of_fd: dict[int, float]) -> list:
    dummy = mlines.Line2D(
        [], [], color="none", linestyle="none", marker="none", label=""
    )
    solver_handles = []
    for s in _HSL_SOLVER_ORDER:
        if s not in present:
            continue
        lb, co, li, _ = _hsl_solver_style(s)
        solver_handles.append(
            mlines.Line2D(
                [],
                [],
                color=co,
                linestyle=li,
                marker="o",
                markersize=5,
                markeredgewidth=0,
                linewidth=1.6,
                label=lb,
            )
        )
    if of_fd:
        solver_handles.append(
            mlines.Line2D(
                [],
                [],
                color=_HSL_OF_COLOR,
                linestyle=_HSL_OF_LS,
                marker="h",
                markersize=5,
                markeredgewidth=0,
                linewidth=1.6,
                label="OpenFOAM (FD est.)",
            )
        )
        solver_handles.append(dummy)
    return solver_handles


def _hsl_build_failure_handles(failure_types_seen: set[str]) -> list:
    failure_handles = []
    for ft in ["OOM", "nan", "error", "timeout"]:
        if ft in failure_types_seen:
            failure_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    marker=_HSL_FAILURE_MARKER[ft],
                    color="0.4",
                    linestyle="none",
                    markersize=7,
                    markeredgewidth=1.0,
                    markeredgecolor="white",
                    label=_HSL_FAILURE_LABEL[ft],
                )
            )
    return failure_handles


def _hsl_attach_legend(
    fig, present: set[str], of_fd: dict[int, float], failure_types_seen: set[str]
) -> None:
    solver_handles = _hsl_build_solver_handles(present, of_fd)
    failure_handles = _hsl_build_failure_handles(failure_types_seen)

    dummy = mlines.Line2D(
        [], [], color="none", linestyle="none", marker="none", label=""
    )
    if len(solver_handles) % 2 == 1:
        solver_handles.append(dummy)

    all_handles = solver_handles + failure_handles
    ncol = -(-len(all_handles) // 2)  # ceil → 2 rows
    fig.legend(
        handles=all_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=ncol,
        fontsize=7.5,
        framealpha=0.7,
        handlelength=2.0,
    )


def _hsl_save_figure(fig, out_dir: Path) -> None:
    for ext in ("pdf", "png"):
        out = out_dir / f"horizon_sweep_limits.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")


def _plot_horizon_sweep_limits(cfg: Problem, **_kw):
    """``_extra/horizon_sweep_limits`` — VJP rollout-length limit figure."""
    out_dir = _extra_out_dir(cfg)
    path = (
        results_dir()
        / "ns-3d-grid"
        / "gradient"
        / "horizon_sweep_limits"
        / "result.json"
    )
    if not path.exists():
        print(f"[horizon_sweep_limits] {path} not found — skipping")
        return None
    data = load_json(path)
    # ``by_solver`` from result.json is keyed by spec.name (display form).
    # Re-key to canonical aliases up-front so downstream helpers (which
    # compare against _HSL_SOLVER_ORDER / SOLVER_STYLES, both alias-keyed)
    # match correctly. Drop unresolved entries silently — they would not
    # appear in any ordering list anyway.
    _raw_by_solver = data["by_solver"]
    by_solver: dict = {}
    for display_name, sv in _raw_by_solver.items():
        a = resolve_solver_alias(display_name)
        by_solver[a if a is not None else display_name] = sv

    with plt.rc_context(PAPER_RCPARAMS):
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.27), dpi=300)
        gs = GridSpec(1, 3, figure=fig)
        gs.update(hspace=0.38, wspace=0.5, bottom=0.19, top=0.93, left=0.09, right=0.97)
        ax_vr = fig.add_subplot(gs[0, 0])
        ax_wt = fig.add_subplot(gs[0, 1])
        ax_gn = fig.add_subplot(gs[0, 2])

        present, ordered = _hsl_order_solvers(by_solver)

        all_sweep_steps = sorted(
            {int(k) for sv in data["by_solver"].values() for k in sv}
        )
        of_fd = _hsl_openfoam_fd_vjp_estimate(all_sweep_steps, N_sweep=20)

        solver_data = _hsl_parse_all_solvers(by_solver, ordered)

        gn_display, max_log10 = _hsl_build_gn_scale(solver_data)
        wt_display, max_wt_log10, wt_upper_factor, wt_upper_data = _hsl_build_wt_scale(
            solver_data, of_fd
        )

        jitter_x = _hsl_compute_jitter(solver_data, ordered)

        failure_types_seen = _hsl_plot_solvers(
            (ax_vr, ax_wt, ax_gn),
            solver_data,
            ordered,
            jitter_x,
            gn_display,
            wt_display,
        )

        _hsl_plot_openfoam_fd(ax_wt, of_fd, wt_display)

        _hsl_decorate_vram_panel(ax_vr)
        _hsl_set_panel_titles(ax_wt, ax_gn)

        _hsl_set_wt_yticks(ax_wt, max_wt_log10, wt_upper_factor, wt_upper_data)
        _hsl_set_gn_yticks(ax_gn, gn_display, max_log10)

        _hsl_set_piecewise_x_axis(ax_vr, ax_wt, ax_gn, all_sweep_steps)

        _hsl_attach_legend(fig, present, of_fd, failure_types_seen)

        _hsl_save_figure(fig, out_dir)
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# lid_cavity — 1×3 panels per sweep value (specialised recovery variant)
# ─────────────────────────────────────────────────────────────────────────────


_LC_SOLVER_ORDER = ["ins_jl", "exponax", "phiflow", "xlb", "warp_ns", "pict"]
_LC_SWEEP_VALS = ["0.5", "1.0", "2.0"]


def _plot_lid_cavity(cfg: Problem, **_kw) -> None:
    """``_extra/lid_cavity`` — 1×3 convergence figure for the lid-cavity sweep."""
    out_dir = _extra_out_dir(cfg)
    path = results_dir() / "ns-3d-grid" / "optimization" / "lid_cavity" / "result.json"
    if not path.exists():
        print(f"[lid_cavity] {path} not found — skipping")
        return
    with plt.rc_context(PAPER_RCPARAMS):
        data = load_json(path)
        by_sweep = data["by_sweep"]

        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.38))
        fig.subplots_adjust(bottom=0.22, wspace=0.35)

        present: set[str] = set()

        for col, sv in enumerate(_LC_SWEEP_VALS):
            ax = axes[col]

            for display_name, sweep_data in by_sweep.items():
                alias = resolve_solver_alias(display_name)
                if alias in {"fenics_ns", "su2"} or display_name in {
                    "fenics_ns",
                    "su2",
                }:
                    continue
                if sv not in sweep_data:
                    continue
                _label, color, ls, _mk = SOLVER_STYLES.get(
                    alias or display_name, (display_name, "#888888", "-", "o")
                )

                losses = sweep_data[sv]["losses"]
                iters = list(range(len(losses)))
                kw = {"color": color, "linestyle": ls, "marker": "", "linewidth": 1.6}
                ax.semilogy(iters, losses, **kw)
                if alias is not None:
                    present.add(alias)

            ax.set_title(f"$U_x^\\mathrm{{true}} = {sv}$")
            ax.set_xlabel("Iteration")
            if col == 0:
                ax.set_ylabel("Loss")

        handles = [
            mlines.Line2D(
                [],
                [],
                color=SOLVER_STYLES[s][1],
                linestyle=SOLVER_STYLES[s][2],
                linewidth=1.6,
                label=SOLVER_STYLES[s][0],
            )
            for s in _LC_SOLVER_ORDER
            if s in present
        ]

        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            fontsize=7.5,
            framealpha=0.7,
            handlelength=2.0,
        )

        out = out_dir / "lid_cavity_convergence.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# cost_overview — single-column per-N forward / VJP cost figure
# ─────────────────────────────────────────────────────────────────────────────


def _plot_cost_overview(cfg: Problem, **_kw) -> None:
    plot_cost_overview_for(cfg, steady_state=False)


# ─────────────────────────────────────────────────────────────────────────────
# scaling — 3D NS forward / VJP / ratio
# ─────────────────────────────────────────────────────────────────────────────


def _n_to_elements_3d(N: int) -> int:
    return N**3


def _scaling_extract(by_n: dict) -> dict[int, float]:
    out = {}
    for k, v in by_n.items():
        if v is not None and isinstance(v, dict) and v.get("mean") is not None:
            out[int(k)] = float(v["mean"])
    return out


def _scaling_load_cost(experiment: str) -> dict[str, dict[int, float]]:
    p = results_dir() / "ns-3d-grid" / "cost" / experiment / "result.json"
    if not p.exists():
        return {}
    data = load_json(p)
    return {s: _scaling_extract(nd) for s, nd in data.get("by_N", {}).items()}


def _plot_scaling(cfg: Problem, **_kw) -> None:
    out_dir = _extra_out_dir(cfg)
    plt.rcParams.update(PAPER_RCPARAMS)

    fwd_data = _scaling_load_cost("spatial_cost")
    vjp_data = _scaling_load_cost("vjp_cost")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(TEXTWIDTH, TEXTWIDTH * 0.34),
        sharey=False,
        dpi=300,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.36, top=0.88, wspace=0.40)
    ax_fwd, ax_vjp, ax_ratio = axes

    all_els: set[int] = set()
    seen: set[str] = set()

    # ``fwd_data`` / ``vjp_data`` are keyed by spec.name (display form).
    _display_names = set(fwd_data) | set(vjp_data)
    alias_to_display: dict[str, str] = {}
    for display_name in _display_names:
        a = resolve_solver_alias(display_name)
        if a is not None:
            alias_to_display[a] = display_name

    for alias in NS_ORDER:
        display_name = alias_to_display.get(alias)
        if display_name is None:
            continue
        fwd_pts = fwd_data.get(display_name, {})
        vjp_pts = vjp_data.get(display_name, {})
        if not fwd_pts and not vjp_pts:
            continue

        _label, color, ls, mk = solver_props(alias)
        kw = {
            "color": color,
            "linestyle": ls,
            "marker": mk,
            "markersize": 4,
            "markeredgewidth": 0,
            "linewidth": 1.5,
        }

        if fwd_pts:
            ns_f = sorted(fwd_pts)
            els_f = [_n_to_elements_3d(n) for n in ns_f]
            ax_fwd.loglog(els_f, [fwd_pts[n] for n in ns_f], **kw)
            all_els.update(els_f)

        if vjp_pts:
            ns_v = sorted(vjp_pts)
            els_v = [_n_to_elements_3d(n) for n in ns_v]
            ax_vjp.loglog(els_v, [vjp_pts[n] for n in ns_v], **kw)
            all_els.update(els_v)

        common_ns = sorted(set(fwd_pts) & set(vjp_pts))
        if len(common_ns) >= 2:
            els_c = [_n_to_elements_3d(n) for n in common_ns]
            ratios = [vjp_pts[n] / fwd_pts[n] for n in common_ns]
            ax_ratio.loglog(els_c, ratios, **kw)
            all_els.update(els_c)

        seen.add(alias)

    ax_ratio.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, zorder=0)

    ax_fwd.set_title("Forward time")
    ax_vjp.set_title("VJP time")
    ax_ratio.set_title("VJP / forward")
    ax_fwd.set_ylabel("3D NS\nTime (s)", fontsize=7.5)
    for ax in axes:
        ax.set_xlabel("DOFs", fontsize=7.5)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    tick_els = sorted(all_els)
    if len(tick_els) > 4:
        idx = np.round(np.linspace(0, len(tick_els) - 1, 4)).astype(int)
        tick_els = [tick_els[i] for i in idx]

    fmt = mticker.FuncFormatter(
        lambda x, _: f"{round(x / 1000):.0f}k" if x >= 1000 else str(int(x))
    )
    for ax in axes:
        ax.set_xticks(tick_els)
        ax.xaxis.set_major_formatter(fmt)
        ax.tick_params(axis="x", labelsize=7, rotation=35)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.tick_params(axis="y", labelsize=7)

    handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in seen])
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        fontsize=6.5,
        framealpha=0.8,
        handlelength=2.0,
        borderpad=0.5,
        labelspacing=0.3,
    )

    out = out_dir / "scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ucurves — F3 (3D NS) FD U-curves
# ─────────────────────────────────────────────────────────────────────────────


def _plot_ucurve_3d(cfg_dict: dict, out_dir: Path) -> None:
    path: Path = cfg_dict["path"]
    if not path.exists():
        print(f"[ucurves] {path} not found — skipping")
        return

    data = load_json(path)
    by_solver: dict = data["by_solver"]

    all_steps: list[int] = sorted(
        {int(s) for sv in by_solver.values() for s in sv},
        key=int,
    )
    ncols: int = cfg_dict["ncols"]
    nrows: int = int(np.ceil(len(all_steps) / ncols))

    panel_w = TEXTWIDTH / ncols
    panel_h = panel_w * 0.92
    fig_h = nrows * panel_h + 0.55

    fig = plt.figure(figsize=(TEXTWIDTH, fig_h))
    gs = gridspec.GridSpec(
        nrows,
        ncols,
        figure=fig,
        left=0.10,
        right=0.98,
        top=1.0 - 0.12 / fig_h,
        bottom=0.52 / fig_h,
        hspace=0.65,
        wspace=0.40,
    )

    seen: set[str] = set()

    # ``by_solver`` is keyed by spec.name (display form); build alias→display.
    alias_to_display: dict[str, str] = {}
    for display_name in by_solver:
        a = resolve_solver_alias(display_name)
        if a is not None:
            alias_to_display[a] = display_name

    for idx, steps in enumerate(all_steps):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        for alias in NS_ORDER:
            display_name = alias_to_display.get(alias)
            if display_name is None:
                continue
            sv = by_solver.get(display_name)
            if sv is None:
                continue
            entry = sv.get(str(steps))
            if entry is None:
                continue
            eps_sweep: dict = entry.get("eps_sweep", {})
            if not eps_sweep:
                continue

            eps_vals = sorted(eps_sweep.keys(), key=float)
            xs = [float(e) for e in eps_vals]
            ys = [eps_sweep[e]["rel_error_mean"] for e in eps_vals]

            if not all(np.isfinite(y) and y > 0 for y in ys):
                pairs = [
                    (x, y)
                    for x, y in zip(xs, ys, strict=False)
                    if np.isfinite(y) and y > 0
                ]
                if not pairs:
                    continue
                xs, ys = zip(*pairs, strict=False)

            _, color, ls, mk = solver_props(alias)
            ax.loglog(
                xs,
                ys,
                color=color,
                linestyle=ls,
                marker=mk,
                markersize=3.5,
                markeredgewidth=0,
                linewidth=1.4,
            )
            seen.add(alias)

        ax.set_title(f"$T={steps}$", fontsize=8)
        ax.set_xlabel(r"$\varepsilon$", fontsize=7.5)
        if col == 0:
            ax.set_ylabel("Rel. FD error", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    for idx in range(len(all_steps), nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    handles = dedup_handles([make_handle(s) for s in NS_ORDER if s in seen])
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(handles), 6) if handles else 1,
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="0.8",
        handlelength=2.0,
    )

    out = out_dir / cfg_dict["out"]
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def _plot_ucurves(cfg: Problem, **_kw) -> None:
    out_dir = _extra_out_dir(cfg)
    cfg_dict = {
        "path": results_dir()
        / "ns-3d-grid"
        / "gradient"
        / "horizon_sweep"
        / "result.json",
        "out": "ucurves.pdf",
        "ncols": 5,
    }
    with paper_rc_context():
        _plot_ucurve_3d(cfg_dict, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────


def register(problem: Problem) -> None:
    """Register all paper-figure extras as ``_extra/<key>`` plot fns."""
    problem.add_extra_plot("_extra/recovery_overview", _plot_recovery_overview)
    problem.add_extra_plot("_extra/grad_divergence", _plot_grad_divergence)
    problem.add_extra_plot("_extra/horizon_sweep_limits", _plot_horizon_sweep_limits)
    problem.add_extra_plot("_extra/lid_cavity", _plot_lid_cavity)
    problem.add_extra_plot("_extra/cost_overview", _plot_cost_overview)
    problem.add_extra_plot("_extra/scaling", _plot_scaling)
    problem.add_extra_plot("_extra/ucurves", _plot_ucurves)
