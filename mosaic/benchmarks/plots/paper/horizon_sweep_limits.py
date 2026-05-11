"""Generate Figure: VJP rollout-length limit sweep for ns-3d-grid.

2-row layout:
  top-left    — peak (V)RAM (MiB) vs rollout steps  (log-log)
  top-right   — VJP wall time (s)  vs rollout steps  (log-log)
  bottom      — gradient norm vs rollout steps (log-log)

Failure modes are shown with large open markers at the failing step:
  OOM           → down-triangle  (▼)
  NaN gradient  → ×  (x-marker)
  error         → diamond  (◆)
When multiple solvers fail at the same step, markers are jittered in
log-space (±JITTER_LOG log10 units) so all symbols remain visible.
A dashed horizontal line at 16 384 MiB marks the single V100 VRAM limit;
its label is placed inline at the right edge of the (V)RAM axis.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory

from mosaic.benchmarks.core.utils import results_dir
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import RCPARAMS, SOLVER_STYLES

SOLVER_ORDER = [
    "phiflow",
    "xlb",
    "pict",
    "warp_ns",
    "exponax",
    "ins_jl",
]
_OF_COLOR = SOLVER_STYLES.get("openfoam", ("OpenFOAM", "#DDCC77", "--", "h"))[1]
_OF_LS = SOLVER_STYLES.get("openfoam", ("OpenFOAM", "#DDCC77", "--", "h"))[2]

_FAILURE_MARKER = {"OOM": "v", "nan": "X", "error": "D", "timeout": "s"}
_FAILURE_LABEL = {
    "OOM": "OOM (VRAM)",
    "nan": "NaN gradient",
    "error": "error",
    "timeout": "timeout",
}
_VRAM_LIMIT_MIB = 16_384


# Gradient norm panel: 3-section piecewise y-scale.
# [_GN_YMIN, _GN_LOWER_BREAK]: compressed (bottom noise floor)
# [_GN_LOWER_BREAK, _BREAK_LOG]: fine-grained (region of interest)
# above _BREAK_LOG: compressed (exploding gradients)
_GN_YMIN = 1.0  # bottom of y-axis (log10 units, i.e. start at 10)
_GN_LOWER_BREAK = 1.6  # start of fine-grained region
_GN_LOWER_FACTOR = 0.15  # compression factor for bottom region
_GN_MIDDLE_FACTOR = 4.0  # stretch factor for fine-grained middle region
_BREAK_LOG = 3.0  # end of fine-grained region

_WT_YMIN = 0.0  # bottom of wall time y-axis (log10 seconds, i.e. start at 1 s)
_WT_BREAK_LOG = 2.0  # log10(seconds) transition point (~100 s)

# Piecewise log x-axis (Rollout steps T) for wall-time and gradient-norm panels:
# normal log up to 10**_X_BREAK_LOG (~316), compressed log above so that
# very-long-rollout failures don't push the points of interest into a corner.
_X_BREAK_LOG = 2.5
_X_UPPER_FACTOR = 0.4


def _x_log_forward(steps):
    steps = np.asarray(steps, dtype=float)
    log_x = np.log10(np.maximum(steps, 1e-10))
    return np.where(
        log_x <= _X_BREAK_LOG,
        log_x,
        _X_BREAK_LOG + (log_x - _X_BREAK_LOG) * _X_UPPER_FACTOR,
    )


def _x_log_inverse(disp):
    disp = np.asarray(disp, dtype=float)
    log_x = np.where(
        disp <= _X_BREAK_LOG,
        disp,
        _X_BREAK_LOG + (disp - _X_BREAK_LOG) / _X_UPPER_FACTOR,
    )
    return np.power(10.0, log_x)


# Half-spread in log10 units for jittering coincident failure markers.
# 0.04 → factor of ~1.10, so 3 markers at step=160 land at ≈145, 160, 176.
_JITTER_LOG = 0.04


def _solver_style(name: str) -> tuple:
    return SOLVER_STYLES.get(name, (name, "#888888", "-", "o"))


def _openfoam_fd_vjp_estimate(
    sweep_steps: list[int], N_sweep: int = 20
) -> dict[int, float]:
    """Estimate FD VJP wall time for OpenFOAM: N_inputs × forward_cost(T).

    Fits a linear model to temporal_cost at N=16, scales the per-step
    coefficient to N_sweep via N³ volume scaling.  Startup overhead is kept
    constant (file I/O is N-independent).
    N_inputs = N_sweep³ × 3  (one FD perturbation per IC velocity component).
    """
    _temporal_cost_path = (
        results_dir() / "ns-3d-grid" / "cost" / "temporal_cost" / "result.json"
    )
    if not _temporal_cost_path.exists():
        return {}
    td = json.loads(_temporal_cost_path.read_text())
    of_data = td.get("by_steps", {}).get("openfoam", {})
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


def generate(out_dir: Path) -> None:
    path = (
        results_dir()
        / "ns-3d-grid"
        / "gradient"
        / "horizon_sweep_limits"
        / "result.json"
    )
    data = json.loads(path.read_text())
    by_solver = data["by_solver"]

    with plt.rc_context(RCPARAMS):
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.27), dpi=300)
        gs = GridSpec(1, 3, figure=fig)
        gs.update(hspace=0.38, wspace=0.5, bottom=0.19, top=0.93, left=0.09, right=0.97)
        ax_vr = fig.add_subplot(gs[0, 0])
        ax_wt = fig.add_subplot(gs[0, 1])
        ax_gn = fig.add_subplot(gs[0, 2])

        failure_types_seen: set[str] = set()

        _EXCLUDED = {"fenics_ns", "fenics_ns_3d", "su2"}
        present = set(by_solver.keys())
        ordered = [s for s in SOLVER_ORDER if s in present] + [
            s for s in present if s not in SOLVER_ORDER and s not in _EXCLUDED
        ]

        # ── Phase 0: pre-compute OpenFOAM FD estimate ────────────────────────
        _all_sweep_steps = sorted(
            {int(k) for sv in data["by_solver"].values() for k in sv}
        )
        _of_fd = _openfoam_fd_vjp_estimate(_all_sweep_steps, N_sweep=20)

        # ── Phase 1: parse all solver data ───────────────────────────────────
        solver_data: dict[str, dict] = {}
        for solver in ordered:
            step_results = by_solver[solver]
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

            solver_data[solver] = {
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

        # ── Phase 1b: piecewise y-scale parameters ────────────────────────────
        _all_log10 = [
            np.log10(max(g, 1e-30)) for d in solver_data.values() for g in d["ok_gnorm"]
        ]
        _max_log10 = max(_all_log10) if _all_log10 else _BREAK_LOG + 1.0
        _upper_data = max(_max_log10 - _BREAK_LOG, 0.1)
        _middle_height = _BREAK_LOG - _GN_LOWER_BREAK
        _middle_disp = _middle_height * _GN_MIDDLE_FACTOR
        _upper_factor = (_middle_disp / 3.0) / _upper_data
        _lower_disp_height = (_GN_LOWER_BREAK - _GN_YMIN) * _GN_LOWER_FACTOR
        _gn_lower_break_disp = _GN_YMIN + _lower_disp_height
        _gn_upper_break_disp = _gn_lower_break_disp + _middle_disp

        def _gn_display(v: float) -> float:
            if v <= _GN_LOWER_BREAK:
                return _GN_YMIN + (v - _GN_YMIN) * _GN_LOWER_FACTOR
            elif v <= _BREAK_LOG:
                return _gn_lower_break_disp + (v - _GN_LOWER_BREAK) * _GN_MIDDLE_FACTOR
            else:
                return _gn_upper_break_disp + (v - _BREAK_LOG) * _upper_factor

        _all_wt_log10 = [
            np.log10(max(t, 1e-10)) for d in solver_data.values() for t in d["ok_wall"]
        ]
        if _of_fd:
            _all_wt_log10 += [np.log10(max(c, 1e-10)) for c in _of_fd.values()]
        _max_wt_log10 = max(_all_wt_log10) if _all_wt_log10 else _WT_BREAK_LOG + 1.0
        _wt_lower_height = _WT_BREAK_LOG - _WT_YMIN
        _wt_upper_data = max(_max_wt_log10 - _WT_BREAK_LOG, 0.1)
        _wt_upper_factor = (_wt_lower_height / 3.0) / _wt_upper_data

        def _wt_display(v: float) -> float:
            if v <= _WT_BREAK_LOG:
                return v
            return _WT_BREAK_LOG + (v - _WT_BREAK_LOG) * _wt_upper_factor

        # ── Phase 2: compute log-space jitter for coincident failure steps ────
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
                    log_off = (2 * t - 1) * _JITTER_LOG
                    jitter_x[(sv, step)] = step * 10**log_off

        # ── Phase 3: plot ────────────────────────────────────────────────────
        for solver in ordered:
            d = solver_data[solver]
            label, color, ls, _ = _solver_style(solver)

            ok_steps = d["ok_steps"]
            ok_vram = d["ok_vram"]
            ok_wall = d["ok_wall"]
            ok_gnorm = d["ok_gnorm"]
            ok_ram = d["ok_ram"]
            cpu_only = d["cpu_only"]
            has_ram = d["has_ram"]
            fail_step = d["fail_step"]
            fail_vram = d["fail_vram"]
            fail_wall = d["fail_wall"]
            fail_ram = d["fail_ram"]
            fail_ft = d["fail_ft"]

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

            if ok_steps:
                # ── (V)RAM ───────────────────────────────────────────────────
                if not cpu_only:
                    ax_vr.loglog(ok_steps, [max(v, 1) for v in ok_vram], **kw)
                    if fail_step:
                        ax_vr.loglog(
                            [ok_steps[-1], fail_step],
                            [max(ok_vram[-1], 1), max(fail_vram, 1)],
                            **kw_line,
                        )
                elif has_ram:
                    ax_vr.loglog(ok_steps, [max(r, 1) for r in ok_ram], **kw)
                    if fail_step and fail_ram:
                        ax_vr.loglog(
                            [ok_steps[-1], fail_step],
                            [max(ok_ram[-1], 1), max(fail_ram, 1)],
                            **kw_line,
                        )
                else:
                    ax_vr.loglog([], [], **kw)

                # ── Wall time (piecewise-scaled y) ────────────────────────────
                log_wt = [np.log10(max(t, 1e-10)) for t in ok_wall]
                disp_wt = [_wt_display(v) for v in log_wt]
                ax_wt.semilogx(ok_steps, disp_wt, **kw)
                if fail_step:
                    _last_wt_disp = _wt_display(np.log10(max(ok_wall[-1], 1e-10)))
                    _fail_wt_disp = _wt_display(np.log10(max(fail_wall, 1e-10)))
                    ax_wt.semilogx(
                        [ok_steps[-1], fail_step],
                        [_last_wt_disp, _fail_wt_disp],
                        **kw_line,
                    )

                # ── Gradient norm (piecewise-scaled y) ────────────────────────
                log_gnorm = [np.log10(max(g, 1e-30)) for g in ok_gnorm]
                disp_gnorm = [_gn_display(v) for v in log_gnorm]
                ax_gn.semilogx(ok_steps, disp_gnorm, **kw)
                if fail_step and ok_gnorm:
                    _last_disp = _gn_display(np.log10(max(ok_gnorm[-1], 1e-30)))
                    ax_gn.semilogx(
                        [ok_steps[-1], fail_step], [_last_disp, _last_disp], **kw_line
                    )
                    ax_gn.semilogx(
                        [jx],
                        [_last_disp],
                        **{
                            **kw,
                            "marker": _FAILURE_MARKER.get(fail_ft, "D"),
                            "markersize": 9,
                            "markeredgewidth": 1.2,
                            "markeredgecolor": "white",
                            "linestyle": "none",
                            "zorder": 6,
                        },
                    )
            else:
                for ax in (ax_vr, ax_wt, ax_gn):
                    ax.loglog([], [], **kw)

            # ── Failure markers on VRAM and wall-time axes ────────────────────
            if fail_step is not None:
                fm = _FAILURE_MARKER.get(fail_ft, "D")
                mk_kw = {
                    "marker": fm,
                    "color": color,
                    "markersize": 9,
                    "markeredgewidth": 1.2,
                    "markeredgecolor": "white",
                    "linestyle": "none",
                    "zorder": 6,
                }
                if not cpu_only:
                    ax_vr.loglog([jx], [max(fail_vram, 1)], **mk_kw)
                ax_wt.semilogx(
                    [jx], [_wt_display(np.log10(max(fail_wall, 1e-10)))], **mk_kw
                )
                failure_types_seen.add(fail_ft)

        # ── OpenFOAM FD VJP estimate — wall-time panel only ──────────────────
        if _of_fd:
            _of_steps = sorted(_of_fd)
            _of_disp = [_wt_display(np.log10(max(_of_fd[s], 1e-10))) for s in _of_steps]
            ax_wt.semilogx(
                _of_steps,
                _of_disp,
                color=_OF_COLOR,
                linestyle=_OF_LS,
                marker="h",
                markersize=4,
                markeredgewidth=0,
                linewidth=1.6,
                label="OpenFOAM (FD est.)",
                zorder=3,
            )

        # ── 16 GiB limit line ─────────────────────────────────────────────────
        ax_vr.axhline(
            _VRAM_LIMIT_MIB, color="0.35", linestyle="--", linewidth=1.0, zorder=2
        )
        _trans = blended_transform_factory(ax_vr.transAxes, ax_vr.transData)
        ax_vr.text(
            0.28,
            _VRAM_LIMIT_MIB,
            "16 GiB",
            transform=_trans,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="0.35",
            clip_on=True,
        )

        ax_vr.set_title("Peak (V)RAM")
        ax_vr.set_xlabel("Rollout steps $T$")
        ax_vr.set_ylabel("MiB")

        ax_wt.set_title("Wall time")
        ax_wt.set_xlabel("Rollout steps $T$")
        ax_wt.set_ylabel("Seconds")

        ax_gn.set_title("Gradient norm")
        ax_gn.set_xlabel("Rollout steps $T$")
        ax_gn.set_ylabel(r"$\|\nabla\mathcal{L}\|$")

        # ── Wall time y-axis ticks ────────────────────────────────────────────
        _wt_below_ticks = [v for v in [0, 1, 2] if v >= _WT_YMIN]
        _wt_above_ticks = [v for v in [4, 6, 8] if v <= _max_wt_log10 + 0.5]
        _wt_above_disp = [
            _WT_BREAK_LOG + (v - _WT_BREAK_LOG) * _wt_upper_factor
            for v in _wt_above_ticks
        ]
        ax_wt.set_yticks(list(_wt_below_ticks) + _wt_above_disp)
        ax_wt.set_yticklabels(
            [rf"$10^{{{v}}}$" for v in _wt_below_ticks + _wt_above_ticks]
        )
        _wt_ymax = _WT_BREAK_LOG + _wt_upper_data * _wt_upper_factor
        ax_wt.set_ylim(_WT_YMIN, _wt_ymax + 0.1)

        # ── Gradient norm y-axis ticks ────────────────────────────────────────
        _lower_ticks = [v for v in [1] if _GN_YMIN <= v < _GN_LOWER_BREAK]
        _middle_ticks = [2, 3]
        _above_ticks = [v for v in [5, 7, 9] if v <= _max_log10 + 0.5]
        _all_gn_ticks = _lower_ticks + _middle_ticks + _above_ticks
        _all_gn_disp = [_gn_display(v) for v in _all_gn_ticks]
        ax_gn.set_yticks(_all_gn_disp)
        ax_gn.set_yticklabels([rf"$10^{{{v}}}$" for v in _all_gn_ticks])
        _gn_ymax = _gn_display(_max_log10)
        ax_gn.set_ylim(_GN_YMIN, _gn_ymax + 0.05)

        # ── Piecewise log x-axis on wall-time and gradient-norm panels ────────
        _x_min_data = min(_all_sweep_steps) if _all_sweep_steps else 1
        _x_max_data = max(_all_sweep_steps) if _all_sweep_steps else 1e4
        # Small log-space padding so tick labels at the edges aren't clipped.
        _x_pad = 0.06 * (np.log10(_x_max_data) - np.log10(_x_min_data))
        _x_lim_lo = 10 ** (np.log10(_x_min_data) - _x_pad)
        _x_lim_hi = 10 ** (np.log10(_x_max_data) + _x_pad)
        for _ax in (ax_wt, ax_gn):
            _ax.set_xscale(
                "function",
                functions=(_x_log_forward, _x_log_inverse),
            )
            _ax.set_xticks([10, 100, 1000, 10000])
            _ax.set_xticklabels([r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$"])
            _ax.axvline(
                10**_X_BREAK_LOG,
                color="0.7",
                linestyle=":",
                linewidth=0.6,
                zorder=0,
            )
            _ax.set_xlim(_x_lim_lo, _x_lim_hi)
        # Match VRAM panel x-range to actual data, too.
        ax_vr.set_xlim(_x_lim_lo, _x_lim_hi)

        # ── Legend ────────────────────────────────────────────────────────────
        _dummy = mlines.Line2D(
            [], [], color="none", linestyle="none", marker="none", label=""
        )
        solver_handles = []
        for s in SOLVER_ORDER:
            if s not in present:
                continue
            lb, co, li, _ = _solver_style(s)
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
        if _of_fd:
            solver_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=_OF_COLOR,
                    linestyle=_OF_LS,
                    marker="h",
                    markersize=5,
                    markeredgewidth=0,
                    linewidth=1.6,
                    label="OpenFOAM (FD est.)",
                )
            )
            solver_handles.append(_dummy)

        failure_handles = []
        for ft in ["OOM", "nan", "error", "timeout"]:
            if ft in failure_types_seen:
                failure_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        marker=_FAILURE_MARKER[ft],
                        color="0.4",
                        linestyle="none",
                        markersize=7,
                        markeredgewidth=1.0,
                        markeredgecolor="white",
                        label=_FAILURE_LABEL[ft],
                    )
                )

        # matplotlib fills legends column-first: entries 2k and 2k+1 share a column.
        # Pad solver handles to an even count so OOM and NaN (consecutive) land
        # in the same column on different rows → vertically aligned.
        _dummy = mlines.Line2D(
            [], [], color="none", linestyle="none", marker="none", label=""
        )
        if len(solver_handles) % 2 == 1:
            solver_handles.append(_dummy)

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

        for ext in ("pdf", "png"):
            out = out_dir / f"horizon_sweep_limits.{ext}"
            fig.savefig(out)
            print(f"Saved {out}")
        plt.close(fig)
    return fig
