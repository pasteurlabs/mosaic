"""Plots for the cost suite (forward and VJP wall-clock timing + peak (V)RAM)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import load_json, results_dir
from mosaic.benchmarks.problems.shared.plots.style import (
    apply_style,
    fig_shared_legend,
    save_fig,
    solver_plot_props,
    solver_styles,
)

apply_style()

_SUITE = "cost"

_FAILURE_MARKER = {
    "OOM": "v",
    "nan": "X",
    "error": "D",
    "timeout": "s",
    "container_died": "D",
}
_FAILURE_LABEL = {
    "OOM": "OOM (VRAM)",
    "nan": "NaN gradient",
    "error": "error",
    "timeout": "timeout",
    "container_died": "error",
}


def _hardware_str(result: dict) -> str:
    hw = result.get("hardware", {})
    parts = []
    gpus = hw.get("gpus")
    if gpus:
        parts.append(gpus[0].split(",")[0].strip())
    cpu = hw.get("cpu", "")
    if cpu:
        parts.append(cpu)
    ram = hw.get("ram_gb")
    if ram:
        parts.append(f"{ram} GB RAM")
    return "  |  ".join(parts)


def _time_vals(row: dict, keys: list) -> list[float]:
    return [(row.get(k) or {}).get("mean", np.nan) for k in keys]


def _mem_vals(row: dict, keys: list) -> list[float]:
    """Peak (V)RAM per key — VRAM if non-zero, else RAM, else NaN."""
    out = []
    for k in keys:
        v = row.get(k) or {}
        if not isinstance(v, dict):
            out.append(np.nan)
            continue
        vram = v.get("vram_peak_mib") or 0.0
        ram = v.get("ram_peak_mib") or 0.0
        mem = vram if vram > 50 else ram
        out.append(float(mem) if mem > 0 else np.nan)
    return out


def _first_failure(row: dict, keys: list) -> tuple[str | None, str | None]:
    """Return (key, failure_type) of the first failed entry, or (None, None)."""
    for k in keys:
        v = row.get(k)
        if isinstance(v, dict) and v.get("status") == "failed":
            return k, v.get("failure_type", "error")
    return None, None


def _draw_failure(
    ax,
    x_arr: np.ndarray,
    keys: list,
    fail_k: str,
    ys: list[float],
    color: str,
    ls: str,
    ft: str,
) -> None:
    """Connector from last ok point → fail N, then failure marker."""
    fail_i = list(keys).index(fail_k)
    fail_x = float(x_arr[fail_i])
    ok_pairs = [(float(x_arr[i]), v) for i, v in enumerate(ys) if np.isfinite(v)]
    if not ok_pairs:
        return
    lx, ly = ok_pairs[-1]
    fm = _FAILURE_MARKER.get(ft, "D")
    ax.loglog(
        [lx, fail_x],
        [ly, ly],
        color=color,
        linestyle=ls,
        linewidth=1.0,
        marker="none",
        zorder=3,
    )
    ax.loglog(
        [fail_x],
        [ly],
        marker=fm,
        color=color,
        markersize=9,
        markeredgewidth=1.2,
        markeredgecolor="white",
        linestyle="none",
        zorder=6,
    )


def _has_inner_data(top: dict) -> bool:
    if not isinstance(top, dict):
        return False
    return any(isinstance(v, dict) and v for v in top.values())


def _first_nonempty_keys(top: dict) -> list[str]:
    for v in top.values():
        if isinstance(v, dict) and v:
            return sorted(v.keys(), key=int)
    return []


def _load_cost_inputs(suite_dir, suffix: str):
    """Load the three result.json files; return (data_dicts, hw_str) or (None, "")."""
    spatial_path = suite_dir / f"spatial_cost{suffix}" / "result.json"
    temporal_path = suite_dir / f"temporal_cost{suffix}" / "result.json"
    vjp_path = suite_dir / f"vjp_cost{suffix}" / "result.json"

    has_spatial = spatial_path.exists()
    has_temporal = temporal_path.exists()
    has_vjp = vjp_path.exists()

    if not has_spatial and not has_temporal and not has_vjp:
        return None, ""

    spatial_data = load_json(spatial_path) if has_spatial else {}
    temporal_data = load_json(temporal_path) if has_temporal else {}
    vjp_data = load_json(vjp_path) if has_vjp else {}

    hw_str = ""
    for data in (spatial_data, temporal_data, vjp_data):
        if data:
            hw_str = _hardware_str(data)
            break

    return (spatial_data, temporal_data, vjp_data), hw_str


def _build_columns(
    spatial_data: dict, temporal_data: dict, vjp_data: dict, res_key: str
) -> list[tuple[str, dict, str, str, list]]:
    """Assemble the column specs in display order, skipping empty ones."""
    columns: list[tuple[str, dict, str, str, list]] = []

    if _has_inner_data(spatial_data.get("by_N", {})):
        columns.append(
            (
                "spatial_N",
                spatial_data["by_N"],
                res_key,
                "Forward — N scaling",
                _first_nonempty_keys(spatial_data["by_N"]),
            )
        )

    if _has_inner_data(temporal_data.get("by_steps", {})):
        columns.append(
            (
                "temporal_steps",
                temporal_data["by_steps"],
                "steps",
                "Forward — steps scaling",
                _first_nonempty_keys(temporal_data["by_steps"]),
            )
        )

    if _has_inner_data(vjp_data.get("by_N", {})):
        columns.append(
            (
                "vjp_N",
                vjp_data["by_N"],
                res_key,
                "VJP — N scaling",
                _first_nonempty_keys(vjp_data["by_N"]),
            )
        )

    if _has_inner_data(vjp_data.get("by_steps", {})):
        columns.append(
            (
                "vjp_steps",
                vjp_data["by_steps"],
                "steps",
                "VJP — steps scaling",
                _first_nonempty_keys(vjp_data["by_steps"]),
            )
        )

    return columns


def _column_x(panel_id: str, keys: list, n_to_cells, xlabel: str):
    """Compute x-axis array and display label for a column."""
    is_N_panel = panel_id in ("spatial_N", "vjp_N")
    if is_N_panel and n_to_cells is not None:
        x_arr = np.array([n_to_cells(int(k)) for k in keys], dtype=float)
        return x_arr, "cells"
    x_arr = np.array([int(k) for k in keys], dtype=float)
    return x_arr, xlabel


def _draw_solver_series(
    ax_time,
    ax_mem,
    x_arr: np.ndarray,
    keys: list,
    row: dict,
    style: dict,
    cfg: Problem,
    name: str,
    failure_types_seen: set[str],
) -> bool:
    """Draw one solver's time + memory series on a column. Return True if mem was drawn."""
    color = style.get("color", "black")
    ls = style.get("linestyle", "-")
    spec = cfg.solver(name)
    tag = "(GPU)" if spec.uses_gpu else "(CPU)"
    label = f"{spec.name} {tag}"
    kw = dict(
        label=label,
        markersize=5,
        markeredgewidth=0,
        **solver_plot_props(style),
    )

    t_ys = _time_vals(row, keys)
    m_ys = _mem_vals(row, keys)
    fail_k, fail_ft = _first_failure(row, keys)

    if any(np.isfinite(v) for v in t_ys):
        ax_time.loglog(x_arr, t_ys, **kw)
        if fail_k is not None:
            _draw_failure(ax_time, x_arr, keys, fail_k, t_ys, color, ls, fail_ft)
            failure_types_seen.add(fail_ft)

    mem_drawn = False
    if any(np.isfinite(v) for v in m_ys):
        ax_mem.loglog(x_arr, m_ys, **kw)
        if fail_k is not None:
            _draw_failure(ax_mem, x_arr, keys, fail_k, m_ys, color, ls, fail_ft)
        mem_drawn = True

    return mem_drawn


def _draw_column(
    axes_grid,
    col: int,
    col_spec: tuple,
    cfg: Problem,
    styles: dict,
    failure_types_seen: set[str],
    *,
    n_to_cells,
) -> bool:
    """Draw all solvers on one column. Return True if any memory series was drawn."""
    panel_id, by_data, xlabel, title, keys = col_spec
    ax_time = axes_grid[0, col]
    ax_mem = axes_grid[1, col]

    x_arr, xlabel_disp = _column_x(panel_id, keys, n_to_cells, xlabel)

    col_has_mem = False
    for spec in cfg.solvers:
        name = spec.name
        row = by_data.get(name) or {}
        style = styles.get(name, {"color": cfg.solver(name).color})
        if _draw_solver_series(
            ax_time, ax_mem, x_arr, keys, row, style, cfg, name, failure_types_seen
        ):
            col_has_mem = True

    ax_time.set_xlabel(xlabel_disp)
    ax_time.set_ylabel("Wall-clock time (s)")
    ax_time.set_title(title)

    ax_mem.set_xlabel(xlabel_disp)
    ax_mem.set_ylabel("Peak (V)RAM (MiB)")

    if not col_has_mem:
        ax_mem.set_visible(False)

    return col_has_mem


def _add_failure_legend_entries(ax, failure_types_seen: set[str]) -> None:
    """Add phantom plots for failure markers so fig_shared_legend picks them up."""
    for ft in ["OOM", "nan", "error", "timeout"]:
        if ft in failure_types_seen:
            ax.plot(
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


def plot_cost(
    cfg: Problem,
    *,
    n_to_cells=None,
    resolution_key: str = "N",
    save: bool = True,
    suffix: str = "",
    **_kw,
):
    """Cost plots: wall-clock timing + peak (V)RAM (log-log), 2 rows × N columns.

    Row 0 — wall-clock time.  Row 1 — peak GPU VRAM (or RAM for CPU solvers).
    Row 1 is hidden when no memory data is available.
    Failure markers (OOM ▼, error ◆, NaN ×) with connectors from last ok point.
    """
    suite_dir = results_dir() / cfg.name / _SUITE

    loaded, hw_str = _load_cost_inputs(suite_dir, suffix)
    if loaded is None:
        return None
    spatial_data, temporal_data, vjp_data = loaded

    columns = _build_columns(spatial_data, temporal_data, vjp_data, resolution_key)
    n_cols = len(columns)
    if n_cols == 0:
        return None

    styles = solver_styles(cfg)

    fig, axes_grid = plt.subplots(
        2,
        n_cols,
        figsize=(5 * n_cols, 7),
        squeeze=False,
    )

    failure_types_seen: set[str] = set()
    mem_row_used = False

    for col, col_spec in enumerate(columns):
        if _draw_column(
            axes_grid,
            col,
            col_spec,
            cfg,
            styles,
            failure_types_seen,
            n_to_cells=n_to_cells,
        ):
            mem_row_used = True

    # hide memory row label axes if the whole row is empty
    if not mem_row_used:
        for ax in axes_grid[1]:
            ax.set_visible(False)

    _add_failure_legend_entries(axes_grid[0, 0], failure_types_seen)

    # ── legend, title, hardware footnote ─────────────────────────────────────
    fig.suptitle(f"{cfg.name} — cost")
    fig_shared_legend(fig, axes_grid, bottom=0.16)

    if hw_str:
        fig.text(
            0.5,
            0.07,
            hw_str,
            ha="center",
            fontsize=7,
            color="gray",
            transform=fig.transFigure,
        )

    if save:
        save_fig(fig, f"cost{suffix}", suite_dir)
    return fig
