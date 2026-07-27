# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-problem plots for 2D Navier--Stokes optimisation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.animation as manimation
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    experiment_dir,
    load_json,
    results_dir,
    try_load_npz,
    v1_to_legacy,
)
from mosaic.benchmarks.problems.shared.plots.optimization import _save_animation
from mosaic.benchmarks.problems.shared.plots.style import (
    NS_ORDER,
    RCPARAMS,
    SOLVER_STYLES,
    TEXTWIDTH,
    dedup_handles,
    imshow_with_cbar,
    make_handle,
    paper_image_grid,
    paper_row,
    resolve_solver_alias,
    save_fig,
    solver_legend,
    solver_props,
    solver_styles,
)

# Solvers shown in the drag_opt paper panel, in display order.
_DRAG_OPT_SOLVER_ORDER = ["xlb", "phiflow", "pict"]


def _solver_loop_legend(
    fig: plt.Figure,
    names: list[str],
    *,
    extra_handles: list[Any] | None = None,
) -> None:
    """Identify solvers by colour/marker while reserving line style for semantics."""
    by_alias = {
        alias: name
        for name in names
        if (alias := resolve_solver_alias(name)) is not None
    }
    handles: list[Any] = []
    for alias in NS_ORDER:
        name = by_alias.get(alias)
        if name is None:
            continue
        label, color, _linestyle, marker = solver_props(name)
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                marker=marker,
                linestyle="-",
                label=label,
            )
        )
    handles.extend(extra_handles or [])
    if handles:
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=min(len(handles), 6),
            handlelength=2.0,
        )


def plot_solver_in_loop(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "solver_in_loop",
    title: str = "Solver-in-the-loop neural correction",
    solver_specific_reference: bool = False,
    **_kw: Any,
) -> list:
    """Plot corrector trainability, rollout quality, and time-to-quality."""
    out_dir = experiment_dir(
        results_dir(),
        cfg.name,
        "optimization",
        f"{exp_key}{suffix}",
    )
    result_path = out_dir / "result.json"
    fields_path = out_dir / "corrector_fields.npz"
    if not result_path.exists() or not fields_path.exists():
        print(f"[solver_in_loop] missing results in {out_dir} — skipping")
        return []

    data = v1_to_legacy(load_json(result_path))
    arrays = try_load_npz(fields_path)
    names = [str(v) for v in arrays.get("solver_names", np.array([])).tolist()]
    if not names:
        return []

    times = np.asarray(arrays.get("evaluation_times", np.array([])))
    fig, axes = paper_row(3, squeeze=False)
    ax_loss, ax_roll, ax_cost = np.atleast_1d(axes).ravel()
    present: list[str] = []
    has_stopped_rollout = False

    for idx, name in enumerate(names):
        metrics = data.get("by_solver", {}).get(name, {})
        label, color, _linestyle, marker = solver_props(name)
        loss = np.asarray(arrays.get(f"loss_{idx}", np.array([])))
        loss_std = np.asarray(arrays.get(f"loss_seed_std_{idx}", np.array([])))
        stopped_loss = np.asarray(arrays.get(f"loss_stop_gradient_{idx}", np.array([])))
        corrected = np.asarray(arrays.get(f"error_corrected_{idx}", np.array([])))
        corrected_std = np.asarray(
            arrays.get(f"error_corrected_seed_std_{idx}", np.array([]))
        )
        corrected_ic_std = np.asarray(
            arrays.get(f"error_corrected_ic_std_{idx}", np.array([]))
        )
        stopped = np.asarray(arrays.get(f"error_stop_gradient_{idx}", np.array([])))
        uncorrected = np.asarray(arrays.get(f"error_uncorrected_{idx}", np.array([])))
        if solver_specific_reference and uncorrected.size:
            # Each self-reference cell has a different target.  Divide by that
            # cell's raw trajectory so shared axes show within-solver correction
            # gain rather than incomparable absolute target errors.
            baseline = np.maximum(uncorrected, 1e-12)
            if corrected.size:
                corrected = corrected / baseline
            if corrected_std.size:
                corrected_std = corrected_std / baseline
            if corrected_ic_std.size:
                corrected_ic_std = corrected_ic_std / baseline
            if stopped.size:
                stopped = stopped / baseline
            uncorrected = np.ones_like(uncorrected)
        if loss.size:
            updates = np.arange(1, loss.size + 1)
            ax_loss.plot(
                updates,
                loss,
                color=color,
                linestyle="-",
                label=label,
            )
            if loss_std.shape == loss.shape:
                ax_loss.fill_between(
                    updates,
                    np.maximum(loss - loss_std, 0.2 * loss),
                    loss + loss_std,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
        if stopped_loss.size:
            ax_loss.plot(
                np.arange(1, stopped_loss.size + 1),
                stopped_loss,
                color=color,
                linestyle="--",
                alpha=0.55,
            )
        if corrected.size:
            x = times[: corrected.size] if times.size else np.arange(corrected.size)
            ax_roll.plot(
                x[1:],
                corrected[1:],
                color=color,
                linestyle="-",
                label=label,
            )
            if corrected_std.shape == corrected.shape:
                rollout_uncertainty = corrected_std
                if corrected_ic_std.shape == corrected.shape:
                    rollout_uncertainty = np.hypot(
                        corrected_std,
                        corrected_ic_std,
                    )
                ax_roll.fill_between(
                    x[1:],
                    np.maximum(corrected[1:] - rollout_uncertainty[1:], 1e-12),
                    corrected[1:] + rollout_uncertainty[1:],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
        if stopped.size:
            has_stopped_rollout = True
            x = times[: stopped.size] if times.size else np.arange(stopped.size)
            ax_roll.plot(
                x[1:],
                stopped[1:],
                color=color,
                linestyle="--",
                alpha=0.55,
            )
        if uncorrected.size:
            x = times[: uncorrected.size] if times.size else np.arange(uncorrected.size)
            ax_roll.plot(
                x[1:],
                uncorrected[1:],
                color=color,
                linestyle=":",
                alpha=0.45,
            )

        wall = metrics.get("training_wall_time_s")
        admitted = bool(metrics.get("valid_for_vjp_ranking", False))
        quality = (
            100.0 * (float(metrics["solver_vjp_geometric_lift"]) - 1.0)
            if solver_specific_reference
            and admitted
            and metrics.get("solver_vjp_geometric_lift") is not None
            else (
                None
                if solver_specific_reference
                else metrics.get("final_rollout_error")
            )
        )
        if wall is not None and quality is not None:
            ax_cost.scatter(
                [wall],
                [quality],
                color=color,
                marker=marker,
                s=30,
                label=label,
            )
        present.append(name)

    if any(
        np.any(np.asarray(arrays.get(f"loss_{idx}", np.array([]))) > 0)
        for idx in range(len(names))
    ):
        ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Optimizer update")
    ax_loss.set_ylabel("Normalized training loss")
    ax_loss.set_title("Corrector training")

    ax_roll.set_yscale("log")
    ax_roll.set_xlabel("Physical time")
    ax_roll.set_ylabel(
        "Error / solver-only error"
        if solver_specific_reference
        else "Relative $L^2$ error"
    )
    ax_roll.set_title("Held-out rollout")
    ax_roll.text(
        0.98,
        0.04,
        (
            "solid: full VJP\n dashed: stop-gradient\n dotted: solver only"
            if has_stopped_rollout
            else "solid: solver + corrector\n dotted: solver only"
        ),
        transform=ax_roll.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )

    ax_cost.set_xlabel("Training wall time [s]")
    ax_cost.set_ylabel(
        "Solver-VJP lift [%]" if solver_specific_reference else "Final rollout error"
    )
    ax_cost.set_title(
        "VJP benefit versus cost (admitted only)"
        if solver_specific_reference
        else "Time to quality"
    )
    if solver_specific_reference:
        ax_cost.axhline(0.0, color="0.45", linestyle=":", linewidth=1.0)

    _solver_loop_legend(fig, present)
    fig.suptitle(title)
    figs = [fig]
    if save:
        save_fig(fig, "solver_in_loop", out_dir)
    fields_fig = _plot_solver_in_loop_fields(arrays, names, out_dir, save=save)
    if fields_fig is not None:
        figs.append(fields_fig)
    fairness_fig = _plot_solver_in_loop_fairness(
        data,
        names,
        out_dir,
        save=save,
        solver_specific_reference=solver_specific_reference,
    )
    if fairness_fig is not None:
        figs.append(fairness_fig)
    physics_fig = _plot_solver_in_loop_physics(
        arrays,
        names,
        out_dir,
        save=save,
        solver_specific_reference=solver_specific_reference,
    )
    if physics_fig is not None:
        figs.append(physics_fig)
    if not solver_specific_reference:
        diagnostics_fig = _plot_solver_in_loop_diagnostics(
            data,
            names,
            out_dir,
            save=save,
        )
        if diagnostics_fig is not None:
            figs.append(diagnostics_fig)
    if save:
        _save_solver_in_loop_animation(arrays, names, out_dir)
    return figs


def plot_solver_in_loop_tgv(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    **kwargs: Any,
) -> list:
    """Plot the analytic Taylor--Green solver-in-the-loop sanity regime."""
    return plot_solver_in_loop(
        cfg,
        save=save,
        suffix=suffix,
        exp_key="solver_in_loop_tgv",
        **kwargs,
    )


def plot_solver_in_loop_self_reference(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    **kwargs: Any,
) -> list:
    """Plot the recurrence-admitted, solver-specific refined-reference task."""
    return plot_solver_in_loop(
        cfg,
        save=save,
        suffix=suffix,
        exp_key="solver_in_loop_self_reference",
        title="Solver-specific refined-reference neural correction",
        solver_specific_reference=True,
        **kwargs,
    )


def _solver_reference_rollout(
    arrays: dict[str, np.ndarray],
    solver_index: int,
) -> np.ndarray:
    """Return a solver-specific target when present, else the shared target."""
    return np.asarray(
        arrays.get(
            f"reference_rollout_{solver_index}",
            arrays.get("reference_rollout", np.array([])),
        )
    )


def _periodic_vorticity_2d(
    velocity: np.ndarray,
    *,
    domain_extent: float = 2.0 * np.pi,
) -> np.ndarray:
    """Return centered-difference vorticity for a periodic 2-D velocity field."""
    ux, uy = _vel_components_2d(np.asarray(velocity))
    dx = domain_extent / ux.shape[0]
    dy = domain_extent / ux.shape[1]
    d_uy_dx = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2.0 * dx)
    d_ux_dy = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2.0 * dy)
    return d_uy_dx - d_ux_dy


def _periodic_divergence_2d(
    velocity: np.ndarray,
    *,
    domain_extent: float = 2.0 * np.pi,
    spectral: bool,
) -> np.ndarray:
    """Return a common spectral or centered periodic divergence diagnostic."""
    ux, uy = _vel_components_2d(np.asarray(velocity))
    dx = domain_extent / ux.shape[0]
    dy = domain_extent / ux.shape[1]
    if spectral:
        kx = 2.0 * np.pi * np.fft.fftfreq(ux.shape[0], d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ux.shape[1], d=dy)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
        divergence_hat = 1j * (kx_grid * np.fft.fft2(ux) + ky_grid * np.fft.fft2(uy))
        return np.fft.ifft2(divergence_hat).real
    du_dx = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2.0 * dx)
    dv_dy = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2.0 * dy)
    return du_dx + dv_dy


def _trajectory_diagnostics(
    trajectory: np.ndarray,
    *,
    domain_extent: float = 2.0 * np.pi,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return energy, enstrophy, and two divergence curves for a trajectory."""
    energy: list[float] = []
    enstrophy_values: list[float] = []
    spectral_divergence: list[float] = []
    centered_divergence: list[float] = []
    for frame in np.asarray(trajectory):
        ux, uy = _vel_components_2d(frame)
        vorticity = _periodic_vorticity_2d(frame, domain_extent=domain_extent)
        energy.append(float(0.5 * np.mean(ux**2 + uy**2)))
        enstrophy_values.append(float(0.5 * np.mean(vorticity**2)))
        spectral_divergence.append(
            float(
                np.sqrt(
                    np.mean(
                        _periodic_divergence_2d(
                            frame,
                            domain_extent=domain_extent,
                            spectral=True,
                        )
                        ** 2
                    )
                )
            )
        )
        centered_divergence.append(
            float(
                np.sqrt(
                    np.mean(
                        _periodic_divergence_2d(
                            frame,
                            domain_extent=domain_extent,
                            spectral=False,
                        )
                        ** 2
                    )
                )
            )
        )
    return (
        np.asarray(energy),
        np.asarray(enstrophy_values),
        np.asarray(spectral_divergence),
        np.asarray(centered_divergence),
    )


def _plot_solver_in_loop_fields(
    arrays: dict[str, np.ndarray],
    names: list[str],
    out_dir: Path,
    *,
    save: bool,
) -> plt.Figure | None:
    """Render final held-out reference, raw-solver, and corrected vorticity.

    Each solver gets one row. The reference column is deliberately repeated so
    every raw/corrected pair can be read horizontally without looking across
    rows. All panels share one robust symmetric colour scale.
    """
    rows = _ordered_solver_rollouts(arrays, names)
    if not rows:
        return None

    rendered_rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    scale_fields: list[np.ndarray] = []
    for name, raw, corrected in rows:
        reference = _solver_reference_rollout(arrays, names.index(name))
        if reference.size == 0 or reference.shape[0] == 0:
            continue
        reference_vorticity = _periodic_vorticity_2d(reference[-1])
        raw_vorticity = _periodic_vorticity_2d(raw[-1])
        corrected_vorticity = _periodic_vorticity_2d(corrected[-1])
        rendered_rows.append(
            (name, reference_vorticity, raw_vorticity, corrected_vorticity)
        )
        scale_fields.extend((reference_vorticity, raw_vorticity, corrected_vorticity))
    if not rendered_rows:
        return None

    magnitudes = np.concatenate([np.abs(field).ravel() for field in scale_fields])
    vmax = float(np.percentile(magnitudes, 99.0)) or 1.0

    plt.rcParams.update(RCPARAMS)
    fig, axes = plt.subplots(
        len(rendered_rows),
        3,
        figsize=(TEXTWIDTH, max(1.8, 0.95 * len(rendered_rows))),
        squeeze=False,
        layout="constrained",
    )
    column_titles = ("Reference", "Solver only", "Solver + corrector")
    image = None
    for row_idx, (
        name,
        reference_vorticity,
        raw_vorticity,
        corrected_vorticity,
    ) in enumerate(rendered_rows):
        label, _color, _linestyle, _marker = solver_props(name)
        for col_idx, field in enumerate(
            (reference_vorticity, raw_vorticity, corrected_vorticity)
        ):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                field.T,
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(column_titles[col_idx])
        axes[row_idx, 0].set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )

    if image is not None:
        colorbar = fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            location="right",
            shrink=0.82,
            pad=0.02,
        )
        colorbar.set_label(r"Vorticity $\omega$")
    times = np.asarray(arrays.get("evaluation_times", np.array([])))
    time_suffix = f" at $t={float(times[-1]):g}$" if times.size else ""
    fig.suptitle(f"Held-out final-frame flow{time_suffix}")

    if save:
        save_fig(fig, "solver_in_loop_fields", out_dir)
    return fig


def _ordered_solver_rollouts(
    arrays: dict[str, np.ndarray],
    names: list[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return complete raw/corrected rollouts in canonical solver order."""
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for idx, name in enumerate(names):
        raw = np.asarray(arrays.get(f"rollout_uncorrected_{idx}", np.array([])))
        corrected = np.asarray(arrays.get(f"rollout_corrected_{idx}", np.array([])))
        if raw.size and corrected.size and raw.shape[0] and corrected.shape[0]:
            rows.append((name, raw, corrected))
    solver_order = {alias: idx for idx, alias in enumerate(NS_ORDER)}
    rows.sort(
        key=lambda row: solver_order.get(
            resolve_solver_alias(row[0]) or row[0],
            len(solver_order),
        )
    )
    return rows


def _plot_solver_in_loop_physics(
    arrays: dict[str, np.ndarray],
    names: list[str],
    out_dir: Path,
    *,
    save: bool,
    solver_specific_reference: bool = False,
) -> plt.Figure | None:
    """Compare energy, enstrophy, and divergence along held-out rollouts."""
    rows = _ordered_solver_rollouts(arrays, names)
    solver_references = {
        name: _solver_reference_rollout(arrays, names.index(name))
        for name, _raw, _corrected in rows
    }
    rows = [row for row in rows if solver_references[row[0]].size]
    if not rows:
        return None

    n_frames = min(
        [
            min(
                solver_references[name].shape[0],
                raw.shape[0],
                corrected.shape[0],
            )
            for name, raw, corrected in rows
        ]
    )
    times = np.asarray(arrays.get("evaluation_times", np.array([])))[:n_frames]
    if times.size != n_frames:
        times = np.arange(n_frames, dtype=float)

    plt.rcParams.update(RCPARAMS)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(TEXTWIDTH, 4.5),
        squeeze=False,
        layout="constrained",
    )
    ax_energy, ax_enstrophy, ax_spectral, ax_centered = axes.ravel()
    diagnostic_axes = (ax_energy, ax_enstrophy, ax_spectral, ax_centered)

    present: list[str] = []
    for row_index, (name, raw, corrected) in enumerate(rows):
        _label, color, _linestyle, _marker = solver_props(name)
        reference_diagnostics = _trajectory_diagnostics(
            solver_references[name][:n_frames]
        )
        energy_scale = reference_diagnostics[0][0] + 1e-12
        enstrophy_scale = reference_diagnostics[1][0] + 1e-12
        reference_curves = (
            reference_diagnostics[0] / energy_scale,
            reference_diagnostics[1] / enstrophy_scale,
            reference_diagnostics[2],
            reference_diagnostics[3],
        )
        raw_diagnostics = _trajectory_diagnostics(raw[:n_frames])
        corrected_diagnostics = _trajectory_diagnostics(corrected[:n_frames])
        raw_curves = (
            raw_diagnostics[0] / energy_scale,
            raw_diagnostics[1] / enstrophy_scale,
            raw_diagnostics[2],
            raw_diagnostics[3],
        )
        corrected_curves = (
            corrected_diagnostics[0] / energy_scale,
            corrected_diagnostics[1] / enstrophy_scale,
            corrected_diagnostics[2],
            corrected_diagnostics[3],
        )
        for ax, reference_curve, raw_curve, corrected_curve in zip(
            diagnostic_axes,
            reference_curves,
            raw_curves,
            corrected_curves,
            strict=True,
        ):
            if solver_specific_reference or row_index == 0:
                ax.plot(
                    times,
                    np.maximum(reference_curve, 1e-12),
                    color=color if solver_specific_reference else "0.25",
                    linestyle="--",
                    alpha=0.7,
                )
            ax.plot(times, np.maximum(raw_curve, 1e-12), color=color, linestyle=":")
            ax.plot(
                times,
                np.maximum(corrected_curve, 1e-12),
                color=color,
                linestyle="-",
            )
        present.append(name)

    ax_energy.set_ylabel("Energy / initial reference")
    ax_energy.set_title("Kinetic energy")
    ax_enstrophy.set_ylabel("Enstrophy / initial reference")
    ax_enstrophy.set_title("Resolved enstrophy")
    ax_spectral.set_ylabel("RMS divergence")
    ax_spectral.set_title("Common spectral divergence")
    ax_centered.set_ylabel("RMS divergence")
    ax_centered.set_title("Common centered-grid divergence")
    for ax in diagnostic_axes:
        ax.set_xlabel("Physical time")
    ax_spectral.set_yscale("log")
    ax_centered.set_yscale("log")

    _solver_loop_legend(
        fig,
        present,
        extra_handles=[
            mlines.Line2D(
                [],
                [],
                color="0.4",
                linestyle="--",
                label=(
                    "solver-specific reference"
                    if solver_specific_reference
                    else "shared reference"
                ),
            ),
            mlines.Line2D([], [], color="0.4", linestyle=":", label="solver only"),
            mlines.Line2D([], [], color="0.4", linestyle="-", label="full corrector"),
        ],
    )
    fig.suptitle("Held-out trajectory physics (divergence is operator-dependent)")
    if save:
        save_fig(fig, "solver_in_loop_physics", out_dir)
    return fig


def _plot_solver_in_loop_fairness(
    data: dict[str, Any],
    names: list[str],
    out_dir: Path,
    *,
    save: bool,
    solver_specific_reference: bool = False,
) -> plt.Figure | None:
    """Separate raw quality, correctability, and benefit from the solver VJP."""
    by_solver = data.get("by_solver", {})
    rows: list[tuple[str, dict[str, Any]]] = []
    solver_order = {alias: idx for idx, alias in enumerate(NS_ORDER)}
    for name in names:
        metrics = by_solver.get(name, {})
        if (
            metrics.get("uncorrected_mean_rollout_error") is not None
            and metrics.get("mean_rollout_error") is not None
        ):
            rows.append((name, metrics))
    rows.sort(
        key=lambda row: solver_order.get(
            resolve_solver_alias(row[0]) or row[0],
            len(solver_order),
        )
    )
    if not rows:
        return None

    plt.rcParams.update(RCPARAMS)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(TEXTWIDTH, 4.5),
        squeeze=False,
        layout="constrained",
    )
    ax_quality, ax_gain, ax_vjp, ax_efficiency = axes.ravel()
    positions = np.arange(len(rows), dtype=float)
    labels: list[str] = []

    all_errors: list[float] = []
    normalized_full_errors: list[float] = []
    normalized_stopped_errors: list[float] = []
    full_gain: list[float] = []
    stopped_gain: list[float] = []
    vjp_lift: list[float] = []
    full_gain_log_std: list[float] = []
    stopped_gain_log_std: list[float] = []
    vjp_lift_log_std: list[float] = []
    update_times: list[float] = []
    has_counterfactual = True
    has_cost = True
    has_seed_uncertainty = False
    has_ic_uncertainty = False
    for name, metrics in rows:
        label, color, _linestyle, marker = solver_props(name)
        labels.append(label)
        raw = float(metrics["uncorrected_mean_rollout_error"])
        corrected = float(metrics["mean_rollout_error"])
        normalized_full_errors.append(corrected / max(raw, 1e-12))
        stopped_error = metrics.get("stop_gradient_mean_rollout_error")
        if solver_specific_reference:
            has_counterfactual &= stopped_error is not None
        normalized_stopped_errors.append(
            float(stopped_error) / max(raw, 1e-12)
            if stopped_error is not None
            else np.nan
        )
        if not solver_specific_reference:
            all_errors.extend((raw, corrected))
            ax_quality.scatter(raw, corrected, color=color, marker=marker, s=30)
        update_time = metrics.get("median_update_time_s")
        has_cost &= update_time is not None
        update_times.append(float(update_time) if update_time is not None else np.nan)

        full = metrics.get("geometric_error_reduction")
        if full is None:
            full = raw / max(corrected, 1e-12)
        full_gain.append(float(full))
        full_gain_log_std.append(
            float(
                np.hypot(
                    float(metrics.get("rollout_log_gain_seed_std", 0.0)),
                    float(metrics.get("rollout_log_gain_ic_std", 0.0)),
                )
            )
        )
        has_seed_uncertainty |= "rollout_log_gain_seed_std" in metrics
        has_ic_uncertainty |= "rollout_log_gain_ic_std" in metrics

        stopped = metrics.get("stop_gradient_geometric_error_reduction")
        lift = metrics.get("solver_vjp_geometric_lift")
        if stopped is None or lift is None:
            has_counterfactual = False
        stopped_gain.append(float(stopped) if stopped is not None else np.nan)
        vjp_lift.append(float(lift) if lift is not None else np.nan)
        stopped_gain_log_std.append(
            float(
                np.hypot(
                    float(
                        metrics.get(
                            "stop_gradient_rollout_log_gain_seed_std",
                            0.0,
                        )
                    ),
                    float(
                        metrics.get(
                            "stop_gradient_rollout_log_gain_ic_std",
                            0.0,
                        )
                    ),
                )
            )
        )
        seed_lift_std = float(metrics.get("solver_vjp_log_lift_seed_std", 0.0))
        ic_lift_std = float(metrics.get("solver_vjp_log_lift_ic_std", 0.0))
        vjp_lift_log_std.append(float(np.hypot(seed_lift_std, ic_lift_std)))
        has_ic_uncertainty |= "solver_vjp_log_lift_ic_std" in metrics

    width = 0.36 if has_counterfactual else 0.62
    colors = [solver_props(name)[1] for name, _metrics in rows]
    if solver_specific_reference:
        ax_quality.bar(
            positions - (width / 2 if has_counterfactual else 0.0),
            normalized_full_errors,
            width=width,
            color=colors,
            alpha=0.9,
            label="full solver VJP",
        )
        if has_counterfactual:
            ax_quality.bar(
                positions + width / 2,
                normalized_stopped_errors,
                width=width,
                color=colors,
                alpha=0.35,
                hatch="//",
                label="stop-gradient",
            )
            ax_quality.legend(loc="best", fontsize=6.5)
        ax_quality.axhline(1.0, color="0.45", linestyle=":", linewidth=1.0)
        ax_quality.set_xticks(positions, labels, rotation=35, ha="right")
        ax_quality.set_ylabel("Mean error / solver-only error")
        ax_quality.set_title("Target-normalized quality")
    else:
        error_min = max(min(all_errors) * 0.8, 1e-4)
        error_max = max(all_errors) * 1.25
        ax_quality.plot(
            [error_min, error_max],
            [error_min, error_max],
            color="0.45",
            linestyle=":",
            linewidth=1.0,
        )
        if error_max / error_min >= 10.0:
            ax_quality.set_xscale("log")
            ax_quality.set_yscale("log")
            ax_quality.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax_quality.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax_quality.set_xlim(error_min, error_max)
        ax_quality.set_ylim(error_min, error_max)
        ax_quality.set_xlabel("Solver-only mean error")
        ax_quality.set_ylabel("Corrected mean error")
        ax_quality.set_title("Absolute quality")

    full_error = _log_scale_errorbars(full_gain, full_gain_log_std)
    ax_gain.bar(
        positions - (width / 2 if has_counterfactual else 0.0),
        full_gain,
        width=width,
        color=colors,
        alpha=0.9,
        label="full solver VJP",
        yerr=full_error,
        capsize=2,
    )
    if has_counterfactual:
        ax_gain.bar(
            positions + width / 2,
            stopped_gain,
            width=width,
            color=colors,
            alpha=0.35,
            hatch="//",
            label="stop-gradient",
            yerr=_log_scale_errorbars(stopped_gain, stopped_gain_log_std),
            capsize=2,
        )
        ax_gain.legend(loc="best", fontsize=6.5)
    ax_gain.axhline(1.0, color="0.45", linestyle=":", linewidth=1.0)
    ax_gain.set_xticks(positions, labels, rotation=35, ha="right")
    ax_gain.set_ylabel("Geometric error reduction [×]")
    ax_gain.set_title("Correctability")

    if has_counterfactual:
        admitted = [
            idx
            for idx, (_name, metrics) in enumerate(rows)
            if metrics.get("valid_for_vjp_ranking", False)
        ]
        admitted_positions = np.arange(len(admitted), dtype=float)
        admitted_labels = [labels[idx] for idx in admitted]
        admitted_colors = [colors[idx] for idx in admitted]
        vjp_lift_pct = [100.0 * (vjp_lift[idx] - 1.0) for idx in admitted]
        vjp_error = _log_scale_errorbars(vjp_lift, vjp_lift_log_std)
        if vjp_error is not None:
            vjp_error = 100.0 * vjp_error[:, admitted]
        ax_vjp.bar(
            admitted_positions,
            vjp_lift_pct,
            color=admitted_colors,
            alpha=0.9,
            yerr=vjp_error,
            capsize=2,
        )
        ax_vjp.axhline(0.0, color="0.45", linestyle=":", linewidth=1.0)
        ax_vjp.set_xticks(
            admitted_positions,
            admitted_labels,
            rotation=35,
            ha="right",
        )
        ax_vjp.set_ylabel("Solver-VJP lift [%]")
        title_suffix = " (admitted only)" if len(admitted) < len(rows) else ""
        ax_vjp.set_title(f"Benefit from solver VJP{title_suffix}")
        if has_cost:
            for ranked_idx, idx in enumerate(admitted):
                name, _metrics = rows[idx]
                _label, color, _linestyle, marker = solver_props(name)
                error = (
                    None
                    if vjp_error is None
                    else vjp_error[:, ranked_idx].reshape(2, 1)
                )
                ax_efficiency.errorbar(
                    update_times[idx],
                    vjp_lift_pct[ranked_idx],
                    yerr=error,
                    color=color,
                    marker=marker,
                    linestyle="none",
                    capsize=2,
                )
            positive_times = [
                update_times[idx] for idx in admitted if update_times[idx] > 0
            ]
            if positive_times and max(positive_times) / min(positive_times) >= 10.0:
                ax_efficiency.set_xscale("log")
            ax_efficiency.axhline(
                0.0,
                color="0.45",
                linestyle=":",
                linewidth=1.0,
            )
            ax_efficiency.set_xlabel("Full-VJP update time [s]")
            ax_efficiency.set_ylabel("Solver-VJP lift [%]")
            ax_efficiency.set_title(f"VJP benefit versus cost{title_suffix}")
        else:
            ax_efficiency.axis("off")
    else:
        ax_vjp.axis("off")
        ax_vjp.text(
            0.5,
            0.5,
            "VJP counterfactual\nnot available in this run",
            ha="center",
            va="center",
            transform=ax_vjp.transAxes,
        )
        ax_efficiency.axis("off")

    _solver_loop_legend(fig, [name for name, _metrics in rows])
    uncertainty_suffix = ""
    if has_seed_uncertainty and has_ic_uncertainty:
        uncertainty_suffix = " (error bars: combined seed/IC SD)"
    elif has_seed_uncertainty:
        uncertainty_suffix = " (error bars: ±1 seed SD)"
    title = (
        "Solver-specific self-consistency decomposition"
        if solver_specific_reference
        else "Fair solver-in-the-loop decomposition"
    )
    fig.suptitle(f"{title}{uncertainty_suffix}")
    if save:
        save_fig(fig, "solver_in_loop_fairness", out_dir)
    return fig


def _plot_solver_in_loop_diagnostics(
    data: dict[str, Any],
    names: list[str],
    out_dir: Path,
    *,
    save: bool,
) -> plt.Figure | None:
    """Separate solver-interface effects, IC generalization, and extrapolation."""
    by_solver = data.get("by_solver", {})
    required = (
        "first_interval_rollout_error",
        "native_final_rollout_error",
        "uncorrected_rollout_error",
        "recurrent_to_native_error_ratio",
        "seen_ic_matched_horizon_error",
        "heldout_ic_matched_horizon_error",
        "seen_ic_long_horizon_error",
        "heldout_ic_long_horizon_error",
    )
    solver_order = {alias: idx for idx, alias in enumerate(NS_ORDER)}
    rows = [
        (name, by_solver[name])
        for name in names
        if name in by_solver
        and all(by_solver[name].get(key) is not None for key in required)
    ]
    rows.sort(
        key=lambda row: solver_order.get(
            resolve_solver_alias(row[0]) or row[0],
            len(solver_order),
        )
    )
    if not rows:
        return None

    labels = [solver_props(name)[0] for name, _metrics in rows]
    colors = [solver_props(name)[1] for name, _metrics in rows]
    markers = [solver_props(name)[3] for name, _metrics in rows]
    positions = np.arange(len(rows), dtype=float)
    width = 0.36

    def values(key: str) -> list[float]:
        return [float(metrics.get(key, 0.0)) for _name, metrics in rows]

    plt.rcParams.update(RCPARAMS)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(TEXTWIDTH, 5.0),
        squeeze=False,
        layout="constrained",
    )
    ax_first, ax_recurrent, ax_closure, ax_matched, ax_long, ax_gaps = axes.ravel()

    ax_first.bar(
        positions,
        values("first_interval_rollout_error"),
        color=colors,
        yerr=values("first_interval_rollout_error_ic_std"),
        capsize=2,
    )
    ax_first.set_ylabel("Relative $L^2$ error")
    ax_first.set_title("First canonical interval")

    native = values("native_final_rollout_error")
    recurrent = values("uncorrected_rollout_error")
    lower = max(min(native + recurrent) * 0.8, 1e-5)
    upper = max(native + recurrent) * 1.25
    ax_recurrent.plot(
        [lower, upper],
        [lower, upper],
        color="0.45",
        linestyle=":",
        linewidth=1.0,
    )
    for x_value, y_value, color, marker in zip(
        native,
        recurrent,
        colors,
        markers,
        strict=True,
    ):
        ax_recurrent.scatter(x_value, y_value, color=color, marker=marker, s=30)
    if upper / lower >= 10.0:
        ax_recurrent.set_xscale("log")
        ax_recurrent.set_yscale("log")
    ax_recurrent.set(xlim=(lower, upper), ylim=(lower, upper))
    ax_recurrent.set_xlabel("Native final error")
    ax_recurrent.set_ylabel("Repeated-call final error")
    ax_recurrent.set_title("Repeated calls vs native")

    if all(
        metrics.get("long_closure_error_p95") is not None for _name, metrics in rows
    ):
        closure_pct = 100.0 * np.asarray(values("long_closure_error_p95"))
        tolerance_pct = 100.0 * max(values("long_closure_tolerance"))
        ax_closure.bar(positions, closure_pct, color=colors)
        ax_closure.axhline(
            tolerance_pct,
            color="0.45",
            linestyle=":",
            linewidth=1.0,
            label="admission limit",
        )
        positive = closure_pct[closure_pct > 0]
        if positive.size and max(positive) / min(positive) >= 10.0:
            ax_closure.set_yscale("log")
        ax_closure.set_ylabel("95th percentile residual [%]")
        ax_closure.set_title("K-step closure gate")
    elif all(metrics.get("semigroup_error_p95") is not None for _name, metrics in rows):
        closure_pct = 100.0 * np.asarray(values("semigroup_error_p95"))
        tolerance_pct = 100.0 * max(values("semigroup_p95_tolerance"))
        ax_closure.bar(positions, closure_pct, color=colors)
        ax_closure.axhline(
            tolerance_pct,
            color="0.45",
            linestyle=":",
            linewidth=1.0,
            label="admission limit",
        )
        positive = closure_pct[closure_pct > 0]
        if positive.size and max(positive) / min(positive) >= 10.0:
            ax_closure.set_yscale("log")
        ax_closure.set_ylabel("95th percentile residual [%]")
        ax_closure.set_title("Two-step closure admission")
    else:
        ax_closure.bar(
            positions,
            values("recurrent_to_native_error_ratio"),
            color=colors,
        )
        ax_closure.axhline(1.0, color="0.45", linestyle=":", linewidth=1.0)
        ax_closure.set_ylabel("Repeated / native error [×]")
        ax_closure.set_title("Repeated-call penalty")

    for ax, horizon, title in (
        (ax_matched, "matched_horizon", "Matched training horizon"),
        (ax_long, "long_horizon", "Long rollout horizon"),
    ):
        seen = values(f"seen_ic_{horizon}_error")
        heldout = values(f"heldout_ic_{horizon}_error")
        ax.bar(
            positions - width / 2,
            seen,
            width,
            color=colors,
            alpha=0.9,
            label="seen IC",
            yerr=values(f"seen_ic_{horizon}_error_ic_std"),
            capsize=2,
        )
        ax.bar(
            positions + width / 2,
            heldout,
            width,
            color=colors,
            alpha=0.35,
            hatch="//",
            label="held-out IC",
            yerr=values(f"heldout_ic_{horizon}_error_ic_std"),
            capsize=2,
        )
        positive = [value for value in (*seen, *heldout) if value > 0]
        if positive and max(positive) / min(positive) >= 10.0:
            ax.set_yscale("log")
        ax.set_xticks(positions, labels, rotation=35, ha="right")
        ax.set_ylabel("Corrected relative $L^2$ error")
        ax.set_title(title)
    ax_matched.legend(loc="best", fontsize=6.5)

    gap_width = 0.24
    for offset, key, label, alpha in (
        (
            -gap_width,
            "ic_generalization_ratio_at_matched_horizon",
            "held-out / seen",
            0.9,
        ),
        (0.0, "seen_ic_temporal_extrapolation_ratio", "seen long / matched", 0.6),
        (
            gap_width,
            "heldout_ic_temporal_extrapolation_ratio",
            "held-out long / matched",
            0.3,
        ),
    ):
        ax_gaps.bar(
            positions + offset,
            values(key),
            gap_width,
            color=colors,
            alpha=alpha,
            label=label,
        )
    ax_gaps.axhline(1.0, color="0.45", linestyle=":", linewidth=1.0)
    ax_gaps.set_ylabel("Error ratio [×]")
    ax_gaps.set_title("Generalization gaps")
    ax_gaps.legend(loc="lower left", fontsize=6.0)

    for ax in (ax_first, ax_closure, ax_matched, ax_long, ax_gaps):
        ax.set_xticks(positions, labels, rotation=35, ha="right")

    matched_time = rows[0][1].get("matched_horizon_time")
    long_time = rows[0][1].get("rollout_final_time")
    suffix = ""
    if matched_time is not None and long_time is not None:
        suffix = f" ($t_\\mathrm{{match}}={matched_time:g}$, $t_\\mathrm{{long}}={long_time:g}$)"
    fig.suptitle(f"Solver interface and corrector generalization{suffix}")
    if save:
        save_fig(fig, "solver_in_loop_diagnostics", out_dir)
    return fig


def _log_scale_errorbars(
    values: list[float],
    log_standard_deviations: list[float],
) -> np.ndarray | None:
    """Convert symmetric log-space uncertainty to asymmetric linear bars."""
    centers = np.asarray(values, dtype=float)
    deviations = np.asarray(log_standard_deviations, dtype=float)
    if not np.any(deviations > 0):
        return None
    lower = centers - centers * np.exp(-deviations)
    upper = centers * np.exp(deviations) - centers
    return np.stack((lower, upper))


def _save_solver_in_loop_animation(
    arrays: dict[str, np.ndarray],
    names: list[str],
    out_dir: Path,
) -> None:
    """Animate reference, solver-only, and fully corrected held-out rollouts."""
    rows = _ordered_solver_rollouts(arrays, names)
    solver_references = {
        name: _solver_reference_rollout(arrays, names.index(name))
        for name, _raw, _corrected in rows
    }
    rows = [row for row in rows if solver_references[row[0]].size]
    if not rows:
        return

    available_frames = min(
        [
            min(
                solver_references[name].shape[0],
                raw.shape[0],
                corrected.shape[0],
            )
            for name, raw, corrected in rows
        ]
    )
    # Keep the full simulated horizon in the animation while bounding the GIF
    # payload embedded in PRs.  Evenly spaced samples retain both endpoints.
    frame_indices = np.unique(
        np.linspace(
            0,
            available_frames - 1,
            num=min(available_frames, 24),
            dtype=int,
        )
    )
    n_frames = len(frame_indices)
    rollout_vorticity = [
        (
            name,
            [
                _periodic_vorticity_2d(solver_references[name][index])
                for index in frame_indices
            ],
            [_periodic_vorticity_2d(raw[index]) for index in frame_indices],
            [_periodic_vorticity_2d(corrected[index]) for index in frame_indices],
        )
        for name, raw, corrected in rows
    ]
    scale_fields: list[np.ndarray] = []
    for _name, reference_fields, raw_fields, corrected_fields in rollout_vorticity:
        scale_fields.extend(reference_fields)
        scale_fields.extend(raw_fields)
        scale_fields.extend(corrected_fields)
    magnitudes = np.concatenate([np.abs(field).ravel() for field in scale_fields])
    vmax = float(np.percentile(magnitudes, 99.0)) or 1.0

    plt.rcParams.update(RCPARAMS)
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(TEXTWIDTH, max(1.8, 0.95 * len(rows))),
        dpi=100,
        squeeze=False,
        layout="constrained",
    )
    images: list[tuple[Any, int, int]] = []
    for row_idx, (
        name,
        reference_fields,
        raw_fields,
        corrected_fields,
    ) in enumerate(rollout_vorticity):
        label, _color, _linestyle, _marker = solver_props(name)
        for col_idx, fields in enumerate(
            (reference_fields, raw_fields, corrected_fields)
        ):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                fields[0].T,
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
                animated=True,
            )
            images.append((image, row_idx, col_idx))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(
                    ("Reference", "Solver only", "Solver + corrector")[col_idx]
                )
        axes[row_idx, 0].set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )

    colorbar = fig.colorbar(
        images[-1][0],
        ax=axes.ravel().tolist(),
        location="right",
        shrink=0.82,
        pad=0.02,
    )
    colorbar.set_label(r"Vorticity $\omega$")
    all_times = np.asarray(arrays.get("evaluation_times", np.array([])))
    times = (
        all_times[frame_indices] if all_times.size >= available_frames else np.array([])
    )
    title = fig.suptitle("Held-out trajectory at $t=0$")

    def _update(frame: int) -> tuple[Any, ...]:
        for image, row_idx, col_idx in images:
            field = rollout_vorticity[row_idx][col_idx + 1][frame]
            image.set_data(field.T)
        time_value = float(times[frame]) if frame < times.size else float(frame)
        title.set_text(f"Held-out trajectory at $t={time_value:g}$")
        artists = [image for image, _row, _col in images]
        artists.append(title)
        return tuple(artists)

    animation = manimation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=180,
        blit=False,
    )
    _save_animation(animation, "solver_in_loop_trajectory", out_dir, fps=6)


def plot_drag_opt(
    cfg: Problem,
    *,
    save: bool = True,
    suffix: str = "",
    exp_key: str = "drag_opt",
    **_kw: Any,
) -> list:
    """Drag-optimisation per-experiment plot — styled figure + extras.

    Inlined styled 3-column figure (drag reduction, optimised inlet
    profile, profile history) plus:

      * ``drag_opt_fields`` — per-solver flow field panels (u_x, u_y) when
        ``flow_fields.npz`` is present.
      * ``drag_opt_evolution.gif`` — one combined inflow-profile animation
        with a panel per solver when ``profiles.npz`` carries
        ``profile_history_*``.

    Supports both single-run (drag_opt/result.json) and multi-run
    (drag_opt/<name>/result.json) layouts.
    """
    base_dir = results_dir() / cfg.name / "optimization" / f"{exp_key}{suffix}"
    styles = solver_styles(cfg)
    figs: list = []

    def _plot_one(
        data: Any,
        profiles_path: Any,
        out_dir: Any,
        *,
        paper_exp_key: Any,
        paper_suffix: Any,
    ) -> None:
        by_solver = data.get("by_solver", {})
        if not by_solver:
            return
        run_name = data.get("run_name", "")
        title_suffix = f" — {run_name}" if run_name else ""

        # ── Canonical figure (inlined) ───────────────────────────────────────
        fig = _drag_opt_figure(
            cfg,
            exp_key=paper_exp_key,
            suffix=paper_suffix,
            save=save,
        )
        if fig is not None:
            figs.append(fig)

        profiles = try_load_npz(profiles_path) if profiles_path.exists() else {}
        solver_names = list(by_solver.keys())

        # ── Flow field visualisation (velocity + vorticity) ──────────────────
        _plot_drag_opt_fields(data, out_dir, run_name, title_suffix, styles, save, figs)

        # ── Inflow profile evolution GIF (combined, one panel per solver) ───
        if save and "initial" in profiles:
            _render_drag_opt_evolution_gifs(
                profiles, out_dir, solver_names, styles, run_name
            )

    # Single-run layout — figure resolves the experiment dir from cfg.
    single_path = base_dir / "result.json"
    single_result = (
        v1_to_legacy(load_json(single_path)) if single_path.exists() else None
    )
    if single_result is not None:
        _plot_one(
            single_result,
            base_dir / "profiles.npz",
            base_dir,
            paper_exp_key=exp_key,
            paper_suffix=suffix,
        )
        return figs

    # Multi-run layout: one canonical figure per run subdir.
    if base_dir.is_dir():
        for sub in sorted(base_dir.iterdir()):
            sub_data = v1_to_legacy(load_json(sub / "result.json"))
            if sub_data is not None:
                _plot_one(
                    sub_data,
                    sub / "profiles.npz",
                    sub,
                    paper_exp_key=f"{exp_key}/{sub.name}",
                    paper_suffix=suffix,
                )
    return figs


def _drag_alias_to_display(prefix: str, container: Any) -> dict[str, str]:
    """Build an alias→display-name map from npz/dict keys matching *prefix*."""
    if container is None:
        return {}
    keys = container.files if hasattr(container, "files") else container.keys()
    out: dict[str, str] = {}
    for k in keys:
        if not k.startswith(prefix):
            continue
        display = k[len(prefix) :]
        alias = resolve_solver_alias(display)
        if alias is not None:
            out[alias] = display
    return out


def _drag_panel_drag_reduction(
    ax: Any, data: dict, alias_to_display: dict, present: set[str]
) -> None:
    """Draw the drag-reduction-vs-iteration panel; updates ``present`` aliases."""
    for alias in _DRAG_OPT_SOLVER_ORDER:
        display_name = alias_to_display.get(alias, alias)
        sdata = data["by_solver"].get(display_name)
        if sdata is None:
            continue
        drags = sdata.get("drags", [])
        if not drags or not drags[0] or np.isnan(drags[0]) or drags[0] == 0:
            continue
        drag_0 = drags[0]
        step = max(1, len(drags) // 50)
        indices = list(range(0, len(drags), step))
        if indices[-1] != len(drags) - 1:
            indices.append(len(drags) - 1)
        reductions = [(drag_0 - drags[i]) / drag_0 * 100 for i in indices]
        _label, color, ls, _mk = solver_props(alias)
        ax.plot(indices, reductions, color=color, linestyle=ls, linewidth=1.6)
        present.add(alias)

    ax.set_title("Drag reduction")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Drag reduction (%)")
    ax.set_ylim(bottom=0)


def _drag_panel_profiles(
    ax: Any, profiles: Any, alias_to_display: dict, present: set[str]
) -> None:
    """Draw the initial + final inlet profiles panel; updates ``present`` aliases."""
    if profiles is not None and "initial" in profiles:
        y_arr = np.linspace(0, 1, profiles["initial"].shape[0])
        ax.plot(
            profiles["initial"], y_arr, color="#999999", linestyle="--", linewidth=1.4
        )
        for alias in _DRAG_OPT_SOLVER_ORDER:
            display_name = alias_to_display.get(alias, alias)
            if f"final_{display_name}" not in profiles:
                continue
            _label, color, ls, _mk = solver_props(alias)
            ax.plot(
                profiles[f"final_{display_name}"],
                y_arr,
                color=color,
                linestyle=ls,
                linewidth=1.6,
            )
            present.add(alias)

    ax.set_title("Optimised profile")
    ax.set_xlabel(r"$u_x$")
    ax.set_ylabel("$y$")


def _drag_panel_history(
    imshow_axes: Any,
    hist_solvers: Any,
    hist_alias_to_display: Any,
    profiles: Any,
    data: Any,
) -> None:
    """Draw the profile-history imshow panels (one per solver row)."""
    for idx, (ax_im, alias) in enumerate(zip(imshow_axes, hist_solvers, strict=False)):
        display_name = hist_alias_to_display.get(alias, alias)
        hist = profiles[f"profile_history_{display_name}"]
        label, color, _ls, _mk = solver_props(alias)
        ax_im.imshow(
            hist.T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            interpolation="bilinear",
        )
        n_snaps = hist.shape[0]
        n_iters = len(data["by_solver"].get(display_name, {}).get("drags", [1]))
        snap_step = n_iters / max(n_snaps - 1, 1)
        tick_pos = [0, n_snaps // 2, n_snaps - 1]
        ax_im.set_xticks(tick_pos)
        ax_im.set_xticklabels([f"{int(t * snap_step)}" for t in tick_pos], fontsize=6.5)
        ax_im.tick_params(labelsize=6.5)
        ax_im.set_yticks([])
        ax_im.text(
            0.03,
            0.95,
            label,
            transform=ax_im.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=color,
            bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1.0},
        )
        if idx == 0:
            ax_im.set_title("Profile history")
        if idx < len(hist_solvers) - 1:
            ax_im.tick_params(labelbottom=False)
        else:
            ax_im.set_xlabel("Iteration", fontsize=7.0)

    for ax_im in imshow_axes[len(hist_solvers) :]:
        ax_im.set_visible(False)


def _drag_opt_figure(
    cfg: Problem,
    *,
    exp_key: str = "drag_opt",
    suffix: str = "",
    save: bool = True,
) -> plt.Figure | None:
    """Single-experiment styled drag-optimisation figure.

    Layout (3-column GridSpec):
      * col 0 — drag reduction (%) vs iteration
      * col 1 — final + initial inlet profiles
      * col 2 — profile-history imshow, one row per solver
    """
    out_dir = experiment_dir(results_dir(), cfg.name, "optimization", exp_key + suffix)
    result_path = out_dir / "result.json"
    profiles_path = out_dir / "profiles.npz"

    if not result_path.exists():
        print(f"[drag_opt] {result_path} not found — skipping")
        return None

    plt.rcParams.update(RCPARAMS)

    data = v1_to_legacy(load_json(result_path))
    profiles = try_load_npz(profiles_path) if profiles_path.exists() else None

    # Both ``by_solver`` and ``profiles`` are keyed by spec.name (display form).
    # Build alias→display maps so the alias-ordered _DRAG_OPT_SOLVER_ORDER loop
    # can index display-keyed data.
    hist_alias_to_display = _drag_alias_to_display("profile_history_", profiles)
    final_alias_to_display = _drag_alias_to_display("final_", profiles)
    by_solver_alias_to_display: dict[str, str] = {}
    for name in data.get("by_solver", {}):
        alias = resolve_solver_alias(name)
        if alias is not None:
            by_solver_alias_to_display[alias] = name
    alias_to_display = {**by_solver_alias_to_display, **final_alias_to_display}

    hist_solvers = [s for s in _DRAG_OPT_SOLVER_ORDER if s in hist_alias_to_display]
    n_rows = max(len(hist_solvers), 1)

    fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * (0.14 + 0.13 * n_rows)), dpi=300)
    gs = gridspec.GridSpec(
        n_rows,
        3,
        figure=fig,
        width_ratios=[1.4, 0.9, 1.1],
        left=0.10,
        right=0.97,
        top=0.93,
        bottom=0.22,
        hspace=0.12,
        wspace=0.45,
    )
    ax_drag = fig.add_subplot(gs[:, 0])
    ax_prof = fig.add_subplot(gs[:, 1])
    imshow_axes = [fig.add_subplot(gs[r, 2]) for r in range(n_rows)]

    present: set[str] = set()
    _drag_panel_drag_reduction(ax_drag, data, alias_to_display, present)
    _drag_panel_profiles(ax_prof, profiles, alias_to_display, present)
    _drag_panel_history(
        imshow_axes, hist_solvers, hist_alias_to_display, profiles, data
    )

    handles = [
        mlines.Line2D(
            [], [], color="#999999", linestyle="--", linewidth=1.4, label="Initial"
        ),
        *dedup_handles(
            [make_handle(s) for s in NS_ORDER if s in present and s in SOLVER_STYLES]
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(handles), 5),
        fontsize=7.5,
        framealpha=0.7,
        edgecolor="0.8",
        handlelength=2.0,
    )

    if save:
        out = out_dir / f"{exp_key}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


def _vel_components_2d(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_x, u_y) 2-D arrays from field (N, N, 1, 2) or (N, N, 2)."""
    if v.ndim == 4:
        v = v[:, :, 0, :]
    return v[..., 0], v[..., 1]


def _plot_drag_opt_fields(
    data: dict,
    out_dir: Any,
    run_name: str,
    title_suffix: str,
    styles: dict,
    save: bool,
    figs: list,
) -> None:
    """Render u_x and u_y velocity component fields from ``flow_fields.npz``.

    The npz is expected to contain keys ``flow_initial`` and
    ``flow_final_{solver_name}`` with shape (N, N, 1, 2) or (N, N, 2).
    If the file does not exist the function silently returns.

    One row per solver (plus initial), two columns: u_x | u_y.
    Solvers that did not converge (converged=False in by_solver) are annotated
    with a red border so the failure is immediately visible.
    """
    fields_path = Path(out_dir) / "flow_fields.npz"
    if not fields_path.exists():
        return

    npz = try_load_npz(fields_path)
    by_solver = data.get("by_solver", {})
    # ``try_load_npz`` returns a plain dict; tolerate both dict and NpzFile.
    npz_keys = set(npz.files if hasattr(npz, "files") else npz.keys())

    # Drive the rows from ``by_solver`` (the canonical solver set for this
    # result) rather than scanning every ``flow_final_*`` npz key.  The npz can
    # accumulate stale per-solver entries from earlier runs (e.g. differently
    # cased aliases like ``flow_final_XLB`` alongside ``flow_final_xlb``); those
    # have no ``by_solver`` metadata and would render as blank/duplicate rows.
    # Additionally dedup by canonical alias: if ``by_solver`` carries both a
    # display name and an alias mapping to the same solver, keep only the first.
    solver_names_clean: list[str] = []
    seen_aliases: set[str] = set()
    for s in by_solver:
        if f"flow_final_{s}" not in npz_keys:
            continue
        alias = resolve_solver_alias(s) or s
        if alias in seen_aliases:
            continue
        seen_aliases.add(alias)
        solver_names_clean.append(s)

    if not solver_names_clean:
        return

    # Rows: initial + one per solver.  Columns: u_x | u_y.
    n_rows = 1 + len(solver_names_clean)
    ncols = 2
    fig_fld, axes_fld = paper_image_grid(n_rows, ncols)
    # Widen the figure to host a dedicated left gutter for the (multi-line) row
    # labels without stealing width from the panels/colorbars.  ``IMG_PANEL`` is
    # ~1.45" per panel; reserve a fixed ~1.5" gutter on the left.
    panel_w = fig_fld.get_size_inches()[0] / ncols
    gutter_w = 1.5
    fig_w = ncols * panel_w + gutter_w
    fig_h = fig_fld.get_size_inches()[1]
    fig_fld.set_size_inches(fig_w, fig_h)
    left_frac = gutter_w / fig_w
    # Reserve the left gutter and keep generous vertical/horizontal spacing so
    # labels, column titles, suptitle and colorbars never collide.  Done before
    # rendering so ``ax.get_position()`` reflects the final layout.
    fig_fld.subplots_adjust(
        left=left_frac,
        right=0.97,
        top=0.90,
        bottom=0.03,
        hspace=0.30,
        wspace=0.55,
    )

    # Compute shared colour scales from the initial flow so all panels are comparable.
    flow_init = npz.get("flow_initial")
    if flow_init is not None:
        ux_init, uy_init = _vel_components_2d(flow_init)
        ux_vmax = float(np.percentile(np.abs(ux_init), 99)) or 1.0
        uy_vmax = float(np.percentile(np.abs(uy_init), 99)) or 1.0
    else:
        ux_vmax, uy_vmax = 1.0, 0.5

    def _render_row(
        row_idx: int, label: str, field: np.ndarray, converged: bool | None
    ):
        ux, uy = _vel_components_2d(field)

        for col, (arr, cmap, vmin, vmax, col_title) in enumerate(
            [
                (ux, "RdBu_r", -ux_vmax, ux_vmax, "$u_x$"),
                (uy, "RdBu_r", -uy_vmax, uy_vmax, "$u_y$"),
            ]
        ):
            ax = axes_fld[row_idx, col]
            imshow_with_cbar(
                ax,
                fig_fld,
                arr.T,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            if row_idx == 0:
                ax.set_title(col_title, fontweight="bold", pad=6)
            ax.axis("off")

        # Row label placed in the left figure gutter (reserved via
        # subplots_adjust above).  Right-aligned so the label always ends a
        # fixed gap left of the panel regardless of its length — multi-line
        # solver labels never overlap the panels, colorbars or adjacent rows.
        ax0 = axes_fld[row_idx, 0]
        pos = ax0.get_position()
        fig_fld.text(
            pos.x0 - 0.02,
            (pos.y0 + pos.y1) / 2,
            label,
            fontsize=7,
            va="center",
            ha="right",
            linespacing=1.3,
        )

        # Red border annotation for non-converged / poor solvers
        if converged is False:
            for col in range(ncols):
                ax_c = axes_fld[row_idx, col]
                for spine in ax_c.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)

    # Initial row
    if flow_init is not None:
        _render_row(0, "Initial flow", flow_init, converged=None)
    else:
        for col in range(ncols):
            axes_fld[0, col].set_visible(False)

    # One row per solver
    for i, sname in enumerate(solver_names_clean):
        key = f"flow_final_{sname}"
        field = npz.get(key)
        if field is None:
            for col in range(ncols):
                axes_fld[i + 1, col].set_visible(False)
            continue
        solver_info = by_solver.get(sname, {})
        converged = solver_info.get("converged")
        final_drag = solver_info.get("final_drag")
        sty = styles.get(sname, {})
        label = sty.get("label", sname)
        if final_drag is not None:
            label += f"\ndrag={final_drag:.4g}"
        if converged is False:
            label += "\n[NOT CONVERGED]"
        _render_row(i + 1, label, field, converged)

    fig_fld.suptitle("Optimised flow fields", fontweight="bold", y=0.97)
    if save:
        save_fig(fig_fld, "drag_opt_fields", out_dir)
    figs.append(fig_fld)


def _render_drag_opt_evolution_gifs(
    profiles: Any,
    out_dir: Any,
    solver_names: list,
    styles: dict,
    run_name: str,
) -> None:
    """Write a single combined ``drag_opt_evolution.gif`` for all solvers.

    Builds one figure with a row of subplots — one panel per solver that has
    a recorded ``profile_history_<name>``. Each panel animates that solver's
    inflow profile u_x(y) over BFGS iterations, with the initial profile drawn
    dashed as a reference. Frames are synchronised across panels: at frame *k*
    every panel shows its iteration-*k* state, and solvers with fewer snapshots
    hold (clamp to) their last frame. A shared solver legend is placed below
    the row. Solvers are deduplicated by canonical alias so each appears once.
    """
    initial = np.asarray(profiles["initial"])
    N = initial.size
    y = np.linspace(0, 1, N, endpoint=False) + 0.5 / N

    # Treat np.load NpzFile *and* plain dict uniformly via .files / keys.
    keys = set(profiles.files) if hasattr(profiles, "files") else set(profiles.keys())

    # Collect one entry per solver that has a usable history, deduped by alias.
    panels: list[dict] = []
    seen_aliases: set[str] = set()
    for name in solver_names:
        hkey = f"profile_history_{name}"
        if hkey not in keys:
            continue
        hist = np.asarray(profiles[hkey])  # (n_snaps, N)
        if hist.ndim != 2 or hist.shape[0] == 0:
            continue
        alias = resolve_solver_alias(name)
        dedup_key = alias if alias is not None else name
        if dedup_key in seen_aliases:
            continue
        seen_aliases.add(dedup_key)

        label, color, _ls, _mk = solver_props(name)
        panels.append(
            {
                "name": name,
                "alias": alias,
                "label": label,
                "color": color,
                "hist": hist,
                "n": int(hist.shape[0]),
            }
        )

    if not panels:
        return

    n_panels = len(panels)
    n_frames = max(p["n"] for p in panels)

    fig, axes = paper_row(n_panels, squeeze=False)
    axes = np.atleast_1d(axes).ravel()

    lines: list = []
    for ax, p in zip(axes, panels, strict=True):
        hist = p["hist"]
        extrema = [
            float(hist.min()),
            float(hist.max()),
            float(initial.min()),
            float(initial.max()),
        ]
        xlo, xhi = min(extrema), max(extrema)
        pad = 0.05 * (xhi - xlo + 1e-12)
        xlo -= pad
        xhi += pad

        ax.plot(initial, y, "k--", lw=1.4, label="initial")
        (line,) = ax.plot(hist[0], y, color=p["color"], lw=2.0, label=p["label"])
        lines.append(line)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$u_x$")
        ax.set_ylabel("$y$")
        ax.set_title(p["label"])

    sup = (
        f"Inflow profile evolution — {run_name}"
        if run_name
        else "Inflow profile evolution"
    )
    fig.suptitle(sup)

    # Shared solver legend below the row (canonical order, deduped by alias).
    initial_handle = mlines.Line2D(
        [], [], color="k", linestyle="--", lw=1.4, label="initial"
    )
    solver_legend(
        fig,
        [p["alias"] for p in panels if p["alias"] is not None],
        order=NS_ORDER,
        extra_handles=[initial_handle],
    )

    def _update(
        idx: Any,
        _lines: Any = lines,
        _panels: Any = panels,
    ) -> Any:
        for _line, _p in zip(_lines, _panels, strict=True):
            k = min(idx, _p["n"] - 1)  # clamp: hold last frame
            _line.set_xdata(_p["hist"][k])
        return tuple(_lines)

    anim = manimation.FuncAnimation(
        fig, _update, frames=n_frames, interval=250, blit=False
    )
    _save_animation(anim, "drag_opt_evolution", out_dir, fps=4)
