"""Topology-optimisation single-experiment + paper-figure generator.

Two public entry points:

  * :func:`plot_experiment(cfg, *, exp_key, suffix, save)` — the canonical
    single-experiment paper figure: compliance vs iteration + final
    optimised density volumes (one 3-D voxel panel per solver) for one
    problem (e.g. structural-mesh). Reads ``result.json`` from
    ``<results>/<cfg.name>/optimization/<exp_key><suffix>/`` and writes a
    paper-quality PDF in the same directory.
    Used both as the per-experiment plot delegate (called from
    :func:`mosaic.benchmarks.problems.structural_mesh.plots.plot_topopt`)
    and as the source figure for the paper-output pipeline.
  * :func:`generate(out_dir)` — paper-output entry point. Produces
    ``topopt_convergence.pdf`` (structural+thermal combined),
    ``conductivity_recovery.pdf``, and per-domain ``*_topopt_fields.pdf``
    files for the build pipeline, plus the canonical single-experiment
    figure for structural-mesh.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.io import (
    experiment_dir,
    load_json,
    results_dir,
    try_load_npz,
)
from mosaic.benchmarks.plots.paper import TEXTWIDTH
from mosaic.benchmarks.plots.paper.style import (
    RCPARAMS,
    SOLVER_STYLES,
    STRUCTURAL_ORDER,
    THERMAL_ORDER,
    dedup_handles,
    make_handle,
    solver_props,
)

# ── 3-D voxel rendering helpers (shared) ─────────────────────────────────────

_THRESH = 0.35
_ELEV = 22
_AZIM = 35
_CLR_FIXED = "#888888"
_CLR_LOAD = "#FF1744"


def _add_bcs(ax, nx: int, ny: int, nz: int, ph: dict) -> None:
    """Overlay fixed-face patch and load arrow on a voxel Axes3D."""
    wall = Poly3DCollection(
        [[(0, 0, 0), (0, ny, 0), (0, ny, nz), (0, 0, nz)]],
        alpha=0.55,
        facecolor=_CLR_FIXED,
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax.add_collection3d(wall)

    corner_z_high = ph.get("corner_z_high", False)
    corner_y_high = ph.get("corner_y_high", False)
    load_axis = ph.get("load_axis", "z")
    ly = (ny - 0.5) if corner_y_high else 0.5
    lz = (nz - 0.5) if corner_z_high else 0.5

    arrow_len = nz * 0.65
    if load_axis == "y":
        y_sign = -1 if corner_y_high else 1
        dx, dy_a, dz_a = 0, y_sign * arrow_len, 0
    else:
        z_sign = -1 if corner_z_high else 1
        dx, dy_a, dz_a = 0, 0, z_sign * arrow_len

    ox = nx + 0.6 if load_axis != "y" else nx
    ax.quiver(
        ox,
        ly,
        lz,
        dx,
        dy_a,
        dz_a,
        color=_CLR_LOAD,
        linewidth=2.5,
        arrow_length_ratio=0.28,
    )


def _voxel_facecolors(
    rho_xyz: np.ndarray,
    filled: np.ndarray,
    base_color: str,
) -> np.ndarray:
    """Return RGBA facecolor array; empty voxels are fully transparent."""
    import matplotlib.colors as mcolors

    r, g, b, _ = mcolors.to_rgba(base_color)
    fc = np.zeros((*rho_xyz.shape, 4))
    norm = np.where(filled, (rho_xyz - _THRESH) / (1.0 - _THRESH), 0.0)
    fc[..., 0] = r + (1 - r) * (1 - norm) * 0.45
    fc[..., 1] = g + (1 - g) * (1 - norm) * 0.45
    fc[..., 2] = b + (1 - b) * (1 - norm) * 0.45
    fc[..., 3] = filled.astype(float)
    return fc


# ── Single-experiment canonical figure ───────────────────────────────────────


def _solver_order_for(cfg_name: str) -> list[str]:
    """Best-effort solver-ordering pick based on problem name."""
    if "structural" in cfg_name:
        return STRUCTURAL_ORDER
    if "thermal" in cfg_name:
        return THERMAL_ORDER
    return STRUCTURAL_ORDER


def plot_experiment(
    cfg: Problem,
    *,
    exp_key: str = "topopt",
    suffix: str = "",
    save: bool = True,
    **_kw,
) -> plt.Figure | None:
    """Single-experiment topopt figure (compliance convergence + final fields).

    Layout: two-row composite —
      * top row spans full width: compliance vs iteration per solver
      * bottom row: 3-D voxel renders of the final optimised density,
        one panel per solver (only when ``topopt_fields.npz`` is present)

    Reads ``result.json`` (+ optional ``topopt_fields.npz``,
    ``params.json``) from the experiment directory and writes
    ``<exp_key>.pdf`` next to them when ``save`` is True.
    """
    out_dir = experiment_dir(results_dir(), cfg.name, "optimization", exp_key + suffix)
    result_path = out_dir / "result.json"
    if not result_path.exists():
        print(f"[topopt] {result_path} not found — skipping")
        return None

    plt.rcParams.update(RCPARAMS)
    data = load_json(result_path)
    by_solver = data.get("by_solver", {})

    solver_order = _solver_order_for(cfg.name)

    # Optional fields npz + physics params
    fields_path = out_dir / "topopt_fields.npz"
    params_path = out_dir / "params.json"
    npz = try_load_npz(fields_path) if fields_path.exists() else None
    params = load_json(params_path) if params_path.exists() else None
    ph = (params or data.get("params", {}) or {}).get("physics", {}) or {}

    have_fields = (
        npz is not None
        and "solver_names" in npz
        and ph.get("nx")
        and ph.get("ny")
        and ph.get("nz")
    )

    if have_fields:
        npz_solvers = list(npz["solver_names"])
        n_field_panels = len(npz_solvers)
        ncols = max(1, n_field_panels)
        fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.62), dpi=300)
        gs = fig.add_gridspec(
            2, ncols, height_ratios=[1.0, 1.35], hspace=0.45, top=0.93, bottom=0.18
        )
        ax_c = fig.add_subplot(gs[0, :])
        field_axes = [
            fig.add_subplot(gs[1, i], projection="3d") for i in range(n_field_panels)
        ]
    else:
        fig, ax_c = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.38), dpi=300)
        fig.subplots_adjust(bottom=0.32, top=0.92)
        field_axes = []
        npz_solvers = []

    # ── Compliance vs iteration ──────────────────────────────────────────────
    present: set[str] = set()
    for solver, sdata in by_solver.items():
        _label, color, ls, _mk = solver_props(solver)
        compliances = sdata.get("compliances", [])
        if compliances:
            ax_c.semilogy(
                range(len(compliances)),
                compliances,
                color=color,
                linestyle=ls,
                linewidth=1.6,
            )
            present.add(solver)

    ax_c.set_title(f"Compliance — {cfg.category_label or cfg.name}")
    ax_c.set_xlabel("Iteration")
    ax_c.set_ylabel("Compliance")

    # ── 3-D voxel field panels ───────────────────────────────────────────────
    if have_fields:
        nx, ny, nz = int(ph["nx"]), int(ph["ny"]), int(ph["nz"])
        for i, sname in enumerate(npz_solvers):
            ax = field_axes[i]
            rho_flat = npz[f"rho_final_{i}"]
            rho_xyz = rho_flat.reshape(nz, ny, nx).transpose(2, 1, 0)
            filled = rho_xyz > _THRESH

            _label, color, _ls, _mk = solver_props(sname)
            fc = _voxel_facecolors(rho_xyz, filled, color)
            ax.voxels(filled, facecolors=fc, edgecolors=fc, shade=True)
            _add_bcs(ax, nx, ny, nz, ph)

            label = SOLVER_STYLES.get(sname, (sname,))[0]
            ax.set_title(label, fontsize=7.5, pad=-4)
            ax.view_init(elev=_ELEV, azim=_AZIM)
            ax.set_axis_off()
            present.add(sname)

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = dedup_handles(
        [make_handle(s) for s in solver_order if s in present and s in SOLVER_STYLES]
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(handles), 5),
            fontsize=7.5,
            framealpha=0.7,
            handlelength=2.0,
        )

    if save:
        out = out_dir / f"{exp_key}.pdf"
        fig.savefig(out)
        print(f"Saved {out}")
    return fig


# ── Cross-paper combined convergence figure ──────────────────────────────────


def _plot_combined_convergence(out_path: Path) -> None:
    domains = [
        (
            "Structural",
            results_dir()
            / "structural-mesh"
            / "optimization"
            / "topopt"
            / "result.json",
            STRUCTURAL_ORDER,
        ),
        (
            "Thermal",
            results_dir() / "thermal-mesh" / "optimization" / "topopt" / "result.json",
            THERMAL_ORDER,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.38))
    fig.subplots_adjust(bottom=0.42, wspace=0.35)

    all_present: set[str] = set()
    all_order: list[str] = []

    for ax, (domain_label, result_path, solver_order) in zip(
        axes, domains, strict=False
    ):
        if not result_path.exists():
            print(f"[topopt] {result_path} not found — skipping {domain_label}")
            ax.set_title(f"{domain_label} (no data)")
            ax.set_visible(False)
            continue
        data = load_json(result_path)
        by_solver = data["by_solver"]

        for solver, sdata in by_solver.items():
            _label, color, ls, _mk = SOLVER_STYLES.get(
                solver, (solver, "#888888", "-", "o")
            )
            compliances = sdata.get("compliances", [])
            if compliances:
                ax.semilogy(
                    range(len(compliances)),
                    compliances,
                    color=color,
                    linestyle=ls,
                    marker="",
                    linewidth=1.6,
                )
            all_present.add(solver)

        for s in solver_order:
            if s not in all_order:
                all_order.append(s)

        ax.set_title(domain_label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Compliance" if ax is axes[0] else "")

    handles = [
        mlines.Line2D(
            [],
            [],
            color=SOLVER_STYLES[s][1],
            linestyle=SOLVER_STYLES[s][2],
            linewidth=1.6,
            label=SOLVER_STYLES[s][0],
        )
        for s in all_order
        if s in all_present and s in SOLVER_STYLES
    ]
    ncols = max(1, len(handles) // 2)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=ncols,
        fontsize=7.5,
        framealpha=0.7,
        handlelength=2.0,
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_fields(
    domain_label: str,
    fields_path: Path,
    params_path: Path,
    out_path: Path,
) -> None:
    if not fields_path.exists():
        print(f"Skipping fields figure — {fields_path} not found")
        return

    npz = try_load_npz(fields_path)
    params = load_json(params_path)
    ph = params["physics"]
    nx, ny, nz = ph["nx"], ph["ny"], ph["nz"]

    npz_solvers = list(npz["solver_names"])

    n = len(npz_solvers)
    nrows, ncols = 2, math.ceil(n / 2)

    fig = plt.figure(figsize=(TEXTWIDTH, TEXTWIDTH * 0.58))

    for i, sname in enumerate(npz_solvers):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")

        rho_flat = npz[f"rho_final_{i}"]
        rho_xyz = rho_flat.reshape(nz, ny, nx).transpose(2, 1, 0)
        filled = rho_xyz > _THRESH

        _, color, _, _ = SOLVER_STYLES.get(sname, (sname, "#555555", "-", "o"))
        fc = _voxel_facecolors(rho_xyz, filled, color)
        ax.voxels(filled, facecolors=fc, edgecolors=fc, shade=True)
        _add_bcs(ax, nx, ny, nz, params["physics"])

        label = SOLVER_STYLES.get(sname, (sname,))[0]
        ax.set_title(label, fontsize=7.5, pad=-4)
        ax.view_init(elev=_ELEV, azim=_AZIM)
        ax.set_axis_off()

    fig.subplots_adjust(wspace=0.0, hspace=-0.08, top=0.88, bottom=0.0)
    fig.suptitle(domain_label, fontsize=8, y=0.97)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_conductivity_recovery(out_path: Path) -> None:
    result_path = (
        results_dir()
        / "thermal-mesh"
        / "optimization"
        / "conductivity_recovery"
        / "result.json"
    )
    fields_png = (
        results_dir()
        / "thermal-mesh"
        / "optimization"
        / "conductivity_recovery"
        / "conductivity_recovery_fields.png"
    )

    if not result_path.exists():
        print(f"[topopt] {result_path} not found — skipping conductivity recovery")
        return

    data = load_json(result_path)
    by_solver = data["by_solver"]

    has_fields = fields_png.exists()
    ncols = 2 if has_fields else 1
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(TEXTWIDTH, TEXTWIDTH * 0.38),
        gridspec_kw={"width_ratios": [1, 1.6]} if has_fields else {},
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    ax_conv = axes[0] if has_fields else axes

    present: set[str] = set()
    for solver, sdata in by_solver.items():
        errors = sdata.get("errors", [])
        if not errors:
            continue
        label, color, ls, _mk = SOLVER_STYLES.get(solver, (solver, "#888888", "-", "o"))
        ax_conv.semilogy(
            range(len(errors)),
            errors,
            color=color,
            linestyle=ls,
            linewidth=1.6,
            label=label,
        )
        present.add(solver)

    ax_conv.set_title("Thermal — conductivity recovery")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Identification error")

    handles = [
        mlines.Line2D(
            [],
            [],
            color=SOLVER_STYLES[s][1],
            linestyle=SOLVER_STYLES[s][2],
            linewidth=1.6,
            label=SOLVER_STYLES[s][0],
        )
        for s in THERMAL_ORDER
        if s in present and s in SOLVER_STYLES
    ]
    ax_conv.legend(
        handles=handles,
        fontsize=7.5,
        framealpha=0.7,
        handlelength=2.0,
        loc="upper right",
    )

    if has_fields:
        ax_fields = axes[1]
        img = plt.imread(str(fields_png))
        ax_fields.imshow(img, aspect="auto", interpolation="bilinear")
        ax_fields.set_title("Recovered conductivity fields")
        ax_fields.axis("off")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate(out_dir: Path) -> None:
    """Paper-output entry point: cross-domain combined + per-domain field figures."""
    from mosaic.benchmarks.problems import get_config

    with plt.rc_context(RCPARAMS):
        _plot_combined_convergence(out_path=out_dir / "topopt_convergence.pdf")
        _plot_conductivity_recovery(out_path=out_dir / "conductivity_recovery.pdf")
        _plot_fields(
            domain_label="Structural",
            fields_path=results_dir()
            / "structural-mesh"
            / "optimization"
            / "topopt"
            / "topopt_fields.npz",
            params_path=results_dir()
            / "structural-mesh"
            / "optimization"
            / "topopt"
            / "params.json",
            out_path=out_dir / "structural_topopt_fields.pdf",
        )
        _plot_fields(
            domain_label="Thermal",
            fields_path=results_dir()
            / "thermal-mesh"
            / "optimization"
            / "topopt"
            / "topopt_fields.npz",
            params_path=results_dir()
            / "thermal-mesh"
            / "optimization"
            / "topopt"
            / "params.json",
            out_path=out_dir / "thermal_topopt_fields.pdf",
        )

        # Canonical single-experiment figure (structural-mesh).
        try:
            cfg = get_config("structural-mesh")
        except Exception:
            return
        fig = plot_experiment(cfg, exp_key="topopt", suffix="", save=False)
        if fig is not None:
            plt.close(fig)
