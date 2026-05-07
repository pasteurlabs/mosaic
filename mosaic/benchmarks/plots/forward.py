"""Plots for the forward suite (agreement, physical_laws)."""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

from mosaic.benchmarks.core.config import ProblemConfig
from mosaic.benchmarks.core.utils import load_json, results_dir
from mosaic.benchmarks.plots.style import (
    apply_style,
    field_grid,
    fig_shared_legend,
    save_fig,
    solver_plot_props,
    solver_styles,
    subplots_grid,
    unit_label,
    vorticity_2d,
)

apply_style()


def _smooth(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian smooth a 1-D array (for noisy per-snapshot observables like RDF)."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(arr, k, mode="same")


def _field_to_2d(cfg):
    """Return the field→2D callable for *cfg*, falling back to vorticity_2d."""
    return cfg.field_to_2d if cfg.field_to_2d is not None else vorticity_2d


def _field_grid_kw(cfg) -> dict:
    """Return field_grid keyword overrides for the problem's colormap / symmetry."""
    return {"cmap": cfg.field_cmap, "symmetric": cfg.field_symmetric}


def _is_field(arr: np.ndarray) -> bool:
    """True if *arr* is a plottable field (at least 2-D and not a scalar stub)."""
    return arr.ndim >= 2


_SUITE = "forward"


# ── agreement ─────────────────────────────────────────────────────────────────


def plot_agreement(
    cfg: ProblemConfig, save: bool = True, suffix: str = "", exp_key: str = "agreement"
):
    """Field-error grid (rows=solvers × cols=sweep values) + optional power spectra."""
    out_dir = results_dir() / cfg.name / _SUITE / f"{exp_key}{suffix}"
    fields_path = out_dir / "fields.npz"
    if not fields_path.exists():
        # Multi-IC layout: each IC lands in a subdir; plot each one.
        subdirs = (
            sorted(
                p
                for p in out_dir.iterdir()
                if p.is_dir() and (p / "fields.npz").exists()
            )
            if out_dir.exists()
            else []
        )
        if subdirs:
            for sub in subdirs:
                plot_agreement(
                    cfg, save=save, suffix=f"{suffix}/{sub.name}", exp_key=exp_key
                )
            return None
        return None

    data = load_json(out_dir / "result.json")
    sweep_key = data.get("sweep_key", "param")
    reference_label = data.get("reference_label", "consensus")
    styles = solver_styles(cfg)
    f2d = _field_to_2d(cfg)

    npz = np.load(fields_path, allow_pickle=True)
    sweep_vals = npz["sweep_values"].tolist()
    solver_names = npz["solver_names"].tolist()
    n_vals = len(sweep_vals)

    # ── detect comparison type from consensus shape ───────────────────────────
    sample_consensus = npz["consensus_0"] if "consensus_0" in npz else None
    curve_mode = sample_consensus is not None and sample_consensus.ndim == 1

    # Scalar outputs (ndim == 0): plot the scalar value vs sweep parameter.
    if sample_consensus is not None and sample_consensus.ndim == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        all_y: list[float] = []
        solver_series: list[tuple] = []
        for j, name in enumerate(solver_names):
            y_vals = [
                float(npz[f"{name}_{i}"]) if f"{name}_{i}" in npz else np.nan
                for i in range(n_vals)
            ]
            style = styles.get(name, {})
            props = solver_plot_props(style)
            lbl = style.get("label", name)
            solver_series.append((y_vals, props, lbl))
            all_y.extend(v for v in y_vals if np.isfinite(v) and v > 0)
        # Use log-y when the output spans more than one decade (e.g. SIMP compliance)
        if all_y and max(all_y) / min(all_y) > 10:
            ax.set_yscale("log")
        for y_vals, props, lbl in solver_series:
            ax.plot(sweep_vals, y_vals, label=lbl, **props)
        ax.set_xlabel(unit_label(sweep_key, cfg.units))
        ylabel = cfg.output_key.replace("_", " ")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{cfg.name} — {ylabel} vs {sweep_key}")
        ax.grid(True, alpha=0.3)
        fig_shared_legend(fig, [ax])
        if save:
            save_fig(fig, "curves", out_dir)
        return fig

    if curve_mode:
        # 1-D observable agreement (e.g. RDF g(r) vs r, or P(k) vs k)
        # Layout: row 0 = absolute curves, row 1 = residual (solver − consensus).
        # The residual row makes %-level differences visible — mandatory in
        # code-comparison papers (Euclid, HACC, etc.) where the absolute panel
        # shows everything agreeing.
        x = npz["x_axis"] if "x_axis" in npz else np.arange(len(sample_consensus))
        fig_agr, ax_grid = plt.subplots(
            2,
            n_vals,
            figsize=(4 * n_vals, 7),
            sharex="col",
            sharey="row",
            squeeze=False,
        )

        for i, val in enumerate(sweep_vals):
            ax_top = ax_grid[0, i]
            ax_bot = ax_grid[1, i]
            cons_key = f"consensus_{i}"
            consensus = _smooth(npz[cons_key]) if cons_key in npz else None

            for j, name in enumerate(solver_names):
                key_s = f"{name}_{i}"
                if key_s not in npz:
                    continue
                curve = _smooth(npz[key_s])
                style = styles.get(name, {})
                props = solver_plot_props(style, marker=False)
                lbl = style.get("label", name)
                ax_top.plot(x, curve, label=lbl, **props)
                if consensus is not None:
                    ax_bot.plot(x, curve - consensus, label=lbl, **props)

            if consensus is not None:
                ax_bot.axhline(0, color="0.55", lw=0.8, ls="--")

            ax_top.set_title(f"{sweep_key}={val:.3g}")
            ax_bot.set_xlabel(cfg.agreement_xlabel)
            if i == 0:
                ax_top.set_ylabel(cfg.agreement_ylabel)
                ax_bot.set_ylabel(f"Δ {cfg.agreement_ylabel}")

        fig_agr.suptitle(f"{cfg.name} — agreement ({cfg.agreement_ylabel})")
        fig_shared_legend(fig_agr, list(ax_grid.flat))
        if save:
            save_fig(fig_agr, "curves", out_dir)
        return fig_agr

    # ── raw field grid: rows=solvers, cols=sweep values ──────────────────────
    raw_panels = []
    for name in solver_names:
        for i, val in enumerate(sweep_vals):
            key_s = f"{name}_{i}"
            if key_s not in npz or not _is_field(npz[key_s]):
                continue
            label = f"{styles.get(name, {}).get('label', name)}\n{sweep_key}={val:.3g}"
            raw_panels.append((label, f2d(npz[key_s])))

    if raw_panels:
        fig_raw = field_grid(
            raw_panels,
            f"{cfg.name} — solver fields",
            ncols=n_vals,
            **_field_grid_kw(cfg),
        )
        if save:
            save_fig(fig_raw, "fields_raw", out_dir)

    # ── field error grid: rows=solvers, cols=sweep values ────────────────────
    panels = []
    for name in solver_names:
        for i, val in enumerate(sweep_vals):
            key_s = f"{name}_{i}"
            key_c = f"consensus_{i}"
            if key_s not in npz or key_c not in npz:
                continue
            if not _is_field(npz[key_s]) or not _is_field(npz[key_c]):
                continue
            err = f2d(npz[key_s]) - f2d(npz[key_c])
            label = f"{styles.get(name, {}).get('label', name)}\n{sweep_key}={val:.3g}"
            panels.append((label, err))

    fig_err = None
    if panels:
        _ref_desc = (
            "analytic solution" if reference_label == "analytic" else "consensus"
        )
        fig_err = field_grid(
            panels,
            f"{cfg.name} — field error vs {_ref_desc}",
            ncols=n_vals,
            **_field_grid_kw(cfg),
        )
        if save:
            save_fig(fig_err, "fields", out_dir)

    # ── error vs sweep param line chart (baseline: spatial convergence; agreement: error vs ν) ──
    by_param = data.get("by_param", {})
    if by_param:
        fig_conv, ax_conv = plt.subplots(figsize=(6, 5))
        param_vals = sorted(by_param.keys(), key=lambda v: float(v))
        all_errs: list[float] = []
        for name in solver_names:
            style = styles.get(name, {})
            xs, ys = [], []
            for pv in param_vals:
                entry = by_param[pv].get(name, {})
                if entry.get("valid") and entry.get("error") is not None:
                    xs.append(float(pv))
                    ys.append(float(entry["error"]))
                    all_errs.append(float(entry["error"]))
            if xs:
                ax_conv.plot(
                    xs, ys, label=style.get("label", name), **solver_plot_props(style)
                )
        if all_errs:
            ax_conv.set_yscale("log")
            if exp_key == "baseline":
                try:
                    ax_conv.set_xscale("log")
                except Exception:
                    pass
        ref_desc = (
            "analytic solution" if reference_label == "analytic" else "solver consensus"
        )
        ax_conv.set_xlabel(unit_label(sweep_key, cfg.units))
        ax_conv.set_ylabel(f"Relative L₂ error vs {ref_desc}")
        title = (
            f"{cfg.name} — spatial convergence (steps=1)"
            if exp_key == "baseline"
            else f"{cfg.name} — inter-solver agreement vs {sweep_key} (ref: {ref_desc})"
        )
        ax_conv.set_title(title)
        ax_conv.grid(True, alpha=0.3, which="both")
        fig_shared_legend(fig_conv, [ax_conv])
        if save:
            save_fig(
                fig_conv, "convergence" if exp_key == "baseline" else "errors", out_dir
            )

    # ── power spectra (one subplot per sweep value, all solvers overlaid) ─────
    ps_fn = cfg.power_spectrum_fn
    if ps_fn is None:
        return fig_err

    fig_ps, axes = subplots_grid(n_vals, panel_w=4, panel_h=4, sharey=True)

    for i, (val, ax) in enumerate(zip(sweep_vals, axes)):
        key_c = f"consensus_{i}"
        for j, name in enumerate(solver_names):
            key_s = f"{name}_{i}"
            if key_s not in npz or not _is_field(npz[key_s]):
                continue
            k, Pk = ps_fn(npz[key_s], domain_extent=cfg.domain_extent)
            style = styles.get(name, {})
            ax.loglog(
                k,
                Pk,
                label=style.get("label", name),
                **solver_plot_props(style, marker=False),
            )
        ax.set_title(f"{sweep_key}={val:.3g}")
        ax.set_xlabel("k  [h/Mpc]")
        if i == 0:
            ax.set_ylabel("P(k)  [(Mpc/h)³]")

    fig_ps.suptitle(f"{cfg.name} — matter power spectrum")
    fig_shared_legend(fig_ps, axes)
    if save:
        save_fig(fig_ps, "power_spectra", out_dir)
    return fig_err


# ── convergence ───────────────────────────────────────────────────────────────


def plot_convergence(cfg: ProblemConfig, save: bool = True, suffix: str = ""):
    """Log-log error vs N + output field panels at each resolution."""
    out_dir = results_dir() / cfg.name / _SUITE / f"convergence{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg)

    # ── error vs N ────────────────────────────────────────────────────────────
    fig_err, ax = plt.subplots(figsize=(6, 5))
    all_N = []
    for name, n_errs in data["errors"].items():
        Ns = sorted(n_errs.keys(), key=int)
        all_N.extend(Ns)
        valid = [
            (int(N), e) for N, e in zip(Ns, [n_errs[N] for N in Ns]) if e is not None
        ]
        if valid:
            Nv, ev = zip(*valid)
            ax.loglog(
                Nv,
                ev,
                label=styles[name]["label"],
                **solver_plot_props(styles[name]),
            )
    ax.set_xlabel(unit_label(cfg.resolution_key, cfg.units))
    ax.set_ylabel("Relative L2 error")
    ax.set_title(f"{cfg.name} — spatial convergence")
    fig_shared_legend(fig_err, [ax])
    if save:
        save_fig(fig_err, "error", out_dir)

    # ── field panels at each N ────────────────────────────────────────────────
    fields_path = out_dir / "fields.npz"
    if not fields_path.exists():
        return fig_err

    npz = np.load(fields_path)
    N_values = npz["N_values"].tolist()
    f2d = _field_to_2d(cfg)
    panels = [
        (f"N={N}", f2d(npz[f"f_{k}"]))
        for k, N in enumerate(N_values)
        if f"f_{k}" in npz and _is_field(npz[f"f_{k}"])
    ]
    if panels:
        fig_fld = field_grid(
            panels,
            f"{cfg.name} — output field by resolution",
            ncols=len(panels),
            **_field_grid_kw(cfg),
        )
        if save:
            save_fig(fig_fld, "fields", out_dir)
    return fig_err


# ── diagnostics ───────────────────────────────────────────────────────────────


def plot_diagnostics(cfg: ProblemConfig, save: bool = True, suffix: str = ""):
    """Scalar diagnostics bars, energy spectra, and density field panels."""
    out_dir = results_dir() / cfg.name / _SUITE / f"diagnostics{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg)

    by_ic = data["by_ic"]
    ic_labels = list(by_ic.keys())
    solvers = list(cfg.solvers)
    x = np.arange(len(ic_labels))
    width = 0.8 / len(solvers)

    # ── scalar diagnostics (one subplot per scalar metric) ────────────────────
    sample_diag = next(
        (v for row in by_ic.values() for v in row.values() if isinstance(v, dict)), {}
    )
    scalar_keys = [
        k
        for k, v in sample_diag.items()
        if isinstance(v, (int, float)) and k != "energy_spectrum"
    ]

    _ZERO_REF_METRICS = {"eigval_min", "n_negative_modes"}

    if scalar_keys:
        fig_sc, axes_sc = subplots_grid(len(scalar_keys), panel_w=5, panel_h=4)
        for col, (ax, dname) in enumerate(zip(axes_sc, scalar_keys)):
            for i, name in enumerate(solvers):
                vals = [
                    (by_ic[lbl].get(name) or {}).get(dname, np.nan)
                    if isinstance(by_ic[lbl].get(name), dict)
                    else np.nan
                    for lbl in ic_labels
                ]
                ax.bar(
                    x + i * width,
                    vals,
                    width,
                    label=styles[name]["label"],
                    color=styles[name]["color"],
                )
            ax.set_xticks(x + width * len(solvers) / 2)
            ax.set_xticklabels(ic_labels, rotation=20, ha="right")
            ax.set_title(dname)
            ax.grid(axis="y")
            ax.grid(False, axis="x")
            if dname in _ZERO_REF_METRICS:
                ax.axhline(0, color="0.35", lw=1.0, ls="--", zorder=3)
            if dname == "eigval_min":
                ax.text(
                    0.98,
                    0.96,
                    "solid",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="0.35",
                    ha="right",
                    va="top",
                )
                ax.text(
                    0.98,
                    0.04,
                    "liquid / gas",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="0.35",
                    ha="right",
                    va="bottom",
                )
        fig_sc.suptitle(f"{cfg.name} — scalar diagnostics")
        fig_shared_legend(fig_sc, axes_sc)
        if save:
            save_fig(fig_sc, "scalars", out_dir)

    # ── energy spectra (one subplot per IC) ───────────────────────────────────
    has_spectra = any(
        isinstance((by_ic[lbl].get(n) or {}).get("energy_spectrum"), dict)
        for lbl in ic_labels
        for n in solvers
    )
    if has_spectra:
        fig_sp, axes_sp = subplots_grid(
            len(ic_labels), panel_w=5, panel_h=4, sharey=True
        )
        for col, (ax, lbl) in enumerate(zip(axes_sp, ic_labels)):
            for name in solvers:
                spec = (by_ic[lbl].get(name) or {}).get("energy_spectrum")
                if isinstance(spec, dict):
                    k = np.array(spec["k"])
                    Ek = np.array(spec["E_k"])
                    ax.loglog(
                        k,
                        Ek,
                        label=styles[name]["label"],
                        **solver_plot_props(styles[name], marker=False),
                    )
            ax.set_xlabel("k")
            ax.set_title(lbl)
        axes_sp[0].set_ylabel("E(k)")
        fig_sp.suptitle(f"{cfg.name} — energy spectra")
        fig_shared_legend(fig_sp, axes_sp)
        if save:
            save_fig(fig_sp, "spectra", out_dir)

    # ── RDF / generic curve diagnostics (one subplot per IC, solvers overlaid) ──
    # Triggered when a diagnostic value is a dict with "r" + "g_r" keys.
    # Plots all solvers on the same axes with a grey confidence band between them.
    rdf_keys = [
        k
        for k, v in sample_diag.items()
        if isinstance(v, dict) and "r" in v and "g_r" in v
    ]
    for dname in rdf_keys:
        fig_rdf, axes_rdf = subplots_grid(
            len(ic_labels), panel_w=5, panel_h=4, sharey=True
        )
        for i_ax, (ax, lbl) in enumerate(zip(axes_rdf, ic_labels)):
            all_g: list[np.ndarray] = []
            r_ref: np.ndarray | None = None
            for name in solvers:
                rdf_data = (by_ic[lbl].get(name) or {}).get(dname)
                if not isinstance(rdf_data, dict):
                    continue
                r = np.array(rdf_data["r"])
                g = _smooth(np.array(rdf_data["g_r"]), sigma=1.0)
                ax.plot(
                    r,
                    g,
                    label=styles[name]["label"],
                    **solver_plot_props(styles[name], marker=False),
                )
                all_g.append(g)
                r_ref = r
            if len(all_g) > 1 and r_ref is not None:
                lo = np.min(all_g, axis=0)
                hi = np.max(all_g, axis=0)
                ax.fill_between(
                    r_ref, lo, hi, alpha=0.15, color="0.5", label="solver spread"
                )
            ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
            ax.set_xlabel("r / σ")
            ax.set_title(lbl)
            if i_ax == 0:
                ax.set_ylabel("g(r)")
        fig_rdf.suptitle(f"{cfg.name} — {dname}")
        fig_shared_legend(fig_rdf, axes_rdf)
        if save:
            save_fig(fig_rdf, dname.replace(" ", "_"), out_dir)

    # ── output field panels ───────────────────────────────────────────────────
    if cfg.diagnostic_fields:
        fields_path = out_dir / "fields.npz"
        if fields_path.exists():
            npz = np.load(fields_path, allow_pickle=True)
            ic_lbl_arr = npz["ic_labels"].tolist()
            solver_names = npz["solver_names"].tolist()
            n_sol = len(solver_names)

            f2d = _field_to_2d(cfg)
            for i, lbl in enumerate(ic_lbl_arr):
                panels = [
                    (styles.get(n, {}).get("label", n), f2d(npz[f"f_{i}_{j}"]))
                    for j, n in enumerate(solver_names)
                    if f"f_{i}_{j}" in npz and _is_field(npz[f"f_{i}_{j}"])
                ]
                if panels:
                    stem = "fields_" + re.sub(r"[ /]", "_", re.sub(r"[()=]", "", lbl))
                    fig_fld = field_grid(
                        panels,
                        f"{cfg.name} — density | {lbl}",
                        ncols=min(len(panels), n_sol),
                        **_field_grid_kw(cfg),
                    )
                    if save:
                        save_fig(fig_fld, stem, out_dir)

    # ── pairwise diagnostics ─────────────────────────────────────────────────
    by_ic_pw = data.get("by_ic_pairwise") or {}
    if not by_ic_pw:
        return

    sample_pw = next(iter(by_ic_pw.values()), {})
    pw_names = list(sample_pw.keys())

    _tab10 = plt.cm.get_cmap("tab10")

    for dname in pw_names:
        sample_val = next(
            (v for ic_res in by_ic_pw.values() for v in ic_res.get(dname, {}).values()),
            None,
        )
        if sample_val is None:
            continue

        if isinstance(sample_val, dict) and "k" in sample_val and "r_k" in sample_val:
            # ── curve-mode: one subplot per IC, all pairs overlaid ────────────
            fig, axes = subplots_grid(len(ic_labels), panel_w=5, panel_h=4, sharey=True)
            ylabel = cfg.pairwise_ylabels.get(dname, "r(k)")
            for i_ax, (ax, lbl) in enumerate(zip(axes, ic_labels)):
                ic_res = by_ic_pw.get(lbl, {}).get(dname, {})
                for p_idx, (pair_label, pdata) in enumerate(ic_res.items()):
                    if not isinstance(pdata, dict):
                        continue
                    x = np.array(pdata.get("k", []))
                    y = np.array(pdata.get("r_k", []))
                    if len(x) == 0:
                        continue
                    ax.plot(x, y, color=_tab10(p_idx % 10), lw=1.5, label=pair_label)
                ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
                ax.set_xlabel(cfg.pairwise_xlabel)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(lbl)
                if i_ax == 0:
                    ax.set_ylabel(ylabel)
            fig.suptitle(f"{cfg.name} — {dname}")
            fig_shared_legend(fig, axes)
            if save:
                safe_stem = dname.replace(" ", "_")
                save_fig(fig, f"pairwise_{safe_stem}", out_dir)

        else:
            # ── scalar-mode: grouped bar chart ────────────────────────────────
            all_pairs = sorted(
                {pl for ic_res in by_ic_pw.values() for pl in ic_res.get(dname, {})}
            )
            x = np.arange(len(ic_labels))
            width = 0.8 / max(len(all_pairs), 1)
            fig, ax = plt.subplots(figsize=(max(6, 2 * len(ic_labels)), 4))
            for p_idx, pair_label in enumerate(all_pairs):
                vals = [
                    by_ic_pw.get(lbl, {}).get(dname, {}).get(pair_label, np.nan)
                    for lbl in ic_labels
                ]
                ax.bar(
                    x + p_idx * width,
                    vals,
                    width,
                    label=pair_label,
                    color=_tab10(p_idx % 10),
                )
            ax.set_xticks(x + width * len(all_pairs) / 2)
            ax.set_xticklabels(ic_labels, rotation=20, ha="right")
            ax.set_title(f"{cfg.name} — {dname}")
            ax.set_ylabel(dname)
            ax.grid(axis="y")
            ax.grid(False, axis="x")
            fig_shared_legend(fig, [ax])
            if save:
                safe_stem = dname.replace(" ", "_")
                save_fig(fig, f"pairwise_{safe_stem}", out_dir)


# ── stability ─────────────────────────────────────────────────────────────────


def plot_stability(cfg: ProblemConfig, save: bool = True, suffix: str = ""):
    """Compound energy figure + Hessian phase figure + generic per-metric figures.

    Energy (PE / KE / total) are grouped into a single multi-row figure and
    normalised per atom so different N values are comparable.  For the total
    energy row a dotted horizontal reference at the first-chunk value is drawn
    so integrator drift is immediately visible (flat = perfect NVE conservation;
    monotone = thermostat pumping energy in/out).

    Hessian metrics (eigval_min, n_negative_modes) get a horizontal y=0 line
    labelled "solid" (above) / "liquid/gas" (below) so phase transitions appear
    as zero-crossings.
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"stability{suffix}"
    data = load_json(out_dir / "result.json")
    styles = solver_styles(cfg)

    by_param = data["by_param"]
    sweep_key = data.get("sweep_key", "param")
    vals = sorted(by_param.keys(), key=float)
    N = data.get("params", {}).get("N", 1)

    sample_entry = next(
        (
            r
            for sv in by_param.values()
            for ts in sv.values()
            for r in ts
            if r.get("valid")
        ),
        {},
    )
    all_keys = [k for k in sample_entry if k not in ("t", "valid")]

    _ENERGY_KEYS = ["potential_energy", "kinetic_energy"]
    _HESSIAN_KEYS = ["eigval_min", "n_negative_modes"]
    energy_present = [k for k in _ENERGY_KEYS if k in all_keys]
    hessian_present = [k for k in _HESSIAN_KEYS if k in all_keys]
    other_keys = [
        k for k in all_keys if k not in _ENERGY_KEYS and k not in _HESSIAN_KEYS
    ]

    def _ensemble(name):
        scheme = (cfg.solvers[name].scheme or "").lower()
        return (
            "NVT"
            if ("nvt" in scheme or "nose" in scheme or "hoover" in scheme)
            else "NVE"
        )

    first_fig = None

    # ── 1. Compound energy figure ─────────────────────────────────────────────
    # Layout: row 0 = KE/atom (thermometer + target line), row 1 = total_E/atom
    # (conservation / thermostat drift).  PE row is dropped — PE = total − KE
    # so it adds no independent information.
    if energy_present:
        has_both = len(energy_present) == 2
        row_keys = ["kinetic_energy", "total_energy"] if has_both else energy_present
        _ylabels = {
            "kinetic_energy": "KE / atom  [ε]",
            "total_energy": "E_total / atom  [ε]",
        }
        n_rows = len(row_keys)
        n_cols = len(vals)
        fig, ax_grid = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.8 * n_cols, 3.5 * n_rows),
            sharex="col",
            sharey="row",
            squeeze=False,
        )
        for row, rkey in enumerate(row_keys):
            for col, val in enumerate(vals):
                ax = ax_grid[row, col]
                solver_handles = []
                for name, ts in by_param[val].items():
                    valid_rows = [r for r in ts if r["valid"]]
                    t_arr = [r["t"] for r in valid_rows]
                    if rkey == "total_energy":
                        pe = [r.get("potential_energy") for r in valid_rows]
                        ke = [r.get("kinetic_energy") for r in valid_rows]
                        vals_raw = (
                            [p + k for p, k in zip(pe, ke)]
                            if all(v is not None for v in pe + ke)
                            else []
                        )
                    else:
                        vals_raw = [
                            r.get(rkey) for r in valid_rows if r.get(rkey) is not None
                        ]
                    if not vals_raw:
                        continue
                    per_atom = [v / N for v in vals_raw]
                    label = f"{styles[name]['label']} ({_ensemble(name)})"
                    (ln,) = ax.plot(
                        t_arr[: len(per_atom)],
                        per_atom,
                        lw=1.5,
                        label=label,
                        **solver_plot_props(styles[name], marker=False),
                    )
                    solver_handles.append(ln)
                    if rkey == "total_energy" and per_atom:
                        ax.axhline(
                            per_atom[0],
                            color=styles[name]["color"],
                            lw=0.7,
                            ls=":",
                            alpha=0.45,
                        )

                # KE target reference: 3/2 · kT per atom
                if rkey == "kinetic_energy" and sweep_key == "kT":
                    try:
                        ke_target = 1.5 * float(val)
                        ref_line = ax.axhline(
                            ke_target,
                            color="0.45",
                            lw=0.9,
                            ls="--",
                            zorder=0,
                        )
                        if col == 0:
                            ref_line.set_label("³⁄₂ kT  (target)")
                    except (ValueError, TypeError):
                        pass

                if col == 0:
                    ax.set_ylabel(_ylabels.get(rkey, rkey))
                if row == 0:
                    ax.set_title(f"{sweep_key} = {val}")
                if row == n_rows - 1:
                    ax.set_xlabel("t  [τ]")
        fig.suptitle(f"{cfg.name} — stability — energy components")
        fig_shared_legend(fig, list(ax_grid.flat))
        if save:
            save_fig(fig, "energy", out_dir)
        first_fig = fig

    # ── 2. Hessian / phase figures ────────────────────────────────────────────
    for metric in hessian_present:
        fig, axes = subplots_grid(len(vals), panel_w=4, panel_h=4, sharey=False)
        for ax, val in zip(axes, vals):
            for name, ts in by_param[val].items():
                t_arr = [r["t"] for r in ts if r["valid"]]
                m_arr = [
                    r.get(metric)
                    for r in ts
                    if r["valid"] and r.get(metric) is not None
                ]
                if t_arr and m_arr:
                    ax.plot(
                        t_arr[: len(m_arr)],
                        m_arr,
                        label=f"{styles[name]['label']} ({_ensemble(name)})",
                        **solver_plot_props(styles[name]),
                    )
            ax.axhline(
                0,
                color="0.4",
                lw=1.0,
                ls="--",
                zorder=0,
                label="solid / liquid threshold",
            )
            if metric == "eigval_min":
                ax.text(
                    0.03,
                    0.96,
                    "solid",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="0.35",
                    va="top",
                )
                ax.text(
                    0.03,
                    0.04,
                    "liquid / gas",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="0.35",
                    va="bottom",
                )
            ax.set_title(f"{sweep_key} = {val}")
            ax.set_xlabel("t  [τ]")
        axes[0].set_ylabel(metric)
        fig.suptitle(f"{cfg.name} — stability — {metric}")
        fig_shared_legend(fig, axes)
        if save:
            save_fig(fig, metric, out_dir)
        first_fig = first_fig or fig

    # ── 3. Other metrics (generic) ─────────────────────────────────────────────
    for metric in other_keys:
        fig, axes = subplots_grid(len(vals), panel_w=4, panel_h=4, sharey=False)
        for ax, val in zip(axes, vals):
            for name, ts in by_param[val].items():
                t_arr = [r["t"] for r in ts if r["valid"]]
                m_arr = [
                    r.get(metric)
                    for r in ts
                    if r["valid"] and r.get(metric) is not None
                ]
                if t_arr and m_arr:
                    ax.plot(
                        t_arr[: len(m_arr)],
                        m_arr,
                        label=styles[name]["label"],
                        **solver_plot_props(styles[name], marker=False),
                    )
            ax.set_title(f"{sweep_key} = {val}")
            ax.set_xlabel("t  [τ]")
        axes[0].set_ylabel(metric)
        fig.suptitle(f"{cfg.name} — stability — {metric}")
        fig_shared_legend(fig, axes)
        if save:
            save_fig(fig, metric, out_dir)
        first_fig = first_fig or fig

    # ── 4. Final-state output fields ──────────────────────────────────────────
    fields_path = out_dir / "fields.npz"
    if not fields_path.exists():
        return first_fig

    npz = np.load(fields_path)
    rep_val = float(npz["rep_val"][0])
    solver_names = npz["solver_names"].tolist()
    raw = [
        (n, styles.get(n, {}).get("label", n), npz[f"f_{j}"])
        for j, n in enumerate(solver_names)
        if f"f_{j}" in npz and _is_field(npz[f"f_{j}"])
    ]
    if raw:
        f2d = _field_to_2d(cfg)
        panels = [(lbl_, f2d(arr)) for _, lbl_, arr in raw]
        fig_fld = field_grid(
            panels,
            f"{cfg.name} — density at {sweep_key}={rep_val:.4g}",
            **_field_grid_kw(cfg),
        )
        if save:
            save_fig(fig_fld, "fields", out_dir)
    return first_fig


# ── physical_laws ──────────────────────────────────────────────────────────────


def _plot_physical_laws_single(cfg, data, out_dir, styles, save):
    """Render one physical-laws sweep result (one sweep key)."""
    by_param = data.get("by_param", {})
    sweep_key = data.get("sweep_key", "param")
    vals = sorted(by_param.keys(), key=float)
    if not vals:
        return None

    diag_names: list[str] = []
    for val in vals:
        for solver_data in by_param[val].values():
            if isinstance(solver_data, dict):
                diag_names = list(solver_data.keys())
                break
        if diag_names:
            break

    if not diag_names:
        return None

    x = np.array([float(v) for v in vals])
    n_diags = len(diag_names)
    fig, axes = subplots_grid(n_diags, panel_w=5, panel_h=4)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax, dname in zip(axes, diag_names):
        for name in cfg.solvers:
            y = []
            for val in vals:
                sd = by_param[val].get(name)
                y.append(sd.get(dname, np.nan) if isinstance(sd, dict) else np.nan)
            y = np.array(y, dtype=float)
            style = styles.get(name, {})
            if not np.all(np.isnan(y)):
                ax.plot(
                    x,
                    y,
                    label=style.get("label", name),
                    **solver_plot_props(style),
                )
        ax.set_xlabel(unit_label(sweep_key, cfg.units))
        ax.set_ylabel(dname)
        ax.set_title(f"{dname} vs {sweep_key}")
        if np.any(x > 0):
            ax.set_xscale("log")

    fig.suptitle(f"{cfg.name} — physical laws ({sweep_key} sweep)")
    fig_shared_legend(fig, axes)
    if save:
        save_fig(fig, "physical_laws", out_dir)
    return fig


def plot_physical_laws(cfg: ProblemConfig, save: bool = True, suffix: str = ""):
    """One subplot per diagnostic: value vs sweep parameter for each solver.

    Supports both single-run layout (result.json at top level) and multi-run
    layout (one named subdir per sweep, each with its own result.json).
    Also plots analytic_error vs sweep parameter when available.
    """
    out_dir = results_dir() / cfg.name / _SUITE / f"physical_laws{suffix}"
    styles = solver_styles(cfg)

    # Single-run layout: result.json directly in out_dir
    try:
        data = load_json(out_dir / "result.json")
    except FileNotFoundError:
        data = None
    if data is not None:
        return _plot_physical_laws_single(cfg, data, out_dir, styles, save)

    # Multi-run layout: one named subdir per sweep
    figs = []
    if out_dir.is_dir():
        for sub in sorted(out_dir.iterdir()):
            if not sub.is_dir():
                continue
            try:
                sub_data = load_json(sub / "result.json")
            except FileNotFoundError:
                continue
            if sub_data is not None:
                fig = _plot_physical_laws_single(cfg, sub_data, sub, styles, save)
                if fig is not None:
                    figs.append(fig)
    return figs if figs else None
