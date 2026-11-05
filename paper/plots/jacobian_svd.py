"""Generate Figure: Jacobian SVD spectra for 2D and 3D NS.

One panel per solver, one line per (nu, T) configuration.
Produces:
  jacobian_svd_comparison.pdf (3D)
  appendix_jacobian_svd_2d.pdf
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from . import TEXTWIDTH, by_solver, is_ok, load_result
from .style import NS_ORDER, RCPARAMS, solver_props

EXPERIMENTS = [
    "jacobian_svd",
    "jacobian_svd_steps20",
    "jacobian_svd_steps40",
    "jacobian_svd_nu01",
]

VARIANT_STYLES = [
    {"color": "#0077BB", "linestyle": "-"},
    {"color": "#CC3311", "linestyle": "--"},
    {"color": "#009988", "linestyle": "-."},
    {"color": "#EE7733", "linestyle": ":"},
]

NCOLS = 4

# Piecewise-log y-scale: normal log above this threshold, compressed log below.
# Highlights the upper-spectrum behavior (where projection differences live)
# while still showing the lower decades.
_YSCALE_THRESHOLD = 1e-3
_YSCALE_COMPRESS = 0.18


def _piecewise_log_forward(y):
    y = np.asarray(y, dtype=float)
    log_thresh = np.log10(_YSCALE_THRESHOLD)
    safe = np.where(y > 0, y, 1e-30)
    log_y = np.log10(safe)
    return np.where(
        log_y >= log_thresh,
        log_y,
        log_thresh + _YSCALE_COMPRESS * (log_y - log_thresh),
    )


def _piecewise_log_inverse(t):
    t = np.asarray(t, dtype=float)
    log_thresh = np.log10(_YSCALE_THRESHOLD)
    return np.where(
        t >= log_thresh,
        np.power(10.0, t),
        np.power(10.0, log_thresh + (t - log_thresh) / _YSCALE_COMPRESS),
    )


def _variant_label(physics: dict) -> str:
    return f"ν={physics['nu']:g}, T={physics['steps'] * physics['dt']:g}"


def _load_variants(problem: str) -> list[tuple[str, dict[str, list[float]]]]:
    """Return ``[(label, {solver: normalized spectrum})]`` per experiment."""
    variants = []
    for exp in EXPERIMENTS:
        result = load_result(problem, f"gradient/{exp}")
        spectra = {
            solver: metrics["spectrum"]
            for solver, cells in by_solver(result).items()
            for metrics in cells.values()
            if is_ok(metrics)
        }
        variants.append((_variant_label(result["params"]["physics"]), spectra))
    return variants


def _svd_comparison(problem: str, fig_w: float, out_path: Path) -> None:
    variants = _load_variants(problem)
    solvers = [s for s in NS_ORDER if any(s in spectra for _, spectra in variants)]

    nrows = math.ceil(len(solvers) / NCOLS)
    panel_h = fig_w / NCOLS * 0.85
    fig, axes = plt.subplots(
        nrows,
        NCOLS,
        figsize=(fig_w, panel_h * nrows + 0.3),
        squeeze=False,
        sharex=True,
        sharey=True,
        dpi=300,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.12)

    y_min = min(
        min(v for v in spectrum if v > 0)
        for _, spectra in variants
        for spectrum in spectra.values()
    )

    for idx, solver in enumerate(solvers):
        ax = axes[idx // NCOLS][idx % NCOLS]
        for vi, (label, spectra) in enumerate(variants):
            if solver not in spectra:
                continue
            sv = np.asarray(spectra[solver], dtype=float)
            ax.plot(
                np.arange(1, len(sv) + 1),
                sv,
                linewidth=1.5,
                label=label,
                **VARIANT_STYLES[vi],
            )

        ax.set_yscale(
            "function",
            functions=(_piecewise_log_forward, _piecewise_log_inverse),
        )
        ax.set_ylim(bottom=10 ** (np.floor(np.log10(y_min)) - 0.5), top=2.0)
        # Faint divider at the scale-break to flag the change.
        ax.axhline(
            _YSCALE_THRESHOLD, color="0.7", linestyle=":", linewidth=0.6, zorder=0
        )

        ax.set_title(solver_props(solver)[0])
        if idx % NCOLS == 0:
            ax.set_ylabel(r"$\sigma_i\,/\,\sigma_1$")
        if idx + NCOLS >= len(solvers):
            ax.tick_params(labelbottom=True)
        if idx >= (nrows - 1) * NCOLS:
            ax.set_xlabel("Mode index $i$")

        ax.yaxis.set_major_locator(
            mticker.FixedLocator([1.0, 1e-1, 1e-2, 1e-3, 1e-6, 1e-12])
        )
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=True))
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    # The legend takes the place of the first unused panel.
    for idx in range(len(solvers), nrows * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].set_visible(False)
    legend_ax = axes[-1][-1]
    legend_ax.legend(
        handles=[
            mlines.Line2D([], [], linewidth=1.5, label=label, **VARIANT_STYLES[vi])
            for vi, (label, _) in enumerate(variants)
        ],
        loc="center",
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="0.8",
        handlelength=2.5,
    )
    legend_ax.set_visible(True)
    legend_ax.axis("off")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate(out_dir: Path) -> None:
    with plt.rc_context(RCPARAMS):
        _svd_comparison(
            "ns-3d-grid", TEXTWIDTH * 1.15, out_dir / "jacobian_svd_comparison.pdf"
        )
        _svd_comparison("ns-grid", TEXTWIDTH, out_dir / "appendix_jacobian_svd_2d.pdf")
