"""Assembled ``PLOT_FNS`` registry and field-slice helper for ns-3d-grid.

Each ``<suite>/<experiment>`` key here pairs 1:1 with an entry in
:mod:`.experiments` ``EXPERIMENTS``. Sub-IC keys (``forward/agreement/tgv3d``)
reuse the parent's plot function with a specific ``exp_key`` so the per-IC
result directories also get plotted.

``_extra/<suite>/<name>`` keys are bonus plots not tied to a single
experiment — they get called unconditionally by the runner.
"""

from __future__ import annotations

import numpy as np

from mosaic.benchmarks.shared.ics import plot_ic_only
from mosaic.benchmarks.shared.plots.cost import plot_cost
from mosaic.benchmarks.shared.plots.forward import (
    plot_agreement,
    plot_physical_laws,
)
from mosaic.benchmarks.shared.plots.gradient import (
    plot_fd_check,
    plot_horizon_sweep,
    plot_jacobian_svd,
    plot_jacobian_svd_comparison,
)
from mosaic.benchmarks.shared.plots.optimization import plot_recovery

from .ics import MAKE_IC


def _field_to_2d(v: np.ndarray) -> np.ndarray:
    """Extract a 2-D scalar from a 3-D velocity field (N,N,N,3).

    Returns the z-component of vorticity on the middle-z slice,
    shape (N, N).  Used as the primary visualisation slice for 3D field plots.
    """
    N = v.shape[0]
    zmid = N // 2
    vx = np.array(v[:, :, zmid, 0])
    vy = np.array(v[:, :, zmid, 1])
    dvydx = (np.roll(vy, -1, 0) - np.roll(vy, 1, 0)) * 0.5
    dvxdy = (np.roll(vx, -1, 1) - np.roll(vx, 1, 1)) * 0.5
    return (dvydx - dvxdy).astype(np.float32)


def _agreement_plot(exp_key):
    return lambda cfg, **kw: plot_agreement(cfg, exp_key=exp_key, **kw)


def _jsvd_plot(exp_key):
    return lambda cfg, **kw: plot_jacobian_svd(cfg, exp_key=exp_key, **kw)


def _recovery_plot(exp_key):
    return lambda cfg, **kw: plot_recovery(cfg, exp_key=exp_key, **kw)


def _ic_plot(ic_name, plot_params):
    return lambda cfg, **kw: plot_ic_only(
        cfg, ic_name, make_ic=MAKE_IC, params=plot_params
    )


_ICS_PLOT_FNS = {
    f"ics/{name}": _ic_plot(name, dict(getattr(spec, "plot_params", {}) or {}))
    for name, spec in MAKE_IC.items()
}


PLOT_FNS = {
    # Forward
    "forward/baseline": _agreement_plot("baseline"),
    "forward/agreement": plot_agreement,
    "forward/agreement/tgv3d": plot_agreement,
    "forward/physical_laws": plot_physical_laws,
    # Cost
    "cost/spatial_cost": plot_cost,
    "cost/temporal_cost": plot_cost,
    "cost/vjp_cost": plot_cost,
    # Gradient
    "gradient/fd_check": plot_fd_check,
    "gradient/horizon_sweep": plot_horizon_sweep,
    "gradient/horizon_sweep_limits": plot_horizon_sweep,
    "gradient/jacobian_svd": plot_jacobian_svd,
    "gradient/jacobian_svd_steps20": _jsvd_plot("jacobian_svd_steps20"),
    "gradient/jacobian_svd_steps40": _jsvd_plot("jacobian_svd_steps40"),
    "gradient/jacobian_svd_nu01": _jsvd_plot("jacobian_svd_nu01"),
    # Optimization
    "optimization/recovery_constant_ic": _recovery_plot("recovery_constant_ic"),
    "optimization/recovery_constant_ic_bfgs": _recovery_plot(
        "recovery_constant_ic_bfgs"
    ),
    "optimization/recovery_constant_ic_bfgs_proj": _recovery_plot(
        "recovery_constant_ic_bfgs_proj"
    ),
    # ICs
    **_ICS_PLOT_FNS,
    # Suite-wide bonus plots
    "_extra/gradient/jacobian_svd_comparison": lambda cfg, **_kw: (
        plot_jacobian_svd_comparison(cfg)
    ),
}
