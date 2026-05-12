"""Assembled ``PLOT_FNS`` registry and density-slice helper for structural-mesh.

Each ``<suite>/<experiment>`` key here pairs 1:1 with an entry in
:mod:`.experiments` ``EXPERIMENTS``.
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
    plot_jacobian_svd,
    plot_param_sweep,
)
from mosaic.benchmarks.shared.plots.optimization import plot_topopt

from .ics import MAKE_IC
from .physics import _infer_mesh_dims


def _density_to_2d(rho: np.ndarray, **_) -> np.ndarray:
    """Mid-y cross-section of per-cell density field → (nz, nx) image."""
    nx, ny, nz = _infer_mesh_dims(len(rho))
    return rho.reshape(nz, ny, nx)[:, ny // 2, :]  # (nz, nx)


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
    "forward/baseline": lambda cfg, **kw: plot_agreement(cfg, exp_key="baseline", **kw),
    "forward/agreement": plot_agreement,
    "forward/physical_laws": plot_physical_laws,
    # Cost
    "cost/spatial_cost": plot_cost,
    "cost/temporal_cost": plot_cost,
    "cost/vjp_cost": plot_cost,
    # Gradient
    "gradient/fd_check": plot_fd_check,
    "gradient/param_sweep": plot_param_sweep,
    "gradient/jacobian_svd": plot_jacobian_svd,
    # Optimization
    "optimization/topopt": plot_topopt,
    "optimization/topopt_bfgs": lambda cfg, **kw: plot_topopt(
        cfg, exp_key="topopt_bfgs", **kw
    ),
    # ICs
    **_ICS_PLOT_FNS,
}
