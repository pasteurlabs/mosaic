"""Assembled ``PLOT_FNS`` registry for thermal-mesh.

Each ``<suite>/<experiment>`` key here pairs 1:1 with an entry in
:mod:`.experiments` ``EXPERIMENTS``.
"""

from __future__ import annotations

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
from mosaic.benchmarks.shared.plots.optimization import plot_conductivity_recovery

from .ics import MAKE_IC
from .physics import _density_to_2d

# Re-export for `config.py` (kept here so the visualisation transform travels
# with the plot machinery even though the canonical definition lives in
# `physics.py` next to the FEM solve that also uses it).
__all__ = ["PLOT_FNS", "_density_to_2d"]


def _agreement_plot(exp_key):
    return lambda cfg, **kw: plot_agreement(cfg, exp_key=exp_key, **kw)


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
    "forward/physical_laws": plot_physical_laws,
    "forward/source_baseline": _agreement_plot("source_baseline"),
    "forward/source_linearity": _agreement_plot("source_linearity"),
    # Cost
    "cost/spatial_cost": plot_cost,
    "cost/temporal_cost": plot_cost,
    "cost/vjp_cost": plot_cost,
    # Gradient
    "gradient/fd_check": plot_fd_check,
    "gradient/param_sweep": plot_param_sweep,
    "gradient/jacobian_svd": plot_jacobian_svd,
    "gradient/source_fd_check": lambda cfg, **kw: plot_fd_check(
        cfg, exp_key="source_fd_check", **kw
    ),
    "gradient/source_width_sweep": lambda cfg, **kw: plot_param_sweep(
        cfg, exp_key="source_width_sweep", **kw
    ),
    # Optimization
    "optimization/conductivity_recovery": plot_conductivity_recovery,
    "optimization/conductivity_recovery_bfgs": lambda cfg, **kw: (
        plot_conductivity_recovery(cfg, exp_key="conductivity_recovery_bfgs", **kw)
    ),
    # ICs
    **_ICS_PLOT_FNS,
}
