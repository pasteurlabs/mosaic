"""Paper figure generators for the Mosaic benchmark paper.

Each module exposes a ``generate(out_dir: Path) -> None`` function that
reads result JSON files from the benchmark results directory and saves
one or more PDF/PNG figures to *out_dir*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

# NeurIPS textwidth in inches — change here when targeting a different venue.
TEXTWIDTH = 5.5


def pw(frac: float = 1.0) -> float:
    """Return print width in inches for a figure included at *frac* × \\linewidth."""
    return TEXTWIDTH * frac

# Registry: name -> generate function (lazy import to avoid mandatory deps)
_REGISTRY: dict[str, str] = {
    "agreement":          "benchmarks.plots.paper.agreement",
    "cost_overview":      "benchmarks.plots.paper.cost_overview",
    "fd_check":           "benchmarks.plots.paper.fd_check",
    "ics":                "benchmarks.plots.paper.ics_figures",
    "jacobian_svd":       "benchmarks.plots.paper.jacobian_svd",
    "physical_accuracy":  "benchmarks.plots.paper.physical_accuracy",
    "coverage_heatmap":   "benchmarks.plots.paper.coverage_heatmap",
    "architecture":       "benchmarks.plots.paper.architecture",
    "domain_illustrations": "benchmarks.plots.paper.domain_illustrations",
    "scaling":            "benchmarks.plots.paper.scaling",
    "per_iteration_cost": "benchmarks.plots.paper.per_iteration_cost",
    "horizon_sweep":      "benchmarks.plots.paper.horizon_sweep",
    "horizon_sweep_limits": "benchmarks.plots.paper.horizon_sweep_limits",
    "drag_opt":           "benchmarks.plots.paper.drag_opt",
    "drag_opt_overview":  "benchmarks.plots.paper.drag_opt_overview",
    "lid_cavity":         "benchmarks.plots.paper.lid_cavity",
    "topopt":             "benchmarks.plots.paper.topopt",
    "optimization":           "benchmarks.plots.paper.recovery",
    "ic_recovery_fields":           "benchmarks.plots.paper.ic_recovery_fields",
    "jacobian_recovery_correlation": "benchmarks.plots.paper.jacobian_recovery_correlation",
    "recovery_long_steps": "benchmarks.plots.paper.recovery_long_steps",
    "recovery_diagnostics":     "benchmarks.plots.paper.recovery_diagnostics",
    "recovery_final_states_v3": "benchmarks.plots.paper.recovery_final_states_v3",
    "thermal_source_recovery":       "benchmarks.plots.paper.thermal_source_recovery",
    "thermal_conductivity_recovery":     "benchmarks.plots.paper.thermal_conductivity_recovery",
    "thermal_conductivity_recovery_reg": "benchmarks.plots.paper.thermal_conductivity_recovery_reg",
}


def all_names() -> list[str]:
    return list(_REGISTRY.keys())


def get_generate_fn(name: str) -> Callable[[Path], None]:
    import importlib
    module_path = _REGISTRY[name]
    mod = importlib.import_module(module_path)
    return mod.generate
