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
    "agreement": "benchmarks.plots.paper.agreement",
    "cylinder": "benchmarks.plots.paper.cylinder",
    "cost_overview": "benchmarks.plots.paper.cost_overview",
    "fd_check": "benchmarks.plots.paper.fd_check",
    "ics": "benchmarks.plots.paper.ics_figures",
    "jacobian_svd": "benchmarks.plots.paper.jacobian_svd",
    "physical_accuracy": "benchmarks.plots.paper.physical_accuracy",
    "architecture": "benchmarks.plots.paper.architecture",
    "domain_illustrations": "benchmarks.plots.paper.domain_illustrations",
    "scaling": "benchmarks.plots.paper.scaling",
    "horizon_sweep": "benchmarks.plots.paper.horizon_sweep",
    "ucurves": "benchmarks.plots.paper.ucurves",
    "horizon_sweep_limits": "benchmarks.plots.paper.horizon_sweep_limits",
    "drag_opt": "benchmarks.plots.paper.drag_opt",
    "topopt": "benchmarks.plots.paper.topopt",
    "optimization": "benchmarks.plots.paper.recovery",
    "recovery_overview": "benchmarks.plots.paper.recovery_overview",
    "topopt_overview": "benchmarks.plots.paper.topopt_overview",
    "conductivity_overview": "benchmarks.plots.paper.conductivity_overview",
}


def all_names() -> list[str]:
    return list(_REGISTRY.keys())


def get_generate_fn(name: str) -> Callable[[Path], None]:
    import importlib

    module_path = _REGISTRY[name]
    mod = importlib.import_module(module_path)
    return mod.generate
