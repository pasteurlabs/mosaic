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
    "agreement": "mosaic.benchmarks.plots.paper.agreement",
    "cylinder": "mosaic.benchmarks.plots.paper.cylinder",
    "cost_overview": "mosaic.benchmarks.plots.paper.cost_overview",
    "fd_check": "mosaic.benchmarks.plots.paper.fd_check",
    "ics": "mosaic.benchmarks.plots.paper.ics_figures",
    "jacobian_svd": "mosaic.benchmarks.plots.paper.jacobian_svd",
    "physical_accuracy": "mosaic.benchmarks.plots.paper.physical_accuracy",
    "architecture": "mosaic.benchmarks.plots.paper.architecture",
    "domain_illustrations": "mosaic.benchmarks.plots.paper.domain_illustrations",
    "scaling": "mosaic.benchmarks.plots.paper.scaling",
    "horizon_sweep": "mosaic.benchmarks.plots.paper.horizon_sweep",
    "ucurves": "mosaic.benchmarks.plots.paper.ucurves",
    "horizon_sweep_limits": "mosaic.benchmarks.plots.paper.horizon_sweep_limits",
    "drag_opt": "mosaic.benchmarks.plots.paper.drag_opt",
    "lid_cavity": "mosaic.benchmarks.plots.paper.lid_cavity",
    "grad_divergence": "mosaic.benchmarks.plots.paper.grad_divergence",
    "topopt": "mosaic.benchmarks.plots.paper.topopt",
    "optimization": "mosaic.benchmarks.plots.paper.recovery",
    "recovery_overview": "mosaic.benchmarks.plots.paper.recovery_overview",
    "topopt_overview": "mosaic.benchmarks.plots.paper.topopt_overview",
    "conductivity_overview": "mosaic.benchmarks.plots.paper.conductivity_overview",
    "visual_abstract": "mosaic.benchmarks.plots.paper.visual_abstract",
}


def all_names() -> list[str]:
    return list(_REGISTRY.keys())


def get_generate_fn(name: str) -> Callable[[Path], None]:
    import importlib

    module_path = _REGISTRY[name]
    mod = importlib.import_module(module_path)
    return mod.generate
