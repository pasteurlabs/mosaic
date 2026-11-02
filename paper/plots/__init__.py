"""Figure generators for the Mosaic paper.

Each module exposes a ``generate(out_dir: Path) -> None`` function that reads
benchmark results from :func:`results_dir` and saves one or more figures to
*out_dir*.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path

# NeurIPS textwidth in inches; change here when targeting a different venue.
TEXTWIDTH = 5.5

MODULES = [
    "domain_illustrations",
    "visual_abstract",
    "cylinder",
    "fd_check",
    "ics_figures",
    "jacobian_svd",
    "physical_accuracy",
    "scaling",
    "horizon_sweep",
    "horizon_sweep_limits",
    "ucurves",
    "drag_opt",
    "recovery_overview",
    "topopt_overview",
    "conductivity_overview",
]


def results_dir() -> Path:
    """Root results directory: ``$MOSAIC_RESULTS_DIR`` if set, else ``./mosaic-results``."""
    if d := os.environ.get("MOSAIC_RESULTS_DIR"):
        return Path(d)
    return Path.cwd() / "mosaic-results"


def get_generate_fn(name: str) -> Callable[[Path], None]:
    """Return the ``generate`` function of the figure module *name*."""
    return importlib.import_module(f".{name}", __name__).generate
