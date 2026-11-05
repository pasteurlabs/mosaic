"""Figure generators for the Mosaic paper.

Each module exposes a ``generate(out_dir: Path) -> None`` function that reads
benchmark results from :func:`results_dir` and saves one or more figures to
*out_dir*.
"""

from __future__ import annotations

import functools
import importlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import yaml

# NeurIPS textwidth in inches; change here when targeting a different venue.
TEXTWIDTH = 5.5

MODULES = [
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
    "conductivity_overview",
    "memory_rollout",
    "recovery_main",
]

# Tesseract directory per results problem directory.
_TESSERACT_DOMAIN = {
    "ns-grid": "navier-stokes-grid",
    "ns-3d-grid": "navier-stokes-grid",
    "structural-mesh": "structural-mesh",
    "thermal-mesh": "thermal-mesh",
}
_REPO_ROOT = Path(__file__).resolve().parents[2]


def results_dir() -> Path:
    """Root results directory: ``$MOSAIC_RESULTS_DIR`` if set, else ``./mosaic-results``."""
    if d := os.environ.get("MOSAIC_RESULTS_DIR"):
        return Path(d)
    return Path.cwd() / "mosaic-results"


def load_result(problem: str, experiment: str) -> dict:
    """Load ``<results>/<problem>/<experiment>/result.json`` (schema v1)."""
    path = results_dir() / problem / experiment / "result.json"
    return json.loads(path.read_text())


def by_solver(result: dict) -> dict[str, dict[str, dict]]:
    """Index a schema-v1 result as ``{solver: {sweep_value: metrics}}``.

    Sweep values stay strings as stored; failed entries keep their metrics
    dict, which carries ``status`` and ``failure_type``.
    """
    out: dict[str, dict[str, dict]] = {}
    for r in result["results"]:
        out.setdefault(r["solver"], {})[str(r["sweep_value"])] = r.get("metrics") or {}
    return out


def is_ok(metrics: dict) -> bool:
    """True if a result entry succeeded."""
    return bool(metrics) and metrics.get("status", "ok") == "ok"


# A container that exits while its GPU is full ran out of memory. PICT's
# server exits at the device limit without raising an error (exit code 0), so
# the harness records `container_died` instead of `OOM`.
MASKED_OOM_FRACTION = 0.95


@functools.cache
def device_mib(root: Path, default: float | None = None) -> float | None:
    """GPU memory of a run's device in MiB, from the first result that records it."""
    for path in sorted(root.glob("*/cost/*/result.json")):
        hardware = json.loads(path.read_text()).get("extras", {}).get("hardware") or {}
        if gpus := hardware.get("gpus"):
            return float(gpus[0].rsplit(",", 1)[1].split()[0])
    return default


def failure_type(metrics: dict, device: float | None) -> str | None:
    """The recorded failure type, with a container death at a full GPU counted as OOM."""
    ft = metrics.get("failure_type")
    vram = metrics.get("vram_peak_mib")
    if (
        ft == "container_died"
        and device is not None
        and vram is not None
        and vram >= MASKED_OOM_FRACTION * device
    ):
        return "OOM"
    return ft


@functools.cache
def _tesseracts(problem: str) -> list[tuple[str, dict]]:
    """``(wrapper directory, metadata.mosaic)`` of every Tesseract of *problem*."""
    domain = _TESSERACT_DOMAIN[problem]
    return [
        (f.parent.name, yaml.safe_load(f.read_text())["metadata"]["mosaic"])
        for f in sorted(
            (_REPO_ROOT / "mosaic" / "tesseracts" / domain).glob(
                "*/tesseract_config.yaml"
            )
        )
    ]


def solver_metadata(problem: str) -> dict[str, dict]:
    """``metadata.mosaic`` of every Tesseract of *problem*, keyed by solver name."""
    return {meta["name"]: meta for _, meta in _tesseracts(problem)}


def solver_wrappers(problem: str) -> dict[str, tuple[str, str]]:
    """``(domain directory, wrapper directory)`` of every Tesseract of *problem*, keyed by solver name."""
    domain = _TESSERACT_DOMAIN[problem]
    return {meta["name"]: (domain, wrapper) for wrapper, meta in _tesseracts(problem)}


def get_generate_fn(name: str) -> Callable[[Path], None]:
    """Return the ``generate`` function of the figure module *name*."""
    return importlib.import_module(f".{name}", __name__).generate
