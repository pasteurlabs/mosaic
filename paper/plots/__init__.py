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
import statistics
from collections.abc import Callable
from pathlib import Path

import yaml

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
    "drag_opt",
    "recovery_overview",
    "topopt_overview",
    "conductivity_overview",
    "memory_rollout",
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


# Experiments of the extended mesh cost sweeps, keyed by the experiment whose
# size range they extend.
EXTRA_EXPERIMENT = {
    "cost/spatial_cost": "cost/spatial_cost_large",
    "cost/vjp_cost/by_N": "cost/vjp_cost_large",
}
_extra_results: Path | None = None


def set_extra_results_dir(path: Path | None) -> None:
    """Set the results directory of the extended mesh cost sweeps."""
    global _extra_results
    _extra_results = path


def extra_results_dir() -> Path | None:
    """Results directory of the extended mesh cost sweeps, if set."""
    return _extra_results


def load_extra_result(
    problem: str, experiment: str, root: Path | None = None
) -> dict | None:
    """Load the extended sweep of *experiment* from *root* (default: :func:`extra_results_dir`), or None."""
    root = root or extra_results_dir()
    if root is None or experiment not in EXTRA_EXPERIMENT:
        return None
    path = root / problem / EXTRA_EXPERIMENT[experiment] / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def by_solver(result: dict) -> dict[str, dict[str, dict]]:
    """Index a schema-v1 result as ``{solver: {sweep_value: metrics}}``.

    Sweep values stay strings as stored; failed entries keep their metrics
    dict, which carries ``status`` and ``failure_type``.
    """
    out: dict[str, dict[str, dict]] = {}
    for r in result["results"]:
        out.setdefault(r["solver"], {})[str(r["sweep_value"])] = r.get("metrics") or {}
    return out


def median_time(metrics: dict) -> float | None:
    """Median wall time over the timed trials (the mean if trials are missing)."""
    trials = metrics.get("trials_s")
    return float(statistics.median(trials)) if trials else metrics.get("mean")


def trial_range(metrics: dict) -> tuple[float, float] | None:
    """Fastest and slowest timed trial, or None if trials are missing."""
    trials = metrics.get("trials_s")
    return (min(trials), max(trials)) if trials else None


def trial_cv(metrics: dict) -> float | None:
    """Coefficient of variation of the timed trials (sample standard deviation)."""
    trials = metrics.get("trials_s")
    if not trials or len(trials) < 2:
        return None
    return statistics.stdev(trials) / statistics.fmean(trials)


def is_ok(metrics: dict) -> bool:
    """True if a result entry succeeded."""
    return bool(metrics) and metrics.get("status", "ok") == "ok"


# A container that exits while its GPU is full ran out of memory. PICT's
# server exits at the device limit without raising an error (exit code 0), so
# the harness records `container_died` instead of `OOM`.
MASKED_OOM_FRACTION = 0.95

# The harness caps each solver container at this fraction of host RAM
# (MOSAIC_CONTAINER_MEM_FRACTION, default 0.9). At the cap, the kernel kills a
# process in the container, which the harness records as `container_died` if it
# was the Tesseract server and as `error` if it was a solver subprocess.
CONTAINER_MEM_FRACTION = 0.9


def _hardware(root: Path) -> list[dict]:
    """Hardware records of a run's cost results."""
    return [
        json.loads(path.read_text()).get("extras", {}).get("hardware") or {}
        for path in sorted(root.glob("*/cost/*/result.json"))
    ]


@functools.cache
def device_mib(root: Path, default: float | None = None) -> float | None:
    """GPU memory of a run's device in MiB, from the first result that records it."""
    for hardware in _hardware(root):
        if gpus := hardware.get("gpus"):
            return float(gpus[0].rsplit(",", 1)[1].split()[0])
    return default


@functools.cache
def container_ram_mib(root: Path) -> float | None:
    """Memory cap of a run's solver containers in MiB, from the first result that records the host RAM."""
    for hardware in _hardware(root):
        if ram_gb := hardware.get("ram_gb"):
            return CONTAINER_MEM_FRACTION * ram_gb * 1e9 / 2**20
    return None


def failure_type(
    metrics: dict, device: float | None, ram_cap: float | None = None
) -> str | None:
    """The recorded failure type, with a failure at a full GPU or at the container's RAM cap counted as OOM."""
    ft = metrics.get("failure_type")
    vram = metrics.get("vram_peak_mib")
    ram = metrics.get("ram_peak_mib")
    if (
        ft == "container_died"
        and device is not None
        and vram is not None
        and vram >= MASKED_OOM_FRACTION * device
    ):
        return "OOM"
    if (
        ft in ("container_died", "error")
        and ram_cap is not None
        and ram is not None
        and ram >= MASKED_OOM_FRACTION * ram_cap
    ):
        return "host_OOM"
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
