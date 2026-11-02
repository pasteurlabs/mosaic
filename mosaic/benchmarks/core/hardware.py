# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware detection and resource sampling utilities."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from mosaic.benchmarks.core.memory import _nvml_ready, sample_vram_mib

if TYPE_CHECKING:
    from typing import Self

logger = logging.getLogger(__name__)

# ── Hardware info ─────────────────────────────────────────────────────────────


def _gpu_descriptions() -> list[str]:
    """``["<name>, <total_mib> MiB", ...]`` for each visible GPU via NVML.

    Returns an empty list on any failure (no driver, NVML init error).
    Format mirrors the previous ``nvidia-smi --query-gpu=name,memory.total
    --format=csv,noheader`` output so existing consumers (status display,
    plot captions) don't change.
    """
    nvml = _nvml_ready()
    if nvml is None:
        return []
    out: list[str] = []
    try:
        for i in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            total_mib = nvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 * 1024)
            out.append(f"{name}, {total_mib:.0f} MiB")
    except Exception:
        return []
    return out


def _nvidia_versions() -> dict:
    """NVIDIA driver version and the CUDA version it supports, via NVML.

    The CUDA version is the newest runtime the driver can serve, not the
    runtime inside a solver image (each image ships its own). Returns an
    empty dict on any failure.
    """
    nvml = _nvml_ready()
    if nvml is None:
        return {}
    out: dict = {}
    try:
        driver = nvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8", errors="replace")
        out["nvidia_driver"] = driver
        cuda = nvml.nvmlSystemGetCudaDriverVersion_v2()
        out["cuda_driver"] = f"{cuda // 1000}.{cuda % 1000 // 10}"
    except Exception:
        logger.debug("NVIDIA version probe failed", exc_info=True)
    return out


# Environment variables that set thread counts: OpenMP (and Intel's and GNU's
# extensions), MKL, and anything with THREAD in its name (OPENBLAS_NUM_THREADS,
# JULIA_NUM_THREADS, NUMEXPR_MAX_THREADS, ...).
_THREAD_ENV = re.compile(r"^(OMP|KMP|GOMP|MKL)_|THREAD")


def get_hardware_info() -> dict:
    """Return a dict describing the machine a result was measured on.

    Keys:
      * ``gpus`` (list[str]), ``nvidia_driver`` (str), ``cuda_driver`` (str):
        GPU models and memory, the driver version and the CUDA version the
        driver supports.
      * ``cpu`` (str), ``cpu_logical_cores`` (int), ``cpu_physical_cores``
        (int): CPU model and core counts. Solver containers get no CPU limit,
        so these are the cores a CPU solver can use.
      * ``ram_gb`` (float): total RAM.
      * ``thread_env`` (dict[str, str]): thread-count environment variables of
        the harness process, empty when none is set. Containers do not
        inherit them; a solver's own settings are in the ``env:`` of its
        ``tesseract_config.yaml``.

    Any key is omitted if the underlying query fails.
    """
    info: dict = {}

    gpus = _gpu_descriptions()
    if gpus:
        info["gpus"] = gpus
        info.update(_nvidia_versions())

    # CPU model from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        logger.debug("CPU model probe failed", exc_info=True)

    # Core counts and total RAM via psutil
    try:
        import psutil

        info["cpu_logical_cores"] = psutil.cpu_count(logical=True)
        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        logger.debug("CPU core / RAM probe failed", exc_info=True)

    info["thread_env"] = {
        k: v for k, v in sorted(os.environ.items()) if _THREAD_ENV.search(k)
    }
    return info


def hardware_by_solver(result: dict) -> dict[str, dict]:
    """Map each solver in a schema_version=1 result to its hardware record.

    A result written by one run carries a single ``extras.hardware`` record
    for every solver that produced metrics. Merging results from different
    machines (the CPU and GPU jobs in CI) replaces it with a per-solver
    ``extras.hardware_by_solver`` map, see :func:`merge_hardware`.
    """
    extras = result.get("extras", {})
    if "hardware_by_solver" in extras:
        return extras["hardware_by_solver"]
    hw = extras.get("hardware")
    if not hw:
        return {}
    return {e["solver"]: hw for e in result.get("results", []) if e.get("metrics")}


def merge_hardware(merged_extras: dict, a: dict, b: dict) -> None:
    """Record each solver's hardware from results *a* and *b* in *merged_extras*.

    *b* wins per solver. The single ``hardware`` record is dropped, since it
    can describe only one of the machines.
    """
    hw = {**hardware_by_solver(a), **hardware_by_solver(b)}
    if hw:
        merged_extras.pop("hardware", None)
        merged_extras["hardware_by_solver"] = hw


def has_gpu() -> bool:
    """Return True if at least one NVIDIA GPU is visible via NVML."""
    nvml = _nvml_ready()
    if nvml is None:
        return False
    try:
        return nvml.nvmlDeviceGetCount() > 0
    except Exception:
        return False


# ── Resource sampler ──────────────────────────────────────────────────────────


class ResourceSampler:
    """Before/after GPU memory delta sampler.  Use as a context manager.

    Snapshots whole-GPU memory usage at ``__enter__`` and ``__exit__`` and
    reports the delta as ``vram_peak_mib``. Also reports harness-process
    RAM (RSS) at exit as ``ram_peak_mib`` if psutil is available.

    The complementary tool, :class:`mosaic.benchmarks.core.memory.MemoryPoller`,
    polls in a background thread and reports the absolute peak over the
    block — pick that one when you need true peak or when the workload
    may be killed mid-call. ResourceSampler is the cheaper choice for
    steady-state timing.

    The summary key shape (``vram_peak_mib`` / ``ram_peak_mib``) matches
    :class:`MemoryPoller.summary` so both samplers are drop-in compatible
    downstream (suite results, plots).

    Args:
        gpu_id:    GPU index to query (int or str). ``None`` → queries GPU 0.
        interval:  Ignored (kept for API compatibility with MemoryPoller).
        image_tag: Ignored (kept for API compatibility).
    """

    def __init__(
        self,
        gpu_id: int | str | None = None,
        interval: float = 0.5,
        image_tag: str | None = None,
    ) -> None:
        self.gpu_id = 0 if gpu_id is None else int(gpu_id)
        self._baseline_vram_mib: float | None = None
        self._final_vram_mib: float | None = None
        self._ram_mib: float | None = None

    def __enter__(self) -> Self:
        self._baseline_vram_mib = sample_vram_mib(self.gpu_id)
        return self

    def __exit__(self, *_: object) -> None:
        self._final_vram_mib = sample_vram_mib(self.gpu_id)
        try:
            import psutil

            self._ram_mib = psutil.Process().memory_info().rss / 1e6
        except Exception:
            logger.debug("RSS sampling failed", exc_info=True)

    @property
    def summary(self) -> dict:
        """Resource stats collected during the context.

        Returns:
            ``vram_peak_mib``: VRAM delta (final − baseline), whole-GPU,
                or ``None`` if either snapshot failed.
            ``ram_peak_mib``: harness-process RSS at exit, in MiB, or
                ``None`` if psutil is unavailable.
        """
        if self._baseline_vram_mib is not None and self._final_vram_mib is not None:
            vram = max(0.0, self._final_vram_mib - self._baseline_vram_mib)
        else:
            vram = None
        return {"vram_peak_mib": vram, "ram_peak_mib": self._ram_mib}
