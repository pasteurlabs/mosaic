# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware detection and resource sampling utilities."""

from __future__ import annotations

from mosaic.benchmarks.core.memory import _nvml_ready, sample_vram_mib

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


def get_hardware_info() -> dict:
    """Return a dict with GPU model(s), CPU model, and total RAM.

    Keys: "gpus" (list[str]), "cpu" (str), "ram_gb" (float).
    Any key is omitted if the underlying query fails.
    """
    info: dict = {}

    gpus = _gpu_descriptions()
    if gpus:
        info["gpus"] = gpus

    # CPU model from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    # Total RAM via psutil
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass

    return info


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

    def __enter__(self) -> ResourceSampler:
        self._baseline_vram_mib = sample_vram_mib(self.gpu_id)
        return self

    def __exit__(self, *_: object) -> None:
        self._final_vram_mib = sample_vram_mib(self.gpu_id)
        try:
            import psutil

            self._ram_mib = psutil.Process().memory_info().rss / 1e6
        except Exception:
            pass

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
