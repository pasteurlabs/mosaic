"""Hardware detection and resource sampling utilities."""

from __future__ import annotations

import subprocess

from mosaic.benchmarks.core.memory import sample_vram_mib

# ── Hardware info ─────────────────────────────────────────────────────────────


def get_hardware_info() -> dict:
    """Return a dict with GPU model(s), CPU model, and total RAM.

    Keys: "gpus" (list[str]), "cpu" (str), "ram_gb" (float).
    Any key is omitted if the underlying query fails.
    """
    info: dict = {}

    # GPUs via nvidia-smi
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            gpus = [
                line.strip() for line in out.stdout.strip().splitlines() if line.strip()
            ]
            if gpus:
                info["gpus"] = gpus
    except Exception:
        pass

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
    """Return True if at least one NVIDIA GPU is visible via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
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
        gpu_id=None,
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

    def __exit__(self, *_) -> None:
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
