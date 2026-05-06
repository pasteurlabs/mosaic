"""Hardware detection and resource sampling utilities."""

from __future__ import annotations

import subprocess

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


def _read_gpu_mem_mb(gpu_id: int = 0) -> float | None:
    """Read whole-GPU memory.used (MiB) from nvidia-smi."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_id),
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode != 0:
            return None
        return float(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


class ResourceSampler:
    """Simple before/after GPU memory sampler.  Use as a context manager.

    Snapshots whole-GPU memory usage at __enter__ and __exit__, reporting the
    peak delta as ``peak_gpu_mem_mb``.  Also reports harness-process RAM via
    psutil if available.

    Args:
        gpu_id:    GPU index to query (int or str).  None → queries GPU 0.
        interval:  Ignored (kept for API compatibility).
        image_tag: Ignored (kept for API compatibility).
    """

    def __init__(
        self,
        gpu_id=None,
        interval: float = 0.5,
        image_tag: str | None = None,
    ) -> None:
        self.gpu_id = 0 if gpu_id is None else int(gpu_id)
        self._baseline_gpu_mem_mb: float | None = None
        self._final_gpu_mem_mb: float | None = None
        self._peak_ram_mb: float | None = None

    def __enter__(self) -> "ResourceSampler":
        self._baseline_gpu_mem_mb = _read_gpu_mem_mb(self.gpu_id)
        return self

    def __exit__(self, *_) -> None:
        self._final_gpu_mem_mb = _read_gpu_mem_mb(self.gpu_id)
        try:
            import psutil

            self._peak_ram_mb = psutil.Process().memory_info().rss / 1e6
        except Exception:
            pass

    @property
    def summary(self) -> dict:
        """Resource stats collected during the context.

        Keys (only present when sampled):
            peak_gpu_mem_mb — VRAM delta (final − baseline), whole-GPU
            peak_ram_mb     — harness process RSS at exit
        """
        out: dict = {}
        if self._baseline_gpu_mem_mb is not None and self._final_gpu_mem_mb is not None:
            out["peak_gpu_mem_mb"] = max(
                0.0, self._final_gpu_mem_mb - self._baseline_gpu_mem_mb
            )
        if self._peak_ram_mb is not None:
            out["peak_ram_mb"] = self._peak_ram_mb
        return out
