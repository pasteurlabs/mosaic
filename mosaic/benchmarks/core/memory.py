"""Background-polled memory sampler for GPU VRAM and container RAM.

This module owns three concerns that previously lived inside the gradient
suite:

  * Single-shot memory queries — :func:`sample_vram_mib` (nvidia-smi) and
    :func:`sample_container_ram_mib` (docker stats).
  * Tesseract → container-id extraction — :func:`container_id_from_tesseract`,
    which understands both modern (``_serve_context``) and legacy
    (``_container`` / ``_service`` / ``_backend``) layouts.
  * A polling sampler — :class:`MemoryPoller`, used as a context manager,
    that records the **peak** observed VRAM and RAM across the lifetime of
    the block. This is the right tool for failure-boundary experiments
    such as horizon_sweep_limits, where the container can be OOM-killed
    mid-call: the last sample taken before the kill captures the
    ~OOM threshold (a final-delta read would either be 0 or fail).

The complementary tool, :class:`mosaic.benchmarks.core.hardware.ResourceSampler`,
is a cheaper *before/after delta* sampler. Pick the poller when you need
absolute peak or expect the container to die; pick ResourceSampler when
you only want the workload's incremental cost in the steady state.
"""

from __future__ import annotations

import subprocess
import threading

# ── Single-shot queries ──────────────────────────────────────────────────────


def _parse_mem_mib(s: str) -> float | None:
    """Parse a Docker-style memory string to MiB. e.g. ``'1.23GiB'`` → ``1260.6``."""
    s = s.strip()
    try:
        for suffix, factor in [
            ("GiB", 1024.0),
            ("MiB", 1.0),
            ("KiB", 1.0 / 1024.0),
            ("GB", 953.674),
            ("MB", 0.953674),
            ("kB", 9.537e-4),
        ]:
            if s.endswith(suffix):
                return float(s[: -len(suffix)]) * factor
        return float(s)
    except (ValueError, OverflowError):
        return None


def sample_vram_mib(gpu_id: int | str) -> float | None:
    """Single nvidia-smi query for ``memory.used`` on one GPU, in MiB.

    Returns ``None`` on any failure (no NVIDIA driver, command error,
    parse error) so callers can degrade gracefully.
    """
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return float(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        return None


def sample_container_ram_mib(container_id: str) -> float | None:
    """Single ``docker stats`` query for container memory usage, in MiB.

    Returns ``None`` on any failure (docker not available, container gone,
    parse error).
    """
    try:
        r = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
                container_id,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return _parse_mem_mib(r.stdout.strip().split("/")[0])
    except Exception:
        return None


def container_id_from_tesseract(t) -> str | None:
    """Extract a Docker container name/ID from a Tesseract instance.

    Prefers ``_serve_context['container_name']`` (tesseract_core ≥ 0.9),
    then falls back to scanning legacy Docker-SDK Container object
    attributes. Returns ``None`` if neither path yields an ID.
    """
    try:
        ctx = getattr(t, "_serve_context", None)
        if isinstance(ctx, dict):
            name = ctx.get("container_name") or ctx.get("container_id")
            if name:
                return name
    except Exception:
        pass
    for attr in ("_container", "container", "_service", "_backend"):
        try:
            obj = getattr(t, attr, None)
            if obj is None:
                continue
            cid = getattr(obj, "id", None) or getattr(obj, "short_id", None)
            if isinstance(cid, str) and len(cid) >= 12:
                return cid[:12]
        except Exception:
            continue
    return None


# ── Polling sampler ──────────────────────────────────────────────────────────


class MemoryPoller:
    """Poll GPU VRAM and container RAM in a daemon thread at fixed intervals.

    Records the **peak** observed value over the lifetime of the context.
    Sampling is best-effort: any GPU/container that returns ``None`` from
    its query is silently skipped that round, and the corresponding key in
    :attr:`summary` is ``None`` if no sample ever succeeded.

    Example::

        with MemoryPoller(gpu_id, container_id) as poller:
            try:
                run_workload()
            except Exception:
                ...                                # poller still stops cleanly
        mem = poller.summary
        # {"vram_peak_mib": float | None, "ram_peak_mib": float | None}

    Args:
        gpu_id: GPU index for nvidia-smi (int or str). ``None`` disables
            VRAM sampling.
        container_id: Docker container ID/name for ``docker stats``. ``None``
            disables container-RAM sampling.
        interval: Seconds between samples (default 0.5).
    """

    def __init__(
        self,
        gpu_id: int | str | None,
        container_id: str | None,
        interval: float = 0.5,
    ) -> None:
        self._gpu_id = gpu_id
        self._container_id = container_id
        self._interval = interval
        self._stop = threading.Event()
        self._vram: list[float] = []
        self._ram: list[float] = []
        self._thread: threading.Thread | None = None

    def __enter__(self) -> MemoryPoller:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6.0)

    @property
    def summary(self) -> dict:
        """Peak VRAM and RAM observed during the context.

        Keys:
            ``vram_peak_mib`` — max VRAM across all samples, or ``None`` if
                ``gpu_id`` was ``None`` or every query failed.
            ``ram_peak_mib``  — max container RAM across all samples, or
                ``None`` if ``container_id`` was ``None`` or every query
                failed.
        """
        return {
            "vram_peak_mib": max(self._vram) if self._vram else None,
            "ram_peak_mib": max(self._ram) if self._ram else None,
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._gpu_id is not None:
                v = sample_vram_mib(self._gpu_id)
                if v is not None:
                    self._vram.append(v)
            if self._container_id is not None:
                r = sample_container_ram_mib(self._container_id)
                if r is not None:
                    self._ram.append(r)
            self._stop.wait(self._interval)
