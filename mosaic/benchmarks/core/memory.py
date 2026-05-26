"""Background-polled memory sampler for GPU VRAM and container RAM.

* :func:`sample_vram_mib` — single-shot NVML query.
* :func:`sample_container_ram_mib` — single-shot Docker-SDK query.
* :func:`container_id_from_tesseract` — pull the container name from a
  served Tesseract handle.
* :class:`MemoryPoller` — context manager that records **peak** observed
  VRAM and RAM. Use it when the container may be OOM-killed mid-call so
  the last sample before the kill captures the threshold;
  :class:`~mosaic.benchmarks.core.hardware.ResourceSampler` is the cheaper
  delta variant for healthy steady-state workloads.

Single-shot queries are best-effort (no NVIDIA driver, no Docker socket,
container gone → ``None``) so callers degrade gracefully on CPU-only
hosts.
"""

from __future__ import annotations

import threading

# Lazy / optional imports: NVML and Docker may be absent on CI / CPU-only
# hosts. We import inside the helpers so module import never fails — every
# query has its own try/except that returns ``None`` on missing-driver
# errors so callers don't need to feature-detect.

# Module-level cache for the NVML init state. ``nvmlInit()`` is idempotent
# (refcounted), but caching the success/failure avoids retrying on every
# poll when the driver is genuinely unavailable.
_NVML_OK: bool | None = None


def _nvml_ready():
    """Return the ``pynvml`` module if NVML is initialised, else ``None``.

    The upstream package is ``nvidia-ml-py`` (see ``pyproject.toml``); it
    ships a module named ``pynvml`` so the import statement is unchanged.

    Caches the init result so failing hosts (no driver, CPU-only CI) don't
    re-pay the import + init cost on every sample.
    """
    global _NVML_OK
    if _NVML_OK is False:
        return None
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        _NVML_OK = False
        return None
    if _NVML_OK is None:
        try:
            pynvml.nvmlInit()
            _NVML_OK = True
        except Exception:
            _NVML_OK = False
            return None
    return pynvml


def _docker_client():
    """Return a Docker SDK client, or ``None`` if Docker is unreachable.

    The client is constructed once per call (cheap — it's just a wrapper
    around a Unix socket). Returns ``None`` if the SDK can't connect.
    """
    try:
        import docker  # type: ignore[import-not-found]

        return docker.from_env()
    except Exception:
        return None


# ── Single-shot queries ──────────────────────────────────────────────────────


def sample_vram_mib(gpu_id: int | str) -> float | None:
    """Used VRAM on one GPU, in MiB, via NVML.

    Returns ``None`` on any failure (no NVIDIA driver, invalid ``gpu_id``,
    NVML error) so callers degrade gracefully.
    """
    nvml = _nvml_ready()
    if nvml is None:
        return None
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
        return nvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)
    except Exception:
        return None


def sample_container_ram_mib(container_id: str) -> float | None:
    """Resident RAM of a Docker container, in MiB, via the Docker SDK.

    Returns ``None`` on any failure (no Docker socket, container gone,
    missing stats field).
    """
    client = _docker_client()
    if client is None:
        return None
    try:
        container = client.containers.get(container_id)
        stats = container.stats(stream=False)
        usage = stats.get("memory_stats", {}).get("usage")
        return float(usage) / (1024 * 1024) if usage is not None else None
    except Exception:
        return None


def container_id_from_tesseract(t) -> str | None:
    """Extract the container name/ID from a served Tesseract handle.

    Reads ``_serve_context['container_name']``; returns ``None`` when the
    handle has no serve context (e.g. ``from_tesseract_api`` rather than
    ``from_image``).
    """
    ctx = getattr(t, "_serve_context", None)
    if isinstance(ctx, dict):
        return ctx.get("container_name") or ctx.get("container_id")
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
        gpu_id: GPU index for NVML (int or str). ``None`` disables VRAM
            sampling.
        container_id: Docker container ID/name. ``None`` disables
            container-RAM sampling.
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
