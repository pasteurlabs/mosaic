"""Hardware detection and resource sampling utilities."""

from __future__ import annotations

import fcntl
import os
import subprocess
import tempfile
import threading
from typing import Optional

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


# ── Container helpers ─────────────────────────────────────────────────────────


def _find_container_id(image_tag: str) -> Optional[str]:
    """Return the Docker container ID for the first running container with the
    given image tag, or None if not found."""
    try:
        out = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"ancestor={image_tag}",
                "--format",
                "{{.ID}}",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode != 0:
            return None
        ids = [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
        return ids[0] if ids else None
    except Exception:
        return None


def _container_root_pid(container_id: str) -> Optional[int]:
    """Return the root PID of a running Docker container (host-namespace PID)."""
    try:
        out = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", container_id],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode != 0:
            return None
        pid = int(out.stdout.strip())
        return pid if pid > 0 else None
    except Exception:
        return None


def _container_process_tree(root_pid: int):
    """Return a list of psutil.Process objects for the container's full process tree."""
    try:
        import psutil

        root = psutil.Process(root_pid)
        procs = [root] + root.children(recursive=True)
        # Initialise cpu_percent counters (first call always returns 0)
        for p in procs:
            try:
                p.cpu_percent()
            except Exception:
                pass
        return procs
    except Exception:
        return []


# ── Resource sampler ──────────────────────────────────────────────────────────


class ResourceSampler:
    """Background-thread resource sampler.  Use as a context manager.

    Polls GPU and CPU/RAM metrics every `interval` seconds while the context
    is active and reports peak and mean values via `.summary`.

    When `image_tag` is provided the sampler locates the running Docker
    container for that image and measures container-level CPU % and RAM (by
    walking the container's process tree with psutil) instead of the harness
    process — making CPU/RAM readings meaningful for solvers that run inside
    Docker.

    GPU VRAM: when container PIDs are known, uses
    ``nvidia-smi --query-compute-apps`` to sum per-PID GPU memory, which is
    per-container rather than whole-GPU. Falls back to the whole-GPU delta
    approach when PIDs are unavailable.

    Args:
        gpu_id:    GPU index to query (int or str).  None → queries GPU 0.
        interval:  Polling interval in seconds.
        image_tag: Docker image tag of the solver container (e.g.
                   "xlb_navier_stokes_grid:latest").  When provided, resource
                   measurements target the container instead of the harness.
    """

    def __init__(
        self,
        gpu_id=None,
        interval: float = 0.5,
        image_tag: Optional[str] = None,
    ) -> None:
        self.gpu_id = 0 if gpu_id is None else int(gpu_id)
        self.interval = interval
        self.image_tag = image_tag

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Peaks
        self._peak_gpu_mem_mb: float | None = None
        self._peak_gpu_util_pct: float | None = None
        self._peak_cpu_pct: float | None = None
        self._peak_ram_mb: float | None = None

        # Running mean for GPU util (peak GPU util is not meaningful for efficiency)
        self._sum_gpu_util_pct: float = 0.0
        self._count_gpu_util: int = 0

        # Baseline for whole-GPU delta fallback
        self._baseline_gpu_mem_mb: float | None = None

        # Container state (populated at __enter__ when image_tag is provided)
        self._container_id: str | None = None
        self._container_procs: list = []  # list of psutil.Process
        self._container_pids: set[int] = set()

        # Advisory GPU lock (serialises overlapping samplers on the same GPU)
        self._gpu_lock_fd: int | None = None

    # ── GPU helpers ───────────────────────────────────────────────────────────

    def _read_gpu_util_and_mem_mb(self) -> tuple[float | None, float | None]:
        """Read whole-GPU utilization % and memory.used from nvidia-smi."""
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.gpu_id),
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.returncode != 0:
                return None, None
            line = out.stdout.strip().splitlines()[0]
            util_pct, mem_mb = (float(v.strip()) for v in line.split(","))
            return util_pct, mem_mb
        except Exception:
            return None, None

    def _read_compute_app_gpu_mem_mb(self) -> float | None:
        """Sum GPU memory for all PIDs in the container via --query-compute-apps.

        Returns None when no PIDs are known or the query fails.
        """
        if not self._container_pids:
            return None
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.gpu_id),
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.returncode != 0:
                return None
            total = 0.0
            matched = False
            for line in out.stdout.strip().splitlines():
                parts = line.strip().split(",")
                if len(parts) != 2:
                    continue
                try:
                    pid, mem = int(parts[0].strip()), float(parts[1].strip())
                except ValueError:
                    continue
                if pid in self._container_pids:
                    total += mem
                    matched = True
            return total if matched else None
        except Exception:
            return None

    # ── polling ───────────────────────────────────────────────────────────────

    def _poll_gpu(self) -> None:
        util_pct, mem_mb_whole = self._read_gpu_util_and_mem_mb()

        # GPU utilization — whole-GPU (representative; no per-container metric)
        if util_pct is not None:
            if self._peak_gpu_util_pct is None or util_pct > self._peak_gpu_util_pct:
                self._peak_gpu_util_pct = util_pct
            self._sum_gpu_util_pct += util_pct
            self._count_gpu_util += 1

        # GPU memory — prefer per-PID (container-accurate), fall back to delta
        gpu_mem_mb = self._read_compute_app_gpu_mem_mb()
        if gpu_mem_mb is None and mem_mb_whole is not None:
            # Delta fallback
            if self._baseline_gpu_mem_mb is not None:
                gpu_mem_mb = max(0.0, mem_mb_whole - self._baseline_gpu_mem_mb)
            else:
                gpu_mem_mb = mem_mb_whole

        if gpu_mem_mb is not None:
            if self._peak_gpu_mem_mb is None or gpu_mem_mb > self._peak_gpu_mem_mb:
                self._peak_gpu_mem_mb = gpu_mem_mb

    def _poll_cpu(self) -> None:
        try:
            import psutil

            if self._container_procs:
                # Container-level: sum across the container's process tree
                cpu = 0.0
                ram = 0.0
                live_procs = []
                for p in self._container_procs:
                    try:
                        cpu += p.cpu_percent()
                        ram += p.memory_info().rss / 1e6
                        live_procs.append(p)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self._container_procs = live_procs
            else:
                # Fallback: whole-system CPU % and harness RSS
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.Process().memory_info().rss / 1e6

            if self._peak_cpu_pct is None or cpu > self._peak_cpu_pct:
                self._peak_cpu_pct = cpu
            if self._peak_ram_mb is None or ram > self._peak_ram_mb:
                self._peak_ram_mb = ram
        except Exception:
            pass

    def _sample(self) -> None:
        # One immediate sample so short workloads still yield data
        self._poll_gpu()
        self._poll_cpu()
        while not self._stop.wait(self.interval):
            self._poll_gpu()
            self._poll_cpu()

    # ── context manager ───────────────────────────────────────────────────────

    def _acquire_gpu_lock(self) -> None:
        try:
            lock_path = os.path.join(
                tempfile.gettempdir(), f"mosaic_gpu_{self.gpu_id}.lock"
            )
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._gpu_lock_fd = fd
        except Exception:
            self._gpu_lock_fd = None

    def _release_gpu_lock(self) -> None:
        if self._gpu_lock_fd is not None:
            try:
                fcntl.flock(self._gpu_lock_fd, fcntl.LOCK_UN)
            finally:
                try:
                    os.close(self._gpu_lock_fd)
                except Exception:
                    pass
                self._gpu_lock_fd = None

    def __enter__(self) -> "ResourceSampler":
        self._acquire_gpu_lock()

        # Locate container and build process tree for container-level metrics
        if self.image_tag:
            self._container_id = _find_container_id(self.image_tag)
            if self._container_id:
                root_pid = _container_root_pid(self._container_id)
                if root_pid:
                    self._container_procs = _container_process_tree(root_pid)
                    self._container_pids = {p.pid for p in self._container_procs}

        # Capture GPU memory baseline synchronously before any workload starts.
        # Only used as fallback when per-PID query returns nothing.
        if not self._container_pids:
            _, mem_mb = self._read_gpu_util_and_mem_mb()
            self._baseline_gpu_mem_mb = mem_mb

        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._release_gpu_lock()

    # ── results ───────────────────────────────────────────────────────────────

    @property
    def summary(self) -> dict:
        """Peak (and mean) resource stats collected during the context.

        Keys (only present when sampled):
            peak_gpu_mem_mb    — peak VRAM used by the container (MB); per-PID
                                 when container PIDs known, whole-GPU delta otherwise
            peak_gpu_util_pct  — peak whole-GPU utilization %
            mean_gpu_util_pct  — mean whole-GPU utilization % (more informative
                                 than peak for understanding GPU efficiency)
            peak_cpu_pct       — peak CPU % (container process tree when
                                 image_tag provided, else whole-system)
            peak_ram_mb        — peak RAM (container RSS sum when image_tag
                                 provided, else harness process RSS)
            container_tracked  — True when container PIDs were successfully located
        """
        out: dict = {}
        if self._peak_gpu_mem_mb is not None:
            out["peak_gpu_mem_mb"] = self._peak_gpu_mem_mb
        if self._peak_gpu_util_pct is not None:
            out["peak_gpu_util_pct"] = self._peak_gpu_util_pct
        if self._count_gpu_util > 0:
            out["mean_gpu_util_pct"] = round(
                self._sum_gpu_util_pct / self._count_gpu_util, 1
            )
        if self._peak_cpu_pct is not None:
            out["peak_cpu_pct"] = self._peak_cpu_pct
        if self._peak_ram_mb is not None:
            out["peak_ram_mb"] = self._peak_ram_mb
        if self._container_pids:
            out["container_tracked"] = True
        return out
