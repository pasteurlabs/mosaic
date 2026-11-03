# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the memory poller and its single-shot queries.

Covers:
- ``sample_vram_mib``: excludes driver-reserved memory (NVML v2), with a v1
  fallback for drivers without it.
- ``sample_container_ram_mib``: ``docker stats`` arithmetic (usage minus
  inactive page cache) on cgroup v1 and v2 payloads.
- ``MemoryPoller``: records the peak, samples once more on exit, and leaves a
  key ``None`` when its source is disabled.
- ``run_timed_trials``: reports the poller's peaks for the given device and
  container.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mosaic.benchmarks.core import harness, memory
from mosaic.benchmarks.core.memory import (
    MemoryPoller,
    sample_container_ram_mib,
    sample_vram_mib,
)

MIB = 1024 * 1024


class _FakeContainer:
    def __init__(self, memory_stats: dict) -> None:
        self._memory_stats = memory_stats
        self.stats_kwargs: dict = {}

    def stats(self, **kwargs) -> dict:
        self.stats_kwargs = kwargs
        return {"memory_stats": self._memory_stats}


class _FakeClient:
    def __init__(self, container: _FakeContainer) -> None:
        self.containers = self
        self._container = container

    def get(self, _container_id: str) -> _FakeContainer:
        return self._container


def _patch_docker(monkeypatch, memory_stats: dict) -> _FakeContainer:
    container = _FakeContainer(memory_stats)
    monkeypatch.setattr(memory, "_docker_client", lambda: _FakeClient(container))
    return container


# ── sample_vram_mib ───────────────────────────────────────────────────────────

_V2 = object()


def _patch_nvml(monkeypatch, *, v2: bool) -> None:
    """NVML whose v1 ``used`` counts 768 MiB of driver-reserved memory."""

    def get_memory_info(_handle, version=None):
        if version is _V2:
            if not v2:
                raise RuntimeError("NVML_ERROR_FUNCTION_NOT_FOUND")
            return SimpleNamespace(used=10 * MIB, reserved=768 * MIB)
        return SimpleNamespace(used=778 * MIB)

    nvml = SimpleNamespace(
        nvmlMemory_v2=_V2,
        nvmlDeviceGetHandleByIndex=lambda _i: "h",
        nvmlDeviceGetMemoryInfo=get_memory_info,
    )
    monkeypatch.setattr(memory, "_nvml_ready", lambda: nvml)


def test_vram_excludes_driver_reserved_memory(monkeypatch):
    _patch_nvml(monkeypatch, v2=True)
    assert sample_vram_mib(0) == pytest.approx(10.0)


def test_vram_falls_back_to_v1_without_v2(monkeypatch):
    _patch_nvml(monkeypatch, v2=False)
    assert sample_vram_mib(0) == pytest.approx(778.0)


# ── sample_container_ram_mib ──────────────────────────────────────────────────


@pytest.mark.parametrize("key", ["inactive_file", "total_inactive_file"])
def test_container_ram_excludes_inactive_page_cache(monkeypatch, key):
    """cgroup v2 (``inactive_file``) and v1 (``total_inactive_file``) payloads."""
    container = _patch_docker(
        monkeypatch, {"usage": 900 * MIB, "stats": {key: 300 * MIB}}
    )
    assert sample_container_ram_mib("c") == pytest.approx(600.0)
    # One-shot: without it the daemon waits about a second for a CPU sample.
    assert container.stats_kwargs == {"stream": False, "one_shot": True}


def test_container_ram_without_cache_stats_is_usage(monkeypatch):
    _patch_docker(monkeypatch, {"usage": 512 * MIB})
    assert sample_container_ram_mib("c") == pytest.approx(512.0)


def test_container_ram_without_usage_is_none(monkeypatch):
    _patch_docker(monkeypatch, {})
    assert sample_container_ram_mib("c") is None


# ── MemoryPoller ──────────────────────────────────────────────────────────────


def test_poller_records_peak_including_the_exit_sample(monkeypatch):
    """A block shorter than the interval still gets its exit sample."""
    level = {"vram": 100.0}
    monkeypatch.setattr(memory, "sample_vram_mib", lambda _gpu: level["vram"])
    monkeypatch.setattr(memory, "sample_container_ram_mib", lambda _cid: 7.0)
    with MemoryPoller(0, "c", interval=60.0) as poller:
        level["vram"] = 500.0  # a pool the call leaves allocated
    assert poller.summary == {"vram_peak_mib": 500.0, "ram_peak_mib": 7.0}


def test_poller_disabled_sources_report_none(monkeypatch):
    def _never(_arg):
        raise AssertionError("disabled source was queried")

    monkeypatch.setattr(memory, "sample_vram_mib", _never)
    monkeypatch.setattr(memory, "sample_container_ram_mib", _never)
    with MemoryPoller(None, None, interval=60.0) as poller:
        pass
    assert poller.summary == {"vram_peak_mib": None, "ram_peak_mib": None}


def test_poller_skips_failed_samples(monkeypatch):
    monkeypatch.setattr(memory, "sample_vram_mib", lambda _gpu: None)
    monkeypatch.setattr(memory, "sample_container_ram_mib", lambda _cid: None)
    with MemoryPoller(0, "c", interval=60.0) as poller:
        pass
    assert poller.summary == {"vram_peak_mib": None, "ram_peak_mib": None}


# ── run_timed_trials ──────────────────────────────────────────────────────────


def test_run_timed_trials_reports_peaks_of_device_and_container(monkeypatch):
    """The peak over warmup and trials, queried for the given device and container."""
    queried: list[tuple[str, object]] = []
    level = {"vram": 10.0}

    def fake_vram(gpu_id):
        queried.append(("gpu", gpu_id))
        return level["vram"]

    def fake_ram(container_id):
        queried.append(("container", container_id))
        return 5.0

    monkeypatch.setattr(memory, "sample_vram_mib", fake_vram)
    monkeypatch.setattr(memory, "sample_container_ram_mib", fake_ram)

    def fn():
        level["vram"] += 10.0  # an allocator pool that only grows

    result = harness.run_timed_trials(
        fn, n_trials=3, wall_limit_s=10.0, gpu_id=0, container_id="solver-c"
    )
    assert result.mem == {"vram_peak_mib": 50.0, "ram_peak_mib": 5.0}
    assert set(queried) == {("gpu", 0), ("container", "solver-c")}


def test_run_timed_trials_failure_keeps_peaks(monkeypatch):
    monkeypatch.setattr(memory, "sample_vram_mib", lambda _gpu: 42.0)
    monkeypatch.setattr(memory, "sample_container_ram_mib", lambda _cid: None)

    def fn():
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")

    result = harness.run_timed_trials(fn, n_trials=3, wall_limit_s=10.0, gpu_id=0)
    rec = result.as_record()
    assert rec["failure_type"] == "OOM"
    assert rec["vram_peak_mib"] == 42.0
    assert rec["ram_peak_mib"] is None
