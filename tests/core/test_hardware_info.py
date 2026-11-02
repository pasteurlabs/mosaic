# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What ``get_hardware_info`` records about the machine a result came from."""

from __future__ import annotations

from types import SimpleNamespace

from mosaic.benchmarks.core import hardware


def _fake_nvml(driver: str | bytes = "580.159.03", cuda: int = 12060):
    handle = object()
    return SimpleNamespace(
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda _i: handle,
        nvmlDeviceGetName=lambda _h: b"NVIDIA A100 80GB PCIe",
        nvmlDeviceGetMemoryInfo=lambda _h: SimpleNamespace(total=81920 * 1024 * 1024),
        nvmlSystemGetDriverVersion=lambda: driver,
        nvmlSystemGetCudaDriverVersion_v2=lambda: cuda,
    )


def test_gpu_host_records_driver_and_cuda_version(monkeypatch):
    monkeypatch.setattr(hardware, "_nvml_ready", lambda: _fake_nvml(b"580.159.03"))
    info = hardware.get_hardware_info()
    assert info["gpus"] == ["NVIDIA A100 80GB PCIe, 81920 MiB"]
    assert info["nvidia_driver"] == "580.159.03"
    assert info["cuda_driver"] == "12.6"


def test_cpu_host_records_no_gpu_keys(monkeypatch):
    monkeypatch.setattr(hardware, "_nvml_ready", lambda: None)
    info = hardware.get_hardware_info()
    assert not {"gpus", "nvidia_driver", "cuda_driver"} & info.keys()


def test_core_counts_are_recorded(monkeypatch):
    import psutil

    monkeypatch.setattr(hardware, "_nvml_ready", lambda: None)
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 24 if logical else 12)
    info = hardware.get_hardware_info()
    assert info["cpu_logical_cores"] == 24
    assert info["cpu_physical_cores"] == 12


def test_thread_env_keeps_only_thread_settings(monkeypatch):
    monkeypatch.setattr(hardware, "_nvml_ready", lambda: None)
    for k in list(hardware.os.environ):
        if hardware._THREAD_ENV.search(k):
            monkeypatch.delenv(k)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("JULIA_NUM_THREADS", "1,0")
    monkeypatch.setenv("MOSAIC_TESSERACT_TIMEOUT", "7200")
    assert hardware.get_hardware_info()["thread_env"] == {
        "JULIA_NUM_THREADS": "1,0",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
    }
