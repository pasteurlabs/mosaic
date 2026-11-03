# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PICT's CUDA errors raise in Python instead of exiting the process.

The rewrite itself is checked on handler source copied from PICT 72a6e86. The
image test runs against ``pict_navier_stokes_grid:latest`` built from this
checkout and is skipped where that image, Docker or a GPU is missing.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from mosaic.benchmarks.core.harness import classify_failure

from .helpers import image_available
from .test_pict_cylinder import _gpu_available

_PICT_DIR = (
    Path(__file__).parents[1] / "mosaic" / "tesseracts" / "navier-stokes-grid" / "pict"
)
_IMAGE = "pict_navier_stokes_grid:latest"

_CUDA_HANDLER = """
static void CheckErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(10);
}
"""
_CUBLAS_HANDLER = """
static void CheckErrorAux(const char* file, unsigned line, const char* statement, cublasStatus_t err) {
  if (err == CUBLAS_STATUS_SUCCESS) return;
  std::cerr << statement << " returned " << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(10);
}
"""
_CUSPARSE_HANDLER = _CUBLAS_HANDLER.replace(
    "cublasStatus_t", "cusparseStatus_t"
).replace("CUBLAS_STATUS_SUCCESS", "CUSPARSE_STATUS_SUCCESS")
_SOLVER_SOURCE = (
    _CUDA_HANDLER + "//cublas" + _CUBLAS_HANDLER + "//cusparse" + _CUSPARSE_HANDLER
)
_SOURCES = {
    "cg_solver_kernel.cu": _SOLVER_SOURCE,
    "bicgstab_solver_kernel.cu": _SOLVER_SOURCE,
    "dispatch.h": _CUDA_HANDLER.replace("CheckErrorAux", "CheckCudaErrorAux"),
    "PISO_multiblock_cuda_kernel.cu": _CUDA_HANDLER.replace(
        "CheckErrorAux", "CheckCudaErrorAux"
    ),
}

# Fills the GPU through torch, then asks PICT's CG solver for its workspace.
_SOLVE_ON_FULL_GPU = """
import torch, PISOtorch
dev = torch.device("cuda")
n = 2**24
A = PISOtorch.CSRmatrix(
    torch.ones(n, device=dev),
    torch.arange(n, dtype=torch.int32, device=dev),
    torch.arange(n + 1, dtype=torch.int32, device=dev),
)
rhs, x = torch.ones(n, device=dev), torch.zeros(n, device=dev)
hog = []
for mib in (1024, 64, 4):
    while True:
        try:
            hog.append(torch.empty(mib * 2**20, dtype=torch.uint8, device=dev))
        except torch.OutOfMemoryError:
            break
try:
    PISOtorch.SolveLinear(A, rhs, x, torch.tensor([10], dtype=torch.int32), torch.tensor([1e-6]))
except RuntimeError as e:
    print("RAISED", e)
"""


def _load_patcher():
    spec = importlib.util.spec_from_file_location(
        "patch_cuda_errors", _PICT_DIR / "patch_cuda_errors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_handler_throws_with_its_error_name(tmp_path):
    for name, source in _SOURCES.items():
        (tmp_path / name).write_text(source)

    _load_patcher().patch(tmp_path)

    solver = (tmp_path / "cg_solver_kernel.cu").read_text()
    assert "exit(10)" not in solver
    assert solver.count("throw std::runtime_error(msg.str());") == 3
    for name in ("cudaGetErrorString", "cublasGetStatusName", "cusparseGetErrorName"):
        assert f"<< {name}(err) <<" in solver
    assert solver.startswith("#include <sstream>\n#include <stdexcept>\n")


def test_refuses_source_it_was_not_written_for(tmp_path):
    for name, source in _SOURCES.items():
        (tmp_path / name).write_text(source)
    (tmp_path / "dispatch.h").write_text(_SOLVER_SOURCE)

    with pytest.raises(SystemExit, match=r"dispatch\.h: expected 1 handlers"):
        _load_patcher().patch(tmp_path)


@pytest.mark.skipif(
    not (image_available(_IMAGE) and _gpu_available()),
    reason=f"{_IMAGE} or GPU missing",
)
def test_out_of_memory_in_a_solve_is_classified_as_oom():
    """The process survives a failed workspace allocation, and the error says OOM."""
    proc = subprocess.run(
        [
            *("docker", "run", "--rm", "--gpus", "device=0"),
            *("--entrypoint", "python", _IMAGE, "-c", _SOLVE_ON_FULL_GPU),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    message = proc.stdout.partition("RAISED ")[2]
    assert message, proc.stdout + proc.stderr
    assert classify_failure("RuntimeError", message) == "OOM"
