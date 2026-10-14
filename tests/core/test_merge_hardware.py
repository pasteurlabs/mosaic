# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware records survive merging results from different machines.

In CI the GPU-capable solvers run on a GPU node and the CPU-only solvers on a
separate CPU node, each writing its own ``result.json`` with a single
``extras.hardware`` record. The merged result must still say which machine
each solver's numbers came from.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mosaic.benchmarks.core.experiment import merge_results
from mosaic.benchmarks.core.hardware import hardware_by_solver

_SCRIPTS = Path(__file__).resolve().parents[2] / ".github" / "scripts"
_spec = importlib.util.spec_from_file_location(
    "merge_results", _SCRIPTS / "merge-results.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

GPU_NODE = {
    "gpus": ["Tesla T4, 16384 MiB"],
    "cpu": "AMD EPYC 7V12 64-Core Processor",
    "ram_gb": 29.4,
}
CPU_NODE = {"cpu": "INTEL(R) XEON(R) PLATINUM 8573C", "ram_gb": 67.4}
OTHER_CPU_NODE = {"cpu": "AMD EPYC 9V74 80-Core Processor", "ram_gb": 67.4}


def _result(hw: dict, *solvers: str, empty: tuple[str, ...] = ()) -> dict:
    entries = [
        {"solver": s, "sweep_value": "32", "metrics": {"mean": 1.0}} for s in solvers
    ]
    entries += [{"solver": s, "sweep_value": "32", "metrics": {}} for s in empty]
    return {"schema_version": 1, "results": entries, "extras": {"hardware": hw}}


@pytest.fixture(params=["experiment", "ci-script"])
def merge(request):
    return merge_results if request.param == "experiment" else _mod._merge_result_pair


def test_single_record_applies_to_solvers_with_metrics():
    result = _result(CPU_NODE, "FEniCS", "deal.II", empty=("TopOpt.jl",))
    assert hardware_by_solver(result) == {"FEniCS": CPU_NODE, "deal.II": CPU_NODE}


def test_cpu_and_gpu_jobs_keep_their_own_hardware(merge):
    cpu = _result(CPU_NODE, "FEniCS", "deal.II")
    gpu = _result(GPU_NODE, "JAX-FEM")
    merged = merge(cpu, gpu)
    assert "hardware" not in merged["extras"]
    assert merged["extras"]["hardware_by_solver"] == {
        "FEniCS": CPU_NODE,
        "deal.II": CPU_NODE,
        "JAX-FEM": GPU_NODE,
    }


def test_rerun_updates_only_its_solvers(merge):
    # A PR rerun of one CPU-only solver overlaid on a merged baseline.
    baseline = merge(
        _result(CPU_NODE, "FEniCS", "deal.II"), _result(GPU_NODE, "JAX-FEM")
    )
    merged = merge(baseline, _result(OTHER_CPU_NODE, "deal.II"))
    assert merged["extras"]["hardware_by_solver"] == {
        "FEniCS": CPU_NODE,
        "deal.II": OTHER_CPU_NODE,
        "JAX-FEM": GPU_NODE,
    }


def test_other_extras_are_kept(merge):
    cpu = _result(CPU_NODE, "FEniCS")
    cpu["extras"]["_isolation_note"] = "note"
    merged = merge(cpu, _result(GPU_NODE, "JAX-FEM"))
    assert merged["extras"]["_isolation_note"] == "note"
