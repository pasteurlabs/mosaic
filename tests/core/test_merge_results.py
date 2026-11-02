# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU and GPU result artifacts merged into one tree, as the CI does."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "merge-results.py"
)

_EXP = Path("ns-grid", "forward", "agreement")


def _write_artifact(staging: Path, name: str, hardware: dict, *solvers: str) -> Path:
    exp = staging / name / _EXP
    exp.mkdir(parents=True)
    results = [
        {"solver": s, "sweep_value": "64", "metrics": {"error": 0.01, "valid": True}}
        for s in solvers
    ]
    (exp / "result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "params": {"physics": {"N": 64}},
                "results": results,
                "extras": {"hardware": hardware},
            }
        )
    )
    np.savez(
        exp / "fields.npz",
        solver_names=np.array(solvers),
        **{f"{s}_0": np.array([float(len(s))]) for s in solvers},
    )
    return exp


def _merge(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(_SCRIPT), tmp_path / "staging", out],
        check=True,
        capture_output=True,
    )
    return out


def test_cpu_and_gpu_artifacts_merge_per_solver(tmp_path):
    cpu, gpu = {"cpu": "x86"}, {"gpus": ["T4"]}
    staging = tmp_path / "staging"
    _write_artifact(staging, "results-forward-ns-grid-cpu", cpu, "OpenFOAM")
    _write_artifact(staging, "results-forward-ns-grid-gpu", gpu, "XLB", "PICT")

    exp = _merge(tmp_path) / _EXP

    data = json.loads((exp / "result.json").read_text())
    assert sorted(r["solver"] for r in data["results"]) == ["OpenFOAM", "PICT", "XLB"]
    assert data["extras"]["hardware_by_solver"] == {
        "OpenFOAM": cpu,
        "XLB": gpu,
        "PICT": gpu,
    }
    with np.load(exp / "fields.npz") as fields:
        assert sorted(map(str, fields["solver_names"])) == ["OpenFOAM", "PICT", "XLB"]
        for solver in ("OpenFOAM", "PICT", "XLB"):
            assert float(fields[f"{solver}_0"][0]) == len(solver)


def test_files_in_one_artifact_are_copied(tmp_path):
    staging = tmp_path / "staging"
    exp = _write_artifact(staging, "results-forward-ns-grid-gpu", {}, "XLB")
    (exp / "agreement.png").write_text("gpu")
    cost = staging / "results-cost-ns-grid-cpu" / "ns-grid" / "cost" / "spatial_cost"
    cost.mkdir(parents=True)
    (cost / "result.json").write_text('{"schema_version": 1, "results": []}')
    (staging / "results-forward-ns-grid-cpu" / _EXP).mkdir(parents=True)
    (staging / "results-forward-ns-grid-cpu" / _EXP / "agreement.png").write_text("cpu")
    (staging / "solver-image-xlb").mkdir()
    (staging / "solver-image-xlb" / "image.tar").write_text("")

    out = _merge(tmp_path)

    files = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert files == {
        "ns-grid/cost/spatial_cost/result.json",
        "ns-grid/forward/agreement/agreement.png",
        "ns-grid/forward/agreement/fields.npz",
        "ns-grid/forward/agreement/result.json",
    }
    assert (out / _EXP / "result.json").read_bytes() == (
        exp / "result.json"
    ).read_bytes()
    # Plain files are not merged: the last artifact in name order wins.
    assert (out / _EXP / "agreement.png").read_text() == "gpu"
