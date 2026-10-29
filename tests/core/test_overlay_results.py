# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR results overlaid on the published baseline, as the CI preview and publish do."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "overlay-results.py"
)


def _write(
    root: Path, params: dict, rows: dict[str, list[str]], extra: str = ""
) -> None:
    exp = root / "ns-grid" / "forward" / "cylinder"
    exp.mkdir(parents=True)
    results = [
        {"solver": solver, "sweep_value": value, "metrics": {"error": 0.1}}
        for solver, values in rows.items()
        for value in values
    ]
    (exp / "result.json").write_text(
        json.dumps({"schema_version": 1, "params": params, "results": results})
    )
    if extra:
        (exp / extra).write_text("")


def _overlay(tmp_path: Path) -> tuple[set[tuple[str, str]], set[str]]:
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(_SCRIPT), tmp_path / "base", tmp_path / "pr", out],
        check=True,
        capture_output=True,
    )
    exp = out / "ns-grid" / "forward" / "cylinder"
    rows = json.loads((exp / "result.json").read_text())["results"]
    return {(r["solver"], r["sweep_value"]) for r in rows}, {
        f.name for f in exp.iterdir()
    }


def test_same_params_keep_baseline_only_solvers(tmp_path):
    params = {"sweep": {"key": "resolution", "values": [10, 20]}}
    _write(tmp_path / "base", params, {"XLB": ["10", "20"], "PICT": ["10", "20"]})
    _write(tmp_path / "pr", params, {"XLB": ["10", "20"]})

    rows, _ = _overlay(tmp_path)

    assert rows == {("XLB", "10"), ("XLB", "20"), ("PICT", "10"), ("PICT", "20")}


def test_changed_params_take_the_experiment_from_the_pr(tmp_path):
    old = {"sweep": {"key": "nu", "values": [0.05, 0.01]}}
    new = {"sweep": {"key": "resolution", "values": [10, 20]}}
    _write(
        tmp_path / "base",
        old,
        {"XLB": ["0.05", "0.01"], "PICT": ["0.05"]},
        extra="fields.npz",
    )
    _write(tmp_path / "pr", new, {"XLB": ["10", "20"]})

    rows, files = _overlay(tmp_path)

    assert rows == {("XLB", "10"), ("XLB", "20")}
    assert files == {"result.json"}
