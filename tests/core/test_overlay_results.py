# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR results overlaid on the published baseline, as the CI preview and publish do."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

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


_PARAMS = {"physics": {"N": 64}}
_VALID = {"error": 0.01, "valid": True}
_UNSCORED = {"status": "skipped", "unscored": True, "reason": "no reference"}


def _write_result(exp: Path, metrics: dict[str, dict]) -> None:
    exp.mkdir(parents=True, exist_ok=True)
    results = [
        {"solver": solver, "sweep_value": "64", "metrics": m}
        for solver, m in metrics.items()
    ]
    (exp / "result.json").write_text(
        json.dumps({"schema_version": 1, "params": _PARAMS, "results": results})
    )


def _run_overlay(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(_SCRIPT), tmp_path / "base", tmp_path / "pr", out],
        check=True,
        capture_output=True,
    )
    return out


def _metrics(exp: Path) -> dict[str, dict]:
    rows = json.loads((exp / "result.json").read_text())["results"]
    return {r["solver"]: r["metrics"] for r in rows}


def test_unscored_rerun_keeps_the_baseline_score(tmp_path):
    # A PR that reruns one solver alone has no peers to score it against.
    exp = Path("ns-grid", "forward", "agreement")
    _write_result(tmp_path / "base" / exp, {"XLB": _VALID, "PICT": _VALID})
    failed = {"error": "boom", "valid": False}
    _write_result(tmp_path / "pr" / exp, {"XLB": _UNSCORED, "PICT": failed})

    out = _run_overlay(tmp_path)

    assert _metrics(out / exp) == {"XLB": _VALID, "PICT": failed}


def test_fields_keep_baseline_only_solvers(tmp_path):
    exp = Path("ns-grid", "gradient", "fd_check")
    for root, solvers in (("base", ["XLB", "PICT"]), ("pr", ["PICT"])):
        _write_result(tmp_path / root / exp, dict.fromkeys(solvers, _VALID))
        value = 2.0 if root == "pr" else 1.0
        # Forward fields are keyed by solver name, gradient fields by slot.
        np.savez(
            tmp_path / root / exp / "fields.npz",
            solver_names=np.array(solvers),
            consensus_0=np.array([value]),
            **{f"{s}_0": np.array([value]) for s in solvers},
        )
        np.savez(
            tmp_path / root / exp / "gradient_fields.npz",
            solver_names=np.array(solvers),
            ic=np.array([0.0]),
            **{f"grad_{j}": np.array([value]) for j in range(len(solvers))},
        )

    out = _run_overlay(tmp_path) / exp

    with np.load(out / "fields.npz") as data:
        assert sorted(map(str, data["solver_names"])) == ["PICT", "XLB"]
        assert float(data["XLB_0"][0]) == 1.0
        assert float(data["PICT_0"][0]) == 2.0
    with np.load(out / "gradient_fields.npz") as data:
        grads = {
            str(name): float(data[f"grad_{j}"][0])
            for j, name in enumerate(data["solver_names"])
        }
        assert grads == {"XLB": 1.0, "PICT": 2.0}
        assert float(data["ic"][0]) == 0.0


def test_files_are_overlaid_per_file(tmp_path):
    base, pr = tmp_path / "base" / "ns-grid", tmp_path / "pr" / "ns-grid"
    _write_result(base / "cost" / "spatial_cost", {"XLB": {"mean": 1.0}})
    _write_result(base / "forward" / "agreement", {"XLB": _VALID})
    _write_result(pr / "forward" / "agreement", {"XLB": _VALID})
    _write_result(pr / "forward" / "cylinder", {"XLB": _VALID})
    (base / "forward" / "agreement" / "agreement.png").write_text("base")
    (pr / "forward" / "agreement" / "agreement.png").write_text("pr")
    (tmp_path / "base" / "snapshot.json").write_text("{}")
    (tmp_path / "base" / "status-report.md").write_text("")

    out = _run_overlay(tmp_path)

    files = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert files == {
        "ns-grid/cost/spatial_cost/result.json",
        "ns-grid/forward/agreement/result.json",
        "ns-grid/forward/agreement/agreement.png",
        "ns-grid/forward/cylinder/result.json",
    }
    assert (out / "ns-grid/forward/agreement/agreement.png").read_text() == "pr"


def test_without_a_baseline_the_pr_results_are_kept(tmp_path):
    exp = Path("ns-grid", "forward", "agreement")
    _write_result(tmp_path / "pr" / exp, {"XLB": _UNSCORED})

    out = _run_overlay(tmp_path)

    assert _metrics(out / exp) == {"XLB": _UNSCORED}
