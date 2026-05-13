"""End-to-end framework tests using in-process dummy tesseracts.

These tests exercise the full kernel + framework + per_solver_loop +
apply_tesseract VJP pipeline without Docker. Each registered solver is
pointed at a ``tesseract_api.py`` stub under
:mod:`mosaic.tests.dummy_tesseracts` via the ``inprocess:`` tag prefix
that :func:`run_with_gpu_pool` recognises.

The dummies return constant outputs, so any quantitative result is
expected to be trivial (zeros, NaNs from divide-by-zero in derived
quantities like cosines, …). What we're verifying is *plumbing*:

* The kernel runs without raising.
* The result.json lands at the expected path with the expected schema.
* The NPZ (if any) is written.
* Multi-solver fanout works (same dummy plugged in N times).

For convergence / numerical correctness, see the Docker-backed
integration tests in :mod:`test_integration`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DUMMY_NS_GRID = (
    Path(__file__).parent
    / "dummy_tesseracts"
    / "navier_stokes_grid"
    / "tesseract_api.py"
).resolve()

DUMMY_STRUCTURAL_MESH = (
    Path(__file__).parent / "dummy_tesseracts" / "structural_mesh" / "tesseract_api.py"
).resolve()

DUMMY_THERMAL_MESH = (
    Path(__file__).parent / "dummy_tesseracts" / "thermal_mesh" / "tesseract_api.py"
).resolve()


@pytest.fixture
def ns_grid_tags():
    """Return a tags dict pointing every ns-grid solver at the dummy."""
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-grid")
    tag = f"inprocess:{DUMMY_NS_GRID}"
    return cfg, dict.fromkeys(cfg.solver_names, tag)


@pytest.fixture
def structural_mesh_tags():
    """Return a tags dict pointing every structural-mesh solver at the dummy."""
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("structural-mesh")
    tag = f"inprocess:{DUMMY_STRUCTURAL_MESH}"
    return cfg, dict.fromkeys(cfg.solver_names, tag)


@pytest.fixture
def thermal_mesh_tags():
    """Return a tags dict pointing every thermal-mesh solver at the dummy."""
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("thermal-mesh")
    tag = f"inprocess:{DUMMY_THERMAL_MESH}"
    return cfg, dict.fromkeys(cfg.solver_names, tag)


def test_dummy_apply_loads_via_tesseract_api():
    """Sanity check: the dummy loads via Tesseract.from_tesseract_api."""
    import numpy as np
    from tesseract_core import Tesseract

    with Tesseract.from_tesseract_api(str(DUMMY_NS_GRID)) as t:
        assert "apply" in t.available_endpoints
        assert "vector_jacobian_product" in t.available_endpoints
        v0 = np.zeros((4, 4, 1, 2), dtype=np.float32)
        out = t.apply({"v0": v0})
        assert set(out) == {"result", "drag"}
        assert np.asarray(out["result"]).shape == v0.shape


def test_forward_baseline_runs_with_dummy(ns_grid_tags, tmp_path, monkeypatch):
    """Forward agreement at one small N runs end-to-end against the dummy."""
    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = ns_grid_tags

    # Override the existing forward/baseline registration with a tiny one
    # so we don't sweep N=[64,128,192,256]; we just want plumbing.
    from mosaic.benchmarks.problems.shared.forward import agreement

    cfg.add_experiment(
        "forward/dummy_baseline",
        agreement,
        ic={"name": "multimode", "seed": 0},
        physics={"N": [4, 8], "nu": 0.05, "dt": 0.01, "steps": 1},
    )
    result = cfg.experiments["forward/dummy_baseline"].fn(cfg, tags)
    assert "by_param" in result
    assert "spread" in result
    assert result["by_param"]
    # Result.json should have been written.
    json_path = tmp_path / "ns-grid" / "forward" / "dummy_baseline" / "result.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert "by_param" in on_disk


def test_structural_mesh_forward_runs_with_dummy(
    structural_mesh_tags, tmp_path, monkeypatch
):
    """Structural-mesh forward agreement runs end-to-end against the dummy."""
    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = structural_mesh_tags

    # Tiny resolution sweep — just exercise the plumbing on the canonical
    # thin-slab cantilever geometry (ny=2, nz=nx//2).
    from mosaic.benchmarks.problems.shared.forward import agreement

    cfg.add_experiment(
        "forward/dummy_baseline",
        agreement,
        ic={"name": "uniform", "seed": 0},
        physics={
            "N": [4, 6],
            "ny": 2,
            "nz": 2,
            "Lx": 2.0,
            "Ly": 1.0,
            "Lz": 1.0,
            "F_total": 1.0,
            "corner_load": False,
        },
    )
    result = cfg.experiments["forward/dummy_baseline"].fn(cfg, tags)
    assert "by_param" in result
    assert "spread" in result
    assert result["by_param"]
    json_path = (
        tmp_path / "structural-mesh" / "forward" / "dummy_baseline" / "result.json"
    )
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert "by_param" in on_disk


def test_thermal_mesh_forward_runs_with_dummy(thermal_mesh_tags, tmp_path, monkeypatch):
    """Thermal-mesh forward agreement runs end-to-end against the dummy."""
    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = thermal_mesh_tags

    # Tiny resolution sweep — just exercise the plumbing on the canonical
    # quasi-2D heated-slab geometry.
    from mosaic.benchmarks.problems.shared.forward import agreement

    cfg.add_experiment(
        "forward/dummy_baseline",
        agreement,
        ic={"name": "random", "seed": 0},
        physics={
            "N": [4, 6],
            "nz": 1,
            "Lx": 2.0,
            "Ly": 1.0,
            "Lz": 1.0,
            "Q_total": 1.0,
        },
    )
    result = cfg.experiments["forward/dummy_baseline"].fn(cfg, tags)
    assert "by_param" in result
    assert "spread" in result
    assert result["by_param"]
    json_path = tmp_path / "thermal-mesh" / "forward" / "dummy_baseline" / "result.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert "by_param" in on_disk


# ── Every-experiment parametrized smoke test ─────────────────────────────────

_DUMMY_FOR = {
    "ns-grid": DUMMY_NS_GRID,
    "ns-3d-grid": DUMMY_NS_GRID,
    "structural-mesh": DUMMY_STRUCTURAL_MESH,
    "thermal-mesh": DUMMY_THERMAL_MESH,
}


def _all_experiments():
    """Enumerate (problem, exp_key) pairs for every non-IC experiment.

    ``ics/*`` keys are pure plot-IC registrations that don't invoke a
    tesseract, so they're skipped here.
    """
    from mosaic.benchmarks.problems import get_config

    pairs: list[tuple[str, str]] = []
    for problem in _DUMMY_FOR:
        cfg = get_config(problem)
        for key in sorted(cfg.experiments):
            if key.startswith("ics/"):
                continue
            pairs.append((problem, key))
    return pairs


@pytest.mark.parametrize("problem, exp_key", _all_experiments())
def test_experiment_runs_with_dummy(problem, exp_key, tmp_path, monkeypatch):
    """Every registered experiment runs end-to-end against the dummy.

    The dummy returns constant outputs so numerical results are trivial —
    we only check that the kernel + framework + per_solver_loop +
    apply_tesseract VJP pipeline executes and ``result.json`` lands on
    disk. Optimisation runs against a zero-gradient target will iterate
    their configured ``max_iters`` since loss never decreases; the
    runtime cost is bounded by the dummy's near-zero per-call latency.
    """
    from mosaic.benchmarks.problems import get_config

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg = get_config(problem)
    tag = f"inprocess:{_DUMMY_FOR[problem]}"
    tags = dict.fromkeys(cfg.solver_names, tag)
    cfg.experiments[exp_key].fn(cfg, tags)

    # The framework writes ``result.json`` somewhere under
    # ``<results>/<problem>/<exp_key>[/sub]``. Don't pin the exact path —
    # IC-suffix / debug-suffix conventions vary by harness — just confirm
    # *some* result.json was written under the problem dir.
    written = list((tmp_path / problem).rglob("result.json"))
    assert written, (
        f"{problem}/{exp_key}: no result.json found under {tmp_path / problem}"
    )
