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


@pytest.fixture
def ns_grid_tags():
    """Return a tags dict pointing every ns-grid solver at the dummy."""
    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-grid")
    tag = f"inprocess:{DUMMY_NS_GRID}"
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
