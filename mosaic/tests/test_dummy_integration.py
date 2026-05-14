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


# ── Shared dummy-result corpus ───────────────────────────────────────────────
#
# Building the corpus is the dominant cost in this file — every experiment
# runs through the framework, which pays a JAX warmup tax per kernel.
# Session-scoping the fixture means we build the corpus once, then all the
# per-experiment / per-plot / per-paper-plot tests below read from it.


@pytest.fixture(scope="session")
def dummy_corpus(tmp_path_factory):
    """Run every (problem, exp_key) once against the dummy, yield the shared results dir.

    Session-scoped so the per-experiment, per-plot, and per-paper-plot tests
    all share the same corpus. Individual experiment failures are captured
    on a ``ran_ok`` dict (keyed by ``(problem, exp_key)``) so the
    parametrized ``test_experiment_runs_with_dummy`` below can re-raise the
    exact exception for the failing case.
    """
    import os

    from mosaic.benchmarks.problems import get_config

    results_root = tmp_path_factory.mktemp("dummy_results")
    prev = os.environ.get("MOSAIC_RESULTS_DIR")
    os.environ["MOSAIC_RESULTS_DIR"] = str(results_root)
    errors: dict[tuple[str, str], Exception] = {}
    try:
        for problem, exp_key in _all_experiments():
            cfg = get_config(problem)
            tag = f"inprocess:{_DUMMY_FOR[problem]}"
            tags = dict.fromkeys(cfg.solver_names, tag)
            try:
                cfg.experiments[exp_key].fn(cfg, tags)
            except Exception as exc:
                errors[(problem, exp_key)] = exc
        yield {"results_root": results_root, "errors": errors}
    finally:
        if prev is None:
            os.environ.pop("MOSAIC_RESULTS_DIR", None)
        else:
            os.environ["MOSAIC_RESULTS_DIR"] = prev


@pytest.mark.slow
@pytest.mark.parametrize("problem, exp_key", _all_experiments())
def test_experiment_runs_with_dummy(problem, exp_key, tmp_path, monkeypatch):
    """Every registered experiment runs end-to-end against the dummy.

    The dummy returns constant outputs so numerical results are trivial —
    we only check that the kernel + framework + per_solver_loop +
    apply_tesseract VJP pipeline executes and ``result.json`` lands on
    disk. This test uses its own ``tmp_path`` so a single-experiment run
    stays cheap (the session-scoped ``dummy_corpus`` fixture below is
    reserved for the plot tests, which genuinely need the full corpus).
    """
    from mosaic.benchmarks.problems import get_config

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg = get_config(problem)
    tag = f"inprocess:{_DUMMY_FOR[problem]}"
    tags = dict.fromkeys(cfg.solver_names, tag)
    cfg.experiments[exp_key].fn(cfg, tags)
    written = list((tmp_path / problem).rglob("result.json"))
    assert written, (
        f"{problem}/{exp_key}: no result.json found under {tmp_path / problem}"
    )


# ── Plot generation on dummy results ─────────────────────────────────────────


def _problem_plot_pairs():
    """``(problem, plot_key)`` for every registered ``cfg.plot_fns`` entry."""
    from mosaic.benchmarks.problems import get_config

    pairs: list[tuple[str, str]] = []
    for problem in _DUMMY_FOR:
        cfg = get_config(problem)
        for key in sorted(cfg.plot_fns):
            pairs.append((problem, key))
    return pairs


_DEGENERATE_DATA_HINTS = (
    "no positive values",  # matplotlib log-scale on all-zero data
    "can not be log-scaled",
    "zero-size array",  # np.min / np.max on empty arrays
    "empty array",
    "shape mismatch",  # consensus over a single solver array
)


def _skip_if_degenerate(exc: Exception) -> None:
    """Skip the test when ``exc`` smells like a "dummy constant data" crash.

    The dummies return constant arrays; some plots (log-scale axes,
    relative-error ratios, cosine of zero vectors) hit numerical
    degeneracies that wouldn't occur on real data. We treat those as
    "tested as far as it makes sense" rather than failures — anything
    else is a genuine plot-code bug and propagates normally.
    """
    msg = str(exc).lower()
    if any(hint in msg for hint in _DEGENERATE_DATA_HINTS):
        pytest.skip(f"degenerate dummy data: {exc}")


@pytest.mark.slow
@pytest.mark.parametrize("problem, plot_key", _problem_plot_pairs())
def test_plot_runs_on_dummy_results(problem, plot_key, dummy_corpus):
    """Every registered per-experiment plot fn runs on the dummy result corpus.

    Plot output may be trivial (NaN errors, empty curves) because the
    dummy returns constants, but the plot pipeline itself must execute
    without raising. Paper plots are tested separately below.
    """
    import matplotlib

    matplotlib.use("Agg")
    from mosaic.benchmarks.problems import get_config

    cfg = get_config(problem)
    plot_fn = cfg.plot_fns[plot_key]
    try:
        plot_fn(cfg)
    except (ValueError, ZeroDivisionError) as exc:
        _skip_if_degenerate(exc)
        raise
