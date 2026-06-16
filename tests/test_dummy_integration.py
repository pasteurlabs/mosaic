# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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

import contextlib
import json
import warnings
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
    assert "results" in result
    assert "extras" in result
    assert result["results"]
    # Result.json should have been written.
    json_path = tmp_path / "ns-grid" / "forward" / "dummy_baseline" / "result.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert "results" in on_disk


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
    assert "results" in result
    assert "extras" in result
    assert result["results"]
    json_path = (
        tmp_path / "structural-mesh" / "forward" / "dummy_baseline" / "result.json"
    )
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert "results" in on_disk


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
    assert "results" in result
    assert "extras" in result
    assert result["results"]
    json_path = tmp_path / "thermal-mesh" / "forward" / "dummy_baseline" / "result.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert "results" in on_disk


# ── Every-experiment parametrized smoke test ─────────────────────────────────

_DUMMY_FOR = {
    "ns-grid": DUMMY_NS_GRID,
    "ns-3d-grid": DUMMY_NS_GRID,
    "structural-mesh": DUMMY_STRUCTURAL_MESH,
    "thermal-mesh": DUMMY_THERMAL_MESH,
}


# ── Heavy-experiment shrink overrides ────────────────────────────────────────
#
# The ``jacobian_svd`` kernel builds the *full* Jacobian by looping one VJP
# call per output element (it can't use ``jax.jacrev``/``vmap`` — tesseract
# doesn't support vmap). The row count is ``D_out`` = output-field size, which
# for the registered ns-3d-grid cases is N³·components at N=8 ≈ 1500 rows × 7
# solvers ≈ 10k dummy-VJP calls *per case*. At ~45s each, those four cases
# alone dominated this file's runtime (~180s of ~350s total).
#
# For the dummy plumbing test we only care that the kernel + framework + VJP
# pipeline executes and writes ``result.json`` — a full N=8 Jacobian is
# overkill. We re-register the heavy ns-3d-grid cases at N=4 (8× fewer rows),
# preserving every other physics field so the run is otherwise identical. The
# numerical output is meaningless against a constant dummy either way.
#
# Keyed by ``(problem, exp_key)`` → physics dict (only N differs from the
# production registration in ``navier_stokes_3d_grid/config.py``).
_SHRINK_PHYSICS: dict[tuple[str, str], dict] = {
    ("ns-3d-grid", "gradient/jacobian_svd"): {
        "N": 4,
        "nu": 0.001,
        "dt": 0.05,
        "steps": 10,
    },
    ("ns-3d-grid", "gradient/jacobian_svd_steps20"): {
        "N": 4,
        "nu": 0.001,
        "dt": 0.05,
        "steps": 20,
    },
    ("ns-3d-grid", "gradient/jacobian_svd_steps40"): {
        "N": 4,
        "nu": 0.001,
        "dt": 0.05,
        "steps": 40,
    },
    ("ns-3d-grid", "gradient/jacobian_svd_nu01"): {
        "N": 4,
        "nu": 0.01,
        "dt": 0.05,
        "steps": 10,
    },
}


def _maybe_shrink(cfg, problem: str, exp_key: str) -> None:
    """Re-register ``exp_key`` at a smaller N if it's a known heavy case.

    Mutates ``cfg.experiments[exp_key]`` in place (the config object is a
    fresh per-call ``get_config(problem)``, so this never leaks across tests).
    No-op for experiments not in :data:`_SHRINK_PHYSICS`.
    """
    physics = _SHRINK_PHYSICS.get((problem, exp_key))
    if physics is None:
        return
    from mosaic.benchmarks.problems.shared.gradient import jacobian_svd

    cfg.add_experiment(
        exp_key,
        jacobian_svd,
        ic={"name": "tgv3d", "seed": 0},
        physics=physics,
        jacobian={"n_alphas": 41, "alpha_range": 0.3},
    )


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


# ── Degenerate-dummy warning suppression ─────────────────────────────────────
#
# The dummies return constant / all-zero arrays, so the framework + plot code
# hits numerical degeneracies that never occur on real data: log-scaling axes
# whose data has no positive values, a 0/0 cosine over zero-norm Jacobians, and
# a pydantic field whose ``0.0`` default isn't emitted into the JSON schema.
# These are expected here (see also :data:`_DEGENERATE_DATA_HINTS` /
# :func:`_skip_if_degenerate`), so we silence *exactly these* warnings while
# running against the dummy — without hiding the same categories in real runs.
#
# Each entry is a ``warnings.filterwarnings``-style spec ("ignore", message
# regex, category). Used both as a ``catch_warnings`` block (around the corpus
# build, which runs inside a fixture) and as ``@pytest.mark.filterwarnings``
# strings on the individual tests.
_DUMMY_WARNING_FILTERS: tuple[tuple[str, type[Warning]], ...] = (
    ("Data has no positive values", UserWarning),
    ("invalid value encountered in scalar divide", RuntimeWarning),
    ("Default value .* is not JSON serializable", UserWarning),
)


def _filterwarnings_marks() -> list:
    """``@pytest.mark.filterwarnings`` marks for the degenerate-dummy warnings."""
    return [
        pytest.mark.filterwarnings(f"ignore:{msg}:{cat.__name__}")
        for msg, cat in _DUMMY_WARNING_FILTERS
    ]


# Every test in this module runs against the constant-output dummies, so the
# degenerate-data warnings are expected for all of them. Applying the filters
# module-wide (they match only those specific message/category pairs) keeps the
# suppression scoped to this file without per-test decoration.
pytestmark = _filterwarnings_marks()


@contextlib.contextmanager
def _suppress_dummy_warnings():
    """Silence the known degenerate-dummy warnings within the block."""
    with warnings.catch_warnings():
        for msg, cat in _DUMMY_WARNING_FILTERS:
            warnings.filterwarnings("ignore", message=msg, category=cat)
        yield


# ── Shared dummy-result corpus ───────────────────────────────────────────────
#
# Building the corpus is the dominant cost in this file — every experiment
# runs through the framework, which pays a JAX warmup tax per kernel.
# Session-scoping the fixture means we build the corpus *once*; both the
# per-experiment (``test_experiment_runs_with_dummy``) and per-plot
# (``test_plot_runs_on_dummy_results``) test families read from it, instead of
# each per-experiment case re-running its experiment in a throwaway dir.


@pytest.fixture(scope="session")
def dummy_corpus(tmp_path_factory):
    """Run every (problem, exp_key) once against the dummy, yield the shared results dir.

    Session-scoped so the per-experiment and per-plot tests all share the same
    corpus. Individual experiment failures are captured on an ``errors`` dict
    (keyed by ``(problem, exp_key)``) so ``test_experiment_runs_with_dummy``
    can re-raise the exact exception for the failing case while the rest of the
    corpus still builds.
    """
    import os

    from mosaic.benchmarks.problems import get_config

    results_root = tmp_path_factory.mktemp("dummy_results")
    prev = os.environ.get("MOSAIC_RESULTS_DIR")
    os.environ["MOSAIC_RESULTS_DIR"] = str(results_root)
    errors: dict[tuple[str, str], Exception] = {}
    try:
        with _suppress_dummy_warnings():
            for problem, exp_key in _all_experiments():
                cfg = get_config(problem)
                _maybe_shrink(cfg, problem, exp_key)
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


@pytest.mark.parametrize("problem, exp_key", _all_experiments())
def test_experiment_runs_with_dummy(problem, exp_key, dummy_corpus):
    """Every registered experiment runs end-to-end against the dummy.

    The dummy returns constant outputs so numerical results are trivial —
    we only check that the kernel + framework + per_solver_loop +
    apply_tesseract VJP pipeline executes and ``result.json`` lands on
    disk.

    Both this test and :func:`test_plot_runs_on_dummy_results` read from the
    single session-scoped ``dummy_corpus`` — the experiments run *once* in the
    fixture, then each parametrized case here just asserts on its own cell.
    Re-running every experiment per case (the old approach) built the whole
    corpus twice, which was the dominant cost of this file.
    """
    errors = dummy_corpus["errors"]
    if (problem, exp_key) in errors:
        raise errors[(problem, exp_key)]
    # exp_key is already "<suite>/<experiment>", matching experiment_dir's
    # results/<problem>/<suite>/<experiment>/ layout.
    exp_dir = dummy_corpus["results_root"] / problem / exp_key
    assert (exp_dir / "result.json").exists(), (
        f"{problem}/{exp_key}: no result.json at {exp_dir / 'result.json'}"
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


@pytest.mark.parametrize("problem, plot_key", _problem_plot_pairs())
def test_plot_runs_on_dummy_results(problem, plot_key, dummy_corpus):
    """Every registered per-experiment plot fn runs on the dummy result corpus.

    Plot output may be trivial (NaN errors, empty curves) because the
    dummy returns constants, but the plot pipeline itself must execute
    without raising.
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
