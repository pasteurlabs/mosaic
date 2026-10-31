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


@pytest.fixture(autouse=True)
def _no_png(monkeypatch):
    """Skip the PNG rasterisation of every chart for every test here.

    The plot tests only verify that the plot pipeline *runs* against the dummy
    corpus — the rendered files themselves are never inspected and are thrown
    away with the temp results dir. The chart is still fully built, validated
    against the Vega-Lite schema and written as ``.vl.json``; only the
    vl-convert render to PNG is skipped. Scoped to this file via
    ``monkeypatch``; real runs unaffected.
    """
    import vl_convert

    monkeypatch.setattr(vl_convert, "vegalite_to_png", lambda *_a, **_kw: b"")


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
        assert out["result"] is not None
        assert np.asarray(out["result"]).shape == v0.shape


def test_dummy_raises_on_unsupported_boundary_conditions():
    """The BC contract: a solver raises on conditions it does not implement."""
    import numpy as np
    from tesseract_core import Tesseract

    from .helpers import channel_inputs

    inputs = channel_inputs(10)
    with Tesseract.from_tesseract_api(str(DUMMY_NS_GRID)) as t:
        out = t.apply(inputs)
        assert np.asarray(out["body_force"]).shape == (inputs["steps"], 2)
        bcs = {**inputs["boundary_conditions"], "x_hi": {"type": "no_slip"}}
        with pytest.raises(Exception, match="supports fully periodic"):
            t.apply({**inputs, "boundary_conditions": bcs})


def test_cylinder_scores_the_dummy_against_the_literature(
    ns_grid_tags, tmp_path, monkeypatch
):
    """The dummy returns the reference force, so its Cd matches the reference."""
    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = ns_grid_tags
    from mosaic.benchmarks.problems.navier_stokes_grid import cylinder

    cfg.add_experiment(
        "forward/dummy_cylinder",
        cylinder.cylinder,
        ic={"name": "dfg_inflow", "seed": 0},
        physics={"resolution": [10]},
    )
    result = cfg.experiments["forward/dummy_cylinder"].fn(cfg, tags)
    metrics = [r["metrics"] for r in result["results"] if r["metrics"].get("valid")]
    assert metrics
    for m in metrics:
        assert m["cd"] == pytest.approx(cylinder.CD_REF, rel=1e-5)
        assert m["error"] < 1e-5


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


@pytest.mark.parametrize("kernel_name", ["agreement", "physical_laws"])
def test_forward_nonfinite_output_fails_with_reason(
    kernel_name, ns_grid_tags, tmp_path, monkeypatch
):
    """A solver returning NaN fails its cell, and the reason reaches status."""
    import jax.numpy as jnp

    from mosaic.benchmarks.core import runner
    from mosaic.benchmarks.core.status import FAILED, OK, _classify_result
    from mosaic.benchmarks.problems.shared import forward

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = ns_grid_tags
    bad = cfg.solver_names[0]
    # A distinct tag for the same dummy lets the patched apply tell the
    # poisoned solver apart from its peers.
    tags[bad] = f"inprocess:{DUMMY_NS_GRID.parent}/./{DUMMY_NS_GRID.name}"
    real_apply = runner._apply_tesseract_with_deadline

    def apply_nan_for_bad(t, inputs):
        out = real_apply(t, inputs)
        if runner.current_worker_context().image_tag == tags[bad]:
            out = {**out, "result": jnp.full_like(out["result"], jnp.nan)}
        return out

    monkeypatch.setattr(runner, "_apply_tesseract_with_deadline", apply_nan_for_bad)

    # Register on copies so the throwaway experiment doesn't leak into the
    # shared config other tests iterate over.
    monkeypatch.setattr(cfg, "experiments", dict(cfg.experiments))
    monkeypatch.setattr(cfg, "plot_fns", dict(cfg.plot_fns))
    cfg.add_experiment(
        "forward/dummy_nonfinite",
        getattr(forward, kernel_name),
        ic={"name": "multimode", "seed": 0},
        physics={"N": 4, "nu": [0.01, 0.05], "dt": 0.01, "steps": 1},
        diagnostics={"mean": lambda out, **_: float(jnp.mean(out))},
    )
    cfg.experiments["forward/dummy_nonfinite"].fn(cfg, tags)

    json_path = tmp_path / "ns-grid" / "forward" / "dummy_nonfinite" / "result.json"
    data = json.loads(json_path.read_text())
    bad_entries = [e for e in data["results"] if e["solver"] == bad]
    assert len(bad_entries) == 2
    for e in bad_entries:
        assert e["metrics"]["valid"] is False
        assert "'result' is not finite" in e["metrics"]["error"]

    cells = _classify_result(data, cfg.solver_names, [])
    assert cells[bad].status == FAILED
    assert "not finite" in cells[bad].reason
    assert all(cells[s].status == OK for s in cfg.solver_names if s != bad)


def test_agreement_single_solver_is_unscored(ns_grid_tags, tmp_path, monkeypatch):
    """Alone, a solver has no consensus to be scored against: it is left
    unscored (not run), not failed. With its peers' snapshots cached on disk
    from an earlier run, it is scored against them.
    """
    from mosaic.benchmarks.core.status import NOT_RUN, _classify_result
    from mosaic.benchmarks.problems.shared import forward

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = ns_grid_tags
    solo = cfg.solver_names[0]
    monkeypatch.setattr(cfg, "experiments", dict(cfg.experiments))
    monkeypatch.setattr(cfg, "plot_fns", dict(cfg.plot_fns))
    cfg.add_experiment(
        "forward/dummy_single",
        forward.agreement,
        ic={"name": "multimode", "seed": 0},
        physics={"N": 4, "nu": [0.01, 0.05], "dt": 0.01, "steps": 1},
    )
    run = cfg.experiments["forward/dummy_single"].fn

    alone = run(cfg, {solo: tags[solo]})
    assert [e["metrics"] for e in alone["results"]] == [
        {"status": "skipped", "unscored": True, "reason": forward._NO_REFERENCE}
    ] * 2
    assert _classify_result(alone, [solo], [])[solo].status == NOT_RUN

    run(cfg, tags)
    with_cache = run(cfg, {solo: tags[solo]})
    solo_entries = [e for e in with_cache["results"] if e["solver"] == solo]
    assert [e["metrics"]["valid"] for e in solo_entries] == [True, True]


def test_agreement_only_rerun_uses_cached_peers(ns_grid_tags, tmp_path, monkeypatch):
    """An ``--only`` rerun of one solver keeps every solver in ``tags``; the
    ones it did not rerun still count through their cached snapshots.
    """
    from mosaic.benchmarks.core import cell_filter
    from mosaic.benchmarks.problems.shared import forward

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = ns_grid_tags
    solo = cfg.solver_names[0]
    monkeypatch.setattr(cfg, "experiments", dict(cfg.experiments))
    monkeypatch.setattr(cfg, "plot_fns", dict(cfg.plot_fns))
    cfg.add_experiment(
        "forward/dummy_only",
        forward.agreement,
        ic={"name": "multimode", "seed": 0},
        physics={"N": 4, "nu": [0.01, 0.05], "dt": 0.01, "steps": 1},
    )
    run = cfg.experiments["forward/dummy_only"].fn

    run(cfg, tags)
    cell_filter.set_active({("forward/dummy_only", solo): True})
    try:
        rerun = run(cfg, tags)
    finally:
        cell_filter.set_active(None)
    solo_entries = [e for e in rerun["results"] if e["solver"] == solo]
    assert [e["metrics"].get("valid") for e in solo_entries] == [True, True]


def test_cost_only_rerun_keeps_other_solvers_timings(
    ns_grid_tags, tmp_path, monkeypatch
):
    """An ``--only`` rerun of one solver's cost cell leaves its peers' timings on disk."""
    from mosaic.benchmarks.core import cell_filter
    from mosaic.benchmarks.core.io import load_experiment_result
    from mosaic.benchmarks.problems.shared import cost

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = ns_grid_tags
    solo = cfg.solver_names[0]
    monkeypatch.setattr(cfg, "experiments", dict(cfg.experiments))
    monkeypatch.setattr(cfg, "plot_fns", dict(cfg.plot_fns))
    cfg.add_experiment(
        "cost/dummy_only",
        cost.spatial_cost,
        physics={"N": [4], "nu": 0.01, "dt": 0.01, "steps": 1},
        cost={"n_trials": 1},
    )
    run = cfg.experiments["cost/dummy_only"].fn

    run(cfg, tags)
    cell_filter.set_active({("cost/dummy_only", solo): True})
    try:
        run(cfg, tags)
    finally:
        cell_filter.set_active(None)
    saved = load_experiment_result(tmp_path / cfg.name / "cost" / "dummy_only")
    timed = {e["solver"] for e in saved["results"] if "mean" in e["metrics"]}
    assert timed == set(cfg.solver_names)


@pytest.mark.parametrize("kernel_name", ["fd_check", "param_sweep", "analytic_check"])
def test_gradient_nonfinite_vjp_fails_with_reason(
    kernel_name, ns_grid_tags, tmp_path, monkeypatch
):
    """A NaN VJP fails the cell with its reason; a sweep runs on past it."""
    import jax.numpy as jnp

    from mosaic.benchmarks.core import runner
    from mosaic.benchmarks.core.status import FAILED, _classify_result
    from mosaic.benchmarks.problems.navier_stokes_grid.ics import _tgv_analytic_grad
    from mosaic.benchmarks.problems.shared import gradient

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = ns_grid_tags
    bad = cfg.solver_names[0]
    tags[bad] = f"inprocess:{DUMMY_NS_GRID.parent}/./{DUMMY_NS_GRID.name}"
    real_vjp = gradient._vjp_grad
    bad_calls = []

    # Only the poisoned solver's first VJP is NaN, so a sweep's second point
    # shows whether the sweep carried on.
    def vjp_nan_once_for_bad(*args):
        g = real_vjp(*args)
        if runner.current_worker_context().image_tag == tags[bad]:
            bad_calls.append(None)
            if len(bad_calls) == 1:
                g = jnp.full_like(g, jnp.nan)
        return g

    monkeypatch.setattr(gradient, "_vjp_grad", vjp_nan_once_for_bad)

    # Register on copies so the throwaway experiment doesn't leak into the
    # shared config other tests iterate over.
    monkeypatch.setattr(cfg, "experiments", dict(cfg.experiments))
    monkeypatch.setattr(cfg, "plot_fns", dict(cfg.plot_fns))
    swept = kernel_name in ("param_sweep", "analytic_check")
    cfg.add_experiment(
        "gradient/dummy_nonfinite",
        getattr(gradient, kernel_name),
        ic={
            "name": "tgv" if kernel_name == "analytic_check" else "multimode",
            "seed": 0,
        },
        physics={
            "N": 4,
            "nu": [0.01, 0.05] if swept else 0.01,
            "dt": 0.01,
            "steps": 1,
        },
        fd={"eps_values": [1e-1], "n_dirs": 2},
        reference=_tgv_analytic_grad,
    )
    cfg.experiments["gradient/dummy_nonfinite"].fn(cfg, tags)

    json_path = tmp_path / "ns-grid" / "gradient" / "dummy_nonfinite" / "result.json"
    data = json.loads(json_path.read_text())
    if swept:
        bad_entries = sorted(
            (e for e in data["results"] if e["solver"] == bad),
            key=lambda e: float(e["sweep_value"]),
        )
        assert [e["metrics"].get("status") for e in bad_entries] == ["failed", None]
        assert bad_entries[0]["metrics"]["failure_type"] == "nan"
        assert bad_entries[1]["metrics"]["grad_norm"] is not None

    cells = _classify_result(data, cfg.solver_names, [])
    assert cells[bad].status == FAILED
    assert "VJP returned non-finite gradient" in cells[bad].reason
    # OpenFOAM has no VJP and is never selected, hence not_run.
    assert all(cells[s].status != FAILED for s in cfg.solver_names if s != bad)


def test_analytic_check_scores_every_sweep_point(ns_grid_tags, tmp_path, monkeypatch):
    """The registered analytic check records one scored point per viscosity.

    The dummies' VJP is zero, which gives a relative error of 1 and an
    amplitude error of -1.
    """
    from mosaic.benchmarks.core.status import OK, _classify_result

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = ns_grid_tags
    exp = cfg.experiments["gradient/analytic_check"]
    exp.fn(cfg, tags)

    out_dir = tmp_path / "ns-grid" / "gradient" / "analytic_check"
    data = json.loads((out_dir / "result.json").read_text())
    nus = data["sweep"]["values"]
    solver = cfg.solver_names[0]
    entries = [e for e in data["results"] if e["solver"] == solver]
    assert sorted(float(e["sweep_value"]) for e in entries) == sorted(map(float, nus))
    for e in entries:
        assert e["metrics"]["valid"] is True
        assert e["metrics"]["error"] == pytest.approx(1.0)
        assert e["metrics"]["amplitude_err"] == pytest.approx(-1.0)
    assert _classify_result(data, [solver], [])[solver].status == OK

    cfg.plot_fns["gradient/analytic_check"](cfg)
    assert (out_dir / "analytic_check.vl.json").exists()


@pytest.mark.parametrize(
    ("failure", "only_seed_1"),
    [("target", False), ("optim", False), ("target", True), ("optim", True)],
)
def test_recovery_failure_fails_with_reason(
    failure, only_seed_1, tmp_path, monkeypatch
):
    """A NaN target or a failed optimization fails the cell with that reason,
    whether every seed fails or only one of them.
    """
    import jax.numpy as jnp

    from mosaic.benchmarks.core import runner
    from mosaic.benchmarks.core.status import FAILED, OK, _classify_result
    from mosaic.benchmarks.problems import get_config
    from mosaic.benchmarks.problems.navier_stokes_3d_grid import optimization

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg = get_config("ns-3d-grid")
    bad, good = "PICT", "XLB"
    tags = {
        bad: f"inprocess:{DUMMY_NS_GRID.parent}/./{DUMMY_NS_GRID.name}",
        good: f"inprocess:{DUMMY_NS_GRID}",
    }

    calls = {"apply": 0, "optim": 0}

    def is_bad(kind: str) -> bool:
        if runner.current_worker_context().image_tag != tags[bad]:
            return False
        calls[kind] += 1
        # Seeds run in order, so the second call is ic_seed=1.
        return calls[kind] == 2 if only_seed_1 else True

    if failure == "target":
        real_apply = optimization.apply_tesseract

        def apply_nan_for_bad(t, inputs):
            out = real_apply(t, inputs)
            if is_bad("apply"):
                out = {**out, "result": jnp.full_like(out["result"], jnp.nan)}
            return out

        monkeypatch.setattr(optimization, "apply_tesseract", apply_nan_for_bad)
        expected = (
            "target forward failed for 1 of 3 ic_seeds: ic_seed=1: target is not finite"
            if only_seed_1
            else "target forward failed for every ic_seed: ic_seed=0: target is not finite"
        )
    else:
        real_lbfgs = optimization._run_lbfgs

        def lbfgs_fails_for_bad(*args, **kwargs):
            if is_bad("optim"):
                raise FloatingPointError("optimization loss is not finite (nan)")
            return real_lbfgs(*args, **kwargs)

        monkeypatch.setattr(optimization, "_run_lbfgs", lbfgs_fails_for_bad)
        expected = (
            "optimization failed for 1 of 3 ic_seeds: ic_seed=1: FloatingPointError"
            if only_seed_1
            else "optimization failed for every ic_seed: ic_seed=0: FloatingPointError"
        )

    data = cfg.experiments["optimization/recovery_constant_ic_bfgs_proj"].fn(cfg, tags)

    cells = _classify_result(data, [bad, good], [])
    assert cells[bad].status == FAILED
    assert expected in cells[bad].reason
    assert cells[good].status == OK


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
    "no positive values",  # log-scale on all-zero data
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
    from mosaic.benchmarks.problems import get_config

    cfg = get_config(problem)
    plot_fn = cfg.plot_fns[plot_key]
    try:
        plot_fn(cfg)
    except (ValueError, ZeroDivisionError) as exc:
        _skip_if_degenerate(exc)
        raise


@pytest.mark.parametrize(
    "ic_name,expected_label",
    [("tgv", "analytic"), ("multimode", "consensus")],
    ids=["tgv-has-a-closed-form", "multimode-does-not"],
)
def test_agreement_reference_matches_the_ic(
    ns_grid_tags, tmp_path, monkeypatch, ic_name, expected_label
):
    """An analytic reference must only score the IC it actually solves.

    ``_tgv_analytic`` rebuilds the Taylor-Green field from the grid size and
    ignores the IC it is handed, so before ``supported_ics`` the multimode
    run was scored against a TGV field. That error measures the gap between
    two unrelated flows, which is why every solver reported the same one.
    """
    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = ns_grid_tags

    from mosaic.benchmarks.problems.shared.forward import agreement

    cfg.add_experiment(
        f"forward/dummy_ref_{ic_name}",
        agreement,
        ic={"name": ic_name, "seed": 0},
        physics={"N": [4, 8], "nu": 0.05, "dt": 0.01, "steps": 1},
    )
    result = cfg.experiments[f"forward/dummy_ref_{ic_name}"].fn(cfg, tags)
    assert result["extras"]["reference_label"] == expected_label


def test_continue_keeps_solvers_finished_before_a_crash(
    ns_grid_tags, tmp_path, monkeypatch
):
    """Solvers checkpointed before a crash are not rerun on --continue, and
    their metrics and snapshots still reach the result and the consensus.
    """
    from mosaic.benchmarks.core import runner
    from mosaic.benchmarks.problems.shared import forward

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg, tags = ns_grid_tags
    first, victim = cfg.solver_names[:2]
    tags = {
        first: tags[first],
        victim: f"inprocess:{DUMMY_NS_GRID.parent}/./{DUMMY_NS_GRID.name}",
    }
    monkeypatch.setattr(cfg, "experiments", dict(cfg.experiments))
    monkeypatch.setattr(cfg, "plot_fns", dict(cfg.plot_fns))
    cfg.add_experiment(
        "forward/dummy_resume",
        forward.agreement,
        ic={"name": "multimode", "seed": 0},
        physics={"N": 4, "nu": [0.01, 0.05], "dt": 0.01, "steps": 1},
    )
    run = cfg.experiments["forward/dummy_resume"].fn
    real_apply = forward.safe_apply
    ran = []

    def crash_on_victim(*args, **kwargs):
        tag = runner.current_worker_context().image_tag
        ran.append(tag)
        if tag == tags[victim] and kill:
            raise KeyboardInterrupt
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(forward, "safe_apply", crash_on_victim)
    kill = True
    with pytest.raises(KeyboardInterrupt):
        run(cfg, tags)

    kill = False
    ran.clear()
    resumed = run(cfg, tags, resume=True)
    assert tags[first] not in ran
    assert {e["solver"]: e["metrics"].get("valid") for e in resumed["results"]} == {
        first: True,
        victim: True,
    }


# ── Partial reruns against a full run ────────────────────────────────────────
#
# A partial run (``--only`` on one solver's cells, ``-s`` one solver, or
# ``--continue`` after a crash) merges what it reruns with what is already on
# disk. What it leaves there must be what a full run leaves: the same
# result.json, the same field snapshots and the same ``mosaic status``.
# Timings and memory peaks differ from run to run, so only their presence is
# compared.

_PARTIAL_KERNELS = (
    "agreement",
    "physical_laws",
    "fd_check",
    "param_sweep",
    "analytic_check",
    "spatial_cost",
    "temporal_cost",
    "vjp_cost",
)

# Metrics measured afresh on every run: timings (``cost_time_s`` is the status
# cell's) and memory peaks.
_MEASURED = {
    "mean",
    "std",
    "trials_s",
    "wall_time_s",
    "cost_time_s",
    "ram_peak_mib",
    "vram_peak_mib",
}


def _register_partial_kernel(cfg, kernel_name: str) -> str:
    """Register a small *kernel_name* experiment on *cfg* and return its key."""
    from mosaic.benchmarks.problems.navier_stokes_grid.ics import _tgv_analytic_grad
    from mosaic.benchmarks.problems.shared import cost, forward, gradient

    multimode = {"name": "multimode", "seed": 0}
    physics = {"N": 4, "nu": [0.01, 0.05], "dt": 0.01, "steps": 1}
    fd = {"eps_values": [1e-1], "n_dirs": 2}
    trials = {"n_trials": 1}
    suite, kernel_fn, spec = {
        "agreement": ("forward", forward.agreement, {"ic": multimode}),
        "physical_laws": (
            "forward",
            forward.physical_laws,
            {
                "ic": multimode,
                "diagnostics": {"mean": lambda out, **_: float(out.mean())},
            },
        ),
        "fd_check": (
            "gradient",
            gradient.fd_check,
            {"ic": multimode, "physics": {**physics, "nu": 0.01}, "fd": fd},
        ),
        "param_sweep": ("gradient", gradient.param_sweep, {"ic": multimode, "fd": fd}),
        "analytic_check": (
            "gradient",
            gradient.analytic_check,
            {"ic": {"name": "tgv", "seed": 0}, "reference": _tgv_analytic_grad},
        ),
        "spatial_cost": (
            "cost",
            cost.spatial_cost,
            {"physics": {**physics, "N": [4, 8], "nu": 0.01}, "cost": trials},
        ),
        "temporal_cost": (
            "cost",
            cost.temporal_cost,
            {"physics": {**physics, "nu": 0.01, "steps": [1, 2]}, "cost": trials},
        ),
        "vjp_cost": (
            "cost",
            cost.vjp_cost,
            {"physics": {**physics, "N": [4, 8], "nu": 0.01}, "cost": trials},
        ),
    }[kernel_name]
    exp_key = f"{suite}/dummy_partial_{kernel_name}"
    cfg.add_experiment(exp_key, kernel_fn, **{"physics": physics, **spec})
    return exp_key


def _unmeasured(metrics):
    """*metrics* with each measured value replaced by a marker."""
    if not isinstance(metrics, dict):
        return metrics
    return {k: "measured" if k in _MEASURED else v for k, v in metrics.items()}


def _fields_by_solver(path: Path) -> dict:
    """The arrays in *path*, keyed by solver name rather than by slot.

    A rerun writes its own solvers first in ``solver_names``, so a positional
    key such as ``grad_0`` can name another solver than after a full run.
    """
    import re

    from mosaic.benchmarks.core.io import try_load_npz

    arrays = try_load_npz(path)
    names = [str(n) for n in arrays.pop("solver_names", [])]
    flat = any(key.startswith(f"{n}_") for key in arrays for n in names)
    fields = {"solver_names": sorted(names)}
    for key, arr in arrays.items():
        m = None if flat else re.fullmatch(r"(.+?)_(\d+)(_.*)?", key)
        if m and int(m[2]) < len(names):
            key = f"{m[1]}[{names[int(m[2])]}]{m[3] or ''}"
        fields[key] = arr
    return fields


def _saved_run(cfg, exp_key: str) -> tuple[dict, dict, dict]:
    """What runs of *exp_key* left on disk: result.json, fields and status."""
    import dataclasses

    from mosaic.benchmarks.core.hardware import hardware_by_solver
    from mosaic.benchmarks.core.io import results_dir
    from mosaic.benchmarks.core.status import collect_status

    exp_dir = results_dir() / cfg.name / exp_key
    data = json.loads((exp_dir / "result.json").read_text())
    # The rest of the provenance is when and how long, not what.
    prov = data.pop("provenance")
    data["tesseract_hashes"] = prov["tesseract_hashes"]
    data["harness_hash"] = prov["harness_hash"]
    # A merge records the machine per solver instead of once for the run.
    hardware = hardware_by_solver(data)
    if hardware:
        data["extras"].pop("hardware", None)
        data["extras"]["hardware_by_solver"] = hardware
    data["results"] = sorted(
        ({**e, "metrics": _unmeasured(e["metrics"])} for e in data["results"]),
        key=lambda e: (e["solver"], str(e["sweep_value"])),
    )
    fields = {
        p.name: _fields_by_solver(p)
        for p in sorted(exp_dir.glob("*.npz"))
        if p.name != "result_partial.npz"
    }
    suite, _, name = exp_key.partition("/")
    row = next(r for r in collect_status(cfg, [suite]).rows if r.experiment == name)
    cells = {
        s: dataclasses.replace(c, metrics=_unmeasured(c.metrics))
        for s, c in row.cells.items()
    }
    return data, fields, cells


def _assert_same_run(partial: tuple, full: tuple) -> None:
    """Assert that *partial* and *full*, from :func:`_saved_run`, are the same."""
    import numpy as np

    data, fields, cells = partial
    full_data, full_fields, full_cells = full
    assert data == full_data
    assert cells == full_cells
    assert fields.keys() == full_fields.keys()
    for name, arrays in fields.items():
        assert arrays.keys() == full_fields[name].keys(), name
        for key, arr in arrays.items():
            np.testing.assert_array_equal(arr, full_fields[name][key], err_msg=key)


def _run_partially(how: str, cfg, tags, exp_key, solver: str, monkeypatch) -> None:
    """Rerun *exp_key* the way ``mosaic run`` does for *how*.

    ``only`` reruns *solver*'s cells (``--only``), ``solvers`` runs *solver*
    alone (``-s``), and ``continue`` crashes when *solver* starts and then
    resumes (``--continue``).
    """
    import dataclasses

    from mosaic.benchmarks.core import cell_filter, experiment

    run = cfg.experiments[exp_key].fn
    if how == "only":
        cell_filter.set_active({(exp_key, solver): True})
        try:
            run(cfg, tags)
        finally:
            cell_filter.set_active(None)
        return
    if how == "solvers":
        only_solver = dataclasses.replace(cfg, solvers=[cfg.solver(solver)])
        run(only_solver, {solver: tags[solver]})
        return

    real_loop = experiment.per_solver_loop

    def crash_on_solver(cfg, tags, names, work, **kwargs):
        def work_or_crash(name, t):
            if name == solver:
                raise KeyboardInterrupt
            work(name, t)

        return real_loop(cfg, tags, names, work_or_crash, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(experiment, "per_solver_loop", crash_on_solver)
        with pytest.raises(KeyboardInterrupt):
            run(cfg, tags)
    run(cfg, tags, resume=True)


@pytest.mark.parametrize("how", ["only", "solvers", "continue"])
@pytest.mark.parametrize("kernel_name", _PARTIAL_KERNELS)
def test_partial_rerun_saves_what_a_full_run_saves(
    kernel_name, how, ns_grid_tags, tmp_path, monkeypatch
):
    """``--only``, ``-s`` and ``--continue`` on one solver save what a full run does."""
    from mosaic.benchmarks.core import experiment

    cfg, tags = ns_grid_tags
    monkeypatch.setattr(cfg, "experiments", dict(cfg.experiments))
    monkeypatch.setattr(cfg, "plot_fns", dict(cfg.plot_fns))
    exp_key = _register_partial_kernel(cfg, kernel_name)
    run = cfg.experiments[exp_key].fn
    # The cost checks compare timings, which no two runs share.
    monkeypatch.delitem(cfg.status_checks, "cost", raising=False)

    ran: list[str] = []
    real_loop = experiment.per_solver_loop

    def record_solvers(cfg, tags, names, *args, **kwargs):
        ran.extend(names)
        return real_loop(cfg, tags, names, *args, **kwargs)

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path / "full"))
    with monkeypatch.context() as m:
        m.setattr(experiment, "per_solver_loop", record_solvers)
        run(cfg, tags)
    full = _saved_run(cfg, exp_key)

    # One in the middle, so it has peers that ran before and after it.
    solver = ran[len(ran) // 2]
    if how == "continue":
        monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path / "resumed"))
    _run_partially(how, cfg, tags, exp_key, solver, monkeypatch)
    _assert_same_run(_saved_run(cfg, exp_key), full)
