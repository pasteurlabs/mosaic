"""Unit tests for ``benchmarks.core.runner``.

Covers behaviour reachable without a real Tesseract container:
- the ``_install_tesseract_http_timeout`` monkey-patch (idempotent; safe with
  or without tesseract_core installed),
- the ``MOSAIC_TESSERACT_TIMEOUT`` env-var contract,
- ``safe_apply`` / ``safe_apply_with_extras`` error handling (missing output
  key, non-finite arrays, exception propagation, TimeoutError messaging,
  per-thread error storage), and
- the pure helper ``image_tags_no_build``.
"""

from __future__ import annotations

import importlib
import threading

import jax.numpy as jnp
import pytest

from mosaic.benchmarks.core import runner

# ── Module-level constants ────────────────────────────────────────────────────


def test_tesseract_timeout_default_is_1200():
    """Default timeout is 20 minutes when the env var is unset."""
    # The module-level constant is captured at import time, so this test
    # documents the *baseline* value the codebase ships with. A separate test
    # below verifies the env-var override mechanism via reload.
    assert runner.MOSAIC_TESSERACT_TIMEOUT == 1200.0


def test_tesseract_timeout_env_var_override(monkeypatch):
    """MOSAIC_TESSERACT_TIMEOUT overrides the default after a module reload."""
    monkeypatch.setenv("MOSAIC_TESSERACT_TIMEOUT", "42.5")
    reloaded = importlib.reload(runner)
    try:
        assert reloaded.MOSAIC_TESSERACT_TIMEOUT == 42.5
    finally:
        # Restore the original module state for the rest of the suite.
        monkeypatch.delenv("MOSAIC_TESSERACT_TIMEOUT", raising=False)
        importlib.reload(runner)


def test_connect_timeout_is_short():
    """Connect timeout is always short — a slow connect means the container died."""
    assert runner._MOSAIC_TESSERACT_CONNECT_TIMEOUT == 30.0


# ── HTTP timeout monkey-patch ─────────────────────────────────────────────────


def test_install_http_timeout_is_idempotent():
    """Calling _install_tesseract_http_timeout twice must not double-wrap."""
    # First call (module import already ran it once, this is at least the 2nd).
    runner._install_tesseract_http_timeout()
    runner._install_tesseract_http_timeout()
    runner._install_tesseract_http_timeout()
    # No exception, no infinite recursion — that's the contract.


def test_install_http_timeout_no_op_without_tesseract_core(monkeypatch):
    """If tesseract_core is None at module scope, the patcher returns silently."""
    monkeypatch.setattr(runner, "tesseract_core", None)
    # Must not raise even though there's nothing to patch.
    runner._install_tesseract_http_timeout()


# ── safe_apply_with_extras: success paths ─────────────────────────────────────


def test_safe_apply_with_extras_success_returns_array_only(monkeypatch):
    """Happy path with no extras returns the primary array and empty dicts."""
    arr = jnp.array([1.0, 2.0, 3.0])

    def fake_apply(t, inputs):
        return {"result": arr}

    monkeypatch.setattr(runner, "_apply_tesseract_with_deadline", fake_apply)

    out_arr, extras, state = runner.safe_apply_with_extras(
        t=None, inputs={}, output_key="result", extra_scalar_keys=[], state_keys=[]
    )
    assert out_arr is not None
    assert jnp.array_equal(out_arr, arr)
    assert extras == {}
    assert state == {}


def test_safe_apply_with_extras_extracts_scalar_extras(monkeypatch):
    """extra_scalar_keys are flattened and returned as Python floats."""

    def fake_apply(t, inputs):
        return {
            "result": jnp.zeros(3),
            "potential_energy": jnp.array([2.5]),  # shape-(1,) array
            "kinetic_energy": 1.25,  # bare scalar
            "drag": jnp.asarray(0.75),
        }

    monkeypatch.setattr(runner, "_apply_tesseract_with_deadline", fake_apply)

    _, extras, _ = runner.safe_apply_with_extras(
        None,
        {},
        "result",
        extra_scalar_keys=["potential_energy", "kinetic_energy", "drag"],
        state_keys=[],
    )
    assert extras == pytest.approx(
        {"potential_energy": 2.5, "kinetic_energy": 1.25, "drag": 0.75}
    )


def test_safe_apply_with_extras_drops_non_finite_scalar_extras(monkeypatch):
    """A NaN/inf scalar extra is silently dropped rather than poisoning the dict."""

    def fake_apply(t, inputs):
        return {
            "result": jnp.zeros(3),
            "good": jnp.array(1.0),
            "bad_nan": jnp.array(float("nan")),
            "bad_inf": jnp.array(float("inf")),
        }

    monkeypatch.setattr(runner, "_apply_tesseract_with_deadline", fake_apply)

    _, extras, _ = runner.safe_apply_with_extras(
        None, {}, "result", ["good", "bad_nan", "bad_inf"], []
    )
    assert "good" in extras
    assert "bad_nan" not in extras
    assert "bad_inf" not in extras


def test_safe_apply_with_extras_returns_state_arrays(monkeypatch):
    """state_keys come back as jax arrays for downstream threading."""

    def fake_apply(t, inputs):
        return {
            "result": jnp.zeros(3),
            "velocity": jnp.array([1.0, 2.0]),
            "pressure": jnp.array([10.0]),
        }

    monkeypatch.setattr(runner, "_apply_tesseract_with_deadline", fake_apply)

    _, _, state = runner.safe_apply_with_extras(
        None, {}, "result", [], state_keys=["velocity", "pressure"]
    )
    assert set(state.keys()) == {"velocity", "pressure"}
    assert jnp.array_equal(state["velocity"], jnp.array([1.0, 2.0]))


# ── safe_apply_with_extras: failure paths ─────────────────────────────────────


def test_safe_apply_with_extras_missing_output_key(monkeypatch):
    """When output_key isn't in the response, returns (None, {}, {}) and
    records the error message in thread-local state."""

    def fake_apply(t, inputs):
        return {"other_key": jnp.zeros(3)}

    monkeypatch.setattr(runner, "_apply_tesseract_with_deadline", fake_apply)

    arr, extras, state = runner.safe_apply_with_extras(None, {}, "result", [], [])
    assert arr is None
    assert extras == {}
    assert state == {}
    err = runner.get_last_apply_error()
    assert err is not None
    assert "output_key 'result'" in err
    assert "other_key" in err  # surfaces the available keys for debugging


def test_safe_apply_with_extras_non_finite_array(monkeypatch):
    """A NaN/inf in the primary array means failure — returns (None, {}, {})."""

    def fake_apply(t, inputs):
        return {"result": jnp.array([1.0, float("nan"), 3.0])}

    monkeypatch.setattr(runner, "_apply_tesseract_with_deadline", fake_apply)

    arr, extras, state = runner.safe_apply_with_extras(None, {}, "result", [], [])
    assert arr is None
    assert extras == {}
    assert state == {}


def test_safe_apply_with_extras_exception_wraps_error(monkeypatch):
    """Generic exception is caught, msg stored in thread-local, None returned."""

    def fake_apply(t, inputs):
        raise RuntimeError("solver crashed mid-step")

    monkeypatch.setattr(runner, "_apply_tesseract_with_deadline", fake_apply)

    # Clear any prior state from this thread
    if hasattr(runner._tl, "last_apply_error"):
        del runner._tl.last_apply_error

    arr, _, _ = runner.safe_apply_with_extras(None, {}, "result", [], [])
    assert arr is None
    err = runner.get_last_apply_error()
    assert err is not None
    assert "RuntimeError" in err
    assert "solver crashed mid-step" in err


def test_safe_apply_with_extras_timeout_error_includes_duration(monkeypatch):
    """TimeoutError gets a specialised message mentioning the timeout budget."""

    def fake_apply(t, inputs):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(runner, "_apply_tesseract_with_deadline", fake_apply)

    arr, _, _ = runner.safe_apply_with_extras(None, {}, "result", [], [])
    assert arr is None
    err = runner.get_last_apply_error()
    assert err is not None
    assert "TimeoutError" in err
    assert "did not respond within" in err
    assert "1200" in err  # the default MOSAIC_TESSERACT_TIMEOUT


def test_get_last_apply_error_is_thread_local(monkeypatch):
    """Errors recorded on one thread must not leak into another thread."""

    def fake_apply(t, inputs):
        raise RuntimeError("thread-A error")

    monkeypatch.setattr(runner, "_apply_tesseract_with_deadline", fake_apply)

    # Clear state on the main (this) thread before the experiment.
    if hasattr(runner._tl, "last_apply_error"):
        del runner._tl.last_apply_error

    # Run safe_apply on a side thread; capture what get_last_apply_error
    # returns from that *same* thread and from the main thread.
    seen_from_worker: dict = {}

    def worker():
        runner.safe_apply(None, {}, "result")
        seen_from_worker["err"] = runner.get_last_apply_error()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert seen_from_worker["err"] is not None
    assert "thread-A error" in seen_from_worker["err"]
    # Main thread never called safe_apply, so its TLS slot is still empty.
    assert runner.get_last_apply_error() is None


# ── safe_apply (thin wrapper) ─────────────────────────────────────────────────


def test_safe_apply_returns_only_primary_array(monkeypatch):
    """safe_apply is the array-only wrapper around safe_apply_with_extras.

    The thin wrapper exists so callers that only need the primary array don't
    have to destructure a 3-tuple. Pinning the return type (jax.Array, not
    tuple) guards against an inadvertent API change.
    """
    arr = jnp.array([4.0, 5.0])

    def fake_apply(t, inputs):
        return {"result": arr}

    monkeypatch.setattr(runner, "_apply_tesseract_with_deadline", fake_apply)
    out = runner.safe_apply(None, {}, "result")
    assert not isinstance(out, tuple), "safe_apply must unwrap the 3-tuple"
    assert jnp.array_equal(out, arr)


# ── image_tags_no_build ───────────────────────────────────────────────────────


def test_image_tags_no_build_uses_explicit_image_tag(tmp_path):
    """When SolverSpec.image_tag is set, image_tags_no_build returns it verbatim."""
    from mosaic.benchmarks.core.config import Problem, SolverSpec

    spec = SolverSpec(
        dir="my-solver",
        name="my_solver",
        backend="jax",
        family="spectral",
        scheme="dummy",
        color="#000000",
        linestyle=None,
        marker=None,
        ad_strategy="autodiff",
        differentiable=True,
        uses_gpu=False,
        internal_dtype="float32",
        description="",
        doc_url="",
        image_tag="explicit:tag42",
    )
    cfg = Problem.__new__(Problem)
    cfg.solvers = [spec]
    cfg.tesseract_dir = tmp_path

    tags = runner.image_tags_no_build(cfg)
    assert tags == {"my_solver": "explicit:tag42"}


def test_image_tags_no_build_reads_yaml_name_field(tmp_path):
    """When image_tag isn't set, the image name is read from tesseract_config.yaml."""
    from mosaic.benchmarks.core.config import Problem, SolverSpec

    # Build a fake tesseract dir layout: <tmp>/<solver_dir>/tesseract_config.yaml
    solver_dir = tmp_path / "fancy-solver"
    solver_dir.mkdir()
    (solver_dir / "tesseract_config.yaml").write_text(
        "name: fancy_image\nversion: '0.1.0'\n"
    )

    spec = SolverSpec(
        dir="fancy-solver",
        name="fancy",
        backend="jax",
        family="",
        scheme="",
        color="#000000",
        linestyle=None,
        marker=None,
        ad_strategy=None,
        differentiable=False,
        uses_gpu=False,
        internal_dtype="float32",
        description="",
        doc_url="",
        image_tag="",  # empty → trigger yaml-lookup branch
    )
    cfg = Problem.__new__(Problem)
    cfg.solvers = [spec]
    cfg.tesseract_dir = tmp_path

    tags = runner.image_tags_no_build(cfg)
    assert tags == {"fancy": "fancy_image:latest"}


def test_image_tags_no_build_falls_back_to_dir_name(tmp_path):
    """When no yaml and no image_tag, fall back to <dir>:latest."""
    from mosaic.benchmarks.core.config import Problem, SolverSpec

    spec = SolverSpec(
        dir="lonely-solver",
        name="lonely",
        backend="jax",
        family="",
        scheme="",
        color="#000000",
        linestyle=None,
        marker=None,
        ad_strategy=None,
        differentiable=False,
        uses_gpu=False,
        internal_dtype="float32",
        description="",
        doc_url="",
        image_tag="",
    )
    cfg = Problem.__new__(Problem)
    cfg.solvers = [spec]
    cfg.tesseract_dir = tmp_path  # no subdir created, so no yaml exists

    tags = runner.image_tags_no_build(cfg)
    assert tags == {"lonely": "lonely-solver:latest"}


# ── per_solver_loop ───────────────────────────────────────────────────────────


def _make_dummy_cfg(names: list[str]):
    """Tiny Problem stub with the minimal surface ``per_solver_loop`` reads."""
    from mosaic.benchmarks.core.config import Problem, SolverSpec

    cfg = Problem.__new__(Problem)
    cfg.solvers = [
        SolverSpec(
            dir=f"{n}-dir",
            name=n,
            backend="jax",
            family="",
            scheme="",
            color="#123456",
            linestyle=None,
            marker=None,
            ad_strategy=None,
            differentiable=False,
            uses_gpu=False,
            internal_dtype="float32",
            description="",
            doc_url="",
            image_tag=f"{n}:latest",
        )
        for n in names
    ]
    return cfg


def test_per_solver_loop_records_wall_times_for_each_solver(monkeypatch):
    """The helper returns ``{name: wall_seconds}`` populated by the wrapper.

    Patches ``run_with_gpu_pool`` to call the per-solver wrapper synchronously
    so we don't need Docker; verifies that every successful work_one invocation
    has a positive wall-time entry in the returned dict.
    """
    cfg = _make_dummy_cfg(["a", "b"])
    tags = {"a": "a:latest", "b": "b:latest"}
    seen: list[str] = []

    def _fake_pool(solver_names, _tags, fn, gpu_ids=None, on_error=None):
        for n in solver_names:
            fn(n, object())

    monkeypatch.setattr(runner, "run_with_gpu_pool", _fake_pool)

    def work(name, _t):
        seen.append(name)

    wall = runner.per_solver_loop(cfg, tags, ["a", "b"], work)
    assert seen == ["a", "b"]
    assert set(wall) == {"a", "b"}
    assert all(v >= 0 for v in wall.values())


def test_per_solver_loop_catch_swallows_exceptions(monkeypatch):
    """When ``catch=True``, a failing solver leaves its entry out of wall_times.

    The other (successful) solver's wall-time entry must still appear so the
    caller can report partial progress (the gradient harnesses rely on this).
    """
    cfg = _make_dummy_cfg(["good", "bad"])
    tags = {"good": "g:latest", "bad": "b:latest"}

    def _fake_pool(solver_names, _tags, fn, gpu_ids=None, on_error=None):
        for n in solver_names:
            fn(n, object())

    monkeypatch.setattr(runner, "run_with_gpu_pool", _fake_pool)

    def work(name, _t):
        if name == "bad":
            raise RuntimeError("boom")

    wall = runner.per_solver_loop(
        cfg, tags, ["good", "bad"], work, catch=True, catch_label="VJP failed"
    )
    assert "good" in wall
    assert "bad" not in wall


def test_per_solver_loop_catch_false_reraises(monkeypatch):
    """``catch=False`` (default) lets worker exceptions propagate to the caller."""
    cfg = _make_dummy_cfg(["x"])
    tags = {"x": "x:latest"}

    def _fake_pool(solver_names, _tags, fn, gpu_ids=None, on_error=None):
        for n in solver_names:
            fn(n, object())

    monkeypatch.setattr(runner, "run_with_gpu_pool", _fake_pool)

    def work(_name, _t):
        raise RuntimeError("fatal")

    with pytest.raises(RuntimeError, match="fatal"):
        runner.per_solver_loop(cfg, tags, ["x"], work)


# ── Problem.add_experiment: auto-sweep vs sweep_mode="none" ──────────────────


def _extract_runs_from_experiment(exp) -> list[dict]:
    """Pull the ``runs`` payload baked into the registered experiment's closure.

    ``Problem._register_one_experiment`` stashes the per-experiment kwargs
    (after :meth:`_normalize_run_shorthand`) into the registered closure as
    the ``_kw`` default argument; reading that back is the easiest way to
    assert what the kernel will see at ``ctx.run`` without spinning up a
    real solver.
    """
    import inspect

    sig = inspect.signature(exp.fn)
    kw = sig.parameters["_kw"].default
    return kw["runs"]


def test_add_experiment_sweep_mode_none_preserves_list_payload(tmp_path):
    """A ``sweep_mode="none"`` kernel sees its list-valued config field intact.

    Regression for the auto-sweep bug: ``_normalize_run_shorthand`` used to
    auto-detect any list nested one level inside a dict kwarg as a sweep
    axis, replace it with a scalar placeholder, and stash the original list
    under ``run["sweep"]``. The runner ignores ``run["sweep"]`` for
    ``sweep_mode="none"`` kernels, so the kernel would read the placeholder
    scalar at ``ctx.run["fd"]["eps_values"]`` and crash on
    ``for eps in 5.0``. The list must be passed through verbatim.
    """
    from mosaic.benchmarks.core.config import Problem
    from mosaic.benchmarks.core.experiment import kernel

    @kernel(sweep_mode="none")
    def fake_fd_check(t, ctx):  # pragma: no cover — not invoked
        return {"metrics": {}}

    problem = Problem.__new__(Problem)
    problem.experiments = {}
    problem.make_ic = {}
    problem.error_fn = lambda *a, **kw: 0.0
    problem.output_key = "result"
    problem.ic_key = "v0"
    problem.domain_extent = 1.0
    problem.resolution_key = "N"
    problem.reference = None

    problem.add_experiment(
        "gradient/fd_check",
        fake_fd_check,
        ic={"name": "multimode", "seed": 42},
        physics={"N": 16, "nu": 0.001, "dt": 0.05, "steps": 20},
        fd={"eps_values": [5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4], "n_dirs": 20},
    )

    runs = _extract_runs_from_experiment(problem.experiments["gradient/fd_check"])
    assert len(runs) == 1
    run = runs[0]
    # The kernel reads eps_values off ctx.run["fd"]["eps_values"] — must stay a list.
    assert run["fd"]["eps_values"] == [5e0, 1e0, 1e-1, 1e-2, 1e-3, 1e-4], (
        "sweep_mode='none' kernel had its list-valued fd.eps_values replaced "
        f"with a scalar placeholder: {run['fd']['eps_values']!r}"
    )
    # And no synthetic sweep block should be injected — the runner ignores it
    # for sweep_mode='none' anyway, but a stray entry implies the placeholder
    # substitution ran and would silently shadow the list.
    assert "sweep" not in run, (
        f"sweep_mode='none' kernel got an auto-injected sweep block: {run.get('sweep')!r}"
    )


def test_add_experiment_sweep_mode_default_still_auto_sweeps(tmp_path):
    """``sweep_mode="default"`` kernels keep the auto-sweep behaviour.

    Companion to the regression above: the fix must be scoped to
    ``sweep_mode="none"`` kernels — auto-sweep is the documented happy path
    for ``sweep_mode="default"`` / ``"limits"`` kernels (``param_sweep``,
    ``horizon_sweep_limits``).
    """
    from mosaic.benchmarks.core.config import Problem
    from mosaic.benchmarks.core.experiment import kernel

    @kernel(sweep_mode="default")
    def fake_param_sweep(t, ctx):  # pragma: no cover — not invoked
        return {"metrics": {}}

    problem = Problem.__new__(Problem)
    problem.experiments = {}
    problem.make_ic = {}
    problem.error_fn = lambda *a, **kw: 0.0
    problem.output_key = "result"
    problem.ic_key = "v0"
    problem.domain_extent = 1.0
    problem.resolution_key = "N"
    problem.reference = None

    problem.add_experiment(
        "gradient/param_sweep",
        fake_param_sweep,
        ic={"name": "multimode", "seed": 42},
        physics={"N": 16, "dt": 0.05, "steps": 200, "nu": [0.05, 0.01, 0.005, 0.001]},
    )

    runs = _extract_runs_from_experiment(problem.experiments["gradient/param_sweep"])
    assert len(runs) == 1
    run = runs[0]
    # Auto-sweep: physics.nu list collapses to scalar placeholder + sweep block.
    assert run["physics"]["nu"] == 0.05
    assert run["sweep"] == {"key": "nu", "values": [0.05, 0.01, 0.005, 0.001]}
