# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public contracts and finite-difference checks for the generalized 3D operator."""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = (
    Path(__file__).parents[1] / "mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate"
)
spec = importlib.util.spec_from_file_location(
    "conditioned_operator_api", ROOT / "operator_api.py"
)
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)


@pytest.fixture(params=[False, True])
def weights(monkeypatch, request):
    params = api.model.init_params(width=4, modes=2, layers=1, seed=17)
    settings = {
        "layers": 1,
        "eval_stride": 5,
        "project_output": False,
        "conserve_energy": request.param,
    }
    monkeypatch.setattr(api, "_WEIGHTS", (params, settings))


@pytest.mark.parametrize("n", [8, 12, 16, 20, 32])
def test_shared_weights_and_arbitrary_horizon(n, weights):
    x = np.arange(n) * (2 * np.pi / n)
    u = np.zeros((n, n, n, 3), np.float32)
    u[..., 1] = 0.2 * np.sin(x[:, None, None])
    inputs = api.InputSchema(v0=u, steps=7)
    output = api.apply(inputs)
    assert output["result"].shape == u.shape
    assert np.isfinite(output["result"]).all()
    assert not np.array_equal(output["result"], u)
    np.testing.assert_array_equal(output["drag"], np.zeros(1))
    zero_steps = api.apply(api.InputSchema(v0=u, steps=0))["result"]
    np.testing.assert_array_equal(zero_steps, u)


@pytest.mark.parametrize("key", ["v0", "viscosity", "dt"])
def test_vjp_matches_forward_finite_difference(key, weights):
    rng = np.random.default_rng(44)
    u = rng.normal(scale=0.1, size=(8, 8, 8, 3)).astype(np.float32)
    values = {
        "v0": u,
        "viscosity": np.array([0.05], np.float32),
        "dt": np.array([0.01], np.float32),
        "steps": 7,
    }
    cotangent = rng.normal(size=u.shape).astype(np.float32) / u.size
    direction = rng.normal(size=np.asarray(values[key]).shape).astype(np.float32)
    direction /= np.linalg.norm(direction)
    gradient = api.vector_jacobian_product(
        api.InputSchema(**values), {key}, {"result"}, {"result": cotangent}
    )
    assert set(gradient) == {key}
    assert gradient[key].shape == np.asarray(values[key]).shape
    eps = 1e-4

    def objective(sign):
        output = api.apply(
            api.InputSchema(**(values | {key: values[key] + sign * eps * direction}))
        )
        return float(jnp.vdot(output["result"], cotangent))

    finite_difference = (objective(1) - objective(-1)) / (2 * eps)
    np.testing.assert_allclose(
        float(jnp.vdot(gradient[key], direction)),
        finite_difference,
        rtol=0.025,
        atol=2e-6,
    )


def test_drag_has_zero_gradient_and_empty_inputs_stay_empty(weights):
    inputs = api.InputSchema(v0=np.ones((8, 8, 8, 3), np.float32))
    gradient = api.vector_jacobian_product(
        inputs, {"v0", "dt"}, {"drag"}, {"drag": [1.0]}
    )
    assert all(np.count_nonzero(v) == 0 for v in gradient.values())
    assert api.vector_jacobian_product(inputs, set(), set(), {}) == {}


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((8, 8, 1, 2), np.float32),
        np.zeros((8, 16, 8, 3), np.float32),
        np.full((8, 8, 8, 3), np.nan, np.float32),
    ],
)
def test_unsupported_fields_fail_before_weight_loading(value):
    with pytest.raises(ValueError, match="operator"):
        api.apply(api.InputSchema(v0=value))


@pytest.mark.parametrize(
    "key,value",
    [
        ("viscosity", [0.0]),
        ("dt", [-0.1]),
        ("domain_extent", float("inf")),
        ("steps", -1),
    ],
)
def test_invalid_physics_fails_before_weight_loading(key, value):
    with pytest.raises(ValueError, match="operator"):
        api.apply(api.InputSchema(**{key: value}))


def test_generalized_packaged_checkpoint_through_tesseract_jax():
    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract

    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-3d-grid")
    initial = cfg.make_ic["rand_div_free"](16, seed=0)
    inputs = cfg.make_inputs("XLB 3D surrogate", initial, nu=0.01, dt=0.02, steps=100)
    with Tesseract.from_tesseract_api(str(ROOT / "tesseract_api.py")) as t:
        default_output = t.apply({})
        np.testing.assert_array_equal(
            default_output["result"], np.zeros((16, 16, 16, 3), np.float32)
        )
        output = t.apply(inputs)
        target = jnp.asarray(output["result"])
        assert target.shape == initial.shape
        assert target.dtype == jnp.float32
        assert np.isfinite(target).all()
        assert np.linalg.norm(target) > 0
        np.testing.assert_array_equal(output["drag"], np.zeros(1))

        def loss(velocity):
            result = apply_tesseract(t, {**inputs, "v0": velocity})["result"]
            return jnp.mean((result - target) ** 2)

        # Exercise the actual cold-start recovery objective and the off-zero
        # learned map; checking only the zero fixed point misses the FNO path.
        direction = initial / jnp.linalg.norm(initial)
        for point in (jnp.zeros_like(initial), 0.5 * initial):
            value, gradient = jax.value_and_grad(loss)(point)
            assert np.isfinite(gradient).all()
            assert gradient.shape == initial.shape
            derivative = float(jnp.vdot(gradient, direction))
            assert derivative < 0
            eps = 0.01
            finite_difference = float(
                (loss(point + eps * direction) - loss(point - eps * direction))
                / (2 * eps)
            )
            np.testing.assert_allclose(derivative, finite_difference, rtol=0.02)
            assert float(loss(point - gradient)) < float(value)


def test_generalized_packaged_checkpoint_recovery_harness(tmp_path, monkeypatch):
    import copy
    import json

    from mosaic.benchmarks.problems import get_config
    from mosaic.benchmarks.problems.navier_stokes_3d_grid.optimization import recovery

    monkeypatch.setenv("MOSAIC_RESULTS_DIR", str(tmp_path))
    cfg = copy.deepcopy(get_config("ns-3d-grid"))
    optimizer = "bfgs_proj"
    exp_name = "recovery_constant_ic_bfgs_proj"
    key = f"optimization/{exp_name}"
    # Keep the registered task physics and optimizer; shorten only the
    # optimization budget and seed count for this CPU integration test.
    cfg.add_experiment(
        key,
        recovery,
        optimizer=optimizer,
        _exp_key=exp_name,
        ic={"name": "rand_div_free", "seed": 0},
        physics={"N": 16, "nu": 0.01, "dt": 0.02, "steps": [100]},
        optim={
            "ic_init_type": "zeros",
            "max_iters": 3,
            "ic_seeds": [0],
            "snap_interval": 1,
            "record_diagnostics": True,
        },
    )
    cfg.experiments[key].fn(
        cfg, {"XLB 3D surrogate": f"inprocess:{(ROOT / 'tesseract_api.py')}"}
    )
    path = tmp_path / cfg.name / key / "result.json"
    saved = json.loads(path.read_text())
    assert len(saved["results"]) == 1
    row = saved["results"][0]
    assert row["solver"] == "XLB 3D surrogate"
    assert row["sweep_value"] == "100"
    entry = row["metrics"]
    assert entry["n_trials"] == 1
    assert len(entry["errors"]) == 3
    assert np.isfinite(entry["errors"]).all()
    assert entry["errors"][-1] < entry["errors"][0]
    assert entry["final_ic_error"] < entry["ic_error_init"]
    assert list(path.parent.glob("*.npz"))


def test_all_periodic_3d_benchmarks_admit_surrogate_with_xlb_scaling():
    from mosaic.benchmarks.core.utils import active_solvers
    from mosaic.benchmarks.problems import get_config

    for problem in ("ns-grid", "ns-3d-grid"):
        cfg = get_config(problem)
        admitted = {
            key
            for key in cfg.experiments
            if not key.startswith("ics/")
            and "XLB 3D surrogate" in active_solvers(cfg, *key.split("/", 1))
        }
        expected = (
            {key for key in cfg.experiments if not key.startswith("ics/")}
            if problem == "ns-3d-grid"
            else set()
        )
        assert admitted == expected
    cfg = get_config("ns-3d-grid")
    initial = np.zeros((64, 64, 64, 3), np.float32)
    settings = {"nu": 0.01, "dt": 0.01, "steps": 50, "lbm_N_base": 16}
    surrogate = cfg.make_inputs("XLB 3D surrogate", initial, **settings)
    teacher_name = next(s.name for s in cfg.solvers if s.key == "xlb")
    teacher = cfg.make_inputs(teacher_name, initial, **settings)
    assert surrogate["steps"] == teacher["steps"] == 200
    np.testing.assert_array_equal(surrogate["dt"], teacher["dt"])
    np.testing.assert_allclose(surrogate["dt"], [0.0025])


def test_host_x64_setting_does_not_change_operator(weights):
    inputs = api.InputSchema(
        v0=np.random.default_rng(92).normal(0, 0.1, (8, 8, 8, 3)).astype(np.float32),
        steps=7,
    )
    results, gradients = [], []
    for enabled in (False, True):
        with jax.enable_x64(enabled):
            results.append(np.asarray(api.apply(inputs)["result"]))
            gradients.append(
                np.asarray(
                    api.vector_jacobian_product(
                        inputs, {"v0"}, {"result"}, {"result": inputs.v0}
                    )["v0"]
                )
            )
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(gradients[0], gradients[1])
