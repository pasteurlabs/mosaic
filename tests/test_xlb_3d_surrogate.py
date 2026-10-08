# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fixed-task XLB 3D full-field surrogate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("tesseract_core.runtime")
pytest.importorskip("mosaic_shared")

_API_PATH = (
    Path(__file__).parents[1]
    / "mosaic"
    / "tesseracts"
    / "navier-stokes-grid"
    / "xlb-3d-surrogate"
    / "legacy_api.py"
)
_DATA_PATH = _API_PATH.with_name("training_data.py")
_SPEC = importlib.util.spec_from_file_location("xlb_3d_surrogate_api", _API_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_API = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _API
_SPEC.loader.exec_module(_API)
_DATA_SPEC = importlib.util.spec_from_file_location(
    "xlb_3d_surrogate_training_data",
    _DATA_PATH,
)
assert _DATA_SPEC is not None and _DATA_SPEC.loader is not None
_DATA = importlib.util.module_from_spec(_DATA_SPEC)
_DATA_SPEC.loader.exec_module(_DATA)


def _weights(width: int = 4, modes: int = 2):
    rng = np.random.default_rng(2026)
    weights = {
        "input_scale": jnp.asarray(0.2, dtype=jnp.float32),
        "correction_scale": jnp.full((3,), 0.1, dtype=jnp.float32),
        "w_lift": jnp.asarray(
            rng.normal(scale=0.05, size=(3, width)),
            dtype=jnp.float32,
        ),
        "w_out": jnp.asarray(
            rng.normal(scale=0.05, size=(width, 3)),
            dtype=jnp.float32,
        ),
        "w_local_0": jnp.eye(width, dtype=jnp.float32),
    }
    shape = (modes, modes, modes, width, width)
    for quadrant in range(4):
        weights[f"spec_0_{quadrant}_real"] = jnp.asarray(
            rng.normal(scale=0.01, size=shape),
            dtype=jnp.float32,
        )
        weights[f"spec_0_{quadrant}_imag"] = jnp.asarray(
            rng.normal(scale=0.01, size=shape),
            dtype=jnp.float32,
        )
    return weights


def test_full_field_shape_and_zero_state(monkeypatch):
    monkeypatch.setattr(_API, "_WIDTH", 4)
    monkeypatch.setattr(_API, "_MODES", 2)
    monkeypatch.setattr(_API, "_LAYERS", 1)
    monkeypatch.setattr(_API, "_ROLLOUT_STEPS", 3)
    initial = jnp.zeros((_API._N, _API._N, _API._N, 3), dtype=jnp.float32)
    result = _API._surrogate_forward(initial, _weights())
    assert result.shape == initial.shape
    assert np.all(np.isfinite(np.asarray(result)))
    assert np.allclose(np.asarray(result), 0.0, atol=1e-7)


def test_full_field_vjp_is_finite(monkeypatch):
    monkeypatch.setattr(_API, "_WIDTH", 4)
    monkeypatch.setattr(_API, "_MODES", 2)
    monkeypatch.setattr(_API, "_LAYERS", 1)
    monkeypatch.setattr(_API, "_ROLLOUT_STEPS", 3)
    weights = _weights()
    initial = jnp.zeros((_API._N, _API._N, _API._N, 3), dtype=jnp.float32)
    initial = initial.at[1, 2, 3, 0].set(0.1)

    def field_sum(value):
        return jnp.sum(_API._surrogate_forward(value, weights) ** 2)

    gradient = jax.grad(field_sum)(initial)
    assert gradient.shape == initial.shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_forward_reuses_one_macro_step_operator(monkeypatch):
    monkeypatch.setattr(_API, "_WIDTH", 4)
    monkeypatch.setattr(_API, "_MODES", 2)
    monkeypatch.setattr(_API, "_LAYERS", 1)
    monkeypatch.setattr(_API, "_ROLLOUT_STEPS", 3)
    weights = _weights()
    initial = jnp.zeros((_API._N, _API._N, _API._N, 3), dtype=jnp.float32)
    initial = initial.at[1, 2, 3, 0].set(0.1)
    expected = initial[None]
    for _ in range(3):
        expected = _API._one_step(expected, weights)
    result = _API._surrogate_forward(initial, weights)
    assert np.allclose(np.asarray(result), np.asarray(expected[0]), atol=1e-7)


def test_training_distribution_is_reproducible_and_full_field():
    first = _DATA.make_inputs(10, 123)
    second = _DATA.make_inputs(10, 123)

    fields, amplitudes, families = first
    assert fields.shape == (10, _API._N, _API._N, _API._N, 3)
    assert amplitudes.shape == (10,)
    assert families.shape == (10,)
    assert np.all(np.isfinite(fields))
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)


def test_default_inputs_satisfy_fixed_task_contract():
    _API._validate_contract(_API.InputSchema())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v0", np.zeros((8, 8, 8, 3), np.float32)),
        ("v0", np.full((16, 16, 16, 3), np.nan, np.float32)),
        ("viscosity", np.array([0.05], np.float32)),
        ("dt", np.array([0.01], np.float32)),
        ("steps", 5),
        ("domain_extent", 1.0),
        ("inflow_profile", np.ones(16, np.float32)),
    ],
)
def test_out_of_contract_inputs_fail_before_inference(field, value):
    inputs = _API.InputSchema(**{field: value})
    with pytest.raises(ValueError, match="XLB 3D surrogate"):
        _API.apply(inputs)


def test_packaged_checkpoint_through_tesseract_jax():
    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract

    from mosaic.benchmarks.problems import get_config

    cfg = get_config("ns-3d-grid")
    initial = cfg.make_ic["rand_div_free"](16, seed=0)
    inputs = cfg.make_inputs("XLB 3D surrogate", initial, nu=0.01, dt=0.02, steps=100)
    with Tesseract.from_tesseract_api(str(_API_PATH)) as t:
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


def test_packaged_checkpoint_recovery_harness(tmp_path, monkeypatch):
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
    cfg.experiments[key].fn(cfg, {"XLB 3D surrogate": f"inprocess:{_API_PATH}"})
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


@pytest.mark.parametrize("valid_shape", [True, False])
def test_optional_linear_branch_checkpoint_is_loaded(
    tmp_path, monkeypatch, valid_shape
):
    monkeypatch.setattr(_API, "_WIDTH", 4)
    monkeypatch.setattr(_API, "_MODES", 2)
    monkeypatch.setattr(_API, "_LAYERS", 1)
    monkeypatch.setattr(_API, "_WEIGHTS", None)
    path = tmp_path / "weights.npz"
    monkeypatch.setattr(_API, "_weights_path", lambda: path)
    weights = _weights()
    weights["w_linear"] = np.zeros((193 if valid_shape else 192, 3, 3), np.float32)
    np.savez(
        path,
        **weights,
        width=4,
        modes=2,
        layers=1,
        stride=_API._STRIDE,
        rollout_steps=_API._ROLLOUT_STEPS,
        autoregressive=1,
    )
    if valid_shape:
        assert _API._load_weights()["w_linear"].shape == (193, 3, 3)
    else:
        with pytest.raises(RuntimeError, match="invalid linear correction shape"):
            _API._load_weights()
