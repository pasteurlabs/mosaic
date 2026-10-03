# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check derivative supervision and checkpoint selection independently of XLB."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_PATH = (
    Path(__file__).parents[1] / "mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate"
)
sys.path.insert(0, str(_PATH))
try:
    _SPEC = importlib.util.spec_from_file_location(
        "surrogate_sobolev", _PATH / "sobolev.py"
    )
    _MODULE = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_MODULE)
finally:
    sys.path.pop(0)


def test_same_output_wrong_derivative_has_training_signal():
    initial = jnp.zeros((1, 2))
    cotangent = jnp.asarray([[1.0, -2.0]])

    def objective(weight):
        return _MODULE.losses(
            lambda x: weight * x, initial, initial, cotangent, 3 * cotangent, 1e-4
        )

    field, derivative = objective(1.0)
    assert float(field) == 0.0
    np.testing.assert_allclose(derivative, 4 / 9)
    gradient = jax.grad(lambda w: objective(w)[1])(1.0)
    np.testing.assert_allclose(gradient, -4 / 9)


def test_common_cotangent_matches_nonsymmetric_jacobian():
    matrix = jnp.asarray([[1.0, 3.0], [-2.0, 4.0]])
    initial = jnp.asarray([[0.4, -0.2], [0.1, 0.5]])
    cotangent = jnp.asarray([[2.0, -0.5], [-1.0, 3.0]])
    field, derivative = _MODULE.losses(
        lambda x: x @ matrix.T,
        initial,
        initial @ matrix.T,
        cotangent,
        cotangent @ matrix,
        1e-4,
    )
    assert float(field) == 0.0
    assert float(derivative) == 0.0


def test_fno_mixed_derivative_matches_finite_difference():
    fno = _MODULE.fno
    params = fno.init_params(width=2, modes=1, layers=1, seed=7)
    initial = jnp.asarray(
        np.random.default_rng(4).normal(scale=0.1, size=(1, 16, 16, 16, 3)),
        dtype=jnp.float32,
    )
    cotangent = jnp.ones_like(initial) / np.sqrt(initial.size)

    def objective(scale):
        current = {**params, "w_out": params["w_out"] * scale}

        def forward(x):
            return fno.rollout(
                current,
                x,
                steps=2,
                input_scale=jnp.asarray(0.2),
                correction_scale=jnp.ones(3) * 0.1,
                modes=1,
                layers=1,
            )[:, -1]

        return _MODULE.losses(forward, initial, initial, cotangent, cotangent, 1e-4)[1]

    grad = jax.jit(jax.grad(objective))(jnp.asarray(1.0))
    epsilon = 0.01
    fd = (objective(1.0 + epsilon) - objective(1.0 - epsilon)) / (2 * epsilon)
    assert abs(float(grad)) > 1e-6
    np.testing.assert_allclose(grad, fd, rtol=0.02, atol=1e-5)


@pytest.mark.parametrize("method", ["vjp", "secant"])
def test_training_exports_selected_checkpoint(tmp_path, monkeypatch, method):
    # A linear surrogate makes actual optimizer progress and export inspectable.
    fno = _MODULE.fno
    monkeypatch.setattr(fno, "init_params", lambda **kw: {"gain": jnp.asarray(1.0)})
    monkeypatch.setattr(
        fno, "rollout", lambda params, x, **kw: (params["gain"] * x)[:, None]
    )
    weights = tmp_path / "initial.npz"
    np.savez(
        weights,
        gain=np.float32(1),
        width=1,
        modes=1,
        layers=1,
        input_scale=np.float32(0.2),
        correction_scale=np.ones(3),
    )
    labels = tmp_path / "labels.npz"
    initial = np.ones((3, 2), dtype=np.float32)
    np.savez(
        labels,
        initial=initial,
        target=initial * 2,
        cotangent=initial,
        teacher_vjp=initial * 2,
        displacement=initial * 0.01,
        teacher_secant=initial * 2,
        split=[0, 0, 1],
    )
    # train imports the existing hashing helper from its sibling module.
    monkeypatch.syspath_prepend(str(_PATH))
    output = tmp_path / "selected.npz"
    _MODULE.train(
        argparse.Namespace(
            init_weights=weights,
            labels=labels,
            output=output,
            weight=0.1,
            method=method,
            linear_correction=False,
            lr=0.05,
            updates=3,
            batch_size=1,
            validation_interval=1,
            seed=7,
        )
    )
    metrics = json.loads(output.with_suffix(".metrics.json").read_text())
    assert metrics["best_step"] == 3
    assert metrics["history"][-1][method] < metrics["history"][0][method]
    with np.load(output) as data:
        assert float(data["gain"]) > 1
        assert float(data["input_scale"]) == float(np.float32(0.2))
    with np.load(weights) as data:
        assert float(data["gain"]) == 1


def test_secant_converges_to_directional_derivative():
    initial = jnp.asarray([[0.4, -0.7]])
    direction = jnp.asarray([[1.0, -1.0]])
    target = initial**3
    derivative = 3 * initial**2 * direction
    errors = []
    for epsilon in (0.1, 0.05):
        field, error = _MODULE.secant_losses(
            lambda x: x**3, initial, target, epsilon * direction, derivative, 1e-4
        )
        assert float(field) == 0
        errors.append(float(error))
    # Central-difference bias is O(epsilon**2); squared error is O(epsilon**4).
    np.testing.assert_allclose(errors[0] / errors[1], 16, rtol=0.01)


def test_label_generation_keeps_split_and_secant_scale(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_PATH))
    dataset = tmp_path / "trajectories.npy"
    values = np.ones((4, 1, 16, 16, 16, 3), dtype=np.float32) * 0.2
    np.save(dataset, values)
    np.savez(dataset.with_suffix(".split.npz"), split=[0, 0, 1, 2])
    teacher = tmp_path / "teacher.py"
    teacher.write_text("def xlb_fwd(value, **kwargs):\n    return 2 * value, None\n")
    output = tmp_path / "labels.npz"
    previous_x64 = jax.config.jax_enable_x64
    try:
        _MODULE.generate(
            argparse.Namespace(
                teacher_api=teacher,
                dataset=dataset,
                output=output,
                train_samples=2,
                validation_samples=1,
                seed=11,
                secant_step=0.01,
            )
        )
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
    with np.load(output) as labels:
        assert set(labels["indices"]) == {0, 1, 2}
        np.testing.assert_array_equal(labels["split"], [0, 0, 1])
        np.testing.assert_allclose(labels["target"], labels["initial"] * 2)
        np.testing.assert_allclose(labels["teacher_vjp"], labels["cotangent"] * 2)
        delta = labels["displacement"]
        epsilon = np.sqrt(np.mean(delta**2, axis=(1, 2, 3, 4), keepdims=True))
        np.testing.assert_allclose(
            labels["teacher_secant"], 2 * delta / epsilon, rtol=1e-5, atol=1e-6
        )


def test_linear_correction_preserves_zero_and_learns_zero_state_derivative():
    fno = _MODULE.fno
    params = fno.init_params(width=2, modes=1, layers=1, seed=3)
    zero = jnp.zeros((1, 16, 16, 16, 3))
    direction = jnp.ones_like(zero)

    def response(gain):
        current = {
            **params,
            "w_linear": jnp.broadcast_to(jnp.eye(3) * gain, (193, 3, 3)),
        }

        def forward(x):
            return fno.one_step(
                current,
                x,
                input_scale=jnp.asarray(0.2),
                correction_scale=jnp.ones(3) * 0.1,
                modes=1,
                layers=1,
            )

        value, tangent = jax.jvp(forward, (zero,), (direction,))
        return jnp.sum(value**2), jnp.mean(tangent)

    value, tangent = response(0.1)
    assert float(value) == 0.0
    np.testing.assert_allclose(tangent - response(0.0)[1], 0.1, atol=1e-6)
    np.testing.assert_allclose(
        jax.grad(lambda gain: response(gain)[1])(0.1), 1.0, atol=1e-6
    )


def test_original_trajectory_loss_retains_time_spectral_and_terminal_terms(monkeypatch):
    monkeypatch.syspath_prepend(str(_PATH))
    from train import trajectory_loss

    predicted = jnp.ones((1, 2, 16, 16, 16, 3))
    predicted = predicted.at[:, 1].set(2.0)
    loss, components = trajectory_loss(
        predicted, jnp.zeros_like(predicted), jnp.ones(3)
    )
    np.testing.assert_allclose(components, [3.0, 2.5 * 16 / 9, 4.0], rtol=1e-6)
    np.testing.assert_allclose(loss, 3 + 0.02 * 2.5 * 16 / 9 + 0.25 * 4, rtol=1e-6)


def test_replay_control_uses_trajectory_targets_not_terminal_label_targets(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(_PATH))
    fno = _MODULE.fno
    monkeypatch.setattr(fno, "N", 2)
    monkeypatch.setattr(fno, "ROLLOUT_STEPS", 1)
    monkeypatch.setattr(fno, "init_params", lambda **kw: {"gain": jnp.asarray(1.0)})
    monkeypatch.setattr(
        fno, "rollout", lambda params, x, **kw: (params["gain"] * x)[:, None]
    )
    weights = tmp_path / "initial.npz"
    np.savez(
        weights,
        gain=np.float32(1),
        width=1,
        modes=1,
        layers=1,
        input_scale=np.float32(0.2),
        correction_scale=np.ones(3),
    )
    initial = np.ones((3, 2, 2, 2, 3), np.float32)
    replay = tmp_path / "replay.npy"
    np.save(replay, np.stack([initial, 2 * initial], axis=1))
    np.savez(replay.with_suffix(".split.npz"), split=[0, 0, 1])
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({"output_scale": [1, 1, 1]}))
    gains = []
    for label_target in [2, 9]:
        labels = tmp_path / f"labels-{label_target}.npz"
        np.savez(
            labels,
            initial=initial,
            target=initial * label_target,
            cotangent=initial,
            teacher_vjp=initial * 2,
            split=[0, 0, 1],
        )
        output = tmp_path / f"selected-{label_target}.npz"
        _MODULE.train(
            argparse.Namespace(
                init_weights=weights,
                labels=labels,
                output=output,
                weight=0.0,
                lr=0.05,
                updates=3,
                batch_size=1,
                validation_interval=1,
                seed=7,
                linear_correction=False,
                method="vjp",
                replay_dataset=replay,
                replay_normalization=normalization,
                replay_validation_samples=1,
            )
        )
        metrics = json.loads(output.with_suffix(".metrics.json").read_text())
        assert (
            metrics["history"][-1]["replay_loss"] < metrics["history"][0]["replay_loss"]
        )
        assert metrics["replay_validation_indices"] == [2]
        with np.load(output) as d:
            gains.append(float(d["gain"]))
    np.testing.assert_array_equal(gains[0], gains[1])
