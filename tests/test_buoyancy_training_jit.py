# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiled updates retain schedules, checkpoint choices and failure checks."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mosaic.benchmarks.problems.buoyancy_grid import training


def test_compiled_update_matches_eager_adam_across_cases_and_late_windows():
    references = jnp.arange(2 * 11, dtype=jnp.float32).reshape(2, 11) / 20
    cases, starts = training.balanced_windows(23, 8, 2, 9)
    case_array, start_array = jnp.asarray(cases), jnp.asarray(starts)
    traces = []

    def loss(params, iteration):
        traces.append(None)
        targets = jax.lax.dynamic_slice_in_dim(
            references[case_array[iteration]], start_array[iteration], 4
        )
        return jnp.mean((params["w"] * jnp.arange(4) + params["b"] - targets) ** 2)

    initial = {
        "w": jnp.asarray(0.1, dtype=jnp.float32),
        "b": jnp.asarray(0.0, dtype=jnp.float32),
    }
    validate = lambda p: (p["w"] - 0.03) ** 2 + (p["b"] - 0.1) ** 2
    selected, record = training.fit(initial, loss, validate, 23, 0.003, 5)
    assert len(traces) == 1  # Dynamic iteration/case/start must not retrace.
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.003))
    params, state = initial, optimizer.init(initial)
    best, best_score, best_update = initial, float(validate(initial)), 0
    history = []
    for iteration in range(23):
        targets = references[
            cases[iteration], starts[iteration] : starts[iteration] + 4
        ]
        value, grads = jax.value_and_grad(
            lambda p, targets=targets: jnp.mean(
                (p["w"] * jnp.arange(4) + p["b"] - targets) ** 2
            )
        )(params)
        delta, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, delta)
        history.append(float(value))
        if iteration + 1 in training.checkpoint_updates(23, 5):
            score = float(validate(params))
            if score < best_score:
                best, best_score, best_update = params, score, iteration + 1
    np.testing.assert_allclose(record["training_loss"], history, rtol=3e-6, atol=1e-7)
    for actual, expected in zip(
        jax.tree.leaves(selected), jax.tree.leaves(best), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7)
    assert record["selected_update"] == best_update
    assert record["completed_updates"] == 23
    assert [item["update"] for item in record["validation_checkpoints"]] == [
        0,
        5,
        10,
        15,
        20,
        23,
    ]


@pytest.mark.parametrize("loss", [lambda p, i: p * jnp.nan, lambda p, i: jnp.sqrt(p)])
def test_compiled_update_rejects_nonfinite_loss_or_gradient_before_validation(loss):
    validated = []

    def validate(p):
        validated.append(float(p))
        return 0.0

    with pytest.raises(FloatingPointError, match="nonfinite training at update 0"):
        training.fit(jnp.asarray(0.0, dtype=jnp.float32), loss, validate, 2, 0.003, 1)
    assert validated == [0.0]


def test_compiled_update_rejects_parameter_overflow_before_accepting_checkpoint():
    validated = []

    def validate(p):
        validated.append(float(p))
        return 0.0

    with pytest.raises(
        FloatingPointError, match="nonfinite model parameters at update 1"
    ):
        training.fit(
            jnp.asarray(3e38, dtype=jnp.float32), lambda p, i: -p, validate, 2, 3e38, 1
        )
    assert len(validated) == 1
