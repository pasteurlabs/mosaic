"""Check optional RPC compilation without changing recurrent solver semantics."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

module = importlib.import_module(
    "mosaic.benchmarks.problems.navier_stokes_grid.solver_in_loop"
)


class _Client:
    """Each instance represents an independent, statically identified service."""


def _context(enabled=None):
    run = {} if enabled is None else {"execution": {"jit_solver_rpc": enabled}}
    return SimpleNamespace(
        name="dummy",
        phys={"dt": 0.125},
        output_key="result",
        run=run,
        make_inputs=lambda _name, value, **physics: {
            "v0": value,
            "steps": physics["steps"],
            "dt": jnp.asarray([physics["dt"]], dtype=jnp.float32),
        },
    )


def _fake_service(monkeypatch):
    traces = []

    def apply(client, inputs):
        traces.append((client, inputs["v0"].shape, inputs["steps"], "state" in inputs))
        velocity = inputs["v0"]
        native = inputs.get("state", jnp.zeros_like(velocity))
        return {
            "result": velocity * (inputs["steps"] + 1) + 0.25 * native + inputs["dt"],
            "state": 2 * velocity + native + inputs["steps"] * inputs["dt"],
        }

    monkeypatch.setattr(module, "apply_tesseract", apply)
    monkeypatch.setattr(module, "_supports_native_state", lambda _t: True)
    return traces


@pytest.mark.parametrize("enabled", [None, False])
def test_rpc_compilation_defaults_off(monkeypatch, enabled):
    traces = _fake_service(monkeypatch)
    client, ctx = _Client(), _context(enabled)
    for value in [1.0, 2.0]:
        output, _ = module._solver_advance(
            client, ctx, jnp.full((2, 2, 1, 2), value), frame_steps=1
        )
        np.testing.assert_array_equal(output, 2 * value + 0.125)
    assert len(traces) == 2


def test_rpc_jit_specializes_shapes_steps_and_checkpoint_presence(monkeypatch):
    traces = _fake_service(monkeypatch)
    client, ctx = _Client(), _context(True)
    # Repeated signatures must execute with new dynamic state values while
    # avoiding repeated abstract evaluation. Shape/static changes must retrace.
    cases = [(2, 1, False), (2, 1, True), (3, 1, True), (3, 4, True)]
    for n, steps, continuation in cases:
        for scale in [1.0, 2.0]:
            velocity = jnp.full((n, n, 1, 2), scale)
            native = jnp.full_like(velocity, 3 * scale) if continuation else None
            output, checkpoint = module._solver_advance(
                client, ctx, velocity, frame_steps=steps, native_state=native
            )
            expected_native = 3 * scale if continuation else 0.0
            np.testing.assert_array_equal(
                output, scale * (steps + 1) + 0.25 * expected_native + 0.125
            )
            np.testing.assert_array_equal(
                checkpoint, 2 * scale + expected_native + steps * 0.125
            )
    assert len(traces) == len(cases)
    # A distinct service must not inherit the old service's compiled callback.
    module._solver_advance(_Client(), ctx, jnp.ones((2, 2, 1, 2)), frame_steps=1)
    assert len(traces) == len(cases) + 1


def test_rpc_jit_preserves_both_recurrent_gradient_paths(monkeypatch):
    _fake_service(monkeypatch)
    velocity = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 1, 2) / 8
    native = jnp.full_like(velocity, 0.5)
    results = []
    for enabled in [False, True]:
        client, ctx = _Client(), _context(enabled)

        def loss(value, checkpoint, client=client, ctx=ctx):
            value, checkpoint = module._solver_advance(
                client, ctx, value, frame_steps=1, native_state=checkpoint
            )
            value, checkpoint = module._solver_advance(
                client, ctx, value, frame_steps=3, native_state=checkpoint
            )
            return jnp.sum(value + 0.5 * checkpoint)

        results.append(jax.value_and_grad(loss, argnums=(0, 1))(velocity, native))
    for eager, compiled in zip(
        jax.tree_util.tree_leaves(results[0]),
        jax.tree_util.tree_leaves(results[1]),
        strict=True,
    ):
        np.testing.assert_array_equal(compiled, eager)
    # Analytic derivatives include both velocity and checkpoint recurrence.
    np.testing.assert_array_equal(results[1][1][0], jnp.full_like(velocity, 11.5))
    np.testing.assert_array_equal(results[1][1][1], jnp.full_like(native, 2.0))
