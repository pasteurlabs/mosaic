# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gradient/chain_fd_check`` on closed-form steps in place of a solver."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.benchmarks.problems.navier_stokes_grid import chain as ch
from mosaic.benchmarks.problems.shared.diagnostics import project_divergence_free

N = 16


def _smooth_state() -> np.ndarray:
    rng = np.random.default_rng(3)
    v = project_divergence_free(rng.standard_normal((N, N, 1, 2)), ch.DOMAIN_EXTENT)
    return v.astype(np.float32)


def _step(v: jax.Array) -> jax.Array:
    """A smooth nonlinear map that keeps the state bounded."""
    return 0.9 * v + 0.1 * jnp.tanh(jnp.roll(v, 1, axis=0) * v)


@jax.custom_vjp
def _truncated_step(v: jax.Array) -> jax.Array:
    return _step(v)


def _truncated_fwd(v: jax.Array) -> tuple:
    return _step(v), v


def _truncated_bwd(v: jax.Array, ct: jax.Array) -> tuple:
    # Like an adjoint solve that stops at an absolute tolerance: a cotangent
    # whose RMS is below it is returned unchanged rather than propagated.
    rms = jnp.sqrt(jnp.mean(ct**2))
    _, vjp = jax.vjp(_step, v)
    return (jnp.where(rms < 1e-3, ct, vjp(ct)[0]),)


_truncated_step.defvjp(_truncated_fwd, _truncated_bwd)


def _run(monkeypatch, step) -> dict:
    monkeypatch.setattr(
        ch, "load_ground_truth", lambda domain: {"initial_5": _smooth_state()}
    )
    monkeypatch.setattr(
        ch, "apply_tesseract", lambda t, inputs: {"result": step(inputs["v0"])}
    )
    ctx = SimpleNamespace(
        name="fake",
        cfg=SimpleNamespace(name="ns-grid"),
        phys={"N": N, "intervals": 8, "offset": 0.01, "nu": 0.01},
        run={"fd": {"eps_values": [1e-1, 1e-2], "n_dirs": 4}},
        make_inputs=lambda name, v0, **physics: {"v0": v0, **physics},
        output_key="result",
    )
    return ch.chain_fd_check(None, ctx)["metrics"]


def _best(metrics: dict) -> float:
    return min(float(np.median(e["rel_error"])) for e in metrics["eps_sweep"].values())


def test_chain_loss_is_the_mean_relative_squared_distance():
    t = [np.ones((2, 2)), 2 * np.ones((2, 2))]
    s = [np.zeros((2, 2)), np.ones((2, 2))]
    assert ch.chain_loss(s, t, np) == pytest.approx((1.0 + 0.25) / 2)


def test_directions_are_unit_and_divergence_free():
    d = ch.divergence_free_direction((N, N, 1, 2), np.random.default_rng(0))
    assert np.linalg.norm(d) == pytest.approx(1.0, rel=1e-5)
    assert np.allclose(project_divergence_free(d, ch.DOMAIN_EXTENT), d, atol=1e-6)


def test_a_correct_vjp_matches_the_finite_differences(monkeypatch):
    metrics = _run(monkeypatch, _step)
    assert metrics["loss"] > 0
    assert _best(metrics) < 1e-3
    assert max(e["cosine"] for e in metrics["eps_sweep"].values()) > 0.999


def test_a_vjp_that_drops_small_cotangents_fails(monkeypatch):
    """The loss's cotangents are small, so an absolute cut-off leaves them unpropagated."""
    assert _best(_run(monkeypatch, _truncated_step)) > 0.05
