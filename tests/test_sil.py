# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Kolmogorov solver-in-the-loop benchmark on closed-form transitions.

No Tesseract runs here: steps are small JAX maps standing in for a solver, so
the corrector, the training arms, the trainer, the gates and the status check
can be checked exactly.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.benchmarks.core.status_checks import OptimizationSummary
from mosaic.benchmarks.problems.navier_stokes_grid import kolmogorov as ko
from mosaic.benchmarks.problems.navier_stokes_grid import sil, sil_status
from mosaic.benchmarks.problems.navier_stokes_grid.corrector import (
    apply_corrector,
    init_corrector,
)

N = 8


def _field(n: int = N, seed: int = 0) -> np.ndarray:
    """A band-limited divergence-free field at the cell centres."""
    rng = np.random.default_rng(seed)
    x = ko.cell_centres(n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    v = np.zeros((n, n, 1, 2))
    for kx, ky in ((1, 0), (0, 1), (1, 1), (2, 1)):
        c = rng.normal() * np.cos(kx * xx + ky * yy + rng.uniform(0, 2 * np.pi))
        v[..., 0, 0] += ky * c
        v[..., 0, 1] -= kx * c
    return v.astype(np.float32)


def _random_params(seed: int = 3) -> tuple:
    params = init_corrector(0)
    readout = 0.5 * jax.random.normal(jax.random.PRNGKey(seed), params[-1]["w"].shape)
    return (*params[:-1], {"w": readout, "b": params[-1]["b"]})


def decaying_step(rate: float = 0.9) -> sil.Step:
    """A linear 'solver' that carries its last input as its native state."""

    def step(v, native):
        out = rate * v if native is None else rate * v + 0.05 * native
        return out, v

    return step


def test_untrained_corrector_is_the_identity():
    v = jnp.asarray(_field())
    assert np.array_equal(np.asarray(apply_corrector(init_corrector(0), v)), v)


def test_correction_is_divergence_free_with_zero_mean():
    v = jnp.asarray(_field())
    delta = np.asarray(apply_corrector(_random_params(), v) - v)[:, :, 0, :]
    assert np.abs(delta).max() > 1e-4
    k = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    hat = np.fft.fft2(delta, axes=(0, 1))
    divergence = kx * hat[..., 0] + ky * hat[..., 1]
    assert np.abs(divergence).max() < 1e-5 * np.abs(hat).max()
    np.testing.assert_allclose(delta.mean(axis=(0, 1)), 0, atol=1e-7)


def test_stopped_gradient_cuts_the_solver_and_its_native_state():
    """The stopped arm's gradient is that of a rollout with frozen solver outputs."""
    step = decaying_step()
    targets = jnp.asarray(np.stack([_field(seed=s) for s in range(4)]))
    params = _random_params()
    stopped = jax.grad(sil.window_loss)(
        params, targets, step, differentiate_solver=False
    )
    full = jax.grad(sil.window_loss)(params, targets, step, differentiate_solver=True)

    # The same forward pass with the solver outputs as constants.
    outputs, state, native = [], targets[0], None
    for _ in targets[1:]:
        provisional, native = step(state, native)
        outputs.append(provisional)
        state = apply_corrector(params, provisional)

    def frozen(p):
        losses = []
        for provisional, target in zip(outputs, targets[1:], strict=True):
            s = apply_corrector(p, jax.lax.stop_gradient(provisional))
            losses.append(jnp.sum((s - target) ** 2) / jnp.sum(target**2))
        return jnp.mean(jnp.stack(losses))

    expected = jax.grad(frozen)(params)
    for a, b in zip(jax.tree.leaves(stopped), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8)
    assert sil.global_norm(jax.tree.map(lambda a, b: a - b, full, stopped)) > 1e-6


def test_windows_visit_every_window_once_per_epoch_and_repeat_by_seed():
    draws = sil.windows(trajectories=2, frames=5, unroll=2, updates=16, seed=7)
    assert sorted(draws[:8]) == [(i, s) for i in range(2) for s in range(4)]
    assert draws == sil.windows(2, 5, 2, 16, seed=7)
    assert draws != sil.windows(2, 5, 2, 16, seed=8)
    with pytest.raises(ValueError):
        sil.windows(1, 3, 4, 1, seed=0)


def test_every_trained_arm_sees_the_same_windows(monkeypatch):
    seen = {}

    def fake_fit(params, loss, validate, *, value_and_grad, updates, lr, **_):
        arm = value_and_grad.keywords["_arm"]
        seen[arm] = list(value_and_grad.keywords["_draws"])
        record = {
            "update_seconds": [0.0] * updates,
            "completed_updates": updates,
            "failure": None,
        }
        return params, record

    monkeypatch.setattr(sil, "fit", fake_fit)
    refs = {
        "train": np.stack(
            [np.stack([_field(seed=s + 10 * i) for s in range(6)]) for i in range(2)]
        ),
        "validation": np.stack([np.stack([_field(seed=s) for s in range(4)])]),
    }
    cfg = {**sil.DEFAULT_TRAINING, "unrolls": [2], "updates": 5, "model_seeds": [3]}
    sil._train_all(
        coarse=decaying_step(),
        references_=refs,
        cfg=cfg,
        memory=lambda: _NullPoller(),
    )
    assert seen["supervised"] == seen["stopped"] == seen["sil"]
    assert len(seen["sil"]) == 5


class _NullPoller:
    def __init__(self):
        self.summary = {"vram_peak_mib": None, "ram_peak_mib": None}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _quadratic_fit(scores: list[float], updates: int = 4, interval: int = 2, **kw):
    params = {"x": jnp.asarray(1.0)}
    calls = iter(scores)
    return sil.fit(
        params,
        lambda p, u: (p["x"] - 3.0) ** 2,
        lambda p: next(calls),
        updates=updates,
        lr=0.1,
        validation_interval=interval,
        **kw,
    )


def test_fit_runs_the_whole_budget_and_keeps_the_best_checkpoint():
    _, record = _quadratic_fit([1.0, 0.5, 0.7])
    assert record["completed_updates"] == 4
    assert len(record["loss"]) == len(record["update_seconds"]) == 4
    assert [c["update"] for c in record["checkpoints"]] == [0, 2, 4]
    assert record["selected_update"] == 2
    assert record["validation_score"] == 0.5


def test_fit_keeps_the_earliest_checkpoint_on_a_tie_and_the_identity_without_progress():
    assert _quadratic_fit([1.0, 1.0, 1.0])[1]["selected_update"] == 0
    assert _quadratic_fit([1.0, 0.5, 0.5])[1]["selected_update"] == 2


def test_fit_stops_and_records_a_non_finite_loss():
    def value_and_grad(params, update):
        loss = jnp.nan if update == 2 else (params["x"] - 3.0) ** 2
        return loss, {"x": jnp.asarray(1.0)}

    _, record = _quadratic_fit([1.0, 0.5], value_and_grad=value_and_grad)
    assert record["completed_updates"] == 2
    assert "update 2" in record["failure"]


def test_checkpoint_updates_include_the_identity_and_the_end():
    assert sil.checkpoint_updates(10, 4) == [0, 4, 8, 10]


def _feasible(**overrides) -> dict:
    metrics = {
        key: True
        for key in (
            "audit_passed",
            "closure_passed",
            "completed",
            "fd_passed",
            "finite",
            "selection_valid",
        )
    }
    return {**metrics, **overrides}


@pytest.mark.parametrize(
    "overrides, fires",
    [
        ({}, False),
        ({"audit_passed": False, "reason": "ratio 0.4"}, True),
        ({"closure_passed": False}, True),
        ({"completed": False}, True),
        ({"fd_passed": False}, True),
        ({"finite": False}, True),
        ({"selection_valid": False}, True),
    ],
)
def test_status_gates_feasibility_only(overrides, fires):
    verdict = sil_status.feasible()(OptimizationSummary(metrics=_feasible(**overrides)))
    assert (verdict is not None) == fires
    if "reason" in overrides:
        assert overrides["reason"] in verdict[1]


def _truth() -> dict:
    out = {f"initial_{s}": _field(16, s) for s in range(6)}
    out["horizon_times"] = np.asarray(ko.HORIZON_TIMES)
    return out


def grid_dependent_step(t, ctx, *, dt, steps, native_state):
    """Decay whose rate depends on the grid like a third-order error."""

    def step(v, native):
        h = ko.DOMAIN_EXTENT / v.shape[0]
        return v * jnp.exp(-dt * steps * (0.5 + h**3)), None

    return step


def test_run_gates_trains_and_evaluates(monkeypatch):
    monkeypatch.setattr(ko, "load_ground_truth", lambda domain: _truth())
    optim = {
        "unrolls": [2],
        "updates": 3,
        "model_seeds": [0],
        "learning_rates": [1e-3],
        "validation_interval": 2,
        "seeds": {"train": [0, 1], "validation": [2], "test": [3]},
        "train_frames": 4,
        "eval_frames": 4,
        "audit_frames": 2,
    }
    ctx = SimpleNamespace(
        name="fake",
        cfg=SimpleNamespace(name="ns-grid"),
        run={"optim": optim},
        phys={"N": 4, "dt": 0.05, "reference_factor": 2, "nu": 0.01},
        make_inputs=None,
        output_key="result",
    )
    metrics = sil.run(None, ctx, step_factory=grid_dependent_step)["metrics"]
    assert metrics["eligible"]
    assert metrics["audit_passed"] and metrics["closure_passed"]
    assert metrics["completed"] and metrics["finite"] and metrics["selection_valid"]
    assert set(metrics["evaluation"]) == {
        "uncorrected",
        "K2/supervised",
        "K2/stopped",
        "K2/sil",
    }
    assert len(metrics["paired"]["value_of_gradient/K2"]) == 1
    verdict = sil_status.feasible()(OptimizationSummary(metrics=metrics))
    assert verdict is None or "finite-difference" in verdict[1]
