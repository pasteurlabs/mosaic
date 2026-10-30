# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small mathematical checks; numerical solver validation runs on Slurm/CI."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.benchmarks.problems.buoyancy_grid import training
from mosaic.benchmarks.problems.buoyancy_grid.corrector import (
    apply_corrector,
    init_corrector,
    restrict_state,
)


def test_restriction_preserves_faces_and_conserves_scalar():
    state = np.zeros((8, 8, 3), dtype=np.float32)
    state[..., 0] = np.arange(1, 9)[:, None]
    state[..., 1] = np.arange(1, 9)[None, :]
    state[..., 2] = np.arange(64).reshape(8, 8)
    state[-1, :, 0] = 0
    state[:, -1, 1] = 0
    coarse = restrict_state(state, 2)
    np.testing.assert_array_equal(coarse[:, 0, 0], [2, 4, 6, 0])
    np.testing.assert_array_equal(coarse[0, :, 1], [2, 4, 6, 0])
    assert coarse[..., 2].mean() == state[..., 2].mean()


def test_corrector_starts_at_identity_and_preserves_scalar_and_walls():
    state = jnp.ones((6, 6, 3)).at[-1, :, 0].set(0).at[:, -1, 1].set(0)
    params = init_corrector(0, 2)
    assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(params))
    np.testing.assert_array_equal(apply_corrector(params, state, jnp.ones(3)), state)
    np.testing.assert_array_equal(
        apply_corrector(params, np.asarray(state), jnp.ones(3)), state
    )
    params[-1]["b"] = jnp.ones(1, dtype=jnp.float32)
    corrected = np.asarray(apply_corrector(params, state, jnp.ones(3)))
    np.testing.assert_array_equal(corrected[..., 2], state[..., 2])
    np.testing.assert_array_equal(corrected[-1, :, 0], 0)
    np.testing.assert_array_equal(corrected[:, -1, 1], 0)
    assert corrected[0, 0, 0] > state[0, 0, 0]
    delta = corrected - np.asarray(state)
    ux = np.pad(delta[..., 0], ((1, 0), (0, 0)))
    uy = np.pad(delta[..., 1], ((0, 0), (1, 0)))
    np.testing.assert_allclose(np.diff(ux, axis=0) + np.diff(uy, axis=1), 0, atol=1e-6)


def test_training_gradient_includes_recurrent_solver_derivative(monkeypatch):
    monkeypatch.setattr(
        training, "apply_corrector", lambda p, state, scales, **_: state + p
    )
    targets = jnp.zeros((3, 2, 2, 3))

    def loss(p):
        return training.window_loss(p, targets, lambda x: 2 * x, jnp.ones(3))

    np.testing.assert_allclose(jax.grad(loss)(1.0), 10.0, rtol=1e-6)


def test_directional_fd_checks_recurrent_derivative(monkeypatch):
    monkeypatch.setattr(
        training, "apply_corrector", lambda p, state, scales, **_: state + p
    )
    targets = jnp.zeros((3, 2, 2, 3))
    check = training.directional_fd(
        lambda p: training.window_loss(p, targets, lambda x: 2 * x, jnp.ones(3)),
        jnp.asarray(0.5),
    )
    assert check["relative_error"] < 1e-3


@pytest.mark.parametrize("ic", [{}, {"name": "interacting_plumes"}])
def test_training_uses_configured_ic_for_each_split(monkeypatch, ic):
    calls = []

    class InitialsCaptured(Exception):
        pass

    def initial(**kwargs):
        calls.append(kwargs)
        if len(calls) == 3:
            raise InitialsCaptured

    def wrong_initial(**kwargs):
        raise AssertionError("selected the wrong initial-condition family")

    family = ic.get("name", "plume")
    registry = {"plume": wrong_initial, "interacting_plumes": wrong_initial}
    registry[family] = initial
    ctx = SimpleNamespace(
        run={
            "ic": ic,
            "training": {"train_seed": 7, "validation_seed": 8, "test_seed": 9},
        },
        phys={"N": 32},
        domain_extent=2.0,
        cfg=SimpleNamespace(make_ic=registry),
    )
    # Stop after initial construction; neither a solver nor JAX execution is needed.
    monkeypatch.setattr(training, "trajectory", lambda *args: [None])
    monkeypatch.setattr(training, "restrict_state", lambda state, factor: state)
    with pytest.raises(InitialsCaptured):
        training.solver_in_loop(None, ctx)
    assert calls == [{"N": 64, "L": 2.0, "seed": seed} for seed in (7, 8, 9)]


@pytest.mark.parametrize("updates,count", [(256, 93), (128, 85), (1, 1), (4, 1)])
def test_shuffled_windows_reproduce_seed_and_cover_each_epoch(updates, count):
    starts = training.shuffled_windows(updates, count, 7)
    assert starts == training.shuffled_windows(updates, count, 7)
    assert len(starts) == updates
    for begin in range(0, updates, count):
        epoch = starts[begin : begin + count]
        assert len(set(epoch)) == len(epoch)
        assert all(0 <= start < count for start in epoch)
        if len(epoch) == count:
            assert set(epoch) == set(range(count))
    if count > 1:
        assert starts != training.shuffled_windows(updates, count, 8)


@pytest.mark.parametrize("target,selected", [(0.1, 1), (0.0, 0)])
def test_fit_completes_budget_and_retains_earlier_or_identity_checkpoint(
    target, selected
):
    # A constant training gradient moves p away from zero each update; the
    # independent validation target can favor an earlier point or the identity.
    params, record = training.fit(
        jnp.asarray(0.0),
        lambda p, i: -p,
        lambda p: (p - target) ** 2,
        updates=3,
        lr=0.1,
        validation_interval=1,
    )
    assert record["completed_updates"] == 3
    assert len(record["training_loss"]) == 3
    assert record["selected_update"] == selected
    np.testing.assert_allclose(params, target, atol=1e-5)
    assert [point["update"] for point in record["validation_checkpoints"]] == [
        0,
        1,
        2,
        3,
    ]


def test_validation_ties_keep_identity_and_include_partial_final_interval():
    params, record = training.fit(
        jnp.asarray(0.0),
        lambda p, i: -p,
        lambda p: 1.0,
        updates=3,
        lr=0.1,
        validation_interval=2,
    )
    assert float(params) == 0.0
    assert record["selected_update"] == 0
    assert [point["update"] for point in record["validation_checkpoints"]] == [0, 2, 3]
    assert (
        training.select_candidate(
            [
                {"validation_loss": 1.0, "selected_update": 2},
                {"validation_loss": 1.0, "selected_update": 0},
                {"validation_loss": 1.0, "selected_update": 0},
            ]
        )
        == 1
    )


def test_nonfinite_validation_fails_instead_of_skipping_checkpoint():
    with pytest.raises(FloatingPointError, match="validation"):
        training.fit(
            jnp.asarray(0.0), lambda p, i: -p, lambda p: float("nan"), 3, 0.1, 1
        )


def test_balanced_case_schedule_is_reproducible_and_each_case_covers_windows():
    cases, starts = training.balanced_windows(84, 7, 4, 0)
    assert (cases, starts) == training.balanced_windows(84, 7, 4, 0)
    assert [cases.count(i) for i in range(4)] == [21] * 4
    for begin in range(0, 84, 4):
        assert set(cases[begin : begin + 4]) == set(range(4))
    for case in range(4):
        windows = [
            start for chosen, start in zip(cases, starts, strict=True) if chosen == case
        ]
        for begin in range(0, len(windows), 7):
            assert set(windows[begin : begin + 7]) == set(range(7))
    assert training.balanced_windows(20, 7, 1, 0) == (
        [0] * 20,
        training.shuffled_windows(20, 7, 0),
    )


def test_default_mix_assigns_64_updates_per_case_without_crossing_trajectories():
    cases, starts = training.balanced_windows(256, 100 + 1 - 8, 4, 0)
    assert [cases.count(i) for i in range(4)] == [64] * 4
    assert all(0 <= start and start + 8 <= 100 for start in starts)
    single_cases, single_starts = training.balanced_windows(256, 93, 1, 0)
    assert single_cases == [0] * 256
    assert single_starts == training.shuffled_windows(256, 93, 0)


def test_training_case_jitter_is_never_forwarded_to_heldout_initials(monkeypatch):
    calls = []

    class InitialsCaptured(Exception):
        pass

    def initial(**kwargs):
        calls.append(kwargs)
        if len(calls) == 4:
            raise InitialsCaptured

    cases = [
        {"seed": 0},
        {
            "seed": 2,
            "position_jitter": 0.04,
            "width_jitter": 0.15,
            "amplitude_jitter": 0.15,
        },
    ]
    ctx = SimpleNamespace(
        run={"training": {"train_cases": cases}},
        phys={"N": 24},
        domain_extent=1.0,
        cfg=SimpleNamespace(make_ic={"plume": initial}),
    )
    monkeypatch.setattr(training, "trajectory", lambda *args: [None])
    monkeypatch.setattr(training, "restrict_state", lambda state, factor: state)
    with pytest.raises(InitialsCaptured):
        training.solver_in_loop(None, ctx)
    assert calls == [
        {"N": 48, "L": 1.0, **case} for case in [*cases, {"seed": 1}, {"seed": 100}]
    ]


@pytest.mark.parametrize("joint", [False, True])
def test_optional_temperature_head_starts_at_identity(joint):
    params = init_corrector(0, 8, correct_temperature=joint)
    count = sum(p.size for p in jax.tree.leaves(params))
    assert count == (954 if joint else 881)
    state = jnp.zeros((6, 6, 3)).at[..., 2].set(0.25)
    np.testing.assert_array_equal(apply_corrector(params, state, jnp.ones(3)), state)


def test_temperature_head_can_change_heat_without_changing_velocity():
    params = init_corrector(0, 2, correct_temperature=True)
    params[-1]["b"] = jnp.array([0.0, 0.5])
    state = jnp.zeros((6, 6, 3))
    corrected = np.asarray(apply_corrector(params, state, jnp.array([2.0, 2.0, 4.0])))
    np.testing.assert_array_equal(corrected[..., :2], 0)
    np.testing.assert_array_equal(corrected[..., 2], 2)
    diagnostics = training.temperature_diagnostics(np.stack([state, corrected]))
    assert diagnostics["overshoot"] == [0, 2]
    assert diagnostics["maximum"] == [0, 2]


def test_joint_temperature_correction_receives_recurrent_gradient():
    params = init_corrector(0, 2, correct_temperature=True)
    targets = jnp.zeros((3, 4, 4, 3))
    targets = targets.at[1:, ..., 2].set(1)

    def loss(bias):
        changed = (*params[:-1], {**params[-1], "b": jnp.array([0.0, bias])})
        return training.window_loss(changed, targets, lambda state: state, jnp.ones(3))

    # Temperature predictions are b and 2b, so mean loss derivative at b=0 is -1.5.
    np.testing.assert_allclose(jax.grad(loss)(0.0), -1.5, rtol=1e-6)


def test_temperature_head_scaling_agrees_in_recurrent_loss_and_evaluation():
    params = init_corrector(0, 2, correct_temperature=True)
    targets = jnp.zeros((3, 4, 4, 3))

    def model(bias):
        return (*params[:-1], {**params[-1], "b": jnp.array([0.0, bias])})

    def loss(bias):
        return training.window_loss(
            model(bias),
            targets,
            lambda state: 2 * state,
            jnp.ones(3),
            temperature_head_scale=0.1,
        )

    # The two temperature predictions are 0.1b and 0.3b, so L=0.025b².
    np.testing.assert_allclose(loss(1.0), 0.025, rtol=1e-6)
    np.testing.assert_allclose(jax.grad(loss)(1.0), 0.05, rtol=1e-6)
    rollout = training.trajectory(
        lambda state: 2 * state,
        targets[0],
        2,
        model(1.0),
        jnp.ones(3),
        temperature_head_scale=0.1,
    )
    np.testing.assert_allclose(rollout[:, 0, 0, 2], [0, 0.1, 0.3], rtol=1e-6)
    np.testing.assert_array_equal(rollout[..., :2], 0)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_head_scale_rejected_before_reference_generation(scale):
    ctx = SimpleNamespace(run={"training": {"temperature_head_scale": scale}}, phys={})
    with pytest.raises(ValueError, match="head scale"):
        training.solver_in_loop(None, ctx)


def test_temperature_diagnostics_report_signed_heat_loss_and_extrema():
    states = np.zeros((2, 2, 2, 3))
    states[0, ..., 2] = 1
    states[1, ..., 2] = [[-0.5, 1.5], [0, 0]]
    d = training.temperature_diagnostics(states)
    assert d["relative_integral_drift"] == [0, -0.75]
    assert d["minimum"] == [1, -0.5]
    assert d["undershoot"] == [0, 1.5]
    assert d["overshoot"] == [0, 0.5]
