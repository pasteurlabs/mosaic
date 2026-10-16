# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DFG 2D-1 cylinder benchmark: geometry, inputs, metrics and status checks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from mosaic_shared.problems.navier_stokes_grid import InputSchema
from mosaic_shared.problems.navier_stokes_grid.channel import solid_cells

from mosaic.benchmarks.core.status import ANOMALY, OK, Cell, _refine_for_suite
from mosaic.benchmarks.problems.navier_stokes_grid import cylinder

# ── Geometry and inputs ──────────────────────────────────────────────────────


@pytest.mark.parametrize(("res", "shape"), [(10, (220, 41)), (20, (440, 82))])
def test_grid_matches_the_channel(res, shape):
    nx, ny, dx = cylinder.grid(res)
    assert (nx, ny) == shape
    assert dx == pytest.approx(cylinder.DIAMETER / res)


def test_grid_rejects_a_resolution_that_does_not_tile():
    with pytest.raises(ValueError, match="multiple of 10"):
        cylinder.grid(15)


def test_inflow_profile_has_the_benchmark_flow_rate():
    prof = np.asarray(cylinder.inflow_profile(40))
    assert prof.mean() == pytest.approx(cylinder.U_MEAN, rel=1e-3)
    assert prof.max() == pytest.approx(cylinder.U_MAX, rel=1e-3)


def test_time_step_reaches_the_end_time():
    for res in (10, 20, 40):
        dt, steps = cylinder.time_step(res)
        _, _, dx = cylinder.grid(res)
        assert dt * steps == pytest.approx(cylinder.T_END)
        assert dt * cylinder.U_MAX / dx <= cylinder.CFL + 1e-9
        assert cylinder.VISCOSITY * dt / dx**2 <= cylinder.DIFFUSION_NUMBER + 1e-9


def test_channel_inputs_satisfy_the_solver_schema():
    """What the harness sends is what every solver's schema accepts."""
    inputs = cylinder.channel_inputs(cylinder.inflow_profile(10), 10)
    schema = InputSchema(**inputs)
    assert schema.extent == (cylinder.LENGTH, cylinder.HEIGHT)
    assert schema.dx == pytest.approx(0.01)
    v0 = np.asarray(inputs["v0"])
    assert np.all(v0[cylinder.solid_mask(10)] == 0)


@pytest.mark.parametrize("res", [10, 20, 40])
def test_solid_mask_is_the_solvers_staircase(res):
    """The harness's mask matches the solvers' ``solid_cells`` cell for cell."""
    nx, ny, dx = cylinder.grid(res)
    radius = cylinder.DIAMETER / 2
    expected = solid_cells(nx, ny, dx, cylinder.CENTER, radius)
    np.testing.assert_array_equal(cylinder.solid_mask(res), expected)


# ── Metrics ──────────────────────────────────────────────────────────────────


def _reference_force(steps: int = 100) -> np.ndarray:
    scale = cylinder.U_MEAN**2 * cylinder.DIAMETER / 2
    return np.tile([cylinder.CD_REF * scale, cylinder.CL_REF * scale], (steps, 1))


def test_force_summary_turns_force_into_coefficients():
    out = cylinder.force_summary(_reference_force())
    assert out["cd"] == pytest.approx(cylinder.CD_REF)
    assert out["cl"] == pytest.approx(cylinder.CL_REF)
    assert out["cd_drift"] == pytest.approx(0.0)
    assert float(cylinder.drag_coefficient(jnp.asarray(_reference_force()))) == (
        pytest.approx(cylinder.CD_REF, rel=1e-6)
    )


def test_force_summary_measures_drift_into_the_last_window():
    force = _reference_force()
    force[-20:, 0] *= 1.01  # the final 20% sit 1% above the window before
    assert cylinder.force_summary(force)["cd_drift"] == pytest.approx(0.01 / 1.01)


def test_pressure_difference_is_exact_for_a_linear_field():
    """Linear extrapolation to the cylinder surface is exact for p = −G x."""
    res = 20
    x, _ = cylinder.cell_centers(res)
    p = -3.0 * np.broadcast_to(x[:, None], cylinder.solid_mask(res).shape)
    dp = cylinder.pressure_difference(p, res)
    assert dp == pytest.approx(3.0 * (cylinder.BACK[0] - cylinder.FRONT[0]))


# ── Status checks ────────────────────────────────────────────────────────────


def _cells(by_solver: dict, checks: list) -> dict[str, Cell]:
    cells = {name: Cell(OK) for name in by_solver}
    _refine_for_suite("forward", "cylinder", {"by_solver": by_solver}, cells, checks)
    return cells


def test_max_cd_err_judges_the_canonical_resolution_only():
    by_solver = {
        "good": {
            "10": {"valid": True, "error": 0.2, "cd": 4.5},
            "20": {"valid": True, "error": 0.02, "cd": 5.47},
        },
        "bad": {
            "10": {"valid": True, "error": 0.01, "cd": 5.5},
            "20": {"valid": True, "error": 0.08, "cd": 5.13},
        },
        "missing": {
            "10": {"valid": True, "error": 0.01, "cd": 5.5},
            "20": {"valid": False, "apply_error": "boom"},
        },
    }
    cells = _cells(by_solver, [cylinder.max_cd_err(0.05, resolution=20)])
    assert cells["good"].status == OK
    assert cells["bad"].status == ANOMALY
    assert "5.13" in cells["bad"].reason
    assert cells["missing"].status == ANOMALY


def test_max_cd_drift_flags_an_unsteady_run():
    by_solver = {
        "steady": {"20": {"valid": True, "error": 0.01, "cd_drift": 1e-5}},
        "drifting": {"20": {"valid": True, "error": 0.01, "cd_drift": 0.02}},
    }
    cells = _cells(by_solver, [cylinder.max_cd_drift(1e-3)])
    assert cells["steady"].status == OK
    assert cells["drifting"].status == ANOMALY


# ── drag_opt profile projection ──────────────────────────────────────────────


def test_projection_keeps_flow_rate_and_bounds():
    import jax

    from mosaic.benchmarks.problems.navier_stokes_grid.optimization import (
        U_BOUND,
        _project,
    )

    v = jnp.asarray(np.random.default_rng(0).normal(0.2, 0.3, 41), jnp.float32)
    u = np.asarray(_project(v))
    assert u.mean() == pytest.approx(cylinder.U_MEAN, rel=1e-5)
    assert u.min() >= 0.0 and u.max() <= U_BOUND + 1e-6
    # Already admissible profiles are left alone.
    prof = cylinder.inflow_profile(10)
    np.testing.assert_allclose(_project(prof), prof, atol=1e-4)

    # The gradient is the projection's Jacobian: compare with central differences.
    w = jnp.asarray(np.random.default_rng(1).normal(size=41), jnp.float32)
    grad = np.asarray(jax.grad(lambda x: jnp.sum(w * _project(x)))(v))
    eps = 1e-3
    fd = [
        float(jnp.sum(w * _project(v.at[i].add(eps))))
        - float(jnp.sum(w * _project(v.at[i].add(-eps))))
        for i in range(41)
    ]
    np.testing.assert_allclose(grad, np.asarray(fd) / (2 * eps), atol=1e-3)


def test_adam_stops_on_a_plateau_but_not_on_a_slow_descent():
    """Improvements below the relative threshold do not reset the patience."""
    from mosaic.benchmarks.problems.shared.optimization import _run_optim

    def plateau(x):  # flat to 1e-6: Adam keeps "improving" it by ulps
        return 3.0 + 1e-6 * jnp.sum((x - 1.0) ** 2)

    def descent(x):  # 0.05% per iteration at lr 1e-2, 0.5% per 10
        return 3.0 + 0.04 * jnp.sum(x)

    x0 = jnp.zeros(4)
    _, flat, _ = _run_optim(plateau, x0, 1e-2, 100, 10, min_rel_improvement=1e-3)
    _, down, _ = _run_optim(descent, x0, 1e-2, 100, 10, min_rel_improvement=1e-3)
    assert len(flat) == 11
    assert len(down) == 100
