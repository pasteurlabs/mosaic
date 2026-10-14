# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolution-convergence support: fixed-CFL refinement, order fit, status check."""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.benchmarks.cli.reference import _converged_targets
from mosaic.benchmarks.core.reference import ConvergedSpec
from mosaic.benchmarks.core.status_checks import ForwardSummary, min_order
from mosaic.benchmarks.core.utils import convergence_order, fixed_cfl_dt_steps
from mosaic.benchmarks.problems import get_config


class TestFixedCflDtSteps:
    def test_base_resolution_is_unchanged(self) -> None:
        assert fixed_cfl_dt_steps(0.01, 128, 128, cfl_N_base=128) == (0.01, 128)

    def test_coarser_grid_takes_fewer_larger_steps(self) -> None:
        dt, steps = fixed_cfl_dt_steps(0.01, 128, 16, cfl_N_base=128)
        assert dt == pytest.approx(0.08)
        assert steps == 16

    def test_unset_base_is_a_no_op(self) -> None:
        assert fixed_cfl_dt_steps(0.01, 7, 16, cfl_N_base=None) == (0.01, 7)

    def test_end_time_must_be_exact(self) -> None:
        with pytest.raises(ValueError, match="end time"):
            fixed_cfl_dt_steps(0.01, 100, 16, cfl_N_base=128)


@pytest.mark.parametrize(
    ("problem", "phys"),
    [
        ("ns-grid", {"N": [16, 128], "dt": 0.01, "steps": 128, "cfl_N_base": 128}),
        ("ns-3d-grid", {"N": [8, 32], "dt": 0.04, "steps": 32, "cfl_N_base": 32}),
    ],
)
def test_make_inputs_refines_every_solver_at_fixed_end_time(problem, phys) -> None:
    cfg = get_config(problem)
    ic_name = next(iter(cfg.make_ic))
    for n in phys["N"]:
        curr = {**phys, "N": n, "nu": 0.05}
        ic = cfg.make_ic[ic_name](L=cfg.domain_extent, seed=0, **curr)
        for spec in cfg.solvers:
            inputs = cfg.make_inputs(spec.name, ic, **curr)
            dt = float(np.asarray(inputs["dt"])[0])
            assert dt == pytest.approx(phys["dt"] * phys["cfl_N_base"] / n)
            assert dt * inputs["steps"] == pytest.approx(phys["dt"] * phys["steps"])


class TestConvergenceOrder:
    def test_recovers_power_law_exponent(self) -> None:
        errs = {n: 3.0 * n**-2.0 for n in (16, 32, 64, 128)}
        assert convergence_order(errs) == pytest.approx(2.0)

    def test_uses_the_two_finest_grids(self) -> None:
        # Second order on coarse grids, then a floor: the finest pair shows it.
        errs = {16: 4e-2, 32: 1e-2, 64: 2.5e-3, 128: 2.4e-3}
        assert convergence_order(errs) == pytest.approx(0.059, abs=1e-3)

    def test_points_below_floor_are_ignored(self) -> None:
        errs = {16: 1e-2, 32: 2.5e-3, 64: 1e-8, 128: 1e-8}
        assert convergence_order(errs, floor=1e-6) == pytest.approx(2.0)

    def test_needs_two_usable_points(self) -> None:
        assert convergence_order({16: 1e-2, 32: 1e-9}, floor=1e-6) is None
        assert convergence_order({16: float("nan"), 32: 1e-3}) is None


class TestMinOrder:
    @staticmethod
    def _summary(order: float) -> ForwardSummary:
        errs = {n: n**-order for n in (16, 32, 64)}
        return ForwardSummary(errs_by_pval=errs, n_valid_points=len(errs))

    def test_second_order_passes(self) -> None:
        assert min_order(1.5)(self._summary(2.0)) is None

    def test_first_order_fires(self) -> None:
        outcome = min_order(1.5)(self._summary(1.0))
        assert outcome is not None
        status, reason = outcome
        assert status == "anomaly"
        assert "order 1.00" in reason

    def test_undetermined_order_passes(self) -> None:
        s = ForwardSummary(errs_by_pval={16: 1e-3}, n_valid_points=1)
        assert min_order(1.5)(s) is None


def test_converged_targets_for_resolution_sweep() -> None:
    rp = {
        "sweep_key": "N",
        "sweep_values": [8, 16, 32],
        "phys": {"N": [8, 16, 32], "nu": 0.05},
    }
    targets = _converged_targets(rp, ConvergedSpec(solver="Exponax", high_n=64))
    assert [(i, n) for i, _val, n, _phys in targets] == [(0, 8), (1, 16), (2, 32)]
    assert all(phys["N"] == 64 for *_, phys in targets)


def test_converged_targets_for_parameter_sweep() -> None:
    rp = {"sweep_key": "nu", "sweep_values": [0.01, 0.05], "phys": {"N": 16}}
    targets = _converged_targets(rp, ConvergedSpec(solver="Exponax", high_n=64))
    assert [(val, n) for _i, val, n, _phys in targets] == [(0.01, 16), (0.05, 16)]
    assert [phys["nu"] for *_, phys in targets] == [0.01, 0.05]
    assert all(phys["N"] == 64 for *_, phys in targets)
