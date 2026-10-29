# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kolmogorov-flow fields, statistics and the ``forward/kolmogorov`` kernel."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mosaic.benchmarks.core.status_checks import ForwardSummary, max_error
from mosaic.benchmarks.problems.navier_stokes_grid import kolmogorov as ko

L = ko.DOMAIN_EXTENT


def _smooth(n: int, seed: int = 0) -> np.ndarray:
    """A band-limited divergence-free field at the cell centres of an n-grid."""
    rng = np.random.default_rng(seed)
    x = ko.cell_centres(n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    for kx, ky in ((1, 0), (0, 2), (1, 1), (3, 1), (2, 5)):
        c = rng.normal() * np.cos(kx * xx + ky * yy + rng.uniform(0, 2 * np.pi))
        u += ky * c
        v -= kx * c
    return np.stack([u, v], -1)[:, :, None, :].astype(np.float32)


@pytest.mark.parametrize("n", [16, 128])
def test_forcing_resamples_onto_the_forcing_of_another_grid(n):
    """Cell centres differ between grids, and resampling accounts for that."""
    resampled = np.asarray(ko.resample(ko.forcing_field(64), n))
    np.testing.assert_allclose(resampled, ko.forcing_field(n), atol=1e-5)


def test_resampling_a_band_limited_field_is_exact():
    fine = _smooth(64)
    np.testing.assert_allclose(
        np.asarray(ko.resample(fine, 32)), _smooth(32), atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(ko.resample(ko.resample(fine, 128), 64)), fine, atol=1e-4
    )


def test_laminar_velocity_relaxes_to_the_steady_shear_flow():
    nu, alpha = 0.05, 0.1
    assert np.all(ko.laminar_velocity(16, 0.0, viscosity=nu, drag=alpha) == 0)
    rate = nu * (2 * np.pi * ko.WAVENUMBER / L) ** 2 + alpha
    steady = ko.laminar_velocity(16, 1e3, viscosity=nu, drag=alpha)
    np.testing.assert_allclose(steady, ko.forcing_field(16) / rate, rtol=1e-5)


def test_vorticity_of_the_force_and_its_self_correlation():
    k = 2 * np.pi * ko.WAVENUMBER / L
    w = np.asarray(ko.vorticity(ko.forcing_field(64)))
    expected = -k * np.cos(k * ko.cell_centres(64))[None, :] * np.ones((64, 1))
    np.testing.assert_allclose(w, expected, atol=1e-4)
    field = _smooth(32)
    assert float(ko.vorticity_correlation(field, field)) == pytest.approx(1.0)
    assert float(ko.vorticity_correlation(field, -field)) == pytest.approx(-1.0)


def test_energy_spectrum_sums_to_the_kinetic_energy():
    field = _smooth(32)
    energy = 0.5 * np.mean(np.sum(field.astype(np.float64) ** 2, axis=-1))
    assert ko.energy_spectrum(field).sum() == pytest.approx(energy, rel=1e-6)
    assert ko.spectrum_error([field], [field]) == pytest.approx(0.0)


def test_high_correlation_time_and_first_unstable_frame():
    assert ko.high_correlation_time([1.0, 0.99, 0.9, 0.97], 0.1) == pytest.approx(0.2)
    assert ko.high_correlation_time([1.0, 0.99], 0.1) == pytest.approx(0.2)
    assert ko.first_unstable_frame([0.0, 1.0, np.nan, 0.5]) == 2
    assert ko.first_unstable_frame([0.0, 1.5]) == 2
    assert ko.capped_error(np.full((4, 4, 1, 2), np.nan), np.ones((4, 4, 1, 2))) == 2


def _truth(n_coarse: int = 16, n_ref: int = 32) -> dict:
    seeds = ko.GROUND_TRUTH_SEEDS["test"]
    out = {f"initial_{s}": _smooth(n_ref, s) for s in seeds}
    for s in seeds:
        out[f"truth_{s}"] = np.stack(
            [np.asarray(ko.resample(_smooth(n_ref, s), n_coarse)) * 0.9] * 4
        )
    out["horizon_times"] = np.asarray(ko.HORIZON_TIMES)
    return out


def _context(phys: dict) -> SimpleNamespace:
    def make_inputs(name, v0, **physics):
        return {"v0": v0, **physics}

    return SimpleNamespace(
        name="fake",
        cfg=SimpleNamespace(name="ns-grid"),
        phys=phys,
        make_inputs=make_inputs,
        output_key="result",
    )


def test_forward_scores_every_grid(monkeypatch):
    """A solver that returns its input scores 0.1 at every time and grid."""
    monkeypatch.setattr(ko, "load_ground_truth", lambda domain: _truth())
    monkeypatch.setattr(ko, "safe_apply", lambda t, inputs, key: inputs["v0"])
    phys = {"N": 16, "dt": 0.01, "grids": [64, 16, 32], "nu": 0.01}
    metrics = ko.forward(None, _context(phys))["metrics"]
    assert [r["grid"] for r in metrics["runs"]] == [16, 32, 64]
    for run in metrics["runs"]:
        assert run["dt"] == pytest.approx(0.01 * 16 / run["grid"])
        assert run["median_errors"] == pytest.approx([1 / 9] * 4, rel=1e-3)
        assert not run["diverged"]
        assert run["seconds_per_time"] >= 0
    assert metrics["error"] == pytest.approx(1 / 9, rel=1e-3)
    assert metrics["order"] == pytest.approx(0.0, abs=1e-3)


def test_forward_scores_a_diverged_run_as_left_the_attractor(monkeypatch):
    monkeypatch.setattr(ko, "load_ground_truth", lambda domain: _truth())
    monkeypatch.setattr(
        ko,
        "safe_apply",
        lambda t, inputs, key: None if inputs["v0"].shape[0] == 32 else inputs["v0"],
    )
    phys = {"N": 16, "dt": 0.01, "grids": [16, 32], "nu": 0.01}
    metrics = ko.forward(None, _context(phys))["metrics"]
    finest = metrics["runs"][-1]
    assert finest["diverged"] and finest["seconds_per_time"] is None
    assert metrics["error"] == ko.DIVERGED_ERROR
    verdict = ko.min_refinement_order(0.8)(_summary(metrics))
    assert verdict is not None and "diverged on 32²" in verdict[1]


def _summary(metrics: dict) -> ForwardSummary:
    return ForwardSummary(
        errs_by_pval={"": metrics["error"]}, metrics_by_pval={"": metrics}
    )


def _ladder(order: float, coarse_diverged: bool = False) -> dict:
    runs = [
        {"grid": g, "diverged": d}
        for g, d in ((8, True), (16, coarse_diverged), (64, False))
    ]
    return {"coarse_grid": 16, "runs": runs, "order": order, "error": 0.1}


@pytest.mark.parametrize(
    "metrics, fires",
    [
        (_ladder(1.2), False),
        (_ladder(0.5), True),
        (_ladder(1.2, coarse_diverged=True), True),
    ],
)
def test_min_refinement_order(metrics, fires):
    """Grids below the coarse one may diverge; it and finer ones may not."""
    assert (ko.min_refinement_order(0.8)(_summary(metrics)) is not None) == fires


def test_max_error_at_one_sweep_value():
    summary = ForwardSummary(errs_by_pval={"32": 0.08, "64": 0.02, "128": 0.005})
    assert max_error(0.05)(summary) is not None
    assert max_error(0.05, at=64)(summary) is None
    assert max_error(0.01, at=64)(summary) is not None
