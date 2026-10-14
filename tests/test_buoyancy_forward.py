# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coupled-forward reference timing, staggered diagnostics, and failure reporting."""

from types import SimpleNamespace

import numpy as np
import pytest

from mosaic.benchmarks.problems.buoyancy_grid import forward


def _constant_ic(N=8, **kwargs):
    state = np.zeros((N, N, 3), dtype=np.float32)
    state[..., 2] = 1
    return state


def _context():
    def make_inputs(name, ic, **physics):
        return {
            "state0": ic,
            "dt": np.array([physics["dt"]], np.float32),
            "steps": physics["steps"],
        }

    return SimpleNamespace(
        name="solver",
        cfg=SimpleNamespace(make_ic={"interacting_plumes": _constant_ic}),
        ic=_constant_ic(),
        phys={"N": 8, "dt": 0.01, "steps": 7},
        make_inputs=make_inputs,
        output_key="result",
        domain_extent=1.0,
        run={"ic": {"name": "interacting_plumes", "seed": 42}},
        seed=42,
    )


def test_refinement_preserves_seed_domain_and_physical_time(monkeypatch):
    ctx = _context()
    calls, ic_calls = [], []

    def ic_factory(**kwargs):
        ic_calls.append(kwargs)
        return _constant_ic(**kwargs)

    def apply(solver, inputs, output_key):
        calls.append(inputs)
        return inputs["state0"]

    ctx.cfg.make_ic["interacting_plumes"] = ic_factory
    monkeypatch.setattr(forward, "safe_apply", apply)
    metrics = forward.refinement(None, ctx)["metrics"]
    assert metrics["valid"] and metrics["error"] == 0
    assert [call["state0"].shape for call in calls] == [(8, 8, 3), (16, 16, 3)]
    assert ic_calls[0]["seed"] == 42
    assert ic_calls[0]["L"] == 1.0
    assert calls[1]["steps"] == 2 * calls[0]["steps"]
    np.testing.assert_allclose(
        calls[0]["dt"] * calls[0]["steps"], calls[1]["dt"] * calls[1]["steps"]
    )


def test_refinement_balances_velocity_and_scalar_error(monkeypatch):
    ctx = _context()

    def apply(solver, inputs, output_key):
        result = inputs["state0"].copy()
        # Smooth linear face velocity restricts exactly; double it only at coarse resolution.
        n = result.shape[0]
        result[:-1, :, 0] = np.arange(1, n)[:, None] / n
        if n == 8:
            result[..., 0] *= 2
        return result

    monkeypatch.setattr(forward, "safe_apply", apply)
    metrics = forward.refinement(None, ctx)["metrics"]
    assert metrics["velocity_rel_error"] == pytest.approx(1.0)
    assert metrics["scalar_rel_error"] == 0
    assert metrics["error"] == pytest.approx(np.sqrt(0.5))


def test_failed_reference_is_not_reported_as_valid(monkeypatch):
    calls = iter([_constant_ic(), None])
    monkeypatch.setattr(forward, "safe_apply", lambda *args: next(calls))
    monkeypatch.setattr(forward, "get_last_apply_error", lambda: "reference failed")
    metrics = forward.refinement(None, _context())["metrics"]
    assert metrics == {
        "valid": False,
        "status": "failed",
        "failure_type": "error",
        "stage": "refined",
        "error": "reference failed",
    }


def test_mac_divergence_and_velocity_energy_exclude_scalar_and_padding():
    n = 8
    rng = np.random.default_rng(3)
    psi = np.pad(rng.normal(size=(n - 1, n - 1)), 1)
    state = _constant_ic(n).astype(np.float64)
    state[:-1, :, 0] = n * np.diff(psi[1:-1, :], axis=1)
    state[:, :-1, 1] = -n * np.diff(psi[:, 1:-1], axis=0)
    expected_energy = 0.5 * np.sum(state[..., :2] ** 2) / n**2
    state[..., 2] = 100
    state[-1, :, 0] = 7
    result = forward.physical_diagnostics(state, state)
    assert result["divergence_rms"] < 1e-12
    assert result["kinetic_energy"] == pytest.approx(expected_energy)
    assert result["wall_normal_padding_max"] == 7
    assert result["scalar_mass_rel_drift"] == 0


def test_physical_status_error_is_mass_drift(monkeypatch):
    state = _constant_ic()
    state[..., 2] *= 1.1
    monkeypatch.setattr(forward, "safe_apply", lambda *args: state)
    result = forward.physical_laws(None, _context())["metrics"]
    assert result["valid"]
    assert result["scalar_mass_rel_drift"] == pytest.approx(0.1)
    assert result["error"] == result["scalar_mass_rel_drift"]
    assert result["scalar_min"] == pytest.approx(1.1)
    assert result["scalar_max"] == pytest.approx(1.1)


def test_nonfinite_diagnostic_state_is_rejected():
    state = _constant_ic()
    state[0, 0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        forward.physical_diagnostics(state, _constant_ic())


def test_late_failed_point_is_not_hidden_by_earlier_success(monkeypatch):
    from mosaic.benchmarks.core.status import FAILED, _classify_from_v1

    monkeypatch.setattr(
        forward, "get_last_apply_error", lambda: "late reference failed"
    )
    failure = forward._failure("refined")["metrics"]
    data = {
        "results": [
            {
                "solver": "solver",
                "sweep_value": "16",
                "metrics": {"valid": True, "error": 0.01},
            },
            {"solver": "solver", "sweep_value": "32", "metrics": failure},
        ]
    }
    assert _classify_from_v1(data, ["solver"], [])["solver"].status == FAILED
