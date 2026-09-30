# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thermal-mesh conductivity-recovery kernel.

The Tesseract call is replaced by a loss that is minimised exactly at the
ground-truth density, so these tests exercise the kernel's own optimisation
plumbing (bounds, reparametrisation, reported metrics) without a solver.
"""

from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np
import pytest

import mosaic.benchmarks.problems.thermal_mesh.optimization as recovery
from mosaic.benchmarks.problems.thermal_mesh import physics
from mosaic.benchmarks.problems.thermal_mesh.config import problem
from mosaic.benchmarks.problems.thermal_mesh.ics import _two_gaussians

_PHYS = {
    "nx": 8,
    "ny": 4,
    "nz": 1,
    "Lx": 2.0,
    "Ly": 1.0,
    "rho_0": 0.5,
    "Q_total": 1.0,
    "compliance_key": "identification_error",
    "penalty_weight": 0.0,
    "x_min": 1e-3,
    "snap_interval": 5,
    "target_rho_from_two_gaussians": True,
}


def _run_kernel(monkeypatch, optimizer: str, max_iters: int) -> dict:
    rho_truth = jnp.asarray(_two_gaussians(**_PHYS))

    def fake_apply(_t, inp):
        return {"identification_error": jnp.sum((inp["rho"] - rho_truth) ** 2)}

    monkeypatch.setattr(recovery, "apply_tesseract", fake_apply)
    spec = types.SimpleNamespace(input_overrides={})
    ctx = types.SimpleNamespace(
        name="fake",
        cfg=problem,
        seed=0,
        run={
            "ic": {"name": "uniform", "seed": 0},
            "optimizer": optimizer,
            "optim": {"max_iters": max_iters},
            "physics": _PHYS,
        },
        make_inputs=lambda _name, rho, **kw: physics.make_inputs(spec, rho, **kw),
    )
    return recovery.conductivity_recovery(None, ctx)


@pytest.mark.parametrize("optimizer", ["bfgs", "adam"])
def test_recovery_reports_field_error_and_respects_bounds(monkeypatch, optimizer):
    out = _run_kernel(monkeypatch, optimizer, max_iters=100)
    m = out["metrics"]

    assert m["final_field_error"] < m["initial_field_error"]
    for rho in [out["snapshots"]["rho_final"], *out["snapshots"]["rho_history"]]:
        assert rho.min() >= _PHYS["x_min"] - 1e-6
        assert rho.max() <= 1.0 + 1e-6


def test_bfgs_recovers_the_truth(monkeypatch):
    """With a loss minimised at the truth, L-BFGS should land on it.

    Also checks that snapshots hold densities rather than the latent variable.
    """
    out = _run_kernel(monkeypatch, "bfgs", max_iters=100)
    assert out["metrics"]["final_field_error"] < 1e-2
    np.testing.assert_allclose(
        out["snapshots"]["rho_history"][-1], out["snapshots"]["rho_final"], atol=1e-2
    )
