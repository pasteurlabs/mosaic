# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""vjp_cost times the vector_jacobian_product request alone."""

from __future__ import annotations

import numpy as np

from mosaic.benchmarks.core.experiment import KernelContext
from mosaic.benchmarks.problems.shared.cost import vjp_cost


class _SquareTesseract:
    """out = x**2, counting requests per endpoint."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def apply(self, inputs: dict) -> dict:
        self.calls.append("apply")
        return {"out": np.asarray(inputs["x"]) ** 2}

    def vector_jacobian_product(
        self,
        inputs: dict,
        vjp_inputs: list[str],
        vjp_outputs: list[str],
        cotangent_vector: dict,
    ) -> dict:
        self.calls.append("vjp")
        assert vjp_inputs == ["x"]
        assert vjp_outputs == ["out"]
        return {"x": cotangent_vector["out"] * 2.0 * np.asarray(inputs["x"])}


def _ctx(x: np.ndarray, n_trials: int) -> KernelContext:
    return KernelContext(
        name="square",
        cfg=None,
        ic=x,
        phys={},
        sweep_key=None,
        sweep_value=None,
        ic_key="x",
        output_key="out",
        make_inputs=lambda _name, ic, **_kw: {"x": np.asarray(ic)},
        domain_extent=1.0,
        run={"cost": {"n_trials": n_trials}},
    )


def test_vjp_cost_times_only_the_vjp_request():
    t = _SquareTesseract()
    x = np.array([0.5, -1.0, 2.0])

    result = vjp_cost(t, _ctx(x, n_trials=3))

    # One untimed apply for the cotangent, then warmup + 3 timed VJPs.
    assert t.calls == ["apply", "vjp", "vjp", "vjp", "vjp"]
    assert len(result["metrics"]["trials_s"]) == 3
    # Gradient of sum(out**2) = sum(x**4), as the gradient suite takes it.
    np.testing.assert_allclose(result["snapshot"], 4.0 * x**3)
