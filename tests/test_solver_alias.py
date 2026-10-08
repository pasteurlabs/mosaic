# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for domain-aware solver alias resolution.

Structural and thermal solvers share display labels ("FEniCS", "Firedrake",
"deal.II"), so a bare lookup resolves to whichever SOLVER_STYLES entry comes
first — and plots that filter the result against a domain order list then
silently drop the solver. ``prefer`` breaks the tie toward the caller's domain.
"""

from __future__ import annotations

from mosaic.benchmarks.problems.shared.plots.style import (
    STRUCTURAL_ORDER,
    THERMAL_ORDER,
    resolve_solver_alias,
    solver_order_for_problem,
)


class TestResolveSolverAlias:
    def test_alias_key_returned_as_is(self):
        assert resolve_solver_alias("openfoam") == "openfoam"

    def test_prefer_breaks_label_tie_toward_thermal(self):
        for label, expected in [
            ("FEniCS", "fenics_heat"),
            ("Firedrake", "firedrake_heat"),
            ("deal.II", "dealii_heat"),
        ]:
            alias = resolve_solver_alias(label, prefer=THERMAL_ORDER)
            assert alias == expected
            assert alias in THERMAL_ORDER

    def test_prefer_breaks_label_tie_toward_structural(self):
        for label, expected in [
            ("FEniCS", "fenics_structural"),
            ("Firedrake", "firedrake_structural"),
            ("deal.II", "dealii_structural"),
        ]:
            alias = resolve_solver_alias(label, prefer=STRUCTURAL_ORDER)
            assert alias == expected
            assert alias in STRUCTURAL_ORDER

    def test_prefer_is_noop_for_unambiguous_labels(self):
        assert resolve_solver_alias("JAX-FEM", prefer=THERMAL_ORDER) == "jax_fem"
        assert resolve_solver_alias("torch-fem", prefer=THERMAL_ORDER) == (
            "torch_fem_thermal"
        )
        assert resolve_solver_alias("OpenFOAM") == "openfoam"

    def test_unknown_name_returns_none(self):
        assert resolve_solver_alias("no-such-solver", prefer=THERMAL_ORDER) is None


class TestSolverOrderForProblem:
    def test_domain_pick(self):
        assert solver_order_for_problem("thermal-mesh") is THERMAL_ORDER
        assert solver_order_for_problem("structural-mesh") is STRUCTURAL_ORDER
        # NS problems (and unknown names) fall back to the fluid ordering.
        assert "jax_cfd" in solver_order_for_problem("ns-grid")
        assert "jax_cfd" in solver_order_for_problem("ns-3d-grid")


def test_baseline_plot_includes_surrogate_curve_and_legend(tmp_path):
    import json

    import matplotlib.pyplot as plt
    import numpy as np

    from mosaic.benchmarks.problems import get_config
    from mosaic.benchmarks.problems.shared.plots import forward

    data = {
        "sweep_key": "N",
        "reference_label": "analytic",
        "by_param": {
            str(n): {"XLB 3D surrogate": {"error": error}}
            for n, error in ((8, 0.06), (16, 0.05), (32, 0.04))
        },
    }
    (tmp_path / "result.json").write_text(json.dumps(data))
    fig = forward._agreement_figure(
        get_config("ns-3d-grid"),
        exp_key="baseline",
        suffix="",
        save=False,
        out_dir=tmp_path,
    )
    try:
        assert len(fig.axes[0].lines) == 1
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [8, 16, 32])
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [0.06, 0.05, 0.04])
        assert [t.get_text() for t in fig.legends[0].get_texts()] == [
            "XLB 3D surrogate"
        ]
    finally:
        plt.close(fig)
