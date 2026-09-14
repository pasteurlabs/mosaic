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
