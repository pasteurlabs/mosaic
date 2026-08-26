# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the built-in status-check factories.

The module contract is stated in ``core/status_checks.py``: a check returns
``None``, or ``(status, reason)`` with ``status == "anomaly"``. Only the
exact string ``"anomaly"`` counts — ``_run_checks`` compares against
``status.ANOMALY``, so a check returning any other spelling is silently
inert: it reports no anomaly and the cell publishes as ``ok``.

Nothing enforced that contract, so the tests here pin:

  1. Every built-in factory, when it fires, reports ``ANOMALY`` — and the
     case table is asserted to cover every public factory, so a new check
     added without a case fails rather than slipping through untested.
  2. Every factory stays quiet when its threshold is not crossed.
  3. End-to-end: an anti-aligned gradient reaching ``_refine_fd_check``
     turns the cell into an anomaly.
  4. The blind spot ``min_cosine`` exists to cover: ``_refine_fd_check``
     aggregates rel_error as the *median* across FD directions, so an error
     concentrated in a minority of directions leaves the median clean and
     only the cosine check can see it.
"""

from __future__ import annotations

import unittest
from typing import ClassVar

from mosaic.benchmarks.core import status_checks
from mosaic.benchmarks.core.config import Problem
from mosaic.benchmarks.core.experiment import kernel
from mosaic.benchmarks.core.status import (
    ANOMALY,
    OK,
    Cell,
    _lookup_check,
    _refine_fd_check,
    _refine_for_suite,
    _run_checks,
)
from mosaic.benchmarks.core.status_checks import (
    CostSummary,
    FdCheckSummary,
    ForwardSummary,
    OptimizationSummary,
    max_error,
    max_final_ratio,
    max_peer_k,
    max_rel_err,
    median_k,
    min_cosine,
    peer_final_loss_k,
    rel_err_peer_outlier,
)

# One entry per built-in factory: (name, firing check, summary that trips it,
# summary that must not trip it). ``name`` matches the factory's __name__ so
# the coverage test below can diff this table against the module.
CHECK_CASES = [
    (
        "median_k",
        median_k(3.0),
        ForwardSummary(
            errs_by_pval={"16": 10.0, "32": 12.0},
            peer_medians_by_pval={"16": 1.0, "32": 1.0},
            n_valid_points=2,
        ),
        ForwardSummary(
            errs_by_pval={"16": 1.0, "32": 1.1},
            peer_medians_by_pval={"16": 1.0, "32": 1.0},
            n_valid_points=2,
        ),
    ),
    (
        "max_error",
        max_error(0.5),
        ForwardSummary(errs_by_pval={"16": 0.9}, n_valid_points=1),
        ForwardSummary(errs_by_pval={"16": 0.1}, n_valid_points=1),
    ),
    (
        "max_peer_k",
        max_peer_k(20.0),
        CostSummary(solver_median_time=100.0, peer_median_time=1.0),
        CostSummary(solver_median_time=2.0, peer_median_time=1.0),
    ),
    (
        "min_cosine",
        min_cosine(0.99),
        FdCheckSummary(best_cosine=-0.885),
        FdCheckSummary(best_cosine=0.9999),
    ),
    (
        "max_rel_err",
        max_rel_err(1e-3),
        FdCheckSummary(best_rel_err=0.5),
        FdCheckSummary(best_rel_err=1e-6),
    ),
    (
        "rel_err_peer_outlier",
        rel_err_peer_outlier(50.0),
        FdCheckSummary(best_rel_err=1.0, peer_rel_err_median=1e-3),
        FdCheckSummary(best_rel_err=1e-3, peer_rel_err_median=1e-3),
    ),
    (
        "max_final_ratio",
        max_final_ratio(0.5),
        OptimizationSummary(final_initial_ratio=1.2),
        OptimizationSummary(final_initial_ratio=0.1),
    ),
    (
        "peer_final_loss_k",
        peer_final_loss_k(5.0),
        OptimizationSummary(peer_final_loss_by_sweep={"0.01": 40.0}),
        OptimizationSummary(peer_final_loss_by_sweep={"0.01": 1.2}),
    ),
]


class TestCheckOutcomeContract(unittest.TestCase):
    """Every built-in check must speak the status string the classifier reads."""

    def test_firing_check_reports_the_anomaly_status(self) -> None:
        for name, check, firing, _quiet in CHECK_CASES:
            with self.subTest(check=name):
                outcome = check(firing)
                self.assertIsNotNone(
                    outcome, f"{name} did not fire on a summary that should trip it"
                )
                self.assertEqual(
                    outcome[0],
                    ANOMALY,
                    f"{name} reports {outcome[0]!r}; _run_checks only acts on "
                    f"{ANOMALY!r}, so any other spelling is silently inert",
                )
                self.assertTrue(outcome[1], f"{name} fired with an empty reason")

    def test_check_stays_quiet_below_its_threshold(self) -> None:
        for name, check, _firing, quiet in CHECK_CASES:
            with self.subTest(check=name):
                self.assertIsNone(
                    check(quiet), f"{name} fired on a summary within threshold"
                )

    def test_case_table_covers_every_built_in_factory(self) -> None:
        """A new factory without a case here fails, rather than going untested."""
        documented = {name for name, *_ in CHECK_CASES}
        public = {
            name
            for name in dir(status_checks)
            if not name.startswith("_")
            and name != "normalize"
            and callable(getattr(status_checks, name))
            and getattr(getattr(status_checks, name), "__module__", "")
            == status_checks.__name__
            and not isinstance(getattr(status_checks, name), type)
        }
        self.assertEqual(
            public - documented,
            set(),
            "built-in check factories missing a case in CHECK_CASES",
        )


class TestFdCheckPipeline(unittest.TestCase):
    """``_refine_fd_check`` must actually downgrade the cell, not just compute."""

    @staticmethod
    def _cell(eps_sweep: dict, checks: list) -> Cell:
        cells = {"solver_a": Cell(OK)}
        _refine_fd_check(
            {"by_solver": {"solver_a": {"eps_sweep": eps_sweep}}}, cells, checks
        )
        return cells["solver_a"]

    def test_anti_aligned_gradient_becomes_an_anomaly(self) -> None:
        cell = self._cell(
            {"0.01": {"rel_error": [0.4, 0.5], "cosine": -0.885}},
            [min_cosine(0.99)],
        )
        self.assertEqual(cell.status, ANOMALY)
        self.assertIn("cosine", cell.reason)

    def test_aligned_gradient_stays_ok(self) -> None:
        cell = self._cell(
            {"0.01": {"rel_error": [1e-6], "cosine": 0.99999}},
            [min_cosine(0.99)],
        )
        self.assertEqual(cell.status, OK)

    def test_min_cosine_covers_the_median_blind_spot(self) -> None:
        """An error in a minority of directions is invisible to the rel_err gates.

        ``_refine_fd_check`` takes the median across FD directions, which is
        deliberate — it refuses to let one lucky direction speak for the
        gradient. The cost is that it tolerates a minority of arbitrarily
        wrong directions. Here 8 of 20 directions are badly wrong while the
        median stays at 0.0, so ``max_rel_err`` sees nothing; the whole-vector
        cosine is what catches it.
        """
        rel_error = [0.0] * 12 + [3.0] * 8
        sweep = {"0.01": {"rel_error": rel_error, "cosine": -0.885}}

        self.assertEqual(self._cell(sweep, [max_rel_err(1e-3)]).status, OK)
        self.assertEqual(self._cell(sweep, [min_cosine(0.99)]).status, ANOMALY)


if __name__ == "__main__":
    unittest.main()


class TestForwardPipeline:
    """The forward suite must actually apply its configured checks.

    ``ForwardSummary`` had no producer: ``_refine_for_suite`` dispatched cost,
    gradient and optimization but not forward, and ``_classify_from_v1`` takes
    a ``checks`` argument it never reads. So every ``"forward"`` entry in a
    problem config was inert, including the two in ns-grid.
    """

    DATA: ClassVar[dict] = {
        "by_solver": {
            "peer_a": {
                "16": {"error": 1.0, "valid": True},
                "32": {"error": 1.0, "valid": True},
            },
            "peer_b": {
                "16": {"error": 1.0, "valid": True},
                "32": {"error": 1.0, "valid": True},
            },
            "outlier": {
                "16": {"error": 10.0, "valid": True},
                "32": {"error": 12.0, "valid": True},
            },
        }
    }

    def _run(self, checks, data=None):
        data = self.DATA if data is None else data
        cells = {s: Cell(OK) for s in data["by_solver"]}
        _refine_for_suite("forward", "agreement", data, cells, checks)
        return cells

    def test_peer_outlier_is_flagged(self) -> None:
        cells = self._run([median_k(3.0)])
        assert cells["outlier"].status == ANOMALY
        assert cells["peer_a"].status == OK
        assert cells["peer_b"].status == OK

    def test_absolute_threshold_is_applied(self) -> None:
        cells = self._run([max_error(0.5)])
        assert all(c.status == ANOMALY for c in cells.values())

    def test_no_checks_is_a_noop(self) -> None:
        assert all(c.status == OK for c in self._run([]).values())

    def test_invalid_points_are_ignored(self) -> None:
        data = {
            "by_solver": {
                "peer_a": {"16": {"error": 1.0, "valid": True}},
                "peer_b": {"16": {"error": 1.0, "valid": True}},
                # its only bad point is marked invalid, so nothing should fire
                "outlier": {
                    "16": {"error": 1.0, "valid": True},
                    "32": {"error": 99.0, "valid": False},
                },
            }
        }
        assert self._run([median_k(3.0), max_error(2.0)], data)["outlier"].status == OK

    def test_peer_median_needs_two_solvers(self) -> None:
        """A lone solver has no peers, so a relative check must stay quiet."""
        data = {"by_solver": {"only": {"16": {"error": 99.0, "valid": True}}}}
        assert self._run([median_k(3.0)], data)["only"].status == OK

    def test_non_swept_layout_is_handled(self) -> None:
        """Non-swept forward results are a flat metrics dict per solver."""
        data = {
            "by_solver": {
                "peer_a": {"error": 1.0, "valid": True},
                "peer_b": {"error": 1.0, "valid": True},
                "outlier": {"error": 50.0, "valid": True},
            }
        }
        assert self._run([median_k(3.0)], data)["outlier"].status == ANOMALY


@kernel(sweep_mode="none")
def _noop(t, ctx) -> dict:
    return {"metrics": {}}


class TestLookupCheckPrecedence:
    """A more-specific status-check source *replaces* the suite default.

    The regression this pins: ns-grid gives ``forward/agreement/multimode`` a
    looser ``max_error(1.5)`` override, but ``_lookup_check`` used to *append*
    it to the suite default ``max_error(0.5)``. Since ``_run_checks``
    short-circuits on the first anomaly, the stricter default fired first and
    the override never applied — flagging every solver at error 1.03. Sources
    must replace, not accumulate.
    """

    # An error that trips the strict default (0.5) but not the loose override
    # (1.5) — the exact multimode case from the v0.2.0 run.
    SUMMARY: ClassVar[ForwardSummary] = ForwardSummary(
        errs_by_pval={"0.001": 1.03},
        peer_medians_by_pval={"0.001": 1.03},
        n_valid_points=1,
    )

    def _problem(self, status_checks: dict) -> Problem:
        return Problem(name="t", status_checks=status_checks)

    def test_suite_default_applies_without_an_override(self) -> None:
        p = self._problem({"forward": [median_k(3.0), max_error(0.5)]})
        checks = _lookup_check(p, "forward", "agreement/tgv")
        assert _run_checks(checks, self.SUMMARY) is not None

    def test_per_ic_override_replaces_the_suite_default(self) -> None:
        # The looser per-IC override must win outright — no strict default
        # left behind to fire first.
        p = self._problem(
            {
                "forward": [median_k(3.0), max_error(0.5)],
                "forward/agreement/multimode": [median_k(3.0), max_error(1.5)],
            }
        )
        checks = _lookup_check(p, "forward", "agreement/multimode")
        assert len(checks) == 2
        assert _run_checks(checks, self.SUMMARY) is None

    def test_full_label_beats_leading_token(self) -> None:
        # A per-IC key ("agreement/multimode") is more specific than a
        # per-experiment key ("agreement") and must take precedence.
        p = self._problem(
            {
                "forward/agreement": [max_error(0.5)],
                "forward/agreement/multimode": [max_error(1.5)],
            }
        )
        checks = _lookup_check(p, "forward", "agreement/multimode")
        assert _run_checks(checks, self.SUMMARY) is None

    def test_inline_override_replaces_dict_sources(self) -> None:
        p = self._problem({"forward": [max_error(0.5)]})
        p.add_experiment(
            "forward/loose", _noop, physics={"N": 16}, status_check=[max_error(1.5)]
        )
        checks = _lookup_check(p, "forward", "loose")
        assert _run_checks(checks, self.SUMMARY) is None

    def test_no_config_means_no_checks(self) -> None:
        assert _lookup_check(self._problem({}), "forward", "agreement/tgv") == []
