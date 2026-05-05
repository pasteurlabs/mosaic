"""Unit tests for the shared exclusion-key matching used by both
``active_solvers`` (runtime gating) and ``core.status`` (display).

Run directly with::

    conda run -n gym python -m benchmarks.core.tests.test_exclusions

The root bug these tests guard against (ARCH-20): a solver could RUN in a
specific experiment whose status display said it was excluded — because
``active_solvers`` only matched the leading suite key (e.g. ``"gradient"``)
while ``core/status.py`` matched the full ``"suite/experiment"`` key. Both
paths now share ``exclusion_lookup``, and these tests lock in the
most-specific-first precedence so the two can never drift again.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from mosaic.benchmarks.core.utils import (
    active_solvers,
    exclusion_candidate_keys,
    exclusion_lookup,
)


@dataclass
class _FakeSpec:
    """Minimal stand-in for ``SolverSpec`` — only fields we need here."""

    exclusions: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeCfg:
    """Minimal stand-in for ``ProblemConfig``."""

    solvers: dict[str, _FakeSpec] = field(default_factory=dict)


class TestExclusionCandidateKeys(unittest.TestCase):
    """Ordering of candidate keys: most-specific first."""

    def test_suite_only(self) -> None:
        self.assertEqual(
            exclusion_candidate_keys("recovery"),
            ("recovery",),
        )

    def test_suite_and_experiment(self) -> None:
        self.assertEqual(
            exclusion_candidate_keys("recovery", "drag_opt"),
            ("recovery/drag_opt", "drag_opt", "recovery"),
        )

    def test_suite_experiment_sub(self) -> None:
        # experiment="agreement", sub="tgv" — both granularities visited.
        keys = exclusion_candidate_keys("forward", "agreement", "tgv")
        self.assertEqual(
            keys,
            (
                "forward/agreement/tgv",
                "agreement/tgv",
                "forward/agreement",
                "agreement",
                "forward",
            ),
        )

    def test_inline_subdir_in_experiment(self) -> None:
        # Caller passes "agreement/tgv" inline instead of experiment+sub.
        keys = exclusion_candidate_keys("forward", "agreement/tgv")
        self.assertEqual(
            keys,
            (
                "forward/agreement/tgv",
                "agreement/tgv",
                "forward/agreement",
                "agreement",
                "forward",
            ),
        )

    def test_dedupes_when_suite_equals_experiment(self) -> None:
        # "recovery/recovery" is a real key in configs — candidate_keys should
        # produce "recovery/recovery", "recovery" (no dup suite at end).
        keys = exclusion_candidate_keys("recovery", "recovery")
        self.assertEqual(keys, ("recovery/recovery", "recovery"))


class TestExclusionLookup(unittest.TestCase):
    """Precedence of exclusion_lookup against realistic configs."""

    def test_none_when_no_exclusions(self) -> None:
        self.assertIsNone(exclusion_lookup({}, "recovery", "drag_opt"))

    def test_most_specific_wins_over_suite(self) -> None:
        # Both keys present; the narrower one must win.
        exclusions = {
            "recovery": {"category": "categorical", "reason": "suite-level"},
            "recovery/drag_opt": {"category": "unstable", "reason": "narrow"},
        }
        match = exclusion_lookup(exclusions, "recovery", "drag_opt")
        self.assertIsNotNone(match)
        key, value = match
        self.assertEqual(key, "recovery/drag_opt")
        self.assertEqual(value["category"], "unstable")

    def test_falls_back_to_suite_when_experiment_missing(self) -> None:
        # recovery/recovery (a real ns-grid config shape for su2): the wide
        # "recovery" key isn't set, but "recovery/recovery" is.
        exclusions = {
            "recovery/recovery": {
                "category": "categorical",
                "reason": "no IC-level VJP",
            },
            "recovery/drag_opt": {
                "category": "categorical",
                "reason": "separate",
            },
        }
        # "recovery/recovery" experiment — matches exact narrow key.
        match = exclusion_lookup(exclusions, "recovery", "recovery")
        self.assertIsNotNone(match)
        self.assertEqual(match[0], "recovery/recovery")
        # "drag_opt" experiment — matches the drag-opt-specific key, not the
        # IC-recovery one.
        match = exclusion_lookup(exclusions, "recovery", "drag_opt")
        self.assertIsNotNone(match)
        self.assertEqual(match[0], "recovery/drag_opt")

    def test_suite_level_gates_all_experiments(self) -> None:
        # SU2's "gradient" suite exclusion: every gradient experiment should
        # resolve to the suite-level key when no narrower entry exists.
        exclusions = {
            "gradient": {"category": "categorical", "reason": "no IC VJP"},
        }
        for exp in ("fd_check", "param_sweep", "horizon_sweep", "source_fd_check"):
            match = exclusion_lookup(exclusions, "gradient", exp)
            self.assertIsNotNone(match, f"{exp}: expected suite-level match")
            self.assertEqual(match[0], "gradient")

    def test_bare_experiment_key_legacy(self) -> None:
        # Legacy configs wrote bare "lid_cavity" without a suite prefix.
        exclusions = {"lid_cavity": {"category": "not_implemented"}}
        match = exclusion_lookup(exclusions, "recovery", "lid_cavity")
        self.assertIsNotNone(match)
        self.assertEqual(match[0], "lid_cavity")

    def test_no_match_when_unrelated(self) -> None:
        exclusions = {"cost/spatial_cost": {"category": "infeasible"}}
        self.assertIsNone(exclusion_lookup(exclusions, "gradient", "fd_check"))
        self.assertIsNone(exclusion_lookup(exclusions, "cost", "temporal_cost"))


class TestActiveSolvers(unittest.TestCase):
    """The scenario ARCH-20 guards against: runtime filter and display agree."""

    def test_su2_style_suite_level_gating(self) -> None:
        # Reproduces the ARCH-18 / ARCH-20 scenario from navier_stokes_grid.py
        # (su2 "gradient" exclusion). Every gradient experiment must gate.
        cfg = _FakeCfg(
            solvers={
                "jax_cfd": _FakeSpec(exclusions={}),
                "su2": _FakeSpec(
                    exclusions={
                        "gradient": {
                            "category": "categorical",
                            "reason": "no IC VJP",
                        },
                    }
                ),
            }
        )
        # Before ARCH-20: this would INCORRECTLY include su2 because the old
        # code did `spec.exclusions.get("fd_check")`.
        active = active_solvers(cfg, "gradient", "fd_check")
        self.assertEqual(active, ["jax_cfd"])
        # Also with suite alone.
        self.assertEqual(active_solvers(cfg, "gradient"), ["jax_cfd"])

    def test_recovery_split_exclusions(self) -> None:
        # Mirrors navier_stokes_grid.py su2: `recovery/recovery` is excluded
        # (categorical — no IC VJP) but `recovery/drag_opt` is NOT (the
        # tesseract provides inflow-profile adjoint via SU2_CFD_AD).
        cfg = _FakeCfg(
            solvers={
                "jax_cfd": _FakeSpec(exclusions={}),
                "su2": _FakeSpec(
                    exclusions={
                        "recovery/recovery": {
                            "category": "categorical",
                            "reason": "no IC VJP",
                        },
                    }
                ),
            }
        )
        # The contrived scenario from the task description:
        self.assertEqual(
            active_solvers(cfg, "recovery", "recovery"),
            ["jax_cfd"],
            "recovery/recovery must exclude su2 via the narrow key",
        )
        # drag_opt has no matching exclusion — su2 runs.
        self.assertEqual(
            active_solvers(cfg, "recovery", "drag_opt"),
            ["jax_cfd", "su2"],
            "recovery/drag_opt must NOT exclude su2 — narrow key only gates recovery",
        )

    def test_narrow_key_wins_over_broad_category(self) -> None:
        # A solver that's broadly "recovery"-excluded but explicitly enabled
        # on one experiment via the narrow key. The current helper picks the
        # NARROW key's value (either exclusion flavour), so this pattern only
        # lets users SWITCH CATEGORIES, not un-exclude. We still verify the
        # narrow category wins so downstream category-based logic
        # (status glyph, permanence) sees the intended value.
        cfg = _FakeCfg(
            solvers={
                "foo": _FakeSpec(
                    exclusions={
                        "recovery": {"category": "infeasible", "reason": "slow"},
                        "recovery/drag_opt": {
                            "category": "not_implemented",
                            "reason": "BCs missing",
                        },
                    }
                ),
            }
        )
        # foo is excluded in both cases but the CATEGORY differs.
        self.assertEqual(active_solvers(cfg, "recovery", "recovery"), [])
        self.assertEqual(active_solvers(cfg, "recovery", "drag_opt"), [])
        # The value returned by exclusion_lookup reflects the narrow key.
        spec = cfg.solvers["foo"]
        _, drag_val = exclusion_lookup(spec.exclusions, "recovery", "drag_opt")
        self.assertEqual(drag_val["category"], "not_implemented")
        _, rec_val = exclusion_lookup(spec.exclusions, "recovery", "recovery")
        # "recovery/recovery" isn't set → falls back to "recovery" (the suite).
        self.assertEqual(rec_val["category"], "infeasible")


if __name__ == "__main__":
    unittest.main()
