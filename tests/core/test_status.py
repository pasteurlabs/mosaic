# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the weighted campaign-health score.

Replaces the binary ``% ok`` metric with a per-cell weighted sum. See
``core/status.py::SCORE_WEIGHTS`` for the table. The tests here pin:

  1. Extremes: all-ok → +1.0, all-fail → 0.0, empty → None.
  2. Monotonicity: a cell going ok → fail moves the score by exactly 1.0/N.
  3. Staleness invisibility: the ``*`` marker never moves the score — a
     stale cell scores identically to its fresh counterpart.
  4. Categorical exclusions are off the books (numerator + denominator).
  5. Boundary agreement with ``% ok``: all-ok ⇒ score=1.0 & pct=100;
     all non-ok (fail/anom/missing) ⇒ score in [0, 0.53] & pct=0.
"""

from __future__ import annotations

import unittest

from mosaic.benchmarks.core.config import ExclusionCategory
from mosaic.benchmarks.core.status import (
    ANOMALY,
    EXCLUDED,
    FAILED,
    NOT_RUN,
    OK,
    SCORE_WEIGHTS,
    Cell,
    ExperimentRow,
    ProblemStatus,
    cell_color,
    cell_weight,
    cell_weight_key,
    compute_score,
    tally,
    weight_color,
    weight_emoji,
)

# Shorthand: Cell.category stores the raw string value of an ExclusionCategory
# enum member, so tests can reference categories via the enum members.
EXCL_CATEGORICAL = ExclusionCategory.CATEGORICAL.value
EXCL_INFEASIBLE = ExclusionCategory.INFEASIBLE.value


def _status_from_cells(cells_by_solver: dict[str, Cell]) -> ProblemStatus:
    """Wrap a flat dict of cells into a single-row ProblemStatus."""
    row = ExperimentRow(
        suite="forward",
        experiment="agreement",
        result_path=None,
        cells=dict(cells_by_solver),
    )
    return ProblemStatus(
        problem="test", solvers=list(cells_by_solver.keys()), rows=[row]
    )


class TestScoreExtremes(unittest.TestCase):
    """Extremes and boundary conditions for compute_score / tally."""

    def test_all_ok_gives_plus_one(self) -> None:
        cells = [Cell(OK) for _ in range(5)]
        score, n = compute_score(cells)
        self.assertEqual(n, 5)
        self.assertAlmostEqual(score, 1.0)

    def test_all_fresh_fail_gives_zero(self) -> None:
        cells = [Cell(FAILED) for _ in range(4)]
        score, n = compute_score(cells)
        self.assertEqual(n, 4)
        self.assertAlmostEqual(score, 0.0)

    def test_empty_denominator_returns_none(self) -> None:
        """No cells, and all-categorical input, both return None."""
        self.assertEqual(compute_score([]), (None, 0))
        cat_cells = [Cell(EXCLUDED, category=EXCL_CATEGORICAL) for _ in range(3)]
        self.assertEqual(compute_score(cat_cells), (None, 0))

    def test_tally_score_matches_compute_score(self) -> None:
        """The score surfaced via tally() matches the direct helper."""
        st = _status_from_cells(
            {
                "a": Cell(OK),
                "b": Cell(OK, stale=True),
                "c": Cell(FAILED),
            }
        )
        t = tally(st)
        direct, _ = compute_score([c for r in st.rows for c in r.cells.values()])
        self.assertAlmostEqual(t["score"], direct)
        self.assertEqual(t["score_n"], 3)

    def test_score_range_never_escapes_bounds(self) -> None:
        """Any mix of cells yields a score in [min(weights), max(weights)]."""
        lo = min(SCORE_WEIGHTS.values())
        hi = max(SCORE_WEIGHTS.values())
        mixes: list[list[Cell]] = [
            [Cell(OK), Cell(ANOMALY), Cell(FAILED)],
            [Cell(OK, stale=True), Cell(ANOMALY, stale=True), Cell(FAILED, stale=True)],
            [Cell(NOT_RUN), Cell(EXCLUDED, category=EXCL_INFEASIBLE)],
        ]
        for mix in mixes:
            score, _ = compute_score(mix)
            if score is None:
                continue
            self.assertGreaterEqual(score, lo)
            self.assertLessEqual(score, hi)


class TestScoreTransitions(unittest.TestCase):
    """Direction and magnitude of single-cell status flips."""

    def test_stale_does_not_move_score(self) -> None:
        """The ``*`` marker is invisible to the score.

        Staleness is a "re-run recommended" hint for humans, not a health
        signal — a stale cell scores identically to its fresh counterpart,
        for every underlying status. This is the invariant that keeps the
        headline score from drifting on the incremental PR-preview path when
        carried-over baselines hash-mismatch the current tree.
        """
        for status in (OK, ANOMALY, FAILED):
            fresh, _ = compute_score([Cell(status) for _ in range(4)])
            stale, _ = compute_score([Cell(status, stale=True) for _ in range(4)])
            self.assertAlmostEqual(fresh, stale, msg=f"stale moved score for {status}")

    def test_ok_to_fail_decreases_by_one_over_n(self) -> None:
        """Ok (1.0) → fail (0.0) shifts average by −1.0/N."""
        n = 6
        before = [Cell(OK) for _ in range(n)]
        after = [Cell(FAILED)] + [Cell(OK) for _ in range(n - 1)]
        s_before, _ = compute_score(before)
        s_after, _ = compute_score(after)
        self.assertAlmostEqual(s_before - s_after, 1.0 / n)
        self.assertLess(s_after, s_before)

    def test_transition_across_all_weight_keys(self) -> None:
        """Every mapped weight key is reachable via some (status, category)
        combination — catches silent dead keys.

        Stale probes are included to pin that they collapse onto the same
        (non-star) keys as their fresh counterparts.
        """
        produced: set[str] = set()
        probes: list[Cell] = [
            Cell(OK),
            Cell(OK, stale=True),
            Cell(ANOMALY),
            Cell(ANOMALY, stale=True),
            Cell(FAILED),
            Cell(FAILED, stale=True),
            Cell(NOT_RUN),
            Cell(EXCLUDED, category=EXCL_INFEASIBLE),
        ]
        for c in probes:
            key = cell_weight_key(c)
            self.assertIsNotNone(key, f"unmapped: {c}")
            self.assertNotIn("*", key, f"stale key leaked into score table: {key}")
            produced.add(key)  # type: ignore[arg-type]
        missing = set(SCORE_WEIGHTS.keys()) - produced
        self.assertFalse(
            missing, f"weight keys with no probe mapping: {sorted(missing)}"
        )


class TestCategoricalExclusions(unittest.TestCase):
    """Permanent (categorical) exclusions must not affect the score at all."""

    def test_categorical_excluded_from_denominator(self) -> None:
        """A categorical-excluded cell doesn't shift the score."""
        base = [Cell(OK), Cell(OK), Cell(FAILED)]
        with_cat = [*base, Cell(EXCLUDED, category=EXCL_CATEGORICAL)]
        s_base, n_base = compute_score(base)
        s_with, n_with = compute_score(with_cat)
        self.assertAlmostEqual(s_base, s_with)
        self.assertEqual(n_base, n_with)

    def test_non_categorical_exclusion_does_count(self) -> None:
        """Non-categorical exclusions ARE in the denominator.

        Adding one infeasible-excluded (weight 0.33) to an all-ok mix pulls
        the score down.
        """
        base = [Cell(OK), Cell(OK), Cell(OK)]
        with_slow = [*base, Cell(EXCLUDED, category=EXCL_INFEASIBLE)]
        s_base, n_base = compute_score(base)
        s_with, n_with = compute_score(with_slow)
        self.assertEqual(n_with, n_base + 1)
        self.assertLess(s_with, s_base)
        # Expected: (3·1.0 + 1·0.33) / 4 = 3.33 / 4 = 0.8325.
        self.assertAlmostEqual(s_with, (3.0 * 1.0 + 0.33) / 4.0)

    def test_tally_excl_perm_matches_score_n(self) -> None:
        """tally.score_n = total cells − categorical-excluded cells."""
        cells = {
            "a": Cell(OK),
            "b": Cell(FAILED),
            "c": Cell(EXCLUDED, category=EXCL_CATEGORICAL),
            "d": Cell(EXCLUDED, category=EXCL_CATEGORICAL),
            "e": Cell(EXCLUDED, category=EXCL_INFEASIBLE),
        }
        st = _status_from_cells(cells)
        t = tally(st)
        self.assertEqual(t["score_n"], 3)  # a, b, e
        self.assertEqual(t["excl_perm"], 2)


class TestScoreAgreementWithPctOk(unittest.TestCase):
    """On the binary boundary cases, weighted score and % ok must agree."""

    def test_all_ok_matches_pct_ok_100(self) -> None:
        st = _status_from_cells({s: Cell(OK) for s in ("a", "b", "c", "d")})
        t = tally(st)
        self.assertAlmostEqual(t["score"], 1.0)
        self.assertAlmostEqual(t["pct_ok"], 100.0)

    def test_all_non_ok_matches_pct_ok_0(self) -> None:
        # All-fail: pct_ok=0, score=0.0.
        st_f = _status_from_cells({s: Cell(FAILED) for s in ("a", "b", "c")})
        t_f = tally(st_f)
        self.assertAlmostEqual(t_f["pct_ok"], 0.0)
        self.assertAlmostEqual(t_f["score"], 0.0)

        # All-anom: pct_ok=0, score=0.53.
        st_a = _status_from_cells({s: Cell(ANOMALY) for s in ("a", "b", "c")})
        t_a = tally(st_a)
        self.assertAlmostEqual(t_a["pct_ok"], 0.0)
        self.assertAlmostEqual(t_a["score"], 0.53)
        self.assertLess(t_a["score"], 1.0)

        # All-missing: pct_ok=0, score=0.33.
        st_m = _status_from_cells({s: Cell(NOT_RUN) for s in ("a", "b", "c")})
        t_m = tally(st_m)
        self.assertAlmostEqual(t_m["pct_ok"], 0.0)
        self.assertAlmostEqual(t_m["score"], 0.33)


_HEX_RE = __import__("re").compile(r"^#[0-9a-f]{6}$")


def _red_amount(hex_color: str) -> int:
    return int(hex_color[1:3], 16)


def _green_amount(hex_color: str) -> int:
    return int(hex_color[3:5], 16)


def _blue_amount(hex_color: str) -> int:
    return int(hex_color[5:7], 16)


class TestWeightColorLadder(unittest.TestCase):
    """Unified weight → colour/emoji mapping."""

    def test_weight_color_positive_end_is_bright_green(self) -> None:
        c = weight_color(1.0)
        self.assertTrue(_HEX_RE.match(c))
        self.assertGreater(_green_amount(c), _red_amount(c))

    def test_weight_color_zero_is_red(self) -> None:
        c = weight_color(0.0)
        self.assertTrue(_HEX_RE.match(c))
        self.assertGreater(_red_amount(c), _green_amount(c))

    def test_weight_color_values_in_range(self) -> None:
        for w in (0.0, 0.17, 0.33, 0.43, 0.53, 0.67, 1.0):
            c = weight_color(w)
            self.assertTrue(_HEX_RE.match(c), f"invalid color for w={w}: {c!r}")

    def test_weight_color_clamping(self) -> None:
        self.assertEqual(weight_color(-1.0), weight_color(0.0))
        self.assertEqual(weight_color(2.0), weight_color(1.0))

    def test_weight_color_none(self) -> None:
        self.assertEqual(weight_color(None), "dim")

    def test_weight_emoji_matches_ladder(self) -> None:
        self.assertEqual(weight_emoji(1.0), "🟢")
        self.assertEqual(weight_emoji(0.67), "🟢")
        self.assertEqual(weight_emoji(0.65), "🟢")
        self.assertEqual(weight_emoji(0.53), "🟡")
        self.assertEqual(weight_emoji(0.33), "🟡")
        self.assertEqual(weight_emoji(0.30), "🟡")
        self.assertEqual(weight_emoji(0.17), "🟠")
        self.assertEqual(weight_emoji(0.15), "🟠")
        self.assertEqual(weight_emoji(0.0), "🔴")
        self.assertEqual(weight_emoji(None), "—")

    def test_cell_weight_and_color(self) -> None:
        c = Cell(OK)
        self.assertEqual(cell_weight(c), 1.0)
        self.assertTrue(_HEX_RE.match(cell_color(c)))

        # Staleness doesn't dim the weight — a stale ok weighs the same as a
        # fresh ok. The ``*`` glyph carries the "re-run" hint on its own.
        c = Cell(OK, stale=True)
        self.assertEqual(cell_weight(c), 1.0)

        c_fail = Cell(FAILED)
        self.assertEqual(cell_weight(c_fail), 0.0)
        c_fail_color = cell_color(c_fail)
        self.assertTrue(_HEX_RE.match(c_fail_color))
        self.assertGreater(_red_amount(c_fail_color), _green_amount(c_fail_color))

        c = Cell(NOT_RUN)
        self.assertEqual(cell_weight(c), 0.33)

        c = Cell(EXCLUDED, category=EXCL_CATEGORICAL)
        self.assertIsNone(cell_weight(c))
        self.assertEqual(cell_color(c), "dim")

    def test_every_score_weight_maps_to_a_colour(self) -> None:
        for w in SCORE_WEIGHTS.values():
            color = weight_color(w)
            self.assertTrue(
                color == "dim" or _HEX_RE.match(color),
                f"bad color for w={w}: {color!r}",
            )
            self.assertIn(weight_emoji(w), {"🟢", "🟡", "🟠", "🔴", "—"})


_FAIL_METRICS = {
    "status": "failed",
    "trials_s": [],
    "failure_type": "error",
    "exc_type": "RuntimeError",
    "exc_msg": "tile_fft only supports power-of-two sizes",
    "vram_peak_mib": 0.0,
    "ram_peak_mib": 812.0,
}
_OOM_METRICS = {
    "status": "failed",
    "trials_s": [],
    "failure_type": "OOM",
    "exc_type": "RuntimeError",
    "exc_msg": "RESOURCE_EXHAUSTED: Out of memory",
    "vram_peak_mib": 0.0,
    "ram_peak_mib": 800.0,
}
_OK_METRICS = {"mean": 0.1, "std": 0.0, "trials_s": [0.1], "vram_peak_mib": 10.0}
_SKIP_METRICS = {"status": "skipped", "reason": "skipped after wall-limit hit at N=256"}


class TestClassifyFromV1(unittest.TestCase):
    """Per-sweep-point classification of schema_version=1 cost results.

    Regression cover for PR #132: a solver that crashed at a subset of sweep
    points classified as OK because the crash record still carried finite
    ``vram_peak_mib``/``ram_peak_mib`` (so ``_has_any_finite`` returned True)
    and the classifier never consulted ``metrics["status"] == "failed"``.
    """

    # A failure record as written by TimedResult.as_record on a crash — note
    # the finite mem fields that used to fool _has_any_finite into "ok".
    FAIL = _FAIL_METRICS
    OK_M = _OK_METRICS
    SKIP = _SKIP_METRICS

    # Default sweep values line up with the ns-grid cost sweep so frontier
    # assertions read naturally.
    SWEEP = ("64", "128", "192", "256")

    @classmethod
    def _classify(cls, points: list[dict]):
        from mosaic.benchmarks.core.status import _classify_from_v1

        data = {
            "schema_version": 1,
            "results": [
                {"solver": "s", "sweep_value": cls.SWEEP[i], "metrics": m}
                for i, m in enumerate(points)
            ],
        }
        return _classify_from_v1(data, ["s"], [])["s"]

    def test_partial_crash_is_failed(self) -> None:
        # 64 ok, 128 ok, 192 FAIL, 256 never-ran ({}) — the PR #132 shape.
        cell = self._classify([self.OK_M, self.OK_M, self.FAIL, {}])
        self.assertEqual(cell.status, FAILED)
        self.assertIn("tile_fft", cell.reason)

    def test_all_ok_is_ok(self) -> None:
        self.assertEqual(self._classify([self.OK_M, self.OK_M]).status, OK)

    def test_wall_limit_skips_stay_ok(self) -> None:
        # A slow-but-working solver: real timings then deliberately-skipped
        # larger points. Skipped != failed.
        cell = self._classify([self.OK_M, self.OK_M, self.SKIP, self.SKIP])
        self.assertEqual(cell.status, OK)

    def test_all_skipped_is_not_run(self) -> None:
        self.assertEqual(self._classify([self.SKIP, self.SKIP]).status, NOT_RUN)

    def test_all_empty_is_failed(self) -> None:
        self.assertEqual(self._classify([{}, {}]).status, FAILED)

    def test_crash_at_first_point_is_failed(self) -> None:
        cell = self._classify([self.FAIL, {}, {}])
        self.assertEqual(cell.status, FAILED)
        self.assertIn("tile_fft", cell.reason)

    def test_oom_tail_stays_ok_with_frontier(self) -> None:
        # OOM only at the largest sizes is an expected, platform-dependent
        # resource ceiling — the solver worked everywhere it could. It stays
        # OK, but records the frontier for the diff.
        cell = self._classify([self.OK_M, self.OK_M, self._OOM(), self.SKIP])
        self.assertEqual(cell.status, OK)
        self.assertEqual(cell.resource_frontier, "192")

    def test_oom_everywhere_is_failed(self) -> None:
        # OOM at the very first size (no valid data at all) means the solver
        # can't run this experiment on this platform — surface as FAILED, but
        # keep the frontier.
        cell = self._classify([self._OOM(), self.SKIP])
        self.assertEqual(cell.status, FAILED)
        self.assertEqual(cell.resource_frontier, "64")

    def test_crash_beats_oom_for_reason(self) -> None:
        # A real crash anywhere makes the cell FAILED even if an OOM ceiling
        # is also present — the crash is the actionable signal.
        cell = self._classify([self.OK_M, self.FAIL, self._OOM(), self.SKIP])
        self.assertEqual(cell.status, FAILED)
        self.assertIn("tile_fft", cell.reason)

    @staticmethod
    def _OOM() -> dict:
        return dict(_OOM_METRICS)


class TestFrontierDiff(unittest.TestCase):
    """Resource-ceiling frontier moves surface in the snapshot diff even when
    both cells stay OK — the "did the OOM boundary move?" signal."""

    @staticmethod
    def _snap(frontier: str) -> dict:
        return {
            "score": 0.9,
            "problems": {
                "ns-grid": {
                    "problem": "ns-grid",
                    "solvers": ["warp_ns"],
                    "rows": [
                        {
                            "suite": "cost",
                            "experiment": "spatial_cost",
                            "label": "cost/spatial_cost",
                            "cells": {
                                "warp_ns": {
                                    "status": "ok",
                                    "reason": "",
                                    "category": "",
                                    "stale": False,
                                    "resource_frontier": frontier,
                                }
                            },
                        }
                    ],
                }
            },
        }

    def test_ceiling_moves_earlier_is_regression(self) -> None:
        from mosaic.benchmarks.core.status import diff_snapshots

        d = diff_snapshots(self._snap("256"), self._snap("128"))
        self.assertEqual(len(d["frontier_shifts"]), 1)
        self.assertEqual(d["frontier_shifts"][0]["direction"], "regression")

    def test_ceiling_moves_later_is_improvement(self) -> None:
        from mosaic.benchmarks.core.status import diff_snapshots

        d = diff_snapshots(self._snap("128"), self._snap("256"))
        self.assertEqual(d["frontier_shifts"][0]["direction"], "improvement")

    def test_no_frontier_change_is_silent(self) -> None:
        from mosaic.benchmarks.core.status import diff_snapshots

        d = diff_snapshots(self._snap("128"), self._snap("128"))
        self.assertEqual(d["frontier_shifts"], [])


class TestMetricDiff(unittest.TestCase):
    """Numeric-metric shifts surface between cells that stay OK in both
    snapshots — the signal the coarse status ladder hides."""

    @staticmethod
    def _snap(metrics: dict, status: str = "ok") -> dict:
        return {
            "score": 0.9,
            "problems": {
                "ns-grid": {
                    "problem": "ns-grid",
                    "solvers": ["exponax"],
                    "rows": [
                        {
                            "suite": "gradient",
                            "experiment": "fd_check",
                            "label": "gradient/fd_check",
                            "cells": {
                                "exponax": {
                                    "status": status,
                                    "reason": "",
                                    "category": "",
                                    "stale": False,
                                    "resource_frontier": "",
                                    "metrics": metrics,
                                }
                            },
                        }
                    ],
                }
            },
        }

    def _diff(self, old_m: dict, new_m: dict, **kw) -> dict:
        from mosaic.benchmarks.core.status import diff_snapshots

        return diff_snapshots(self._snap(old_m), self._snap(new_m, **kw))

    def test_firm_metric_worsening_past_threshold_is_regression(self) -> None:
        d = self._diff({"rel_err": 1e-4}, {"rel_err": 3e-4})
        self.assertEqual(len(d["metric_shifts"]), 1)
        s = d["metric_shifts"][0]
        self.assertEqual(s["metric"], "rel_err")
        self.assertEqual(s["direction"], "regression")
        self.assertEqual(s["confidence"], "firm")

    def test_firm_metric_bettering_past_threshold_is_improvement(self) -> None:
        d = self._diff({"rel_err": 3e-4}, {"rel_err": 1e-4})
        self.assertEqual(d["metric_shifts"][0]["direction"], "improvement")

    def test_within_threshold_is_silent(self) -> None:
        # rel_err threshold is 0.15 → a 5% move stays quiet.
        d = self._diff({"rel_err": 1e-4}, {"rel_err": 1.05e-4})
        self.assertEqual(d["metric_shifts"], [])

    def test_higher_is_better_metric_direction(self) -> None:
        # cosine: higher is better (threshold 0.02 relative).
        worse = self._diff({"cosine": 0.99}, {"cosine": 0.90})
        self.assertEqual(worse["metric_shifts"][0]["direction"], "regression")
        better = self._diff({"cosine": 0.90}, {"cosine": 0.99})
        self.assertEqual(better["metric_shifts"][0]["direction"], "improvement")

    def test_timing_is_indicative_and_uses_wide_threshold(self) -> None:
        # cost_time_s threshold is 0.35 → a 20% move stays quiet.
        quiet = self._diff({"cost_time_s": 1.0}, {"cost_time_s": 1.2})
        self.assertEqual(quiet["metric_shifts"], [])
        # A 90% slowdown fires, flagged indicative.
        loud = self._diff({"cost_time_s": 1.0}, {"cost_time_s": 1.9})
        self.assertEqual(len(loud["metric_shifts"]), 1)
        self.assertEqual(loud["metric_shifts"][0]["confidence"], "indicative")

    def test_status_change_suppresses_metric_shift(self) -> None:
        # A cell that changes status is reported by the main diff path; a
        # numeric compare across the boundary would double-report it.
        d = self._diff({"rel_err": 1e-4}, {"rel_err": 3e-4}, status="anomaly")
        self.assertEqual(d["metric_shifts"], [])

    def test_missing_metric_on_either_side_is_silent(self) -> None:
        self.assertEqual(self._diff({}, {"rel_err": 3e-4})["metric_shifts"], [])
        self.assertEqual(self._diff({"rel_err": 1e-4}, {})["metric_shifts"], [])

    def test_legacy_snapshot_without_metrics_key_diffs_cleanly(self) -> None:
        from mosaic.benchmarks.core.status import diff_snapshots

        old = self._snap({"rel_err": 1e-4})
        # Strip the metrics key entirely, as a pre-feature snapshot would lack it.
        del old["problems"]["ns-grid"]["rows"][0]["cells"]["exponax"]["metrics"]
        d = diff_snapshots(old, self._snap({"rel_err": 3e-4}))
        self.assertEqual(d["metric_shifts"], [])

    def test_nonfinite_and_zero_baseline_are_safe(self) -> None:
        # NaN / inf never produce a shift.
        self.assertEqual(
            self._diff({"rel_err": float("nan")}, {"rel_err": 1e-4})["metric_shifts"],
            [],
        )
        # Zero baseline uses a tiny floor rather than dividing by zero.
        d = self._diff({"final_loss": 0.0}, {"final_loss": 1e-4})
        self.assertEqual(len(d["metric_shifts"]), 1)
        self.assertEqual(d["metric_shifts"][0]["direction"], "regression")

    def test_render_splits_firm_from_indicative(self) -> None:
        from mosaic.benchmarks.core.status import diff_snapshots, render_diff_markdown

        old = self._snap({"rel_err": 1e-4})
        new = self._snap({"rel_err": 3e-4})
        # Add an indicative timing shift on a second row.
        for snap, t in ((old, 1.0), (new, 1.9)):
            snap["problems"]["ns-grid"]["rows"].append(
                {
                    "suite": "cost",
                    "experiment": "scaling",
                    "label": "cost/scaling",
                    "cells": {
                        "exponax": {
                            "status": "ok",
                            "reason": "",
                            "category": "",
                            "stale": False,
                            "resource_frontier": "",
                            "metrics": {"cost_time_s": t},
                        }
                    },
                }
            )
        md = render_diff_markdown(diff_snapshots(old, new))
        self.assertIn("### 📊 Metric changes", md)
        self.assertIn("### ⏱ Timing (indicative", md)
        self.assertIn("2 metric change(s)", md)

    def test_render_label_names_the_comparison_point(self) -> None:
        """The diff header names its comparison point (e.g. a release tag)."""
        from mosaic.benchmarks.core.status import diff_snapshots, render_diff_markdown

        diff = diff_snapshots(
            self._snap({"rel_err": 1e-4}), self._snap({"rel_err": 1e-4})
        )
        self.assertIn("## Status diff vs base", render_diff_markdown(diff))
        self.assertIn(
            "## Status diff vs v0.1.1", render_diff_markdown(diff, label="v0.1.1")
        )


class TestMetricCollection(unittest.TestCase):
    """``_collect_metrics_for_suite`` populates ``cell.metrics`` from v1 results."""

    def _collect(
        self, suite: str, exp_label: str, data: dict, solver: str = "s"
    ) -> Cell:
        from mosaic.benchmarks.core.status import _collect_metrics_for_suite

        cells = {solver: Cell(OK)}
        _collect_metrics_for_suite(suite, exp_label, data, cells)
        return cells[solver]

    def test_cost_median_time(self) -> None:
        data = {
            "schema_version": 1,
            "sweep": {"key": "N", "values": ["16", "32"]},
            "results": [
                {"solver": "s", "sweep_value": "16", "metrics": {"mean": 1.0}},
                {"solver": "s", "sweep_value": "32", "metrics": {"mean": 3.0}},
            ],
        }
        cell = self._collect("cost", "scaling", data)
        self.assertAlmostEqual(cell.metrics["cost_time_s"], 2.0)

    def test_fd_check_best_rel_err_and_cosine(self) -> None:
        data = {
            "schema_version": 1,
            "sweep": None,
            "results": [
                {
                    "solver": "s",
                    "sweep_value": None,
                    "metrics": {
                        "eps_sweep": {
                            "0.01": {"rel_error": [0.5, 0.3], "cosine": 0.8},
                            "0.001": {"rel_error": [0.1, 0.2], "cosine": 0.95},
                        }
                    },
                }
            ],
        }
        cell = self._collect("gradient", "fd_check", data)
        # best_rel = min over eps of median rel_error = min(0.4, 0.15) = 0.15
        self.assertAlmostEqual(cell.metrics["rel_err"], 0.15)
        self.assertAlmostEqual(cell.metrics["cosine"], 0.95)

    def test_forward_median_error_skips_invalid(self) -> None:
        data = {
            "schema_version": 1,
            "sweep": {"key": "nu", "values": ["1", "2", "3"]},
            "results": [
                {
                    "solver": "s",
                    "sweep_value": "1",
                    "metrics": {"error": 0.1, "valid": True},
                },
                {
                    "solver": "s",
                    "sweep_value": "2",
                    "metrics": {"error": 0.3, "valid": True},
                },
                {"solver": "s", "sweep_value": "3", "metrics": {"valid": False}},
            ],
        }
        cell = self._collect("forward", "agreement", data)
        self.assertAlmostEqual(cell.metrics["fwd_error"], 0.2)

    def test_optimization_final_ratio_from_trajectory(self) -> None:
        data = {
            "schema_version": 1,
            "sweep": None,
            "results": [
                {
                    "solver": "s",
                    "sweep_value": None,
                    "metrics": {"errors": [10.0, 5.0, 2.0]},
                }
            ],
        }
        cell = self._collect("optimization", "recovery", data)
        self.assertAlmostEqual(cell.metrics["final_ratio"], 0.2)
        self.assertAlmostEqual(cell.metrics["final_loss"], 2.0)

    def test_non_ok_cell_gets_no_metrics(self) -> None:
        from mosaic.benchmarks.core.status import _collect_metrics_for_suite

        data = {
            "schema_version": 1,
            "sweep": None,
            "results": [{"solver": "s", "sweep_value": None, "metrics": {"mean": 1.0}}],
        }
        cells = {"s": Cell(FAILED, "boom")}
        _collect_metrics_for_suite("cost", "scaling", data, cells)
        self.assertEqual(cells["s"].metrics, {})


class TestDiffScoping(unittest.TestCase):
    """Only cells actually re-measured this run are attributed to the PR.

    A ``benchmark:solver`` run re-runs one solver on one problem; every other
    cell in the overlaid tree is carried-over baseline data. Baseline drift on
    those cells (e.g. an infra flake recorded as failed) must not surface as a
    PR regression (F3)."""

    @staticmethod
    def _cell(status: str, **extra) -> dict:
        return {
            "status": status,
            "reason": "",
            "category": "",
            "stale": False,
            "resource_frontier": extra.get("frontier", ""),
            "metrics": extra.get("metrics", {}),
        }

    def _snap(self, cells_by_problem: dict) -> dict:
        problems = {}
        for problem, rows in cells_by_problem.items():
            problems[problem] = {
                "problem": problem,
                "solvers": sorted({s for r in rows.values() for s in r}),
                "rows": [
                    {
                        "suite": "cost",
                        "experiment": lbl,
                        "label": f"cost/{lbl}",
                        "cells": c,
                    }
                    for lbl, c in rows.items()
                ],
            }
        return {"score": 0.9, "problems": problems}

    def _pair(self):
        # jax-cfd (in scope) unchanged; PhiFlow (out of scope) drifts ok->fail.
        old = self._snap(
            {
                "ns-grid": {
                    "spatial": {
                        "jax-cfd": self._cell("ok"),
                        "PhiFlow": self._cell("ok"),
                    }
                }
            }
        )
        new = self._snap(
            {
                "ns-grid": {
                    "spatial": {
                        "jax-cfd": self._cell("ok"),
                        "PhiFlow": self._cell("failed"),
                    }
                }
            }
        )
        return old, new

    def test_unscoped_diff_reports_baseline_drift(self) -> None:
        from mosaic.benchmarks.core.status import diff_snapshots

        old, new = self._pair()
        d = diff_snapshots(old, new)  # measured=None → diff everything
        self.assertEqual(len(d["regressions"]), 1)
        self.assertEqual(d["regressions"][0]["solver"], "PhiFlow")

    def test_scoped_diff_suppresses_out_of_scope_regression(self) -> None:
        from mosaic.benchmarks.core.status import diff_snapshots

        old, new = self._pair()
        d = diff_snapshots(
            old, new, measured={"problems": ["ns-grid"], "solvers": ["jax-cfd"]}
        )
        self.assertEqual(d["regressions"], [])

    def test_scoped_diff_keeps_in_scope_regression(self) -> None:
        from mosaic.benchmarks.core.status import diff_snapshots

        old = self._snap({"ns-grid": {"spatial": {"jax-cfd": self._cell("ok")}}})
        new = self._snap({"ns-grid": {"spatial": {"jax-cfd": self._cell("failed")}}})
        d = diff_snapshots(
            old, new, measured={"problems": ["ns-grid"], "solvers": ["jax-cfd"]}
        )
        self.assertEqual(len(d["regressions"]), 1)
        self.assertEqual(d["regressions"][0]["solver"], "jax-cfd")

    def test_scope_filters_by_problem_too(self) -> None:
        from mosaic.benchmarks.core.status import diff_snapshots

        # jax-cfd drifts on a problem NOT in scope → suppressed.
        old = self._snap({"ns-3d-grid": {"spatial": {"jax-cfd": self._cell("ok")}}})
        new = self._snap({"ns-3d-grid": {"spatial": {"jax-cfd": self._cell("failed")}}})
        d = diff_snapshots(
            old, new, measured={"problems": ["ns-grid"], "solvers": ["jax-cfd"]}
        )
        self.assertEqual(d["regressions"], [])

    def test_empty_problem_list_means_all_problems(self) -> None:
        from mosaic.benchmarks.core.status import _cell_measured

        m = {"problems": [], "solvers": ["jax-cfd"]}
        self.assertTrue(_cell_measured(m, "ns-grid", "jax-cfd"))
        self.assertTrue(_cell_measured(m, "ns-3d-grid", "jax-cfd"))
        self.assertFalse(_cell_measured(m, "ns-grid", "PhiFlow"))

    def test_scope_suppresses_frontier_and_metric_shifts(self) -> None:
        from mosaic.benchmarks.core.status import diff_snapshots

        old = self._snap(
            {
                "ns-grid": {
                    "spatial": {
                        "PhiFlow": self._cell(
                            "ok", frontier="", metrics={"cost_time_s": 1.0}
                        )
                    }
                }
            }
        )
        new = self._snap(
            {
                "ns-grid": {
                    "spatial": {
                        "PhiFlow": self._cell(
                            "ok", frontier="64", metrics={"cost_time_s": 5.0}
                        )
                    }
                }
            }
        )
        d = diff_snapshots(
            old, new, measured={"problems": ["ns-grid"], "solvers": ["jax-cfd"]}
        )
        self.assertEqual(d["frontier_shifts"], [])
        self.assertEqual(d["metric_shifts"], [])


class TestDiffScopeHelper(unittest.TestCase):
    """`_diff_scope` maps a run-scope dict to the diff's ``measured`` filter."""

    def _import(self):
        from mosaic.benchmarks.cli._status_helpers import _diff_scope

        return _diff_scope

    def test_none_scope_is_none(self) -> None:
        self.assertIsNone(self._import()(None))

    def test_all_label_is_none(self) -> None:
        self.assertIsNone(
            self._import()(
                {"label": "all", "is_release_pr": False, "solvers": [], "problems": []}
            )
        )

    def test_release_pr_is_none(self) -> None:
        self.assertIsNone(
            self._import()(
                {"label": "", "is_release_pr": True, "solvers": [], "problems": []}
            )
        )

    def test_solver_label_scopes(self) -> None:
        m = self._import()(
            {
                "label": "solver",
                "is_release_pr": False,
                "solvers": ["jax-cfd"],
                "problems": ["ns-grid"],
            }
        )
        self.assertEqual(m, {"problems": ["ns-grid"], "solvers": ["jax-cfd"]})

    def test_unlabelled_is_none(self) -> None:
        self.assertIsNone(
            self._import()(
                {"label": "", "is_release_pr": False, "solvers": [], "problems": []}
            )
        )


if __name__ == "__main__":
    unittest.main()
