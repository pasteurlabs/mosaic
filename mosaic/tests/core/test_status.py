"""Unit tests for the weighted campaign-health score (ARCH-22).

Replaces the binary ``% ok`` metric with a per-cell weighted sum. See
``core/status.py::SCORE_WEIGHTS`` for the table. The tests here pin:

  1. Extremes: all-ok → +1.0, all-fail → −0.5, empty → None.
  2. Monotonicity: a cell going ok* → ok moves the score by exactly the
     expected ΔN; a cell going ok → fail moves it by (1.0 + 0.5)/N in the
     other direction.
  3. Categorical exclusions are off the books (numerator + denominator).
  4. Boundary agreement with ``% ok``: all-ok ⇒ score=1.0 & pct=100;
     all non-ok (fail/anom/missing) ⇒ score in [−0.5, +0.3] & pct=0.

Run directly with::

    conda run -n gym python -m benchmarks.core.tests.test_status
"""

from __future__ import annotations

import unittest

from mosaic.benchmarks.core.status import (
    ANOMALY,
    EXCL_CATEGORICAL,
    EXCL_INFEASIBLE,
    EXCL_NOT_IMPLEMENTED,
    EXCL_UNSPECIFIED,
    EXCL_UNSTABLE,
    EXCL_UPSTREAM_BUG,
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
    score_color,
    tally,
    weight_color,
    weight_emoji,
)


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
        # All categorical → all excluded from the denominator.
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
            [Cell(EXCLUDED, category=EXCL_UPSTREAM_BUG)],
        ]
        for mix in mixes:
            score, _ = compute_score(mix)
            if score is None:
                continue
            self.assertGreaterEqual(score, lo)
            self.assertLessEqual(score, hi)


class TestScoreTransitions(unittest.TestCase):
    """Direction and magnitude of single-cell status flips."""

    def test_ok_star_to_ok_increases_over_n(self) -> None:
        """ok* (0.67) → ok (1.00) shifts average by (1.0 − 0.67)/N."""
        n = 4
        before = [Cell(OK, stale=True)] + [Cell(OK) for _ in range(n - 1)]
        after = [Cell(OK)] + [Cell(OK) for _ in range(n - 1)]
        s_before, _ = compute_score(before)
        s_after, _ = compute_score(after)
        self.assertAlmostEqual(s_after - s_before, (1.0 - 0.67) / n)
        self.assertGreater(s_after, s_before)  # direction sanity

    def test_ok_to_fail_decreases_by_one_over_n(self) -> None:
        """ok (1.0) → fail (0.0) shifts average by −1.0/N."""
        n = 6
        before = [Cell(OK) for _ in range(n)]
        after = [Cell(FAILED)] + [Cell(OK) for _ in range(n - 1)]
        s_before, _ = compute_score(before)
        s_after, _ = compute_score(after)
        self.assertAlmostEqual(s_before - s_after, 1.0 / n)
        self.assertLess(s_after, s_before)

    def test_transition_across_all_weight_keys(self) -> None:
        """Every mapped weight key is reachable via some (status, stale,
        category) combination — catches silent dead keys."""
        produced: set[str] = set()
        probes: list[Cell] = [
            Cell(OK),
            Cell(OK, stale=True),
            Cell(ANOMALY),
            Cell(ANOMALY, stale=True),
            Cell(FAILED),
            Cell(FAILED, stale=True),
            Cell(NOT_RUN),
            Cell(EXCLUDED, category=EXCL_NOT_IMPLEMENTED),
            Cell(EXCLUDED, category=EXCL_UNSTABLE),
            Cell(EXCLUDED, category=EXCL_INFEASIBLE),
            Cell(EXCLUDED, category=EXCL_UPSTREAM_BUG),
            Cell(EXCLUDED, category=EXCL_UNSPECIFIED),
        ]
        for c in probes:
            key = cell_weight_key(c)
            self.assertIsNotNone(key, f"unmapped: {c}")
            produced.add(key)  # type: ignore[arg-type]
        # Everything except the categorical sentinel (perm → None) should
        # be in SCORE_WEIGHTS.
        missing = set(SCORE_WEIGHTS.keys()) - produced
        self.assertFalse(
            missing, f"weight keys with no probe mapping: {sorted(missing)}"
        )


class TestCategoricalExclusions(unittest.TestCase):
    """Permanent (categorical) exclusions must not affect the score at all."""

    def test_categorical_excluded_from_denominator(self) -> None:
        """A categorical-excluded cell doesn't shift the score."""
        base = [Cell(OK), Cell(OK), Cell(FAILED)]
        with_cat = base + [Cell(EXCLUDED, category=EXCL_CATEGORICAL)]
        s_base, n_base = compute_score(base)
        s_with, n_with = compute_score(with_cat)
        self.assertAlmostEqual(s_base, s_with)
        self.assertEqual(n_base, n_with)  # n unchanged

    def test_non_categorical_exclusion_does_count(self) -> None:
        """'todo' / 'slow' / 'unst' / 'bug' exclusions ARE in the denominator.

        Adding one slow-excluded (weight 0.33) to an all-ok mix pulls the
        score down — it doesn't sit out like categorical does.
        """
        base = [Cell(OK), Cell(OK), Cell(OK)]
        with_slow = base + [Cell(EXCLUDED, category=EXCL_INFEASIBLE)]
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
            "e": Cell(EXCLUDED, category=EXCL_INFEASIBLE),  # work-to-do
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
        """Any mix where no cell is fresh-OK yields pct_ok=0 — the score
        must sit below the all-ok value (strictly < 1.0) in all such
        cases, and bottoms out at 0.0 when everything's fresh-fail."""
        # All-fail: pct_ok=0, score=0.0 (floor of the [0, 1] ladder).
        st_f = _status_from_cells({s: Cell(FAILED) for s in ("a", "b", "c")})
        t_f = tally(st_f)
        self.assertAlmostEqual(t_f["pct_ok"], 0.0)
        self.assertAlmostEqual(t_f["score"], 0.0)

        # All-anom: pct_ok=0, score=0.53 (anomalies carry some signal,
        # while pct_ok claims zero credit).
        st_a = _status_from_cells({s: Cell(ANOMALY) for s in ("a", "b", "c")})
        t_a = tally(st_a)
        self.assertAlmostEqual(t_a["pct_ok"], 0.0)
        self.assertAlmostEqual(t_a["score"], 0.53)
        self.assertLess(t_a["score"], 1.0)

        # All-missing: pct_ok=0, score=0.33 (neutral — no signal yet).
        st_m = _status_from_cells({s: Cell(NOT_RUN) for s in ("a", "b", "c")})
        t_m = tally(st_m)
        self.assertAlmostEqual(t_m["pct_ok"], 0.0)
        self.assertAlmostEqual(t_m["score"], 0.33)


_HEX_RE = __import__("re").compile(r"^#[0-9a-f]{6}$")


def _red_amount(hex_color: str) -> int:
    """Extract the red-channel intensity from a `#RRGGBB` string."""
    return int(hex_color[1:3], 16)


def _green_amount(hex_color: str) -> int:
    return int(hex_color[3:5], 16)


def _blue_amount(hex_color: str) -> int:
    return int(hex_color[5:7], 16)


class TestWeightColorLadder(unittest.TestCase):
    """Unified weight → colour/emoji mapping drives every coloured element
    (cell labels, score headers, progress bar, overall score). The ansi
    output is a continuous health-signal gradient and the emoji output is a
    coarse 4-bucket ladder.

    Ansi: w=0 → red, w=0.33 → orange, w=0.5 → yellow, w=0.67 → green, w=1 → bright green.
    Emoji: 🟢 (w ≥ 0.65) · 🟡 (w ≥ 0.30) · 🟠 (w ≥ 0.15) · 🔴 (w < 0.15).
    """

    def test_weight_color_positive_end_is_bright_green(self) -> None:
        c = weight_color(1.0)
        self.assertTrue(_HEX_RE.match(c))
        # Health top: bright green — green channel is highest.
        self.assertGreater(_green_amount(c), _red_amount(c))
        self.assertGreater(_green_amount(c), _blue_amount(c))

    def test_weight_color_zero_is_red(self) -> None:
        c = weight_color(0.0)
        self.assertTrue(_HEX_RE.match(c))
        # Health bottom: red — red channel is highest.
        self.assertGreater(_red_amount(c), _green_amount(c))
        self.assertGreater(_red_amount(c), _blue_amount(c))

    def test_weight_color_values_in_range(self) -> None:
        """weight_color returns valid hex for all inputs in [0, 1]."""
        for w in (0.0, 0.13, 0.17, 0.33, 0.43, 0.53, 0.67, 1.0):
            c = weight_color(w)
            self.assertTrue(_HEX_RE.match(c), f"invalid color for w={w}: {c!r}")

    def test_weight_color_clamping(self) -> None:
        """Out-of-range inputs clamp to 0 or 1."""
        self.assertEqual(weight_color(-1.0), weight_color(0.0))
        self.assertEqual(weight_color(2.0), weight_color(1.0))

    def test_weight_color_none(self) -> None:
        self.assertEqual(weight_color(None), "dim")

    def test_weight_emoji_matches_ladder(self) -> None:
        self.assertEqual(weight_emoji(1.0), "🟢")
        self.assertEqual(weight_emoji(0.67), "🟢")
        self.assertEqual(weight_emoji(0.65), "🟢")
        self.assertEqual(weight_emoji(0.53), "🟡")
        self.assertEqual(weight_emoji(0.35), "🟡")
        self.assertEqual(weight_emoji(0.33), "🟡")
        self.assertEqual(weight_emoji(0.30), "🟡")
        self.assertEqual(weight_emoji(0.17), "🟠")
        self.assertEqual(weight_emoji(0.15), "🟠")
        self.assertEqual(weight_emoji(0.13), "🔴")
        self.assertEqual(weight_emoji(0.0), "🔴")
        self.assertEqual(weight_emoji(None), "—")

    def test_score_color_delegates_to_weight_color(self) -> None:
        for w in (1.0, 0.67, 0.53, 0.33, 0.17, 0.0, None):
            self.assertEqual(score_color(w), weight_color(w))

    def test_cell_weight_and_color(self) -> None:
        """Every non-categorical cell returns a finite weight and a valid
        colour; categorical cells return None / dim."""
        c = Cell(OK)
        self.assertEqual(cell_weight(c), 1.0)
        self.assertTrue(_HEX_RE.match(cell_color(c)))

        c = Cell(OK, stale=True)
        self.assertEqual(cell_weight(c), 0.67)
        self.assertTrue(_HEX_RE.match(cell_color(c)))

        c_fail = Cell(FAILED)
        self.assertEqual(cell_weight(c_fail), 0.0)
        c_fail_color = cell_color(c_fail)
        self.assertTrue(_HEX_RE.match(c_fail_color))
        # Red-dominant for fail (w=0.0).
        self.assertGreater(_red_amount(c_fail_color), _green_amount(c_fail_color))

        # Missing / not-run: orange-ish (w=0.33).
        c = Cell(NOT_RUN)
        self.assertEqual(cell_weight(c), 0.33)
        self.assertTrue(_HEX_RE.match(cell_color(c)))

        # Categorical excl: no weight, dim.
        c = Cell(EXCLUDED, category=EXCL_CATEGORICAL)
        self.assertIsNone(cell_weight(c))
        self.assertEqual(cell_color(c), "dim")

    def test_every_score_weight_maps_to_a_colour(self) -> None:
        """Every value in SCORE_WEIGHTS produces either a valid hex colour
        or the `dim` sentinel; every emoji is one of the expected glyphs."""
        for _key, w in SCORE_WEIGHTS.items():
            color = weight_color(w)
            self.assertTrue(
                color == "dim" or _HEX_RE.match(color),
                f"bad color for w={w}: {color!r}",
            )
            self.assertIn(weight_emoji(w), {"🟢", "🟡", "🟠", "🔴", "—"})


if __name__ == "__main__":
    unittest.main()
