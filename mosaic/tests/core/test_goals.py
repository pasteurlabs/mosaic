"""Tests for ``Goal`` per-experiment success predicates.

``Goal(name, description, check)`` is registered via
``Problem.add_experiment(goals=[Goal(...), ...])``. After each run,
``run_experiment`` evaluates every goal against the result dict and
writes the result under ``result["goals"][goal.name]``. A raising check
records ``False`` instead of crashing the run.
"""

from __future__ import annotations

import unittest

from mosaic.benchmarks.core.config import Goal, Problem
from mosaic.benchmarks.core.experiment import kernel


@kernel(sweep_mode="none")
def _noop(t, ctx) -> dict:
    return {"metrics": {}}


def _make_problem() -> Problem:
    return Problem(
        name="test",
        tesseract_dir="navier-stokes-grid",
        output_key="result",
        ic_key="v0",
        solvers=[],
        make_ic={"default": lambda **kw: None},
        make_inputs=lambda spec, ic, **kw: {},
        error_fn=lambda *a, **k: 0.0,
    )


class GoalRegistrationTests(unittest.TestCase):
    def test_goal_dataclass_is_frozen(self) -> None:
        import dataclasses

        g = Goal("ok", "doc", check=lambda r: True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            g.name = "renamed"

    def test_no_goals_means_empty_list(self) -> None:
        p = _make_problem()
        p.add_experiment("fwd/baseline", _noop, physics={"N": 32})
        self.assertEqual(p.experiments["fwd/baseline"].goals, [])

    def test_goals_attached_to_single_leaf(self) -> None:
        g = Goal("rel_error_below_1pct", "rel err < 1%", check=lambda r: True)
        p = _make_problem()
        p.add_experiment("fwd/baseline", _noop, physics={"N": 32}, goals=[g])
        self.assertEqual(p.experiments["fwd/baseline"].goals, [g])

    def test_goals_attached_to_each_variant_in_fan_out(self) -> None:
        g1 = Goal("g1", "first", check=lambda r: True)
        g2 = Goal("g2", "second", check=lambda r: False)
        p = _make_problem()
        p.add_experiment(
            "fwd/agreement",
            _noop,
            runs=[
                {"name": "tgv", "ic": {"name": "default"}, "physics": {"N": 32}},
                {"name": "mm", "ic": {"name": "default"}, "physics": {"N": 32}},
            ],
            goals=[g1, g2],
        )
        for k, v in p.experiments.items():
            if k.startswith("fwd/agreement/"):
                self.assertEqual(v.goals, [g1, g2])


class GoalEvaluationTests(unittest.TestCase):
    """``Experiment``-level helper: evaluate goals against a synthetic result."""

    def _eval(self, goals: list[Goal], result: dict) -> dict[str, bool]:
        # Mirror the evaluation block inside ``run_experiment`` so we can
        # test it without standing up the full runner.
        out: dict[str, bool] = {}
        for g in goals:
            try:
                out[g.name] = bool(g.check(result))
            except Exception:
                out[g.name] = False
        return out

    def test_all_pass_pass(self) -> None:
        goals = [
            Goal("a", "", check=lambda r: r["x"] > 0),
            Goal("b", "", check=lambda r: r["y"] < 10),
        ]
        self.assertEqual(self._eval(goals, {"x": 1, "y": 5}), {"a": True, "b": True})

    def test_one_fails_one_passes(self) -> None:
        goals = [
            Goal("ok", "", check=lambda r: True),
            Goal("nope", "", check=lambda r: False),
        ]
        self.assertEqual(self._eval(goals, {}), {"ok": True, "nope": False})

    def test_raising_check_records_false(self) -> None:
        def boom(_):
            raise RuntimeError("predicate blew up")

        goals = [Goal("safe", "", check=lambda r: True), Goal("boom", "", check=boom)]
        out = self._eval(goals, {})
        self.assertEqual(out, {"safe": True, "boom": False})


if __name__ == "__main__":
    unittest.main()
