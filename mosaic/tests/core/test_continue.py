"""Unit tests for ``mosaic run --continue`` skip-completed-experiments logic.

The dispatcher in :func:`mosaic.benchmarks.core.runner.run_suite` consults the
results tree before invoking each experiment callable. When ``skip_completed``
is set, an experiment whose ``result.json`` already exists is bypassed — this
is what powers ``mosaic run --continue`` after an OOM / crash mid-sweep.

We exercise the dispatcher directly with a fake ``experiments`` dict so the
test runs in milliseconds without Docker or any solver image.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mosaic.benchmarks.core.config import Problem, SolverSpec
from mosaic.benchmarks.core.runner import run_suite


def _make_cfg(tesseract_dir: Path) -> Problem:
    spec = SolverSpec(
        dir="dummy",
        color="#000000",
        name="dummy",
        scheme="test",
        backend="python",
    )
    return Problem(
        name="test_problem",
        tesseract_dir=tesseract_dir,
        output_key="result",
        solvers=[spec],
        make_ic={},
        make_inputs=lambda *a, **k: {},
        error_fn=lambda *a, **k: 0.0,
    )


class ContinueFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self._tmpdir.name)
        self.results_root = self.tmp_root / "results"
        # run_suite calls results_dir() — point it at our tmp tree.
        self._env = mock.patch.dict(
            "os.environ", {"MOSAIC_RESULTS_DIR": str(self.results_root)}
        )
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        self._tmpdir.cleanup()

    def _experiments(self, called: list[str]) -> dict:
        """Two fake experiments that just record being invoked."""

        def _exp_a(cfg, tags, **kwargs):
            called.append("exp_a")
            return {"ok": True}

        def _exp_b(cfg, tags, **kwargs):
            called.append("exp_b")
            return {"ok": True}

        return {"exp_a": _exp_a, "exp_b": _exp_b}

    def test_continue_skips_existing_result_json(self) -> None:
        """exp_a has result.json on disk → skipped; exp_b runs."""
        cfg = _make_cfg(self.tmp_root / "tess")
        suite = "fake_suite"
        # Pre-populate exp_a's result.json — simulates a previous run.
        exp_a_dir = self.results_root / cfg.name / suite / "exp_a"
        exp_a_dir.mkdir(parents=True)
        (exp_a_dir / "result.json").write_text("{}")

        called: list[str] = []
        run_suite(
            cfg,
            tags={},
            experiments=self._experiments(called),
            suite_name=suite,
            plots=False,
            skip_completed=True,
        )
        self.assertEqual(called, ["exp_b"])

    def test_continue_off_runs_both(self) -> None:
        """Without --continue, an existing result.json does NOT skip."""
        cfg = _make_cfg(self.tmp_root / "tess")
        suite = "fake_suite"
        exp_a_dir = self.results_root / cfg.name / suite / "exp_a"
        exp_a_dir.mkdir(parents=True)
        (exp_a_dir / "result.json").write_text("{}")

        called: list[str] = []
        run_suite(
            cfg,
            tags={},
            experiments=self._experiments(called),
            suite_name=suite,
            plots=False,
            skip_completed=False,
        )
        self.assertEqual(called, ["exp_a", "exp_b"])

    def test_continue_skips_nested_ic_complete(self) -> None:
        """Multi-IC layout: skip when every IC subdir has result.json."""
        cfg = _make_cfg(self.tmp_root / "tess")
        suite = "fake_suite"
        # exp_a has IC subdirs ic1, ic2 — both with result.json.
        for ic in ("ic1", "ic2"):
            d = self.results_root / cfg.name / suite / "exp_a" / ic
            d.mkdir(parents=True)
            (d / "result.json").write_text("{}")

        called: list[str] = []
        run_suite(
            cfg,
            tags={},
            experiments={"exp_a": lambda c, t, **kw: called.append("exp_a")},
            suite_name=suite,
            plots=False,
            skip_completed=True,
        )
        self.assertEqual(called, [])

    def test_continue_runs_nested_ic_partial(self) -> None:
        """Multi-IC layout: re-run if at least one IC subdir lacks result.json."""
        cfg = _make_cfg(self.tmp_root / "tess")
        suite = "fake_suite"
        # ic1 finished, ic2 only has params.json — incomplete.
        ic1 = self.results_root / cfg.name / suite / "exp_a" / "ic1"
        ic1.mkdir(parents=True)
        (ic1 / "result.json").write_text("{}")
        ic2 = self.results_root / cfg.name / suite / "exp_a" / "ic2"
        ic2.mkdir(parents=True)
        (ic2 / "params.json").write_text("{}")

        called: list[str] = []
        run_suite(
            cfg,
            tags={},
            experiments={"exp_a": lambda c, t, **kw: called.append("exp_a")},
            suite_name=suite,
            plots=False,
            skip_completed=True,
        )
        self.assertEqual(called, ["exp_a"])

    def test_continue_respects_debug_suffix(self) -> None:
        """--debug writes to exp_a_debug/; --continue must check the same path."""
        cfg = _make_cfg(self.tmp_root / "tess")
        suite = "fake_suite"
        # Plain exp_a/result.json exists, but with --debug the dispatcher
        # writes to exp_a_debug/. Skip logic must look at exp_a_debug/, not exp_a/.
        plain = self.results_root / cfg.name / suite / "exp_a"
        plain.mkdir(parents=True)
        (plain / "result.json").write_text("{}")

        called: list[str] = []
        run_suite(
            cfg,
            tags={},
            experiments={"exp_a": lambda c, t, **kw: called.append("exp_a")},
            suite_name=suite,
            plots=False,
            skip_completed=True,
            overrides={"debug": True},
        )
        self.assertEqual(called, ["exp_a"])


if __name__ == "__main__":
    unittest.main()
