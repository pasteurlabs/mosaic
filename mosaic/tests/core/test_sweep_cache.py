"""Tests for per-solver sweep checkpointing — the engine behind sub-experiment
``mosaic run --continue``.

When ``solver_sweep`` is given a ``checkpoint_dir``, it:

* writes ``<dir>/<solver>.pkl`` after each solver completes its full condition
  list (atomic via tmpfile + ``os.replace``), and
* on the next invocation, pre-loads any existing ``<dir>/<solver>.pkl`` and
  skips those solvers from dispatch.

These tests cover the helpers in isolation plus an end-to-end resume scenario
with ``run_with_gpu_pool`` mocked out — no Docker, no Tesseracts.
"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mosaic.benchmarks.core import runner
from mosaic.benchmarks.core.config import ProblemConfig, SolverSpec
from mosaic.benchmarks.core.runner import (
    _load_sweep_cache,
    _save_solver_cache,
    solver_sweep,
)


def _make_cfg(tesseract_dir: Path, solver_names) -> ProblemConfig:
    solvers = {
        n: SolverSpec(dir=n, color="#000000", name=n, scheme="test", backend="python")
        for n in solver_names
    }
    return ProblemConfig(
        name="test_problem",
        tesseract_dir=tesseract_dir,
        output_key="result",
        solvers=solvers,
        make_ic={},
        make_inputs=lambda *a, **k: {},
        error_fn=lambda *a, **k: 0.0,
        diagnostics={},
    )


class SweepCacheHelperTests(unittest.TestCase):
    """``_save_solver_cache`` and ``_load_sweep_cache`` round-trip."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_roundtrip(self) -> None:
        ckpt = self.tmp / "sweep_cache"
        raw_entry = {0.1: np.array([1.0, 2.0]), 0.5: None}
        _save_solver_cache(ckpt, "exponax", raw_entry, wall_time=12.3)

        raw, walls = _load_sweep_cache(ckpt)
        self.assertIn("exponax", raw)
        np.testing.assert_array_equal(raw["exponax"][0.1], np.array([1.0, 2.0]))
        self.assertIsNone(raw["exponax"][0.5])
        self.assertEqual(walls["exponax"], 12.3)

    def test_missing_dir_returns_empty(self) -> None:
        raw, walls = _load_sweep_cache(self.tmp / "does_not_exist")
        self.assertEqual(raw, {})
        self.assertEqual(walls, {})

    def test_corrupt_file_is_skipped(self) -> None:
        ckpt = self.tmp / "sweep_cache"
        ckpt.mkdir()
        (ckpt / "broken.pkl").write_bytes(b"not a valid pickle")
        # Good entry alongside the broken one.
        _save_solver_cache(ckpt, "good", {"k": 1}, wall_time=1.0)

        raw, walls = _load_sweep_cache(ckpt)
        self.assertEqual(set(raw), {"good"})
        self.assertEqual(walls, {"good": 1.0})

    def test_save_is_atomic_via_rename(self) -> None:
        """No half-written .pkl can survive a crash mid-write.

        We assert this by inspecting the temp filename pattern: the helper
        writes ``<name>.pkl.tmp`` then renames to ``<name>.pkl``. We can't
        easily race a crash here, but we can verify there's no stray .tmp
        left after a successful write.
        """
        ckpt = self.tmp / "sweep_cache"
        _save_solver_cache(ckpt, "exponax", {"a": 1}, wall_time=2.0)
        files = sorted(p.name for p in ckpt.iterdir())
        self.assertEqual(files, ["exponax.pkl"])


class SolverSweepResumeTests(unittest.TestCase):
    """End-to-end: solver_sweep skips cached solvers and writes new ones."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _fake_pool(self, names, tags, fn, gpu_ids=None):
        """Stand-in for ``run_with_gpu_pool`` that just calls fn(name, None)."""
        for name in names:
            fn(name, None)

    def test_no_checkpoint_runs_all_solvers(self) -> None:
        cfg = _make_cfg(self.tmp / "tess", ["alpha", "beta"])
        calls: list[str] = []

        def _apply(name, t, val):
            calls.append(name)
            return np.array([float(val)])

        with mock.patch.object(runner, "run_with_gpu_pool", self._fake_pool):
            raw, walls = solver_sweep(cfg, tags={}, conditions=[0.1, 0.5], fn=_apply)
        self.assertEqual(sorted(set(calls)), ["alpha", "beta"])
        self.assertEqual(set(raw), {"alpha", "beta"})

    def test_resume_skips_cached_solver(self) -> None:
        """alpha cached from prior run → only beta executes this time."""
        cfg = _make_cfg(self.tmp / "tess", ["alpha", "beta"])
        ckpt = self.tmp / "sweep_cache"
        _save_solver_cache(
            ckpt,
            "alpha",
            {0.1: np.array([10.0]), 0.5: np.array([50.0])},
            wall_time=7.5,
        )

        calls: list[str] = []

        def _apply(name, t, val):
            calls.append(name)
            return np.array([float(val)])

        with mock.patch.object(runner, "run_with_gpu_pool", self._fake_pool):
            raw, walls = solver_sweep(
                cfg,
                tags={},
                conditions=[0.1, 0.5],
                fn=_apply,
                checkpoint_dir=ckpt,
            )
        self.assertEqual(sorted(set(calls)), ["beta"])
        np.testing.assert_array_equal(raw["alpha"][0.1], np.array([10.0]))
        np.testing.assert_array_equal(raw["alpha"][0.5], np.array([50.0]))
        self.assertEqual(walls["alpha"], 7.5)
        # beta got fresh results from _apply.
        np.testing.assert_array_equal(raw["beta"][0.1], np.array([0.1]))

    def test_resume_writes_cache_for_newly_completed_solvers(self) -> None:
        """After a fresh solver finishes, its .pkl must exist for next time."""
        cfg = _make_cfg(self.tmp / "tess", ["alpha"])
        ckpt = self.tmp / "sweep_cache"

        def _apply(name, t, val):
            return np.array([float(val) * 2])

        with mock.patch.object(runner, "run_with_gpu_pool", self._fake_pool):
            solver_sweep(
                cfg,
                tags={},
                conditions=[1.0, 2.0],
                fn=_apply,
                checkpoint_dir=ckpt,
            )

        cache_file = ckpt / "alpha.pkl"
        self.assertTrue(cache_file.exists())
        with open(cache_file, "rb") as f:
            payload = pickle.load(f)
        np.testing.assert_array_equal(payload["raw"][1.0], np.array([2.0]))
        np.testing.assert_array_equal(payload["raw"][2.0], np.array([4.0]))
        self.assertGreaterEqual(payload["wall_time"], 0.0)

    def test_resume_ignores_cache_for_filtered_out_solvers(self) -> None:
        """Cache for a solver NOT in active set must not re-introduce it.

        Scenario: prior run cached alpha + beta, but this invocation uses
        ``-s beta`` so cfg only contains beta. The alpha cache file must be
        ignored — otherwise --continue would silently un-filter the run.
        """
        cfg = _make_cfg(self.tmp / "tess", ["beta"])  # alpha not in cfg
        ckpt = self.tmp / "sweep_cache"
        _save_solver_cache(ckpt, "alpha", {1.0: np.array([99.0])}, wall_time=1.0)
        _save_solver_cache(ckpt, "beta", {1.0: np.array([42.0])}, wall_time=1.0)

        calls: list[str] = []

        def _apply(name, t, val):
            calls.append(name)
            return np.array([0.0])

        with mock.patch.object(runner, "run_with_gpu_pool", self._fake_pool):
            raw, walls = solver_sweep(
                cfg,
                tags={},
                conditions=[1.0],
                fn=_apply,
                checkpoint_dir=ckpt,
            )
        # beta was already cached → not called; alpha not in cfg → ignored entirely.
        self.assertEqual(calls, [])
        self.assertEqual(set(raw), {"beta"})
        self.assertNotIn("alpha", raw)


if __name__ == "__main__":
    unittest.main()
