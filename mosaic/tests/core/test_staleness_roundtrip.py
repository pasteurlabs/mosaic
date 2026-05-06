"""Write-side ↔ read-side roundtrip tests for the staleness pipeline.

``save_experiment`` stamps each result.json with ``harness_hash``
(AST-normalised fingerprint of the ``run_<experiment>`` function) and a
per-solver ``tesseract_hashes`` dict (content hash of the tesseract source
tree). ``status.collect_status`` re-computes both and flips cells to
``ok*`` / ``anom*`` / ``fail*`` on mismatch.

The bug that motivated this file: the write side moved to the
AST-normalised ``harness_fn_hash`` but the read side in
``status._current_harness_hash`` continued to SHA the raw source. Every fresh
run produced a mismatched pair, every cell stayed ``ok*``, and the campaign
score never budged. No test exercised write → read in one go, so the drift
was silent.

These tests pin the full round-trip for BOTH hash pipelines. If a future
refactor switches one side without the other, at least one test here must
fail. Invariants asserted:

* Fresh save → ``collect_status`` reports the cell as NOT stale.
* Cosmetic edit (whitespace / comment / docstring for harness; mtime-only
  touch for tesseract) → still NOT stale.
* Behavioural edit (rename local / add stmt for harness; modify a tesseract
  file's bytes) → IS stale.

Run directly with::

    conda run -n gym python -m pytest mosaic/tests/core/test_staleness_roundtrip.py -v
"""

from __future__ import annotations

import importlib
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from mosaic.benchmarks.core import status as status_mod
from mosaic.benchmarks.core.config import ProblemConfig, SolverSpec
from mosaic.benchmarks.core.status import OK, collect_status
from mosaic.benchmarks.core.utils import save_experiment

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_module(tmp_root: Path, modname: str, src: str):
    """Write *src* to ``tmp_root/<modname>.py``, import it, return the module.

    Also registers ``tmp_root`` on ``sys.path`` so ``importlib.import_module``
    can find it. Callers are responsible for cleaning up via
    ``_reset_module(modname)`` when they rewrite the source.
    """
    tmp_root.mkdir(parents=True, exist_ok=True)
    (tmp_root / f"{modname}.py").write_text(src)
    if str(tmp_root) not in sys.path:
        sys.path.insert(0, str(tmp_root))
    # Force a clean import so back-to-back edits in the same test load the
    # new source rather than a cached module object.
    sys.modules.pop(modname, None)
    return importlib.import_module(modname)


def _reset_module(modname: str) -> None:
    sys.modules.pop(modname, None)


def _make_cfg(
    tmp_root: Path,
    tesseract_dir: Path,
    solver_name: str,
    solver_subdir: str,
    problem_name: str = "test_problem",
) -> ProblemConfig:
    """Minimal ProblemConfig sufficient for status walking."""
    spec = SolverSpec(
        dir=solver_subdir,
        color="#000000",
        name=solver_name,
        scheme="test",
        backend="python",
    )
    return ProblemConfig(
        name=problem_name,
        tesseract_dir=tesseract_dir,
        output_key="result",
        solvers={solver_name: spec},
        make_ic={},
        make_inputs=lambda *a, **k: {},
        error_fn=lambda *a, **k: 0.0,
        diagnostics={},
    )


class _StalenessRoundtripBase(unittest.TestCase):
    """Shared fixture: tmp results dir, tesseract dir, and harness module."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.tmp_root = Path(cls._tmpdir.name)
        # ``_results_dir`` in status.py delegates to ``results_dir()``
        # (env-var / cwd based). Redirect it so our fake results tree is walked.
        cls.results_root = cls.tmp_root / "results"
        cls.results_root.mkdir(parents=True, exist_ok=True)
        cls._patcher = mock.patch.object(
            status_mod,
            "_results_dir",
            lambda cfg: cls.results_root / cfg.name,
        )
        cls._patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patcher.stop()
        cls._tmpdir.cleanup()


# ── harness_fn_hash round-trip ───────────────────────────────────────────────


class HarnessRoundtripTests(_StalenessRoundtripBase):
    """Write a result.json stamped by ``save_experiment`` and verify
    ``collect_status`` computes the same hash for the current source."""

    def setUp(self) -> None:
        # Each test gets its own module file so rewrites don't leak between
        # tests via import caches.
        self.modname = f"_staleness_harness_mod_{id(self)}"
        self.mod_root = self.tmp_root / f"harness_pkg_{id(self)}"
        self.solver = "solver_a"
        self.tesseract_dir = self.tmp_root / f"tess_{id(self)}"
        # Need at least one file so content_hash is non-empty and the stamp
        # path runs.
        (self.tesseract_dir / self.solver).mkdir(parents=True, exist_ok=True)
        (self.tesseract_dir / self.solver / "tesseract_api.py").write_text(
            "def apply(x): return x\n"
        )
        self.cfg = _make_cfg(
            self.tmp_root,
            self.tesseract_dir,
            self.solver,
            self.solver,
            problem_name=f"harness_p_{id(self)}",
        )
        self.exp_dir = self.results_root / self.cfg.name / "forward" / "baseline"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        _reset_module(self.modname)

    def _write_fn(self, src: str):
        mod = _make_module(self.mod_root, self.modname, src)
        return getattr(mod, "run_baseline")

    def _save_once(self, fn) -> None:
        result = {
            "params": {},
            "by_solver": {
                self.solver: {"error": 0.01, "valid": True},
            },
        }
        save_experiment(result, self.exp_dir, cfg=self.cfg, harness_fn=fn)

    def _cell_for_solver(self):
        st = collect_status(self.cfg, suites=["forward"])
        # Find the row for baseline.
        for row in st.rows:
            if row.suite == "forward" and row.experiment == "baseline":
                return row.cells.get(self.solver)
        return None

    # ── invariant: fresh write → not stale ───────────────────────────────────

    def test_fresh_write_is_not_stale(self) -> None:
        """A result stamped by save_experiment must read back as fresh ok."""
        src = textwrap.dedent(
            """\
            def run_baseline(cfg, tags):
                x = 1
                y = x + 1
                return {"ok": y}
            """
        )
        fn = self._write_fn(src)
        self._save_once(fn)
        cell = self._cell_for_solver()
        self.assertIsNotNone(cell)
        self.assertEqual(cell.status, OK)
        self.assertFalse(
            cell.stale,
            "fresh write must not be flagged stale — indicates write/read hash drift",
        )

    # ── invariant: cosmetic edit → still not stale ───────────────────────────

    def test_whitespace_edit_keeps_cell_fresh(self) -> None:
        src_a = textwrap.dedent(
            """\
            def run_baseline(cfg, tags):
                x = 1
                y = x + 1
                return {"ok": y}
            """
        )
        src_b = (
            "def run_baseline(cfg, tags):\n"
            "\n"
            "    x = 1\n"
            "\n\n"
            "    y = x + 1\n"
            "    return {'ok': y}\n"
        )
        fn = self._write_fn(src_a)
        self._save_once(fn)
        # Rewrite with whitespace-only changes; the new source must still
        # hash-match what was written.
        self._write_fn(src_b)
        cell = self._cell_for_solver()
        self.assertIsNotNone(cell)
        self.assertFalse(
            cell.stale, "whitespace-only edit must not flip the harness hash"
        )

    def test_comment_edit_keeps_cell_fresh(self) -> None:
        src_a = textwrap.dedent(
            """\
            def run_baseline(cfg, tags):
                # original comment
                x = 1
                return {"ok": x}
            """
        )
        src_b = textwrap.dedent(
            """\
            def run_baseline(cfg, tags):
                # an entirely different comment
                x = 1  # trailing comment too
                return {"ok": x}
            """
        )
        fn = self._write_fn(src_a)
        self._save_once(fn)
        self._write_fn(src_b)
        cell = self._cell_for_solver()
        self.assertFalse(cell.stale, "comment-only edit must not flip the harness hash")

    def test_docstring_edit_keeps_cell_fresh(self) -> None:
        src_a = textwrap.dedent(
            '''\
            def run_baseline(cfg, tags):
                """Original docstring."""
                x = 1
                return {"ok": x}
            '''
        )
        src_b = textwrap.dedent(
            '''\
            def run_baseline(cfg, tags):
                """A wholly different docstring spanning several words."""
                x = 1
                return {"ok": x}
            '''
        )
        fn = self._write_fn(src_a)
        self._save_once(fn)
        self._write_fn(src_b)
        cell = self._cell_for_solver()
        self.assertFalse(
            cell.stale, "docstring-only edit must not flip the harness hash"
        )

    # ── sensitivity: behavioural edit → stale ────────────────────────────────

    def test_rename_local_marks_cell_stale(self) -> None:
        src_a = textwrap.dedent(
            """\
            def run_baseline(cfg, tags):
                x = 1
                return {"ok": x}
            """
        )
        src_b = textwrap.dedent(
            """\
            def run_baseline(cfg, tags):
                y = 1
                return {"ok": y}
            """
        )
        fn = self._write_fn(src_a)
        self._save_once(fn)
        self._write_fn(src_b)
        cell = self._cell_for_solver()
        self.assertTrue(
            cell.stale,
            "renaming a local variable is a behavioural edit — must flip the harness hash",
        )

    def test_added_statement_marks_cell_stale(self) -> None:
        src_a = textwrap.dedent(
            """\
            def run_baseline(cfg, tags):
                x = 1
                return {"ok": x}
            """
        )
        src_b = textwrap.dedent(
            """\
            def run_baseline(cfg, tags):
                x = 1
                x = x + 1
                return {"ok": x}
            """
        )
        fn = self._write_fn(src_a)
        self._save_once(fn)
        self._write_fn(src_b)
        cell = self._cell_for_solver()
        self.assertTrue(
            cell.stale, "adding a statement is a behavioural edit — must flip the hash"
        )

    def test_reordered_statements_mark_cell_stale(self) -> None:
        src_a = textwrap.dedent(
            """\
            def run_baseline(cfg, tags):
                x = 1
                y = 2
                return {"ok": x + y}
            """
        )
        src_b = textwrap.dedent(
            """\
            def run_baseline(cfg, tags):
                y = 2
                x = 1
                return {"ok": x + y}
            """
        )
        fn = self._write_fn(src_a)
        self._save_once(fn)
        self._write_fn(src_b)
        cell = self._cell_for_solver()
        self.assertTrue(cell.stale, "reordering statements must flip the hash")


# ── tesseract_content_hash round-trip ────────────────────────────────────────


class TesseractRoundtripTests(_StalenessRoundtripBase):
    """Write a result.json stamped with ``tesseract_hashes`` and verify
    ``collect_status`` agrees on the hash for the current tesseract source."""

    def setUp(self) -> None:
        self.solver = "solver_t"
        # Unique per-test tesseract tree so test-order doesn't matter.
        self.tess_root = self.tmp_root / f"tessroot_{id(self)}"
        self.solver_dir = self.tess_root / self.solver
        self.solver_dir.mkdir(parents=True, exist_ok=True)
        (self.solver_dir / "tesseract_api.py").write_text(
            "def apply(inputs):\n    return {'result': inputs['x']}\n"
        )
        (self.solver_dir / "tesseract_config.yaml").write_text("name: solver_t\n")

        # Stable harness function defined in this very file — its hash won't
        # change during the test run, so any staleness flipping must come
        # from the tesseract side.
        self.harness_fn = _stable_harness_fn

        self.cfg = _make_cfg(
            self.tmp_root,
            self.tess_root,
            self.solver,
            self.solver,
            problem_name=f"tess_p_{id(self)}",
        )
        self.exp_dir = self.results_root / self.cfg.name / "forward" / "baseline"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def _save_once(self) -> None:
        result = {
            "params": {},
            "by_solver": {self.solver: {"error": 0.01, "valid": True}},
        }
        save_experiment(result, self.exp_dir, cfg=self.cfg, harness_fn=self.harness_fn)

    def _cell(self):
        st = collect_status(self.cfg, suites=["forward"])
        for row in st.rows:
            if row.suite == "forward" and row.experiment == "baseline":
                return row.cells.get(self.solver)
        return None

    # ── invariant: fresh write → not stale ───────────────────────────────────

    def test_fresh_write_is_not_stale(self) -> None:
        self._save_once()
        cell = self._cell()
        self.assertIsNotNone(cell)
        self.assertEqual(cell.status, OK)
        self.assertFalse(
            cell.stale,
            "fresh tesseract write must read back as fresh — write/read hash drift?",
        )

    # ── invariant: mtime-only touch → not stale ──────────────────────────────

    def test_mtime_touch_keeps_cell_fresh(self) -> None:
        """tesseract_content_hash hashes file BYTES, not mtimes. Touching a
        file without changing its content must not flip the hash."""
        self._save_once()
        target = self.solver_dir / "tesseract_api.py"
        # Force a new mtime without mutating bytes.
        old_mtime = target.stat().st_mtime
        # bump mtime by 10 seconds
        import os as _os

        _os.utime(target, (old_mtime + 10, old_mtime + 10))
        self.assertNotEqual(target.stat().st_mtime, old_mtime)
        cell = self._cell()
        self.assertFalse(
            cell.stale,
            "mtime-only touch must not flip the tesseract content hash",
        )

    # ── invariant: excluded build artefact mutation → not stale ──────────────

    def test_pycache_change_keeps_cell_fresh(self) -> None:
        """Files under ``__pycache__`` / ``*.pyc`` are on the exclude list —
        mutating them must not mark peers stale."""
        self._save_once()
        cache = self.solver_dir / "__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / "tesseract_api.cpython-311.pyc").write_bytes(b"\x00\x01\x02")
        # Add a lockfile too — also excluded.
        (self.solver_dir / "poetry.lock").write_text("locked\n")
        cell = self._cell()
        self.assertFalse(
            cell.stale,
            "changes inside excluded paths must not flip the tesseract hash",
        )

    # ── sensitivity: byte-level content change → stale ───────────────────────

    def test_source_byte_change_marks_cell_stale(self) -> None:
        self._save_once()
        target = self.solver_dir / "tesseract_api.py"
        target.write_text(
            "def apply(inputs):\n    return {'result': inputs['x'] * 2}\n"
        )
        cell = self._cell()
        self.assertTrue(
            cell.stale,
            "changing a tesseract file's bytes must flip the tesseract hash",
        )

    def test_new_file_marks_cell_stale(self) -> None:
        self._save_once()
        (self.solver_dir / "extra.py").write_text("# new module\nx = 1\n")
        cell = self._cell()
        self.assertTrue(
            cell.stale, "adding a new file to a tesseract dir must flip the hash"
        )

    def test_deleted_file_marks_cell_stale(self) -> None:
        # Seed with an extra file so there's something to delete.
        extra = self.solver_dir / "extra.py"
        extra.write_text("x = 1\n")
        self._save_once()
        extra.unlink()
        cell = self._cell()
        self.assertTrue(
            cell.stale, "removing a file from a tesseract dir must flip the hash"
        )


# Stable module-level harness fn used by TesseractRoundtripTests. Its qualname
# resolves via importlib (module = this test module), so collect_status can
# re-hash it to verify row-level freshness. Keep its body fixed so any
# staleness flipping in those tests can only come from the tesseract side.
def _stable_harness_fn(cfg, tags):  # pragma: no cover — fingerprint only
    return {"ok": True}


if __name__ == "__main__":
    unittest.main()
