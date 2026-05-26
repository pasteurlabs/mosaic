# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``mosaic.benchmarks.core.partial`` — the partial-result
checkpoint helpers that power sub-experiment ``mosaic run --continue``.

Covers:
- ``write_partial`` atomically lands JSON under flock, works concurrently
  from multiple threads.
- ``done_solvers_in_partial`` correctly interprets each schema family
  (by_solver, by_sweep with outer=solver; by_param / by_N / by_steps with
  outer=sweep_value) and applies the ``in_progress`` flag for by_solver
  schemas + the intersection rule for sweep schemas.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from mosaic.benchmarks.core.partial import (
    done_solvers_in_partial,
    filter_resumable_solvers,
    write_partial,
)


class WritePartialTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_writes_json_atomically(self) -> None:
        payload = {"by_solver": {"a": {"foo": 1}}, "params": {"k": "v"}}
        write_partial(self.tmp / "exp", payload)
        path = self.tmp / "exp" / "result_partial.json"
        self.assertTrue(path.exists())
        with open(path) as f:
            self.assertEqual(json.load(f), payload)

    def test_overwrites_existing(self) -> None:
        d = self.tmp / "exp"
        write_partial(d, {"by_solver": {"a": {}}, "round": 1})
        write_partial(d, {"by_solver": {"a": {}, "b": {}}, "round": 2})
        with open(d / "result_partial.json") as f:
            data = json.load(f)
        self.assertEqual(data["round"], 2)
        self.assertEqual(set(data["by_solver"]), {"a", "b"})

    def test_threadsafe_with_lock(self) -> None:
        """Concurrent writers with a shared lock don't corrupt the file."""
        d = self.tmp / "exp"
        lock = threading.Lock()
        results: dict[str, dict] = {}

        def _worker(name: str) -> None:
            for i in range(20):
                results[name] = {"i": i}
                write_partial(
                    d,
                    {"by_solver": dict(results), "params": {}},
                    lock=lock,
                )

        threads = [threading.Thread(target=_worker, args=(n,)) for n in "abc"]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The final file should be parseable JSON with all three solvers.
        with open(d / "result_partial.json") as f:
            data = json.load(f)
        self.assertEqual(set(data["by_solver"]), {"a", "b", "c"})


class DoneSolversBySolverSchemaTests(unittest.TestCase):
    """``by_solver`` schema: outer = solver name, ``in_progress`` flag."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.path = self.tmp / "result_partial.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, data: dict) -> None:
        with open(self.path, "w") as f:
            json.dump(data, f)

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(done_solvers_in_partial(self.path), set())

    def test_unreadable_file_returns_empty(self) -> None:
        self.path.write_text("{not valid json")
        self.assertEqual(done_solvers_in_partial(self.path), set())

    def test_all_solvers_done(self) -> None:
        self._write({"by_solver": {"a": {"x": 1}, "b": {"x": 2}}})
        self.assertEqual(done_solvers_in_partial(self.path), {"a", "b"})

    def test_in_progress_solver_excluded(self) -> None:
        self._write(
            {
                "by_solver": {
                    "a": {"x": 1},
                    "b": {"in_progress": True, "x": 99},
                }
            }
        )
        self.assertEqual(done_solvers_in_partial(self.path), {"a"})

    def test_in_progress_false_is_done(self) -> None:
        """Explicit ``in_progress: False`` counts as done — only truthy excludes."""
        self._write({"by_solver": {"a": {"in_progress": False, "x": 1}}})
        self.assertEqual(done_solvers_in_partial(self.path), {"a"})


class DoneSolversBySweepOuterSolverTests(unittest.TestCase):
    """``by_sweep`` schema: outer = solver name (e.g. recovery_long_work)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.path = self.tmp / "result_partial.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, data: dict) -> None:
        with open(self.path, "w") as f:
            json.dump(data, f)

    def test_outer_solver_keyed(self) -> None:
        # by_sweep[solver][sweep_val] = result
        self._write(
            {
                "by_sweep": {
                    "a": {"0.1": {"ok": True}, "0.5": {"ok": True}},
                    "b": {"in_progress": True, "0.1": {"ok": True}},
                }
            }
        )
        self.assertEqual(done_solvers_in_partial(self.path), {"a"})


class DoneSolversByParamSchemaTests(unittest.TestCase):
    """``by_param``: outer = sweep value, inner = solver name (forward.py)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.path = self.tmp / "result_partial.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, data: dict) -> None:
        with open(self.path, "w") as f:
            json.dump(data, f)

    def test_intersection_across_buckets(self) -> None:
        # Solver "a" in every val; "b" only in 0.1 → only "a" is done.
        self._write(
            {
                "by_param": {
                    "0.1": {"a": {"err": 0.1}, "b": {"err": 0.2}},
                    "0.5": {"a": {"err": 0.3}},
                }
            }
        )
        self.assertEqual(done_solvers_in_partial(self.path), {"a"})

    def test_all_solvers_present_in_all_buckets(self) -> None:
        self._write(
            {
                "by_param": {
                    "0.1": {"a": {}, "b": {}},
                    "0.5": {"a": {}, "b": {}},
                }
            }
        )
        self.assertEqual(done_solvers_in_partial(self.path), {"a", "b"})

    def test_by_N_schema(self) -> None:
        self._write({"by_N": {"32": {"a": {}}, "64": {"a": {}, "b": {}}}})
        self.assertEqual(done_solvers_in_partial(self.path), {"a"})

    def test_by_steps_schema(self) -> None:
        self._write({"by_steps": {"10": {"a": {}}, "20": {"a": {}}}})
        self.assertEqual(done_solvers_in_partial(self.path), {"a"})

    def test_empty_buckets_returns_empty(self) -> None:
        self._write({"by_param": {}})
        self.assertEqual(done_solvers_in_partial(self.path), set())


class FilterResumableSolversTests(unittest.TestCase):
    """The convenience wrapper each harness uses at entry."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_partial(self, data: dict) -> None:
        (self.tmp / "result_partial.json").write_text(json.dumps(data))

    def test_no_resume_passthrough(self) -> None:
        """Without overrides["resume"], the list is unchanged even if a
        partial exists.
        """
        self._write_partial({"by_solver": {"a": {}, "b": {}}})
        self.assertEqual(
            filter_resumable_solvers(["a", "b", "c"], self.tmp, None),
            ["a", "b", "c"],
        )
        self.assertEqual(
            filter_resumable_solvers(["a", "b", "c"], self.tmp, {}),
            ["a", "b", "c"],
        )

    def test_resume_no_partial_passthrough(self) -> None:
        """With --continue but no partial file → run everything."""
        self.assertEqual(
            filter_resumable_solvers(["a", "b"], self.tmp, {"resume": True}),
            ["a", "b"],
        )

    def test_resume_drops_completed(self) -> None:
        self._write_partial({"by_solver": {"a": {}, "b": {"in_progress": True}}})
        self.assertEqual(
            filter_resumable_solvers(["a", "b", "c"], self.tmp, {"resume": True}),
            ["b", "c"],
        )

    def test_preserves_order(self) -> None:
        self._write_partial({"by_solver": {"b": {}}})
        self.assertEqual(
            filter_resumable_solvers(["a", "b", "c", "d"], self.tmp, {"resume": True}),
            ["a", "c", "d"],
        )


class DoneSolversFallbackTests(unittest.TestCase):
    """No recognised schema → empty set (don't skip anything)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.path = self.tmp / "result_partial.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_unknown_schema_returns_empty(self) -> None:
        with open(self.path, "w") as f:
            json.dump({"some_other_key": {"a": 1}}, f)
        self.assertEqual(done_solvers_in_partial(self.path), set())

    def test_non_dict_root_returns_empty(self) -> None:
        with open(self.path, "w") as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(done_solvers_in_partial(self.path), set())


if __name__ == "__main__":
    unittest.main()
