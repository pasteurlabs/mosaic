# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for save_npz_merged.

save_npz_merged is the write path for all field snapshots. Multiple solver
processes write to the same file under FileLock. These tests verify that the
merge logic correctly combines arrays and that concurrent writes don't lose
data.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from mosaic.benchmarks.core.io import save_npz_merged, try_load_npz


class TestSaveNpzMerged:
    def test_creates_new_file(self, tmp_path: Path):
        path = tmp_path / "fields.npz"
        save_npz_merged(path, {"solver_a": np.array([1.0, 2.0])})
        data = try_load_npz(path)
        assert "solver_a" in data
        np.testing.assert_array_equal(data["solver_a"], [1.0, 2.0])

    def test_merge_adds_new_key(self, tmp_path: Path):
        path = tmp_path / "fields.npz"
        save_npz_merged(path, {"solver_a": np.array([1.0])})
        save_npz_merged(path, {"solver_b": np.array([2.0])})
        data = try_load_npz(path)
        assert "solver_a" in data
        assert "solver_b" in data
        np.testing.assert_array_equal(data["solver_a"], [1.0])
        np.testing.assert_array_equal(data["solver_b"], [2.0])

    def test_merge_overwrites_on_collision(self, tmp_path: Path):
        path = tmp_path / "fields.npz"
        save_npz_merged(path, {"solver_a": np.array([1.0])})
        save_npz_merged(path, {"solver_a": np.array([99.0])})
        data = try_load_npz(path)
        np.testing.assert_array_equal(data["solver_a"], [99.0])

    def test_keep_old_predicate(self, tmp_path: Path):
        path = tmp_path / "fields.npz"
        save_npz_merged(
            path,
            {"ref_baseline": np.array([1.0]), "old_key": np.array([2.0])},
        )
        save_npz_merged(
            path,
            {"new_key": np.array([3.0])},
            keep_old=lambda k: k.startswith("ref_"),
        )
        data = try_load_npz(path)
        assert "ref_baseline" in data
        assert "new_key" in data
        assert "old_key" not in data

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "a" / "b" / "fields.npz"
        save_npz_merged(path, {"x": np.array([1.0])})
        assert path.exists()

    def test_concurrent_merges_no_data_loss(self, tmp_path: Path):
        """Two threads writing different keys — both must appear in the final file."""
        path = tmp_path / "fields.npz"
        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def writer(key: str, value: float):
            try:
                barrier.wait(timeout=5)
                save_npz_merged(path, {key: np.array([value])})
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=writer, args=("solver_a", 1.0))
        t2 = threading.Thread(target=writer, args=("solver_b", 2.0))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        data = try_load_npz(path)
        assert "solver_a" in data, f"solver_a missing; keys = {list(data)}"
        assert "solver_b" in data, f"solver_b missing; keys = {list(data)}"
