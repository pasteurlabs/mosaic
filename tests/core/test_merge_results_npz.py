# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the solver-level NPZ merge in .github/scripts/merge-results.py.

In CI the CPU and GPU jobs for the same (suite, problem) each write a
``fields.npz`` containing only the solvers that job ran — including the
``solver_names`` metadata array the field plots read their solver set from.
The merge must union the per-solver arrays *and* ``solver_names``; a naive
key-level merge keeps every array but lets the last artifact's
``solver_names`` hide the other job's solvers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[2] / ".github" / "scripts"
_spec = importlib.util.spec_from_file_location(
    "merge_results", _SCRIPTS / "merge-results.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_merge_npz = _mod._merge_npz


def _write_npz(path: Path, **arrays) -> Path:
    np.savez(path, **arrays)
    return path


class TestMergeNpzSolverUnion:
    def test_unions_solver_names_across_artifacts(self, tmp_path: Path):
        # Thermal-mesh split: CPU job runs the FEM triple, GPU job the rest.
        cpu = _write_npz(
            tmp_path / "cpu.npz",
            solver_names=np.array(["deal.II", "FEniCS", "Firedrake"]),
            sweep_values=np.array([0.05, 0.1]),
            **{"deal.II_0": np.array(1.0), "FEniCS_0": np.array(2.0)},
            Firedrake_0=np.array(3.0),
        )
        gpu = _write_npz(
            tmp_path / "gpu.npz",
            solver_names=np.array(["JAX-FEM", "torch-fem"]),
            sweep_values=np.array([0.05, 0.1]),
            **{"JAX-FEM_0": np.array(4.0), "torch-fem_0": np.array(5.0)},
        )
        out = tmp_path / "merged.npz"
        _merge_npz([cpu, gpu], out)

        with np.load(out) as data:
            names = [str(n) for n in data["solver_names"]]
            assert names == ["deal.II", "FEniCS", "Firedrake", "JAX-FEM", "torch-fem"]
            for key, val in [
                ("deal.II_0", 1.0),
                ("FEniCS_0", 2.0),
                ("Firedrake_0", 3.0),
                ("JAX-FEM_0", 4.0),
                ("torch-fem_0", 5.0),
            ]:
                assert float(data[key]) == val

    def test_later_artifact_wins_per_solver(self, tmp_path: Path):
        first = _write_npz(
            tmp_path / "a.npz",
            solver_names=np.array(["exponax"]),
            exponax_0=np.array([1.0]),
        )
        second = _write_npz(
            tmp_path / "b.npz",
            solver_names=np.array(["exponax"]),
            exponax_0=np.array([9.0]),
        )
        out = tmp_path / "merged.npz"
        _merge_npz([first, second], out)

        with np.load(out) as data:
            assert [str(n) for n in data["solver_names"]] == ["exponax"]
            np.testing.assert_array_equal(data["exponax_0"], [9.0])

    def test_positional_layout_reindexed(self, tmp_path: Path):
        # Gradient-suite layout: grad_{j} indexes into solver_names. After the
        # union the indices must follow the merged ordering, not the per-file one.
        a = _write_npz(
            tmp_path / "a.npz",
            solver_names=np.array(["solver_x"]),
            grad_0=np.array([1.0]),
        )
        b = _write_npz(
            tmp_path / "b.npz",
            solver_names=np.array(["solver_y"]),
            grad_0=np.array([2.0]),
        )
        out = tmp_path / "merged.npz"
        _merge_npz([a, b], out)

        with np.load(out) as data:
            assert [str(n) for n in data["solver_names"]] == ["solver_x", "solver_y"]
            np.testing.assert_array_equal(data["grad_0"], [1.0])
            np.testing.assert_array_equal(data["grad_1"], [2.0])

    def test_shared_arrays_survive(self, tmp_path: Path):
        # consensus_0 carries a trailing integer that must not be misread as a
        # solver index; later artifacts win for shared keys.
        a = _write_npz(
            tmp_path / "a.npz",
            solver_names=np.array(["solver_x"]),
            solver_x_0=np.array([1.0]),
            consensus_0=np.array([0.5]),
            x_axis=np.array([0.0, 1.0]),
        )
        b = _write_npz(
            tmp_path / "b.npz",
            solver_names=np.array(["solver_y"]),
            solver_y_0=np.array([2.0]),
            consensus_0=np.array([0.7]),
        )
        out = tmp_path / "merged.npz"
        _merge_npz([a, b], out)

        with np.load(out) as data:
            np.testing.assert_array_equal(data["consensus_0"], [0.7])
            np.testing.assert_array_equal(data["x_axis"], [0.0, 1.0])
            np.testing.assert_array_equal(data["solver_x_0"], [1.0])
            np.testing.assert_array_equal(data["solver_y_0"], [2.0])

    def test_unreadable_file_skipped(self, tmp_path: Path):
        good = _write_npz(
            tmp_path / "good.npz",
            solver_names=np.array(["solver_x"]),
            solver_x_0=np.array([1.0]),
        )
        bad = tmp_path / "bad.npz"
        bad.write_bytes(b"not an npz")
        out = tmp_path / "merged.npz"
        _merge_npz([bad, good], out)

        with np.load(out) as data:
            assert [str(n) for n in data["solver_names"]] == ["solver_x"]
