# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CLI helper functions that decide which solvers run.

_apply_solver_filter, _resolve_gpu_pool, and _filter_hardware are called on
every ``mosaic run`` invocation. A bug here silently runs the wrong subset
of solvers, producing incomplete benchmark tables.
"""

from __future__ import annotations

import pytest

from mosaic.benchmarks.core.config import Problem, SolverSpec


def _make_spec(name: str, *, uses_gpu: bool = True) -> SolverSpec:
    return SolverSpec(
        dir=name.lower().replace(" ", "-"),
        color="#000000",
        name=name,
        scheme="test",
        backend="test",
        uses_gpu=uses_gpu,
    )


def _make_problem(
    name: str = "test-problem",
    solvers: list[SolverSpec] | None = None,
) -> Problem:
    if solvers is None:
        solvers = [
            _make_spec("XLB", uses_gpu=True),
            _make_spec("jax-cfd", uses_gpu=True),
            _make_spec("Firedrake", uses_gpu=False),
        ]
    return Problem(name=name, solvers=solvers)


# -- _apply_solver_filter ------------------------------------------------------


class TestApplySolverFilter:
    @pytest.fixture(autouse=True)
    def _import(self):
        from mosaic.benchmarks.cli._helpers import _apply_solver_filter

        self.filter = _apply_solver_filter

    def test_none_csv_passthrough(self):
        cfg = _make_problem()
        result = self.filter(cfg, None)
        assert [s.name for s in result.solvers] == ["XLB", "jax-cfd", "Firedrake"]

    def test_empty_csv_passthrough(self):
        cfg = _make_problem()
        result = self.filter(cfg, "")
        assert [s.name for s in result.solvers] == ["XLB", "jax-cfd", "Firedrake"]

    def test_flat_csv_keeps_matching(self):
        cfg = _make_problem()
        result = self.filter(cfg, "XLB,Firedrake")
        assert [s.name for s in result.solvers] == ["XLB", "Firedrake"]

    def test_flat_csv_ignores_unknown_names(self):
        cfg = _make_problem()
        result = self.filter(cfg, "XLB,NonExistent")
        assert [s.name for s in result.solvers] == ["XLB"]

    def test_flat_csv_no_match_returns_none(self):
        cfg = _make_problem()
        result = self.filter(cfg, "DoesNotExist,AlsoMissing")
        assert result is None

    def test_per_problem_map_filters_correct_problem(self):
        cfg = _make_problem(name="ns-grid")
        result = self.filter(cfg, "ns-grid=XLB,Firedrake;thermal=FEniCSx")
        assert [s.name for s in result.solvers] == ["XLB", "Firedrake"]

    def test_per_problem_map_unmentioned_keeps_all(self):
        cfg = _make_problem(name="structural-mesh")
        result = self.filter(cfg, "ns-grid=XLB;thermal=FEniCSx")
        assert [s.name for s in result.solvers] == ["XLB", "jax-cfd", "Firedrake"]

    def test_per_problem_map_no_match_returns_none(self):
        cfg = _make_problem(name="ns-grid")
        result = self.filter(cfg, "ns-grid=DoesNotExist")
        # Unknown solver in addressed problem → returns None after warning.
        assert result is None

    def test_preserves_solver_order(self):
        cfg = _make_problem()
        result = self.filter(cfg, "Firedrake,XLB")
        # Order follows cfg.solvers, not the CSV.
        assert [s.name for s in result.solvers] == ["XLB", "Firedrake"]


# -- _resolve_gpu_pool ---------------------------------------------------------


class TestResolveGpuPool:
    @pytest.fixture(autouse=True)
    def _import(self):
        from mosaic.benchmarks.cli._helpers import _resolve_gpu_pool

        self.resolve = _resolve_gpu_pool

    def test_none_gpus_passthrough(self):
        cfg = _make_problem()
        result_cfg, result_gpus = self.resolve(cfg, None)
        assert [s.name for s in result_cfg.solvers] == ["XLB", "jax-cfd", "Firedrake"]
        assert result_gpus is None

    def test_csv_gpus_passthrough(self):
        cfg = _make_problem()
        result_cfg, result_gpus = self.resolve(cfg, "0,1")
        assert [s.name for s in result_cfg.solvers] == ["XLB", "jax-cfd", "Firedrake"]
        assert result_gpus == "0,1"

    def test_none_string_filters_to_cpu_only(self):
        cfg = _make_problem()
        result_cfg, result_gpus = self.resolve(cfg, "none")
        assert [s.name for s in result_cfg.solvers] == ["Firedrake"]
        assert result_gpus == "cpu-only"

    def test_cpu_string_filters_to_cpu_only(self):
        cfg = _make_problem()
        result_cfg, result_gpus = self.resolve(cfg, "cpu")
        assert [s.name for s in result_cfg.solvers] == ["Firedrake"]
        assert result_gpus == "cpu-only"

    def test_none_string_all_gpu_returns_original(self):
        gpu_only = [
            _make_spec("A", uses_gpu=True),
            _make_spec("B", uses_gpu=True),
        ]
        cfg = _make_problem(solvers=gpu_only)
        result_cfg, result_gpus = self.resolve(cfg, "none")
        # No CPU-only solvers → returns original cfg, gpus=None.
        assert [s.name for s in result_cfg.solvers] == ["A", "B"]
        assert result_gpus is None


# -- _filter_hardware ----------------------------------------------------------


class TestFilterHardware:
    @pytest.fixture(autouse=True)
    def _import(self):
        from mosaic.benchmarks.cli._helpers import _filter_hardware

        self.filter = _filter_hardware

    def test_all_passthrough(self):
        cfg = _make_problem()
        result = self.filter(cfg, "all")
        assert [s.name for s in result.solvers] == ["XLB", "jax-cfd", "Firedrake"]

    def test_cpu_keeps_only_cpu(self):
        cfg = _make_problem()
        result = self.filter(cfg, "cpu")
        assert [s.name for s in result.solvers] == ["Firedrake"]

    def test_gpu_keeps_only_gpu(self):
        cfg = _make_problem()
        result = self.filter(cfg, "gpu")
        assert [s.name for s in result.solvers] == ["XLB", "jax-cfd"]

    def test_cpu_no_cpu_solvers_keeps_all(self):
        gpu_only = [
            _make_spec("A", uses_gpu=True),
            _make_spec("B", uses_gpu=True),
        ]
        cfg = _make_problem(solvers=gpu_only)
        result = self.filter(cfg, "cpu")
        # No CPU solvers → warning, returns unchanged cfg.
        assert [s.name for s in result.solvers] == ["A", "B"]

    def test_gpu_no_gpu_solvers_keeps_all(self):
        cpu_only = [
            _make_spec("A", uses_gpu=False),
            _make_spec("B", uses_gpu=False),
        ]
        cfg = _make_problem(solvers=cpu_only)
        result = self.filter(cfg, "gpu")
        # No GPU solvers → warning, returns unchanged cfg.
        assert [s.name for s in result.solvers] == ["A", "B"]

    def test_none_no_gpu_available_drops_gpu_solvers(self, monkeypatch):
        monkeypatch.setattr("mosaic.benchmarks.core.hardware.has_gpu", lambda: False)
        cfg = _make_problem()
        result = self.filter(cfg, None)
        assert [s.name for s in result.solvers] == ["Firedrake"]

    def test_none_gpu_available_keeps_all(self, monkeypatch):
        monkeypatch.setattr("mosaic.benchmarks.core.hardware.has_gpu", lambda: True)
        cfg = _make_problem()
        result = self.filter(cfg, None)
        assert [s.name for s in result.solvers] == ["XLB", "jax-cfd", "Firedrake"]

    def test_unknown_hardware_value_keeps_all(self):
        cfg = _make_problem()
        result = self.filter(cfg, "tpu")
        assert [s.name for s in result.solvers] == ["XLB", "jax-cfd", "Firedrake"]
