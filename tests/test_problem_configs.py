# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for problem configs: loading, required fields, and consistency."""

from __future__ import annotations

import re

import pytest

from mosaic.benchmarks.problems import PROBLEMS, get_config


@pytest.fixture(params=PROBLEMS)
def problem_config(request):
    return get_config(request.param)


def test_all_problems_load():
    """Every registered problem must load without error."""
    assert len(PROBLEMS) >= 4
    for name in PROBLEMS:
        cfg = get_config(name)
        assert cfg.name == name


def test_solver_required_fields(problem_config):
    """Every solver must have name, dir, scheme, backend, and color populated."""
    for spec in problem_config.solvers:
        assert spec.name, f"{spec.name}: missing name"
        assert spec.dir, f"{spec.name}: missing dir"
        assert spec.scheme, f"{spec.name}: missing scheme"
        assert spec.backend, f"{spec.name}: missing backend"
        assert spec.color, f"{spec.name}: missing color"


def test_solver_colors_are_valid_hex(problem_config):
    for spec in problem_config.solvers:
        assert re.match(r"^#[0-9a-fA-F]{6}$", spec.color), (
            f"{spec.name}: invalid hex color {spec.color!r}"
        )


def test_no_duplicate_dirs(problem_config):
    """Within a problem, no two solvers should share the same tesseract dir."""
    dirs = [spec.dir for spec in problem_config.solvers]
    assert len(dirs) == len(set(dirs)), f"Duplicate dirs: {dirs}"


def test_ad_strategy_values(problem_config):
    """ad_strategy must be one of the known values or None."""
    valid = {"autodiff", "adjoint", "hybrid", None}
    for spec in problem_config.solvers:
        assert spec.ad_strategy in valid, (
            f"{spec.name}: ad_strategy={spec.ad_strategy!r} not in {valid}"
        )


def test_exclusion_keys_are_strings(problem_config):
    """Exclusion keys and values must be well-formed.

    ``cfg.exclusions`` is the single source of truth: solver_name → exp_key →
    :class:`Exclusion`.
    """
    from mosaic.benchmarks.core.config import Exclusion

    for solver_name, per_exp in problem_config.exclusions.items():
        assert isinstance(solver_name, str), f"solver name {solver_name!r}"
        for exc_key, exc_val in per_exp.items():
            assert isinstance(exc_key, str), f"{solver_name}: exclusion key {exc_key!r}"
            assert isinstance(exc_val, Exclusion), (
                f"{solver_name}: exclusion val {exc_val!r} is not an Exclusion"
            )


def test_unknown_problem_raises():
    with pytest.raises(ValueError, match="Unknown problem"):
        get_config("nonexistent-problem")
