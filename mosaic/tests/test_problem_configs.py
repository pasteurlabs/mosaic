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


def test_config_has_solvers(problem_config):
    assert len(problem_config.solvers) >= 1


def test_solver_required_fields(problem_config):
    """Every solver must have name, dir, scheme, backend, and color populated."""
    for key, spec in problem_config.solvers.items():
        assert spec.name, f"{key}: missing name"
        assert spec.dir, f"{key}: missing dir"
        assert spec.scheme, f"{key}: missing scheme"
        assert spec.backend, f"{key}: missing backend"
        assert spec.color, f"{key}: missing color"


def test_solver_colors_are_valid_hex(problem_config):
    for key, spec in problem_config.solvers.items():
        assert re.match(r"^#[0-9a-fA-F]{6}$", spec.color), (
            f"{key}: invalid hex color {spec.color!r}"
        )


def test_no_duplicate_dirs(problem_config):
    """Within a problem, no two solvers should share the same tesseract dir."""
    dirs = [spec.dir for spec in problem_config.solvers.values()]
    assert len(dirs) == len(set(dirs)), f"Duplicate dirs: {dirs}"


def test_make_ic_populated(problem_config):
    """Every problem must define at least one initial condition."""
    assert len(problem_config.make_ic) >= 1


def test_has_error_fn(problem_config):
    assert callable(problem_config.error_fn)


def test_has_make_inputs(problem_config):
    assert callable(problem_config.make_inputs)


def test_ad_strategy_values(problem_config):
    """ad_strategy must be one of the known values or None."""
    valid = {"autodiff", "adjoint", "hybrid", None}
    for key, spec in problem_config.solvers.items():
        assert spec.ad_strategy in valid, (
            f"{key}: ad_strategy={spec.ad_strategy!r} not in {valid}"
        )


def test_exclusion_keys_are_strings(problem_config):
    """Exclusion keys and values must be well-formed."""
    for key, spec in problem_config.solvers.items():
        for exc_key, exc_val in spec.exclusions.items():
            assert isinstance(exc_key, str), f"{key}: exclusion key {exc_key!r}"
            assert isinstance(exc_val, (str, dict)), f"{key}: exclusion val {exc_val!r}"


def test_unknown_problem_raises():
    with pytest.raises(ValueError, match="Unknown problem"):
        get_config("nonexistent-problem")
