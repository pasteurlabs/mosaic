# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for problem configs: loading, required fields, and consistency."""

from __future__ import annotations

import re
import subprocess
import sys

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


def test_problems_load_without_mosaic_shared():
    """The harness environment does not install ``mosaic_shared``, only images do.

    The registry skips a problem config that fails to import, so import each
    one directly.
    """
    code = (
        "import importlib, sys\n"
        "sys.modules['mosaic_shared'] = None\n"
        "from mosaic.benchmarks.problems import _candidate_module_paths\n"
        "for name in _candidate_module_paths():\n"
        "    importlib.import_module(name)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr[-2000:]


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


def test_exclusion_keys_match_experiments(problem_config):
    """Every exclusion key names a registered experiment or a prefix of one."""
    experiments = problem_config.experiments
    for solver_name, per_exp in problem_config.exclusions.items():
        for exc_key in per_exp:
            assert any(
                key == exc_key or key.startswith(f"{exc_key}/") for key in experiments
            ), f"{solver_name}: exclusion key {exc_key!r} matches no experiment"


def test_indicative_metrics_name_experiments_and_metrics(problem_config):
    """Every indicative-metric override names a registered experiment and a known metric."""
    from mosaic.benchmarks.core.status import METRIC_SPECS

    for label, metrics in problem_config.indicative_metrics.items():
        assert label in problem_config.experiments, label
        assert set(metrics) <= METRIC_SPECS.keys(), (label, metrics)


def test_unknown_problem_raises():
    with pytest.raises(ValueError, match="Unknown problem"):
        get_config("nonexistent-problem")


#: Suites that run but carry no pass/fail notion, so a check would have
#: nothing to assert. ``ics`` registrations render initial conditions and are
#: never compared across solvers; the sweeps below publish a spectrum or a
#: curve rather than a value with a correct answer.
_UNGATED_BY_DESIGN: frozenset[str] = frozenset(
    {
        "ns-grid/ics",
        "ns-3d-grid/ics",
        "thermal-mesh/ics",
        "structural-mesh/ics",
        "ns-grid/gradient/jacobian_svd",
        "ns-grid/gradient/jacobian_svd_nu01",
        "ns-grid/gradient/jacobian_svd_steps20",
        "ns-grid/gradient/jacobian_svd_steps40",
        "ns-grid/gradient/horizon_sweep",
        "ns-grid/gradient/param_sweep",
        "ns-3d-grid/gradient/jacobian_svd",
        "ns-3d-grid/gradient/jacobian_svd_nu01",
        "ns-3d-grid/gradient/jacobian_svd_steps20",
        "ns-3d-grid/gradient/jacobian_svd_steps40",
        "ns-3d-grid/gradient/horizon_sweep_limits",
        "thermal-mesh/gradient/param_sweep",
        "thermal-mesh/gradient/source_width_sweep",
        "structural-mesh/gradient/param_sweep",
    }
)


def test_every_experiment_resolves_a_status_check(problem_config):
    """An experiment that runs but resolves no check publishes ok regardless.

    A problem-level check is not enough on its own: the lookup is per suite,
    so a problem can gate ``forward`` and leave ``optimization`` open. Failing
    here makes an ungated experiment opt-in, either by configuring a check or
    by naming it in ``_UNGATED_BY_DESIGN`` with a reason.
    """
    from mosaic.benchmarks.core.status import _lookup_check

    ungated = []
    for key in sorted(problem_config.experiments):
        suite, _, experiment = key.partition("/")
        full = f"{problem_config.name}/{key}"
        if (
            full in _UNGATED_BY_DESIGN
            or f"{problem_config.name}/{suite}" in _UNGATED_BY_DESIGN
        ):
            continue
        if not _lookup_check(problem_config, suite, experiment):
            ungated.append(key)

    assert not ungated, (
        f"{problem_config.name} runs {len(ungated)} experiment(s) that resolve "
        f"no status check, so they publish ok for any solver returning finite "
        f"numbers: {', '.join(ungated)}. Configure a check for the suite, or "
        f"list each in _UNGATED_BY_DESIGN with a reason."
    )


#: Experiments whose resolved checks are looser than their suite default. The
#: reason for each belongs in a comment at the override.
_LOOSENED_BY_DESIGN: frozenset[str] = frozenset(
    {
        "ns-grid/forward/agreement/multimode",
        "ns-grid/forward/cylinder",
        "ns-grid/forward/kolmogorov",
        "ns-grid/forward/kolmogorov_laminar",
        "ns-grid/optimization/drag_opt",
    }
)

#: Whether raising each built-in factory's thresholds lets more results pass.
_LOOSER_WHEN_HIGHER: dict[str, bool] = {
    "median_k": True,
    "max_error": True,
    "max_peer_k": True,
    "min_cosine": False,
    "max_rel_err": True,
    "rel_err_peer_outlier": True,
    "max_final_ratio": True,
}


def _thresholds(check) -> tuple[str, dict] | None:
    """Factory name and captured thresholds of a built-in check, or None."""
    factory, _, local = check.__qualname__.partition(".<locals>.")
    if not local:
        return None
    cells = check.__closure__ or ()
    return factory, {
        name: cell.cell_contents
        for name, cell in zip(check.__code__.co_freevars, cells, strict=True)
    }


def _loosens(default: list, override: list) -> bool:
    """Whether *override* drops a check of *default* or moves one toward passing."""
    built_in = dict(t for t in map(_thresholds, override) if t)
    for check in default:
        spec = _thresholds(check)
        if spec is None:
            if check not in override:
                return True
            continue
        factory, params = spec
        if factory not in built_in:
            return True
        sign = 1 if _LOOSER_WHEN_HIGHER[factory] else -1
        for k, v in params.items():
            w = built_in[factory][k]
            numeric = isinstance(v, int | float) and isinstance(w, int | float)
            # A changed non-numeric parameter, e.g. max_error's ``at``, narrows
            # what the check judges.
            if (numeric and sign * (w - v) > 0) or (not numeric and w != v):
                return True
    return False


def test_loosened_status_checks_are_deliberate(problem_config):
    """Overrides replace the suite default, so loosening it must be opt-in."""
    from mosaic.benchmarks.core.status import _lookup_check
    from mosaic.benchmarks.core.status_checks import normalize

    loosened = []
    for key in sorted(problem_config.experiments):
        suite, _, experiment = key.partition("/")
        if f"{problem_config.name}/{key}" in _LOOSENED_BY_DESIGN:
            continue
        default = normalize(problem_config.status_checks.get(suite))
        if _loosens(default, _lookup_check(problem_config, suite, experiment)):
            loosened.append(key)

    assert not loosened, (
        f"{problem_config.name} loosens its suite-default status checks on "
        f"{', '.join(loosened)}. Give the reason in a comment at the override "
        f"and list each in _LOOSENED_BY_DESIGN, or keep the default checks."
    )
