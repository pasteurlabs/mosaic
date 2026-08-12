# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLI command handlers beyond --help.

These exercise the typer commands through :class:`typer.testing.CliRunner`,
which invokes the registered ``app`` directly in-process. The goal is to cover
argument parsing, dispatch, and error paths (invalid problem, missing JSON
files, malformed input) — not the heavy lifting (which spawns subprocesses /
Docker and is exercised by ``test_integration.py``).
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from mosaic.benchmarks.cli import app

runner = CliRunner()

# ``validate-template`` imports the template's ``schema_module``
# (e.g. ``mosaic_shared.problems.navier_stokes_grid``). Those modules are
# only importable when ``mosaic_shared`` is installed as a top-level
# package — true inside the tesseract runtime / a full dev environment, not
# in a minimal test environment. Mirror the gate used in test_schemas.py.
_missing_runtime_deps = False
try:
    import mosaic_shared.schema_types  # noqa: F401
except ModuleNotFoundError:
    _missing_runtime_deps = True

needs_runtime = pytest.mark.skipif(
    _missing_runtime_deps,
    reason="mosaic_shared not installed as a top-level package",
)


# ── compare ───────────────────────────────────────────────────────────────────


def test_compare_missing_before_file_exits_nonzero(tmp_path):
    """Compare exits 1 with a clear error when 'before' is missing."""
    after = tmp_path / "after.json"
    after.write_text("{}")
    result = runner.invoke(app, ["compare", str(tmp_path / "nope.json"), str(after)])
    assert result.exit_code == 1
    assert "before file not found" in result.output


def test_compare_missing_after_file_exits_nonzero(tmp_path):
    """Compare exits 1 with a clear error when 'after' is missing."""
    before = tmp_path / "before.json"
    before.write_text("{}")
    result = runner.invoke(app, ["compare", str(before), str(tmp_path / "nope.json")])
    assert result.exit_code == 1
    assert "after file not found" in result.output


def test_compare_malformed_json_exits_nonzero(tmp_path):
    """Compare exits 1 when the JSON files are unparseable."""
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text("not valid json {{")
    after.write_text("{}")
    result = runner.invoke(app, ["compare", str(before), str(after)])
    assert result.exit_code == 1
    assert "reading JSON snapshots" in result.output


def test_compare_valid_snapshots_exits_zero(tmp_path):
    """Two empty-but-valid snapshots produce a clean diff and exit 0."""
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    snapshot = {"problems": {}}
    before.write_text(json.dumps(snapshot))
    after.write_text(json.dumps(snapshot))
    result = runner.invoke(app, ["compare", str(before), str(after)])
    assert result.exit_code == 0, result.output


# ── validate-template ─────────────────────────────────────────────────────────


def test_validate_template_unknown_name_exits_nonzero():
    """An unknown template name surfaces FileNotFoundError, not a crash."""
    result = runner.invoke(app, ["validate-template", "no-such-template-xyz"])
    assert result.exit_code != 0
    # FileNotFoundError comes through as the exception text
    assert result.exception is not None
    assert isinstance(result.exception, FileNotFoundError)


@needs_runtime
def test_validate_template_known_template_succeeds():
    """A built-in template passes validation."""
    result = runner.invoke(app, ["validate-template", "ns-periodic"])
    assert result.exit_code == 0, result.output
    assert "is valid" in result.output


# ── templates ─────────────────────────────────────────────────────────────────


def test_templates_lists_all_builtins():
    """Bare `templates` lists each registered template name."""
    result = runner.invoke(app, ["templates"])
    assert result.exit_code == 0, result.output
    for name in ("ns-periodic", "structural-steady", "thermal-steady"):
        assert name in result.output


def test_templates_show_prints_metadata():
    """`templates --show <name>` prints the schema module and output_key."""
    result = runner.invoke(app, ["templates", "--show", "ns-periodic"])
    assert result.exit_code == 0, result.output
    assert "navier_stokes_grid" in result.output
    assert "output_key" in result.output


def test_templates_show_unknown_template_errors():
    """`templates --show <bogus>` raises FileNotFoundError."""
    result = runner.invoke(app, ["templates", "--show", "no-such-template-xyz"])
    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)


# ── validate-domain ───────────────────────────────────────────────────────────


def test_validate_domain_unknown_problem_errors():
    """validate-domain on an unknown problem surfaces the get_config error."""
    result = runner.invoke(app, ["validate-domain", "not-a-real-problem"])
    assert result.exit_code != 0
    # get_config raises a ValueError naming the problem
    assert result.exception is not None


def test_validate_domain_known_problem_runs():
    """validate-domain on a known problem prints check results and exits."""
    result = runner.invoke(app, ["validate-domain", "ns-grid"])
    # Whether all checks pass depends on the environment (schema imports may
    # not all resolve in test envs); the contract is just that the command
    # runs to completion and prints at least one OK or FAIL line.
    assert "Problem.validate" in result.output
    assert result.exit_code in (0, 1)


# ── new-domain ────────────────────────────────────────────────────────────────


def test_new_domain_requires_from_template():
    """`new-domain <name>` without --from-template is a usage error."""
    import re

    result = runner.invoke(app, ["new-domain", "my-test-domain"])
    assert result.exit_code != 0
    # typer reports the missing option in stderr (mixed into output by
    # default). Strip ANSI escapes — under ``--color=always`` Rich inserts
    # them mid-token (``-from\x1b[0m\x1b[1;36m-template``) and a naive
    # substring check misses the split.
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", result.output)
    assert "from-template" in clean or "from_template" in clean


def test_new_domain_unknown_template_errors():
    """`new-domain <name> -t <bogus>` raises FileNotFoundError."""
    result = runner.invoke(
        app, ["new-domain", "my-test-domain", "-t", "no-such-template-xyz"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)


# ── build ─────────────────────────────────────────────────────────────────────


def test_build_unknown_problem_records_failure(monkeypatch):
    """`build -p <bogus>` reports the failure and exits 1 (per-problem isolation)."""
    # Don't actually invoke docker — make build_all a no-op for any
    # successful path. The unknown-problem branch raises in get_config before
    # build_all is reached, so this is just defensive.
    from mosaic.benchmarks import cli as cli_mod

    monkeypatch.setattr(cli_mod, "build_all", lambda cfg, max_workers=2: {})

    result = runner.invoke(app, ["build", "-p", "definitely-not-a-real-problem"])
    assert result.exit_code == 1
    assert "definitely-not-a-real-problem" in result.output


# ── run: -s validation across problems ──────────────────────────────────────────


class _StopBeforeRun(BaseException):
    """Sentinel: raised in place of the per-problem work so ``run`` gets past
    ``-s`` validation without building images or launching a benchmark.

    Subclasses ``BaseException`` (not ``Exception``) so the per-problem
    ``try/except Exception`` in ``run`` — which turns build errors into an
    "error" status and swallows them — lets it propagate to the test.
    """


@pytest.fixture
def stop_before_run(monkeypatch):
    """Make ``run`` reach the per-problem loop and then stop immediately.

    ``_run_prepare_problem`` is the first thing ``run`` calls per problem,
    after suite and ``-s`` validation. Replacing it with a raiser proves a run
    cleared validation while guaranteeing no Docker/build/run work happens.
    """
    from mosaic.benchmarks.cli import run as run_mod

    def _raise(*_args, **_kwargs):
        raise _StopBeforeRun

    monkeypatch.setattr(run_mod, "_run_prepare_problem", _raise)


def test_run_flat_solvers_from_other_problem_passes_validation(stop_before_run):
    """The sharded-benchmark reproducer: a flat ``-s`` list spanning problem
    domains must clear validation on a single-problem leg.

    ``TopOpt.jl`` lives on structural-mesh, not ns-grid; the union-set ``-s``
    is applied per problem downstream, so it must not be rejected up front.
    """
    # Reaching _run_prepare_problem means validation passed; the sentinel then
    # aborts before any real work. A validation failure would exit 1 first,
    # before the loop, so the sentinel would never be raised.
    with pytest.raises(_StopBeforeRun):
        runner.invoke(
            app,
            ["run", "-p", "ns-grid", "-s", "INS.jl,TopOpt.jl", "--plots-only"],
            catch_exceptions=False,
        )


def test_run_flat_solvers_reverse_direction_passes_validation(stop_before_run):
    """The reproducer fails in both directions: structural-mesh must likewise
    accept ``INS.jl`` (an ns-grid solver) in the flat union list."""
    with pytest.raises(_StopBeforeRun):
        runner.invoke(
            app,
            ["run", "-p", "structural-mesh", "-s", "INS.jl,TopOpt.jl", "--plots-only"],
            catch_exceptions=False,
        )


def test_run_genuine_typo_still_fails_fast(stop_before_run):
    """A name on no registered problem is a real typo and must still exit 1 at
    validation — before the per-problem loop (the sentinel) is ever reached."""
    # Validation exits via typer.Exit(1) before the per-problem loop, so the
    # sentinel is never raised — invoke() returns normally with exit_code 1.
    result = runner.invoke(
        app, ["run", "-p", "ns-grid", "-s", "INS.jl,TopOptt.jl", "--plots-only"]
    )
    assert result.exit_code == 1
    assert not isinstance(result.exception, _StopBeforeRun)
    assert "Unknown solver 'TopOptt.jl'" in result.output
    assert "TopOpt.jl" in result.output  # "did you mean" suggestion
