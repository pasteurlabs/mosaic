# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the CLI: verify all subcommands exist and --help works."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

MOSAIC = [sys.executable, "-m", "mosaic.benchmarks.cli"]

# Disable Rich/Click colour output so substring assertions against --help
# stdout aren't broken by ANSI escapes splitting flag names (Rich renders
# ``--flag`` as ``-\x1b[…m-flag`` in narrow non-TTY environments like CI).
_NO_COLOR_ENV = {**os.environ, "NO_COLOR": "1", "TERM": "dumb"}


def _run_help(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*MOSAIC, *args, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        env=_NO_COLOR_ENV,
        check=False,
    )


SUBCOMMANDS = [
    "run",
    "ics",
    "build",
    "status",
    "compare",
    "validate-domain",
    "new-domain",
    "validate-template",
    "templates",
]


def test_top_level_help():
    """Top-level --help must list every registered subcommand."""
    result = _run_help([])
    assert result.returncode == 0
    out = result.stdout
    assert "Usage:" in out
    missing = [s for s in SUBCOMMANDS if s not in out]
    assert not missing, f"Subcommands missing from --help: {missing}"


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_subcommand_help(subcommand: str):
    result = _run_help([subcommand])
    assert result.returncode == 0, f"{subcommand} --help failed:\n{result.stderr}"
    # Each subcommand's --help must show its own Usage line, not the parent's.
    assert f"{subcommand}" in result.stdout
    assert "Usage:" in result.stdout


def test_run_help_exposes_continue_flag():
    """`mosaic run --continue` is the resume-after-crash entrypoint.

    Regression guard: the flag must stay discoverable in --help output so the
    24h-OOM recovery path remains visible to users (and to CI scripts that
    grep for it).
    """
    result = _run_help(["run"])
    assert result.returncode == 0
    assert "--continue" in result.stdout, (
        f"--continue flag missing from `mosaic run --help`:\n{result.stdout}"
    )


# Imported by the harness, but only needed once a command does real work.
# Loading them at startup made every invocation, including ``--help``, take
# seconds (#58).
_HEAVY_MODULES = ("jax", "pandas", "altair", "scipy", "optax", "tesseract_core")


def test_cli_startup_does_not_import_heavy_modules():
    """Importing the CLI must not import JAX, pandas, altair and friends."""
    code = (
        "import sys, mosaic.benchmarks.cli; "
        f"print(' '.join(m for m in {_HEAVY_MODULES!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"imported at CLI startup: {result.stdout.strip()}"
    )


def test_lazy_exports_still_resolve():
    """The names that are now loaded on first access keep working."""
    import mosaic
    from mosaic.benchmarks import cli, problems
    from mosaic.benchmarks.core import runner
    from mosaic.benchmarks.problems.shared import cost

    assert mosaic.cost is cost
    assert mosaic.PROBLEMS == problems.PROBLEMS == problems.problem_names()
    assert len(problems.PROBLEMS) >= 4
    assert cli.build_all is runner.build_all
    assert cli.run_suite is runner.run_suite
    with pytest.raises(AttributeError):
        _ = mosaic.not_an_attribute
