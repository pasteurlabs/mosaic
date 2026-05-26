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
    )


SUBCOMMANDS = [
    "run",
    "ics",
    "build",
    "status",
    "compare",
    "tesseracts",
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
