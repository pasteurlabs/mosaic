# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified CLI entrypoint for Mosaic benchmarks.

This package replaces the former single-file ``cli.py``. The Typer ``app``
object is constructed here; subcommand modules are imported below to trigger
their ``@app.command(...)`` registrations.

The entry point ``mosaic = "mosaic.benchmarks.cli:app"`` in ``pyproject.toml``
keeps working because ``app`` is re-exported at package level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from mosaic.benchmarks.core.runner import build_all, image_tags_no_build, run_suite

app = typer.Typer(name="mosaic", rich_markup_mode="rich")

# Import subcommand modules AFTER ``app`` is created so their
# ``@app.command(...)`` decorators can register against it. The imports are
# side-effectful by design.
from mosaic.benchmarks.cli import build as _build  # noqa: E402, F401
from mosaic.benchmarks.cli import ics as _ics  # noqa: E402, F401
from mosaic.benchmarks.cli import reference as _reference  # noqa: E402, F401
from mosaic.benchmarks.cli import run as _run  # noqa: E402, F401
from mosaic.benchmarks.cli import status as _status  # noqa: E402, F401
from mosaic.benchmarks.cli import templates as _templates  # noqa: E402, F401

__all__ = ["app", "build_all", "image_tags_no_build", "run_suite"]

_RUNNER_EXPORTS = ("build_all", "image_tags_no_build", "run_suite")


def __getattr__(name: str) -> object:
    # ``build_all`` and friends are re-exported so that tests can
    # ``monkeypatch.setattr(cli_mod, "build_all", ...)``: the ``build``
    # subcommand looks ``build_all`` up via this module's namespace at call
    # time. They resolve on first access because ``core.runner`` imports JAX
    # and tesseract-core, which would otherwise slow down every invocation,
    # including ``mosaic --help`` (#58).
    if name in _RUNNER_EXPORTS:
        from mosaic.benchmarks.core import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
