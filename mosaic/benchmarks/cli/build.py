# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`mosaic build` — build solver images for one or more problems."""

from __future__ import annotations

import typer

from mosaic.benchmarks import cli as _cli_pkg
from mosaic.benchmarks.cli import app
from mosaic.benchmarks.cli._helpers import _apply_solver_filter
from mosaic.benchmarks.core.console import console, print_rule, print_warn
from mosaic.benchmarks.problems import PROBLEMS, get_config


@app.command()
def build(
    problems: str = typer.Option(
        "all",
        "--problems",
        "-p",
        help="Comma-separated problem(s) to build, or 'all'.",
    ),
    solvers: str | None = typer.Option(
        None,
        "--solvers",
        "-s",
        help="Comma-separated solver names to build (default: all in each problem).",
    ),
    jobs: int = typer.Option(
        2,
        "--jobs",
        "-j",
        help="Max concurrent docker builds. Default 2 (safe during live campaigns). "
        "On a fresh worker machine 4-8 is usually faster.",
        min=1,
    ),
) -> None:
    """Build solver images for one or more problems.

    Useful to pre-warm tesseract images on a fresh machine. Cached images are
    skipped. Equivalent to `tesseract build` on every solver dir, with sensible
    per-problem defaults from `mosaic/benchmarks/problems/*.py`.
    """
    problem_list = (
        list(PROBLEMS)
        if problems == "all"
        else [p.strip() for p in problems.split(",")]
    )
    if not problem_list:
        print_warn("no problems selected — nothing to build")
        raise typer.Exit(code=1)

    failed: list[tuple[str, str]] = []
    for problem in problem_list:
        print_rule(f"build · {problem}")
        try:
            cfg = get_config(problem)
            cfg = _apply_solver_filter(cfg, solvers)
            # Look ``build_all`` up via the cli package namespace so tests can
            # monkeypatch ``mosaic.benchmarks.cli.build_all`` and have it take
            # effect here (see test_build_unknown_problem_records_failure).
            _cli_pkg.build_all(cfg, max_workers=jobs)
        except Exception as exc:
            print_warn(f"{problem}: {exc}")
            failed.append((problem, str(exc)))

    if failed:
        console.print(f"\n[red]{len(failed)} problem(s) failed to build:[/red]")
        for p, err in failed:
            console.print(f"  [red]{p}[/red] — {err}")
        raise typer.Exit(code=1)
