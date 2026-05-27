# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`mosaic tesseracts` — code-ratio report across tesseract solver modules."""

from __future__ import annotations

import typer

from mosaic.benchmarks.cli import app
from mosaic.benchmarks.cli._helpers import _repo_root
from mosaic.benchmarks.core.console import console


@app.command("tesseracts")
def cmd_tesseracts(
    problem: str = typer.Option(
        None, "--problem", "-p", help="Filter by problem name (substring)."
    ),
    csv: bool = typer.Option(
        False, "--csv", help="Emit CSV to stdout instead of a rich table."
    ),
    vars: bool = typer.Option(
        False,
        "--vars",
        help="Show per-variable gradient implementation pivot table instead.",
    ),
    effort: bool = typer.Option(
        False,
        "--effort",
        help="Show per-solver gradient effort table (lines by implementation type).",
    ),
) -> None:
    """Report solver-specific vs. boilerplate code ratio for every tesseract.

    Classifies each tesseract_api.py by top-level AST node role and counts
    in-repo external source files (.cpp, .cu, .jl).  Detects tier-3 solvers
    whose physics is fetched at image-build time and therefore absent from this
    repo.
    """
    from mosaic.benchmarks.core.code_ratio import (
        collect,
        print_csv,
        print_effort_table,
        print_rich,
        print_variable_table,
    )

    repo = _repo_root()
    tesseracts_root = repo / "mosaic" / "tesseracts"
    mosaic_shared_root = repo / "mosaic" / "mosaic_shared"
    results = collect(
        tesseracts_root,
        problem_filter=problem,
        mosaic_shared_root=mosaic_shared_root if mosaic_shared_root.exists() else None,
    )
    if not results:
        console.print("[red]No tesseracts found.[/red]")
        raise typer.Exit(1)
    if csv:
        print_csv(results)
    elif vars:
        print_variable_table(results)
    elif effort:
        print_effort_table(results)
    else:
        print_rich(results)
