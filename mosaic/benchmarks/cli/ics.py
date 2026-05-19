"""`mosaic ics` — generate initial-condition visualisations."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from mosaic.benchmarks.cli import app
from mosaic.benchmarks.core.console import console, print_rule, print_warn
from mosaic.benchmarks.core.io import RESULTS_DIR_ENV
from mosaic.benchmarks.problems import get_config


@app.command()
def ics(
    problem: str = typer.Option(
        ..., "--problem", "-p", help="Problem config to generate IC plots for"
    ),
    plots_only: bool = typer.Option(
        False,
        "--plots-only",
        help="Regenerate plots without re-running IC generation (uses cached params)",
    ),
    traceback: bool = typer.Option(
        False, "--traceback", "--tb", help="Print full traceback on failure"
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Root directory for benchmark results.",
    ),
):
    """Generate initial-condition visualisations (no solver builds needed)."""
    if output_dir is not None:
        os.environ[RESULTS_DIR_ENV] = str(output_dir.resolve())

    from mosaic.benchmarks.problems.shared.ics import get_experiments, get_plot_fns

    cfg = get_config(problem)
    print_rule("initial conditions")

    if plots_only:
        for name, fn in get_plot_fns(cfg).items():
            try:
                fn(cfg)
                console.print(f"  [green]{name}[/green] ok")
            except Exception as exc:
                if traceback:
                    console.print_exception()
                print_warn(f"{name}: {exc}")
        return

    for name, exp_fn in get_experiments(cfg).items():
        try:
            result = exp_fn(cfg, {})
            console.print(f"  [green]{name}[/green] ok  shape={result['shape']}")
        except Exception as exc:
            if traceback:
                console.print_exception()
            print_warn(f"{name}: {exc}")
