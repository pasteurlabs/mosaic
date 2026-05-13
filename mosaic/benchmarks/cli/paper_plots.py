"""`mosaic paper-plots` — generate paper figures from on-disk results."""

from __future__ import annotations

from pathlib import Path

import typer

from mosaic.benchmarks.cli import app
from mosaic.benchmarks.core.console import console, print_rule
from mosaic.benchmarks.core.io import results_dir


@app.command("paper-plots")
def paper_plots(
    out_dir: Path = typer.Option(
        None,
        "--out-dir",
        "-o",
        help="Output directory for generated figures. Defaults to <results-dir>/figures/.",
    ),
    only: str = typer.Option(
        "all",
        "--only",
        help="Comma-separated plot names to generate, or 'all'. Run "
        "`mosaic paper-plots --only LIST` is not a thing — use Python's "
        "``mosaic.benchmarks.plots.paper.all_names()`` to list, or "
        "rely on the unknown-name error which prints the registry.",
    ),
    list_names: bool = typer.Option(
        False,
        "--list",
        help="Print the available figure names and exit (alternative to --only).",
    ),
) -> None:
    """Generate paper figures from on-disk result.json corpora.

    Reads result JSON files from the benchmark results directory and writes
    PDFs (and PNGs for the coverage heatmap) to the output directory.

    Cross-domain aggregators (cost_overview, scaling, ucurves,
    recovery_overview, …) live exclusively here. Single-experiment
    paper figures (fd_check, jacobian_svd, horizon_sweep, agreement,
    physical_accuracy, …) are also produced as the per-experiment plot
    when running ``mosaic run --plots-only``.
    """
    from mosaic.benchmarks.plots.paper import all_names, get_generate_fn

    if list_names:
        for n in all_names():
            console.print(f"  {n}")
        return

    target_dir: Path = out_dir if out_dir is not None else results_dir() / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)

    names = all_names() if only == "all" else [n.strip() for n in only.split(",")]

    unknown = [n for n in names if n not in all_names()]
    if unknown:
        console.print(f"[red]Unknown plot name(s): {', '.join(unknown)}[/red]")
        console.print(f"Available: {', '.join(all_names())}")
        raise typer.Exit(1)

    print_rule("paper plots")
    failed: list[tuple[str, str]] = []
    for name in names:
        try:
            fn = get_generate_fn(name)
            fn(target_dir)
            console.print(f"  [green]{name}[/green] ok")
        except Exception as exc:
            console.print_exception()
            failed.append((name, str(exc)))

    if failed:
        console.print(f"\n[red]{len(failed)} figure(s) failed:[/red]")
        for n, err in failed:
            console.print(f"  [red]{n}[/red] — {err}")
        raise typer.Exit(1)
