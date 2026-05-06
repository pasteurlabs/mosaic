"""Rich console singleton and terminal output helpers."""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule

console = Console()


def make_build_progress() -> Progress:
    """Indeterminate spinner for Docker builds."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )


def make_sweep_progress(total: int) -> Progress:
    """Progress bar for parameter sweeps (solver x condition grid).

    Shows a determinate bar with percentage, completed/total count, and elapsed
    time.  Thread-safe — works correctly when multiple solvers run concurrently
    in the GPU-pool parallel path.
    """
    from rich.progress import BarColumn, MofNCompleteColumn

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def print_rule(title: str) -> None:
    console.print(Rule(title))


def print_warn(msg: str) -> None:
    console.print(f"[yellow][WARN] {msg}[/yellow]")


def print_skip(msg: str) -> None:
    console.print(f"[dim][SKIP] {msg}[/dim]")


def print_saved(path) -> None:
    console.print(f"[cyan]  Saved \u2192 {path}[/cyan]")
