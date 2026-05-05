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


def print_rule(title: str) -> None:
    console.print(Rule(title))


def print_warn(msg: str) -> None:
    console.print(f"[yellow][WARN] {msg}[/yellow]")


def print_skip(msg: str) -> None:
    console.print(f"[dim][SKIP] {msg}[/dim]")


def print_saved(path) -> None:
    console.print(f"[cyan]  Saved \u2192 {path}[/cyan]")
