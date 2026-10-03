# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
        disable=not console.is_terminal,
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
        disable=not console.is_terminal,
    )


def print_rule(title: str) -> None:
    """Print a horizontal rule with a centred title."""
    console.print(Rule(title))


def print_warn(msg: str) -> None:
    """Print a yellow warning message."""
    console.print(f"[yellow][WARN] {msg}[/yellow]")


def print_skip(msg: str) -> None:
    """Print a dimmed skip message."""
    console.print(f"[dim][SKIP] {msg}[/dim]")


def print_hint(msg: str) -> None:
    """Print an actionable hint. Yellow like a warning, but not itself a failure."""
    console.print(f"[yellow][HINT][/yellow] {msg}")


def github_annotation(level: str, msg: str, *, title: str | None = None) -> None:
    """Emit a GitHub Actions workflow annotation when running under CI.

    ``level`` is ``"warning"`` or ``"error"``. GitHub renders these on the
    Checks tab and inline in the job log, so a per-sweep-point solver failure
    surfaces without anyone downloading result artifacts. Outside CI (no
    ``GITHUB_ACTIONS`` env var) this is a no-op — the plain rich message the
    caller already printed is enough on a developer's terminal.

    The message is flattened to a single line: GitHub's ``::`` command syntax
    is line-oriented and would truncate at the first newline otherwise.
    """
    import os

    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    flat = " ".join(msg.splitlines())
    title_part = f" title={title}" if title else ""
    # Printed raw (not via rich) so GitHub's log parser sees the bare command.
    print(f"::{level}{title_part}::{flat}", flush=True)


def print_saved(path: object) -> None:
    """Print a cyan confirmation that a file was saved at *path*."""
    console.print(f"[cyan]  Saved \u2192 {path}[/cyan]")
