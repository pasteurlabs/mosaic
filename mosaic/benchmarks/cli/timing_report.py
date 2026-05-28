# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`mosaic timing-report` — aggregate per-cell wall times across results."""

from __future__ import annotations

import json as _json
import os
from pathlib import Path
from typing import Any

import typer

from mosaic.benchmarks.cli import app
from mosaic.benchmarks.core.console import console
from mosaic.benchmarks.core.io import RESULTS_DIR_ENV, results_dir


def _walk_results(root: Path) -> list[dict[str, Any]]:
    """Yield one row per (problem, suite, experiment, solver) wall-time entry.

    Reads every ``result*.json`` under ``root`` and emits flat rows keyed by
    the directory layout ``<problem>/<suite>/<experiment>/``. Both
    ``result.json`` and ``result_debug.json`` (audit/debug runs) are scanned;
    each file is tagged with ``mode = "debug"`` if the experiment directory
    has the ``_debug`` suffix, else ``"prod"``.
    """
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for result_path in sorted(root.rglob("result*.json")):
        rel = result_path.relative_to(root)
        parts = rel.parts
        # Expect <problem>/<suite>/<experiment[_debug]>/result*.json
        if len(parts) < 4 or parts[-1] not in ("result.json", "result_debug.json"):
            continue
        problem, suite, exp_dir = parts[0], parts[1], parts[-2]
        mode = "debug" if exp_dir.endswith("_debug") else "prod"
        experiment = exp_dir[: -len("_debug")] if mode == "debug" else exp_dir
        try:
            data = _json.loads(result_path.read_text())
        except Exception:
            continue
        wall = data.get("wall_time_s") or {}
        if not isinstance(wall, dict):
            continue
        for solver, seconds in wall.items():
            if not isinstance(seconds, (int, float)):
                continue
            rows.append(
                {
                    "problem": problem,
                    "suite": suite,
                    "experiment": experiment,
                    "solver": solver,
                    "mode": mode,
                    "seconds": float(seconds),
                }
            )
    return rows


def _format_secs(seconds: float) -> str:
    """Compact human-readable duration: ``1.2s`` / ``45.0s`` / ``3m12s`` / ``1h05m``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _emit_markdown(
    all_rows: list[dict[str, Any]], visible: list[dict[str, Any]]
) -> str:
    """Render visible rows as markdown; ``all_rows`` drives the summary totals."""
    if not all_rows:
        return "_No `wall_time_s` entries found under results directory._\n"

    total = sum(r["seconds"] for r in all_rows)
    n_cells = len(all_rows)
    n_experiments = len({(r["problem"], r["suite"], r["experiment"]) for r in all_rows})

    lines = [
        "## Timing report",
        "",
        f"- **{n_cells}** (experiment, solver) cells across "
        f"**{n_experiments}** experiments",
        f"- **Total CPU/GPU wall-time** (sum of all cells): {_format_secs(total)}",
        "",
        "| problem | suite | experiment | solver | mode | wall |",
        "|---|---|---|---|---|---:|",
    ]
    for r in visible:
        lines.append(
            f"| {r['problem']} | {r['suite']} | {r['experiment']} | "
            f"{r['solver']} | {r['mode']} | {_format_secs(r['seconds'])} |"
        )
    return "\n".join(lines) + "\n"


def _emit_rich(all_rows: list[dict[str, Any]], visible: list[dict[str, Any]]) -> None:
    """Print visible rows as a Rich table; ``all_rows`` drives the summary line."""
    from rich.table import Table

    if not all_rows:
        console.print("[dim]No 'wall_time_s' entries found under results directory.[/]")
        return

    table = Table(title=f"Timing report — slowest {len(visible)} cells")
    table.add_column("problem")
    table.add_column("suite")
    table.add_column("experiment")
    table.add_column("solver")
    table.add_column("mode")
    table.add_column("wall", justify="right")
    for r in visible:
        table.add_row(
            r["problem"],
            r["suite"],
            r["experiment"],
            r["solver"],
            r["mode"],
            _format_secs(r["seconds"]),
        )
    console.print(table)

    total = sum(r["seconds"] for r in all_rows)
    console.print(
        f"[dim]Total wall-time across {len(all_rows)} cells: {_format_secs(total)}[/]"
    )


@app.command("timing-report")
def timing_report(
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        "-o",
        help="Root directory for benchmark results (must match the directory "
        "used by 'mosaic run').",
    ),
    format: str = typer.Option(
        "rich",
        "--format",
        help="Output format: 'rich' (default terminal table), 'md' "
        "(GitHub-flavored markdown), or 'json' (machine-readable rows).",
    ),
    top: int | None = typer.Option(
        None,
        "--top",
        help="Show only the N slowest cells. Default: show all.",
        min=1,
    ),
) -> None:
    """Aggregate per-cell wall times across an existing results tree.

    Crawls ``<results>/<problem>/<suite>/<experiment>/result*.json`` and
    emits a table sorted by ``wall_time_s`` (slowest first). Useful for
    spotting where CI time goes — pair with ``mosaic run --debug`` to get a
    bare-minimum time-audit run that finishes in minutes instead of hours.

    Example workflow::

        mosaic run --debug                   # short run, ~5 iters everywhere
        mosaic timing-report --top 25        # show 25 slowest cells
        mosaic timing-report --format md > timing-report.md
    """
    if output_dir is not None:
        os.environ[RESULTS_DIR_ENV] = str(output_dir.resolve())

    root = results_dir()
    rows = _walk_results(root)
    rows_sorted = sorted(rows, key=lambda r: r["seconds"], reverse=True)
    visible = rows_sorted[:top] if top else rows_sorted

    if format == "json":
        typer.echo(_json.dumps(visible, indent=2))
        return
    if format == "md":
        typer.echo(_emit_markdown(rows, visible), nl=False)
        return
    _emit_rich(rows, visible)
