"""`mosaic status` and `mosaic compare` — report and diff benchmark status."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from mosaic.benchmarks.cli import app
from mosaic.benchmarks.cli._status_helpers import (
    _status_collect_failures,
    _status_emit_snapshot,
    _status_print_failures,
    _status_print_problem_table,
    _status_print_summary,
    _status_tally_problem,
)
from mosaic.benchmarks.core.console import print_warn
from mosaic.benchmarks.core.io import RESULTS_DIR_ENV
from mosaic.benchmarks.problems import PROBLEMS, get_config


@app.command()
def status(
    problems: str = typer.Option(
        "all", "--problems", "-p", help="Comma-separated problems or 'all'"
    ),
    suites: str = typer.Option(
        "all", "--suites", help="Comma-separated suites or 'all'"
    ),
    failures: bool = typer.Option(
        False,
        "--failures",
        "-f",
        help="After the table, list every failed (experiment, solver) with its reason.",
    ),
    only_failures: bool = typer.Option(
        False,
        "--only-failures",
        help="Skip the table; print only the failure list.",
    ),
    format: str = typer.Option(
        "rich",
        "--format",
        help="Output format: rich (default terminal), md (GitHub-flavored markdown), json (machine-readable snapshot).",
    ),
    diff_against: str | None = typer.Option(
        None,
        "--diff-against",
        help="Path to a JSON snapshot (produced by --format json). When set with --format md, "
        "prepend a diff section (regressions, improvements, new/removed rows) before the full tables.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Root directory for benchmark results (must match the directory used by 'mosaic run').",
    ),
):
    """Report per-solver completion status of every experiment on disk.

    Walks ``<results>/<problem>/<suite>/<experiment>/`` and, for each
    solver in the problem config, classifies the cell as:

      [green]ok[/]       solver produced valid data
      [dark_orange]anom[/]     valid but an outlier per the problem's status_checks
      [red]fail[/]      solver was attempted but its entry is empty/invalid/NaN
      [dim]—[/]         no result file, or solver absent from the result
      [dim yellow]perm[/]     excluded (permanent — out of score denominator)
      [yellow]excl[/]     excluded (work-to-do — in score denominator)

    Pass [bold]--failures[/] to list every failed or anomalous cell with its
    reason.  Use [bold]--format md[/] or [bold]--format json[/] to emit a
    PR-friendly snapshot; pair [bold]--format md[/] with
    [bold]--diff-against[/] to render a regression/improvement diff.
    """
    if output_dir is not None:
        os.environ[RESULTS_DIR_ENV] = str(output_dir.resolve())

    from mosaic.benchmarks.core.status import (
        SUITES,
        collect_status,
        compute_score,
    )

    problem_list = (
        PROBLEMS if problems == "all" else [p.strip() for p in problems.split(",")]
    )
    suite_list = (
        list(SUITES) if suites == "all" else [s.strip() for s in suites.split(",")]
    )

    # ── non-rich formats: skip terminal rendering and emit a snapshot ─────
    if format in ("md", "json"):
        _status_emit_snapshot(problem_list, suite_list, format, diff_against)
        return

    failure_records: list[
        tuple[str, str, str, str, str, bool]
    ] = []  # (problem, row, solver, status, reason, stale)
    # tuple layout: (problem, ok, anom, fail, missing, excl_work, excl_perm, stale, stale_ok, score, score_n)
    per_problem_tally: list[tuple] = []

    for problem in problem_list:
        try:
            cfg = get_config(problem)
        except Exception as exc:
            print_warn(f"{problem}: {exc}")
            continue
        st = collect_status(cfg, suites=suite_list)

        counts, all_cells = _status_tally_problem(st)
        score, score_n = compute_score(all_cells)
        per_problem_tally.append(
            (
                problem,
                counts["n_ok"],
                counts["n_anom"],
                counts["n_fail"],
                counts["n_missing"],
                counts["n_excl_work"],
                counts["n_excl_perm"],
                counts["n_stale"],
                counts["n_stale_ok"],
                score,
                score_n,
            )
        )

        if not only_failures:
            _status_print_problem_table(problem, st, counts, score, score_n)

        failure_records.extend(_status_collect_failures(problem, st))

    if failures or only_failures:
        _status_print_failures(failure_records)

    # ── summary table: per-problem + overall ok-rate ─────────────────────────
    if per_problem_tally:
        _status_print_summary(per_problem_tally)


# ── `mosaic compare` ───────────────────────────────────────────────────────


@app.command("compare")
def compare(
    before: Path = typer.Argument(
        ...,
        help="Path to the 'before' JSON snapshot (from `mosaic status --format json`).",
    ),
    after: Path = typer.Argument(
        ...,
        help="Path to the 'after' JSON snapshot (from `mosaic status --format json`).",
    ),
):
    """Compare two status snapshots and print a diff.

    Typical workflow:

        mosaic status --format json > before.json\n
        # … make changes, re-run benchmarks …\n
        mosaic status --format json > after.json\n
        mosaic compare before.json after.json

    The output is a Markdown-formatted summary of regressions,
    improvements, and new/removed rows between the two snapshots.
    """
    import json as _json

    for label, path in [("before", before), ("after", after)]:
        if not path.exists():
            typer.echo(f"Error: {label} file not found: {path}", err=True)
            raise typer.Exit(code=1)

    from mosaic.benchmarks.core.status import diff_snapshots, render_diff_markdown

    try:
        old_snapshot = _json.loads(before.read_text())
        new_snapshot = _json.loads(after.read_text())
    except Exception as exc:
        typer.echo(f"Error reading JSON snapshots: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    diff = diff_snapshots(old_snapshot, new_snapshot)
    typer.echo(render_diff_markdown(diff))
