"""Private helpers for the ``status`` and ``compare`` commands.

Split out of :mod:`mosaic.benchmarks.cli.status` to keep that module under the
~400-line size budget. All functions here are private (``_``-prefixed) and
have no external callers.
"""

from __future__ import annotations

import typer

from mosaic.benchmarks.core.console import console, print_rule, print_warn
from mosaic.benchmarks.problems import get_config


def _status_emit_snapshot(
    problem_list: list[str],
    suite_list: list[str],
    output_format: str,
    diff_against: str | None,
) -> None:
    """Handle --format md/json: build snapshot(s) and emit to stdout.

    For ``json`` writes a dict snapshot; for ``md`` renders a markdown report,
    optionally prepended with a diff against a saved snapshot file.
    """
    import json as _json

    from mosaic.benchmarks.core.status import (
        collect_status,
        diff_snapshots,
        render_diff_markdown,
        render_markdown,
        snapshot_to_dict,
    )

    statuses = []
    for problem in problem_list:
        try:
            cfg = get_config(problem)
        except Exception as exc:
            print_warn(f"{problem}: {exc}")
            continue
        statuses.append(collect_status(cfg, suites=suite_list))
    if output_format == "json":
        typer.echo(_json.dumps(snapshot_to_dict(statuses), indent=2))
        return
    # output_format == "md"
    out_parts: list[str] = []
    if diff_against:
        try:
            with open(diff_against) as f:
                old_snapshot = _json.load(f)
        except Exception as exc:
            print_warn(f"could not read --diff-against file: {exc}")
            old_snapshot = None
        if old_snapshot is not None:
            new_snapshot = snapshot_to_dict(statuses)
            out_parts.append(
                render_diff_markdown(diff_snapshots(old_snapshot, new_snapshot))
            )
    out_parts.append(render_markdown(statuses))
    typer.echo("\n".join(out_parts))


def _status_render_cell(cell) -> str:
    """Render a single status cell as a coloured rich-markup label."""
    from mosaic.benchmarks.core.status import (
        ANOMALY,
        EXCL_PERMANENT,
        EXCLUDED,
        FAILED,
        NOT_RUN,
        OK,
        cell_color,
    )

    if cell is None:
        return "?"
    if cell.status == OK:
        label = "ok"
    elif cell.status == ANOMALY:
        label = "anom"
    elif cell.status == FAILED:
        label = "fail"
    elif cell.status == NOT_RUN:
        label = "—"
    elif cell.status == EXCLUDED:
        label = "perm" if cell.category in EXCL_PERMANENT else "excl"
    else:
        return "?"
    if getattr(cell, "stale", False) and cell.status != EXCLUDED:
        label = f"{label}*"
    return f"[{cell_color(cell)}]{label}[/]"


def _status_tally_problem(st) -> tuple[dict, list]:
    """Count cells per category for one problem's status snapshot.

    Returns ``(counts, all_cells)`` where ``counts`` has keys
    ``n_ok``, ``n_anom``, ``n_fail``, ``n_missing``, ``n_excl_work``,
    ``n_excl_perm``, ``n_stale``, ``n_stale_ok``. ``ok`` is *fresh*-ok only
    — stale-ok cells contribute to ``stale_ok`` (in score denominator but not
    numerator). ``all_cells`` is the flat list used for ``compute_score``.
    """
    from mosaic.benchmarks.core.status import (
        ANOMALY,
        EXCL_PERMANENT,
        EXCLUDED,
        FAILED,
        NOT_RUN,
        OK,
    )

    n_ok = n_anom = n_fail = n_missing = n_excl_work = n_excl_perm = 0
    n_stale = n_stale_ok = 0
    all_cells = []
    for row in st.rows:
        for solver in st.solvers:
            cell = row.cells.get(solver)
            if not cell:
                continue
            all_cells.append(cell)
            is_stale = getattr(cell, "stale", False)
            if is_stale:
                n_stale += 1
            if cell.status == OK:
                if is_stale:
                    n_stale_ok += 1
                else:
                    n_ok += 1
            elif cell.status == ANOMALY:
                n_anom += 1
            elif cell.status == FAILED:
                n_fail += 1
            elif cell.status == NOT_RUN:
                n_missing += 1
            elif cell.status == EXCLUDED:
                if cell.category in EXCL_PERMANENT:
                    n_excl_perm += 1
                else:
                    n_excl_work += 1
    counts = {
        "n_ok": n_ok,
        "n_anom": n_anom,
        "n_fail": n_fail,
        "n_missing": n_missing,
        "n_excl_work": n_excl_work,
        "n_excl_perm": n_excl_perm,
        "n_stale": n_stale,
        "n_stale_ok": n_stale_ok,
    }
    return counts, all_cells


def _status_print_problem_table(problem: str, st, counts: dict, score, score_n) -> None:
    """Render the per-problem rule + experiment x solver table."""
    from rich.table import Table

    from mosaic.benchmarks.core.status import format_score, weight_color

    n_ok = counts["n_ok"]
    n_total = (
        counts["n_ok"]
        + counts["n_anom"]
        + counts["n_fail"]
        + counts["n_missing"]
        + counts["n_excl_work"]
        + counts["n_stale_ok"]
    )
    _hdr_colour = weight_color(score)
    print_rule(
        f"[bold {_hdr_colour}]{problem}[/]  —  {len(st.rows)} experiment(s), "
        f"{n_ok}/{n_total} fresh-ok · score "
        f"[bold {_hdr_colour}]{format_score(score)}[/] "
        f"(n={score_n})"
    )
    if not st.rows:
        console.print("  [dim]no result directories found[/]")
        return
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("experiment", style="bold", no_wrap=True)
    for solver in st.solvers:
        table.add_column(solver, justify="center")
    for row in st.rows:
        cells = [row.label]
        for solver in st.solvers:
            cell = row.cells.get(solver)
            cells.append(_status_render_cell(cell))
        table.add_row(*cells)
    console.print(table)


def _status_collect_failures(problem: str, st) -> list[tuple]:
    """Build the (problem, row, solver, status, reason, stale) records for failures/anomalies."""
    from mosaic.benchmarks.core.status import ANOMALY, FAILED

    records: list[tuple] = []
    for row in st.rows:
        for solver in st.solvers:
            cell = row.cells.get(solver)
            if cell and cell.status in (FAILED, ANOMALY):
                records.append(
                    (
                        problem,
                        row.label,
                        solver,
                        cell.status,
                        cell.reason,
                        getattr(cell, "stale", False),
                    )
                )
    return records


def _status_print_failures(failure_records: list[tuple]) -> None:
    """Render the 'failures & anomalies' list."""
    from mosaic.benchmarks.core.status import FAILED

    print_rule("failures & anomalies")
    if not failure_records:
        console.print("  [green]none recorded[/]")
        return
    for problem, label, solver, status, reason, stale in failure_records:
        reason_str = reason or "[dim]no reason recorded[/]"
        tag = "[red]fail[/]" if status == FAILED else "[dark_orange]anom[/]"
        stale_mark = "  [dim](stale)[/]" if stale else ""
        console.print(
            f"  {tag}  [dim]{problem}[/]  {label}  [bold]{solver}[/]  {reason_str}{stale_mark}"
        )


def _status_progress_bar(
    score: float | None,
    n_ok: int,
    n_anom: int,
    n_fail: int,
    n_missing: int,
    n_excl_work: int,
    n_stale_ok: int = 0,
    width: int = 18,
) -> str:
    """Render a fixed-width coloured progress bar weighted by status category."""
    from mosaic.benchmarks.core.status import weight_color

    segs = [
        (n_ok, 1.00),
        (n_stale_ok, 0.67),
        (n_anom, 0.53),
        (n_missing + n_excl_work, 0.33),
        (n_fail, 0.00),
    ]
    total = sum(c for c, _ in segs)
    if total <= 0:
        return "[dim]" + "░" * width + "[/]"
    raw = [c / total * width for c, _ in segs]
    chars = [int(r) for r in raw]
    remainder = width - sum(chars)
    order = sorted(range(len(segs)), key=lambda i: raw[i] - chars[i], reverse=True)
    for i in order[:remainder]:
        chars[i] += 1
    bar = ""
    for (_, w), n_chars in zip(segs, chars, strict=False):
        if n_chars > 0:
            bar += f"[{weight_color(w)}]{'█' * n_chars}[/]"
    return bar


def _status_print_summary(per_problem_tally: list[tuple]) -> None:
    """Render the bottom summary table with per-problem + overall rows."""
    from rich.table import Table

    from mosaic.benchmarks.core.status import format_score, weight_color

    print_rule("summary")
    console.print(
        "[dim]legend:[/] "
        "[green]ok[/] · "
        "[dark_orange]anom[/] outlier · "
        "[red]fail[/] · "
        "[dim]—[/] missing · "
        "[dim yellow]perm[/] excluded (permanent, out of %) · "
        "[yellow]excl[/] excluded (work-to-do) · "
        "[bold]*[/] stale (predates current tesseract/harness source)"
    )

    def _score_colored(score: float | None) -> str:
        colour = weight_color(score)
        return f"[bold {colour}]{format_score(score)}[/]"

    summary = Table(show_header=True, header_style="bold", show_lines=False)
    summary.add_column("problem", style="bold", no_wrap=True)
    summary.add_column("ok", justify="right")
    summary.add_column("anom", justify="right")
    summary.add_column("fail", justify="right")
    summary.add_column("missing", justify="right")
    summary.add_column("excl·work", justify="right")
    summary.add_column("excl·perm", justify="right")
    summary.add_column("stale", justify="right")
    summary.add_column("progress", no_wrap=True)
    summary.add_column("score", justify="right")
    total_ok = total_anom = total_fail = total_missing = 0
    total_excl_work = total_excl_perm = total_stale = total_stale_ok = 0
    score_num = 0.0
    score_den = 0
    for (
        problem,
        n_ok,
        n_anom,
        n_fail,
        n_missing,
        n_excl_work,
        n_excl_perm,
        n_stale,
        n_stale_ok,
        score,
        score_n,
    ) in per_problem_tally:
        summary.add_row(
            problem,
            f"[green]{n_ok}[/]",
            f"[dark_orange]{n_anom}[/]" if n_anom else "0",
            f"[red]{n_fail}[/]" if n_fail else "0",
            f"[dim]{n_missing}[/]" if n_missing else "0",
            f"[yellow]{n_excl_work}[/]" if n_excl_work else "0",
            f"[dim yellow]{n_excl_perm}[/]" if n_excl_perm else "0",
            f"[dim]{n_stale}[/]" if n_stale else "[dim]0[/]",
            _status_progress_bar(
                score, n_ok, n_anom, n_fail, n_missing, n_excl_work, n_stale_ok
            ),
            _score_colored(score),
        )
        total_ok += n_ok
        total_anom += n_anom
        total_fail += n_fail
        total_missing += n_missing
        total_excl_work += n_excl_work
        total_excl_perm += n_excl_perm
        total_stale += n_stale
        total_stale_ok += n_stale_ok
        if score is not None:
            score_num += score * score_n
            score_den += score_n
    overall_score = (score_num / score_den) if score_den else None
    summary.add_row(
        "[bold]overall[/]",
        f"[green]{total_ok}[/]",
        f"[dark_orange]{total_anom}[/]" if total_anom else "0",
        f"[red]{total_fail}[/]" if total_fail else "0",
        f"[dim]{total_missing}[/]" if total_missing else "0",
        f"[yellow]{total_excl_work}[/]" if total_excl_work else "0",
        f"[dim yellow]{total_excl_perm}[/]" if total_excl_perm else "0",
        f"[dim]{total_stale}[/]" if total_stale else "[dim]0[/]",
        _status_progress_bar(
            overall_score,
            total_ok,
            total_anom,
            total_fail,
            total_missing,
            total_excl_work,
            total_stale_ok,
        ),
        _score_colored(overall_score),
    )
    console.print(summary)
