"""Unified CLI entrypoint for Mosaic benchmarks."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import typer

from mosaic.benchmarks.core.console import console, print_rule, print_skip, print_warn
from mosaic.benchmarks.core.runner import build_all, image_tags_no_build, run_suite
from mosaic.benchmarks.core.utils import _RESULTS_DIR_ENV
from mosaic.benchmarks.problems import PROBLEMS, get_config

app = typer.Typer(name="mosaic", rich_markup_mode="rich", add_completion=False)

_ALL_SUITES = ["ics", "forward", "cost", "gradient", "optimization"]


def _repo_root() -> Path:
    # cli.py is at <repo>/mosaic/benchmarks/cli.py
    return Path(__file__).resolve().parents[2]


def _suite_components(suite: str, cfg=None) -> tuple[dict, callable]:
    """Return (experiments_dict, plot_fns_factory) for the named suite.

    For the 'ics' suite *cfg* is required because experiments are built
    dynamically from the problem config (no fixed module-level dict).
    """
    if suite == "ics":
        from mosaic.benchmarks.suites.ics import get_experiments, get_plot_fns

        if cfg is None:
            raise ValueError("cfg is required for the 'ics' suite")
        exps = get_experiments(cfg)
        return exps, lambda: get_plot_fns(cfg)
    elif suite == "forward":
        from mosaic.benchmarks.suites.forward import _EXPERIMENTS, _plot_fns
    elif suite == "cost":
        from mosaic.benchmarks.suites.cost import _EXPERIMENTS, _plot_fns
    elif suite == "gradient":
        from mosaic.benchmarks.suites.gradient import _EXPERIMENTS, _plot_fns
    elif suite == "optimization":
        from mosaic.benchmarks.suites.optimization import _EXPERIMENTS, _plot_fns
    else:
        raise ValueError(f"Unknown suite {suite!r}. Choose from: {_ALL_SUITES}")
    return _EXPERIMENTS, _plot_fns


def _apply_solver_filter(cfg, solvers_csv: str | None):
    """Return cfg restricted to the requested comma-separated solver names.

    Unknown names are warned about and skipped. If solvers_csv is None/empty
    the original cfg is returned unchanged. Applied BEFORE build_all so that
    excluded/broken solvers (e.g. openfoam, pict) are not built when the user
    passes -s to restrict the run.
    """
    if not solvers_csv:
        return cfg
    requested = {s.strip() for s in solvers_csv.split(",") if s.strip()}
    unknown = requested - cfg.solvers.keys()
    if unknown:
        print_warn(f"unknown solver(s): {', '.join(sorted(unknown))} — skipping")
    keep = {k: v for k, v in cfg.solvers.items() if k in requested}
    if not keep:
        print_warn("no matching solvers after filtering — running all")
        return cfg
    return dataclasses.replace(cfg, solvers=keep)


def _resolve_gpu_pool(cfg, gpus: str | None):
    """Translate a `--gpus` string into a (cfg, gpus_csv) pair.

    Special value ``"none"`` (or ``"cpu"``): filter cfg to solvers with
    ``uses_gpu=False`` and pass ``gpus=None`` so the runner doesn't try to
    pin containers to a GPU. Used on CPU-only hosts that can't expose any
    GPU IDs at all.

    Anything else (a real comma-separated list, or None) passes through.
    """
    if isinstance(gpus, str) and gpus.lower() in ("none", "cpu", "cpu-only"):
        cpu_only = {
            name: spec
            for name, spec in cfg.solvers.items()
            if not getattr(spec, "uses_gpu", True)
        }
        if not cpu_only:
            print_warn(
                "no CPU-only solvers in this problem — --gpus none would run nothing"
            )
            return cfg, None
        # Return sentinel "cpu-only" string so CLI can pass gpu_ids=[] to the runner.
        # gpu_ids=[] in run_with_gpu_pool means: no GPU flags (CPU-only host).
        return dataclasses.replace(cfg, solvers=cpu_only), "cpu-only"
    return cfg, gpus


def _filter_hardware(cfg, hardware: str | None):
    """Filter solvers by hardware target.

    Returns a (possibly filtered) cfg.  ``hardware`` is one of:
      "cpu"  — keep only ``uses_gpu=False`` solvers
      "gpu"  — keep only ``uses_gpu=True``  solvers
      "all"  — keep everything
      None   — auto-detect: if no GPU is available, drop GPU solvers
    """
    if hardware and hardware.lower() == "all":
        return cfg
    if hardware is None:
        from mosaic.benchmarks.core.hardware import has_gpu

        if not has_gpu():
            gpu_solvers = [
                n for n, s in cfg.solvers.items() if getattr(s, "uses_gpu", True)
            ]
            if gpu_solvers:
                print_warn(
                    f"no GPU detected — skipping GPU solvers: {', '.join(gpu_solvers)}. "
                    "Pass --hardware all to override."
                )
                cpu_only = {
                    n: s
                    for n, s in cfg.solvers.items()
                    if not getattr(s, "uses_gpu", True)
                }
                if cpu_only:
                    return dataclasses.replace(cfg, solvers=cpu_only)
                print_warn("no CPU-only solvers either — keeping all solvers")
        return cfg
    if hardware.lower() == "cpu":
        filtered = {
            name: spec
            for name, spec in cfg.solvers.items()
            if not getattr(spec, "uses_gpu", True)
        }
        if not filtered:
            print_warn(
                "no CPU-only solvers in this problem — --hardware cpu would run nothing"
            )
            return cfg
        return dataclasses.replace(cfg, solvers=filtered)
    if hardware.lower() == "gpu":
        filtered = {
            name: spec
            for name, spec in cfg.solvers.items()
            if getattr(spec, "uses_gpu", True)
        }
        if not filtered:
            print_warn(
                "no GPU solvers in this problem — --hardware gpu would run nothing"
            )
            return cfg
        return dataclasses.replace(cfg, solvers=filtered)
    print_warn(f"unknown --hardware value {hardware!r}, ignoring")
    return cfg


def _resolve_cfg_and_tags(
    problem: str,
    no_build: bool,
    plots_only: bool = False,
    solvers_csv: str | None = None,
    gpus: str | None = None,
    hardware: str | None = None,
):
    cfg = get_config(problem)
    # --hardware cpu/gpu filters solvers by target hardware BEFORE build.
    had_gpu_solvers = any(getattr(s, "uses_gpu", True) for s in cfg.solvers.values())
    cfg = _filter_hardware(cfg, hardware)
    no_gpu_solvers_left = not any(
        getattr(s, "uses_gpu", True) for s in cfg.solvers.values()
    )
    # If all GPU solvers were removed (explicit --hardware cpu or auto-detect),
    # force gpus to "none" so the runner never passes --gpus to Docker.
    if had_gpu_solvers and no_gpu_solvers_left and not gpus:
        gpus = "none"
    # `--gpus none` filters cfg to CPU-only solvers BEFORE build so a
    # CPU-only host never tries to build GPU-tagged tesseracts. Returns
    # the (possibly reset) gpus value via the tuple's third element.
    cfg, gpus = _resolve_gpu_pool(cfg, gpus)
    # Restrict cfg to requested solvers BEFORE build so that excluded/broken
    # solvers are never built (prevents e.g. openfoam/pict buildx storms when
    # running `-s ins_jl,lettuce` on ns-grid).
    cfg = _apply_solver_filter(cfg, solvers_csv)
    # --plots-only must never trigger an image build: the operation is purely
    # filesystem-level (load result.json, re-render PNG/PDF), so a rebuild is
    # pure overhead and can wedge the process behind a multi-minute docker
    # buildx stage.  Treat plots_only as implying no_build.
    if no_build or plots_only:
        return cfg, image_tags_no_build(cfg), gpus
    print_rule("build")
    tags = build_all(cfg)
    return cfg, tags, gpus


def _filter_solvers(cfg, tags: dict, solvers_csv: str | None):
    """Backward-compat wrapper: cfg is already filtered by _resolve_cfg_and_tags,
    but callers may still invoke this to get matching tags. A no-op when cfg
    is already narrowed."""
    if not solvers_csv:
        return cfg, tags
    keep_names = set(cfg.solvers.keys())
    filtered_tags = {k: v for k, v in tags.items() if k in keep_names}
    return cfg, filtered_tags


def _plots_only(cfg, to_run, plot_fns, verbose_errors: bool = False):
    if plot_fns is None:
        console.print("No plot functions registered for this suite.")
        return
    print_rule("plots")
    names = to_run or list(plot_fns)
    for name in names:
        if name not in plot_fns:
            print_skip(f"no plot function for '{name}'")
            continue
        try:
            plot_fns[name](cfg)
            console.print(f"  [green]{name}[/green] ok")
        except FileNotFoundError as exc:
            print_skip(f"{name}: results not found ({exc.filename})")
        except Exception as exc:
            if verbose_errors:
                console.print_exception()
            print_warn(f"{name}: {exc}")


@app.command()
def run(
    problems: str = typer.Option(
        "all", "--problems", "-p", help="Comma-separated problems or 'all'"
    ),
    suites: str = typer.Option(
        "all", "--suites", help="Comma-separated suites or 'all'"
    ),
    experiment: str = typer.Option(
        "all",
        "--experiment",
        "-e",
        help="Experiment to run within each suite (default: all). "
        "Use 'drag_opt/re100' for sub-run filtering.",
    ),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip plot generation"),
    plots_only: bool = typer.Option(
        False,
        "--plots-only",
        help="Skip all solver runs; regenerate plots for every (problem, suite, experiment) "
        "from existing result.json files. Implies --no-build. Fast, no Docker, no GPU.",
    ),
    no_build: bool = typer.Option(
        False, "--no-build", help="Skip building solver images"
    ),
    traceback: bool = typer.Option(
        False, "--traceback", "--tb", help="Print full traceback on failure"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Reduce N, steps, sweep counts for fast development iteration",
    ),
    solvers: str | None = typer.Option(
        None,
        "--solvers",
        "-s",
        help="Comma-separated solver names to run (default: all)",
    ),
    gpus: str | None = typer.Option(
        None,
        "--gpus",
        help="Comma-separated GPU IDs for parallel dispatch (e.g. 0,1,2). "
        "Each solver container is pinned to one GPU and run in parallel. "
        "Pass 'none' (or 'cpu') to filter the run to CPU-only solvers and "
        "skip all GPU usage — useful on CPU-only hosts.",
    ),
    hardware: str | None = typer.Option(
        None,
        "--hardware",
        help="Filter solvers by hardware target: 'cpu', 'gpu', or 'all' (default).",
    ),
    ics: str | None = typer.Option(
        None,
        "--ics",
        help="Comma-separated IC names to run (default: use config default). "
        "Results land in {experiment}/{ic_name}/ when multiple ICs are given.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Root directory for benchmark results.  Defaults to ./mosaic-results "
        "in the current working directory.  Can also be set via the "
        f"{_RESULTS_DIR_ENV} environment variable.",
    ),
):
    """Run benchmark suites across problems.

    Build each problem's solver images once, then run all requested suites in
    sequence.  Individual experiment failures are caught and logged; execution
    always continues with the next (problem, suite) pair.

    Examples::

        mosaic run -p ns-grid --suites forward
        mosaic run -p ns-grid --suites gradient -e fd_check
        mosaic run --plots-only
        mosaic run -p thermal-mesh --suites optimization -e drag_opt/re100
        mosaic run -o /tmp/bench-results -p ns-grid --suites forward

    Summary table legend:
      ok        all experiments completed
      N/M       partial — N of M experiments completed
      skip      suite not configured for this problem
      error     suite could not start (e.g. build failed, import error)
    """
    if output_dir is not None:
        os.environ[_RESULTS_DIR_ENV] = str(output_dir.resolve())
    from rich.table import Table

    problem_list = (
        PROBLEMS if problems == "all" else [p.strip() for p in problems.split(",")]
    )
    suite_list = (
        _ALL_SUITES if suites == "all" else [s.strip() for s in suites.split(",")]
    )
    to_run = None if experiment == "all" else [experiment]

    # status[(problem, suite)] = ("ok" | "partial" | "skip" | "error", detail)
    run_status: dict[tuple[str, str], tuple[str, str]] = {}

    for problem in problem_list:
        print_rule(f"problem: {problem}")
        _needs_build = (not plots_only) and any(s != "ics" for s in suite_list)
        try:
            if _needs_build:
                cfg, tags, gpus = _resolve_cfg_and_tags(
                    problem,
                    no_build,
                    solvers_csv=solvers,
                    gpus=gpus,
                    hardware=hardware,
                )
                cfg, tags = _filter_solvers(cfg, tags, solvers)
            else:
                cfg = get_config(problem)
                had_gpu = any(
                    getattr(s, "uses_gpu", True) for s in cfg.solvers.values()
                )
                cfg = _filter_hardware(cfg, hardware)
                no_gpu_left = not any(
                    getattr(s, "uses_gpu", True) for s in cfg.solvers.values()
                )
                if had_gpu and no_gpu_left and not gpus:
                    gpus = "none"
                cfg, gpus = _resolve_gpu_pool(cfg, gpus)
                if solvers:
                    cfg = _apply_solver_filter(cfg, solvers)
                tags = {}
        except Exception as exc:
            msg = f"build failed: {exc}"
            console.print(f"  [red]{msg}[/]")
            for suite in suite_list:
                run_status[(problem, suite)] = ("error", msg)
            continue

        for suite in suite_list:
            print_rule(f"  suite: {suite}")
            try:
                exps, plot_fns_fn = _suite_components(suite, cfg=cfg)
                if plots_only:
                    _plots_only(cfg, to_run, plot_fns_fn(), verbose_errors=traceback)
                    run_status[(problem, suite)] = ("ok", "plots-only")
                    continue
                _overrides: dict = {}
                if debug:
                    _overrides["debug"] = True
                if gpus == "cpu-only":
                    _overrides["gpu_ids"] = []
                elif gpus:
                    _overrides["gpu_ids"] = [g.strip() for g in gpus.split(",")]
                if ics:
                    requested = [s.strip() for s in ics.split(",") if s.strip()]
                    unknown = [ic for ic in requested if ic not in cfg.make_ic]
                    if unknown:
                        print_warn(
                            f"unknown IC(s): {', '.join(sorted(unknown))} — skipping"
                        )
                    valid = [ic for ic in requested if ic in cfg.make_ic]
                    if valid:
                        _overrides["ic_names"] = valid
                # Sub-run path filtering (e.g. "drag_opt/re100").
                if experiment != "all" and "/" in experiment:
                    _parts = experiment.split("/", 1)
                    _top, _sub = _parts[0], _parts[1]
                    if _top in exps and experiment in exps:
                        _overrides["run_names"] = [_sub]
                results = run_suite(
                    cfg,
                    tags,
                    exps,
                    to_run=to_run,
                    plots=not no_plots,
                    plot_fns=plot_fns_fn() if not no_plots else None,
                    suite_name=suite,
                    verbose_errors=traceback,
                    overrides=_overrides or None,
                )
                n_total = len(exps)
                n_ok = len(results)
                if n_ok == n_total:
                    run_status[(problem, suite)] = ("ok", "")
                elif n_ok > 0:
                    run_status[(problem, suite)] = ("partial", f"{n_ok}/{n_total}")
                else:
                    run_status[(problem, suite)] = ("skip", "no experiments ran")
            except Exception as exc:
                if traceback:
                    console.print_exception()
                run_status[(problem, suite)] = ("error", str(exc))
                print_warn(f"{suite} failed: {exc}")

    # ── summary table ─────────────────────────────────────────────────────────
    print_rule("summary")
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("problem", style="bold", no_wrap=True)
    for suite in suite_list:
        table.add_column(suite, justify="center")

    for problem in problem_list:
        cells = [problem]
        for suite in suite_list:
            state, detail = run_status.get((problem, suite), ("skip", "not run"))
            if state == "ok":
                cells.append("[green]ok[/green]")
            elif state == "partial":
                cells.append(f"[yellow]{detail}[/yellow]")
            elif state == "skip":
                cells.append("[dim]skip[/dim]")
            else:
                cells.append("[red]error[/red]")
        table.add_row(*cells)

    console.print(table)


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
        os.environ[_RESULTS_DIR_ENV] = str(output_dir.resolve())

    from mosaic.benchmarks.suites.ics import get_experiments, get_plot_fns

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
):
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
            build_all(cfg, max_workers=jobs)
        except Exception as exc:
            print_warn(f"{problem}: {exc}")
            failed.append((problem, str(exc)))

    if failed:
        console.print(f"\n[red]{len(failed)} problem(s) failed to build:[/red]")
        for p, err in failed:
            console.print(f"  [red]{p}[/red] — {err}")
        raise typer.Exit(code=1)


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
        os.environ[_RESULTS_DIR_ENV] = str(output_dir.resolve())

    from rich.table import Table

    from mosaic.benchmarks.core.status import (
        ANOMALY,
        EXCL_PERMANENT,
        EXCLUDED,
        FAILED,
        NOT_RUN,
        OK,
        SUITES,
        cell_color,
        collect_status,
        diff_snapshots,
        render_diff_markdown,
        render_markdown,
        snapshot_to_dict,
    )

    problem_list = (
        PROBLEMS if problems == "all" else [p.strip() for p in problems.split(",")]
    )
    suite_list = (
        list(SUITES) if suites == "all" else [s.strip() for s in suites.split(",")]
    )

    # ── non-rich formats: skip terminal rendering and emit a snapshot ─────
    if format in ("md", "json"):
        import json as _json

        statuses = []
        for problem in problem_list:
            try:
                cfg = get_config(problem)
            except Exception as exc:
                print_warn(f"{problem}: {exc}")
                continue
            statuses.append(collect_status(cfg, suites=suite_list))
        if format == "json":
            typer.echo(_json.dumps(snapshot_to_dict(statuses), indent=2))
            return
        # format == "md"
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
        return

    def _render_cell(cell) -> str:
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

    failure_records: list[
        tuple[str, str, str, str, str, bool]
    ] = []  # (problem, row, solver, status, reason, stale)
    # tuple layout: (problem, ok, anom, fail, missing, excl_work, excl_perm, stale, stale_ok, score, score_n)
    per_problem_tally: list[tuple] = []

    from mosaic.benchmarks.core.status import compute_score, format_score, weight_color

    for problem in problem_list:
        try:
            cfg = get_config(problem)
        except Exception as exc:
            print_warn(f"{problem}: {exc}")
            continue
        st = collect_status(cfg, suites=suite_list)

        # Count per-problem cells. `ok` now means *fresh* ok (not stale) —
        # a stale-ok cell is work-to-do (re-run needed to verify) and only
        # contributes to `stale`. `excl_work` (non-categorical exclusions)
        # counts toward the denominator because they're fixable; `excl_perm`
        # (categorical — method-intrinsic) stays out.
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
                        n_stale_ok += 1  # counts in denominator, not numerator
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
        score, score_n = compute_score(all_cells)
        per_problem_tally.append(
            (
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
            )
        )

        if not only_failures:
            # Per-problem header shows the weighted score — the single
            # canonical campaign-health metric. Raw ok/total is kept for
            # context but the % is gone; the score table row below carries
            # the colour gradient.  Colour the whole rule by the problem's
            # score via the canonical weight → colour ladder, so a quick
            # vertical scan of the output agrees with the summary table
            # and progress bar.
            n_total = n_ok + n_anom + n_fail + n_missing + n_excl_work + n_stale_ok
            _hdr_colour = weight_color(score)
            print_rule(
                f"[bold {_hdr_colour}]{problem}[/]  —  {len(st.rows)} experiment(s), "
                f"{n_ok}/{n_total} fresh-ok · score "
                f"[bold {_hdr_colour}]{format_score(score)}[/] "
                f"(n={score_n})"
            )
            if not st.rows:
                console.print("  [dim]no result directories found[/]")
            else:
                table = Table(show_header=True, header_style="bold", show_lines=False)
                table.add_column("experiment", style="bold", no_wrap=True)
                for solver in st.solvers:
                    table.add_column(solver, justify="center")
                for row in st.rows:
                    cells = [row.label]
                    for solver in st.solvers:
                        cell = row.cells.get(solver)
                        cells.append(_render_cell(cell))
                    table.add_row(*cells)
                console.print(table)

        for row in st.rows:
            for solver in st.solvers:
                cell = row.cells.get(solver)
                if cell and cell.status in (FAILED, ANOMALY):
                    failure_records.append(
                        (
                            problem,
                            row.label,
                            solver,
                            cell.status,
                            cell.reason,
                            getattr(cell, "stale", False),
                        )
                    )

    if failures or only_failures:
        print_rule("failures & anomalies")
        if not failure_records:
            console.print("  [green]none recorded[/]")
        else:
            for problem, label, solver, status, reason, stale in failure_records:
                reason_str = reason or "[dim]no reason recorded[/]"
                tag = "[red]fail[/]" if status == FAILED else "[dark_orange]anom[/]"
                stale_mark = "  [dim](stale)[/]" if stale else ""
                console.print(
                    f"  {tag}  [dim]{problem}[/]  {label}  [bold]{solver}[/]  {reason_str}{stale_mark}"
                )

    # ── summary table: per-problem + overall ok-rate ─────────────────────────
    if per_problem_tally:
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

        from mosaic.benchmarks.core.status import (
            compute_score,
            format_score,
            weight_color,
        )

        def _progress_bar(
            score: float | None,
            n_ok: int,
            n_anom: int,
            n_fail: int,
            n_missing: int,
            n_excl_work: int,
            n_stale_ok: int = 0,
            width: int = 18,
        ) -> str:
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
            order = sorted(
                range(len(segs)), key=lambda i: raw[i] - chars[i], reverse=True
            )
            for i in order[:remainder]:
                chars[i] += 1
            bar = ""
            for (_, w), n_chars in zip(segs, chars):
                if n_chars > 0:
                    bar += f"[{weight_color(w)}]{'█' * n_chars}[/]"
            return bar

        def _score_colored(score: float | None) -> str:
            """Render the weighted score using the weight → colour ladder. ``None`` renders dim."""
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
                _progress_bar(
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
            # Aggregate score as weighted mean over per-problem scores,
            # weighted by each problem's contributing-cell count.
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
            _progress_bar(
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


# ── `mosaic tesseracts` ────────────────────────────────────────────────────


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
    from mosaic.benchmarks.code_ratio import (
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


# ── `mosaic paper-plots` ──────────────────────────────────────────────────


@app.command("paper-plots")
def paper_plots(
    out_dir: Path = typer.Option(
        None,
        "--out-dir",
        "-o",
        help="Output directory for generated figures. Defaults to paper/figures/ at repo root.",
    ),
    only: str = typer.Option(
        "all",
        "--only",
        help="Comma-separated plot names to generate, or 'all'. "
        "Available: agreement, cost_overview, fd_check, ics, jacobian_svd, "
        "physical_accuracy, coverage_heatmap.",
    ),
) -> None:
    """Generate all paper figures used in the Mosaic benchmark paper.

    Reads result JSON files from the benchmark results directory and writes
    PDFs (and PNGs for the coverage heatmap) to the output directory.
    """
    from mosaic.benchmarks.plots.paper import all_names, get_generate_fn

    repo = _repo_root()
    target_dir: Path = out_dir if out_dir is not None else repo / "paper" / "figures"
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


@app.command("new-domain")
def new_domain(
    name: str = typer.Argument(help="Name for the new domain (e.g. 'my-flow')."),
    from_template: str = typer.Option(
        ...,
        "--from-template",
        "-t",
        help="Template to scaffold from. Use 'mosaic templates' to list available templates.",
    ),
) -> None:
    """Scaffold a new benchmark domain from a template."""
    from mosaic.templates.scaffold import load_template, scaffold_domain

    tpl = load_template(from_template)
    created = scaffold_domain(name, tpl, target_dir=_repo_root() / "mosaic")
    print_rule(f"scaffolded domain: {name}")
    for role, path in created.items():
        console.print(f"  {role}: [green]{path.relative_to(_repo_root())}[/green]")
    console.print(
        f"\nNext steps:\n"
        f"  1. Edit the generated schemas and problem config\n"
        f"  2. Add a solver in mosaic/tesseracts/{name}/\n"
        f"  3. Register the domain in mosaic/benchmarks/problems/__init__.py\n"
    )


@app.command("validate-template")
def validate_template_cmd(
    template: str = typer.Argument(help="Template name or path to a YAML file."),
) -> None:
    """Validate a task template against its schema module."""
    from mosaic.templates.scaffold import load_template, validate_template

    tpl = load_template(template)
    errors = validate_template(tpl)
    if errors:
        console.print(f"[red]{len(errors)} error(s) in template {tpl.name!r}:[/red]")
        for err in errors:
            console.print(f"  [red]- {err}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Template {tpl.name!r} is valid.[/green]")


@app.command("templates")
def list_templates_cmd() -> None:
    """List available task templates."""
    from mosaic.templates.scaffold import list_templates, load_template

    templates = list_templates()
    if not templates:
        console.print("[dim]No templates found.[/dim]")
        return
    for name in templates:
        tpl = load_template(name)
        console.print(f"  [bold]{name}[/bold]  {tpl.description.strip()}")


if __name__ == "__main__":
    app()
