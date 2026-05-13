"""`mosaic run` — execute benchmark suites across problems."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from mosaic.benchmarks.cli import app
from mosaic.benchmarks.cli._helpers import (
    _ALL_SUITES,
    _parse_experiments_path,
    _plots_only,
    _run_build_overrides,
    _run_prepare_problem,
    _run_print_summary,
    _run_validate_suites,
    _suite_components,
)
from mosaic.benchmarks.core.console import console, print_rule, print_warn
from mosaic.benchmarks.core.io import RESULTS_DIR_ENV
from mosaic.benchmarks.core.runner import run_suite
from mosaic.benchmarks.problems import PROBLEMS


@app.command()
def run(
    problems: str = typer.Option(
        "all", "--problems", "-p", help="Comma-separated problems or 'all'"
    ),
    suites: str = typer.Option(
        "all", "--suites", help="Comma-separated suites or 'all'"
    ),
    experiments: str = typer.Option(
        "all",
        "--experiments",
        "-e",
        help="Experiment selector (default: 'all'). Accepts up to three "
        "slash-separated segments: '<suite>', '<suite>/<exp>', or "
        "'<suite>/<exp>/<ic>'. The third segment, when present, filters the "
        "experiment's runs to the named initial condition.",
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
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Root directory for benchmark results.  Defaults to ./mosaic-results "
        "in the current working directory.  Can also be set via the "
        f"{RESULTS_DIR_ENV} environment variable.",
    ),
    jobs: int = typer.Option(
        2,
        "--jobs",
        "-j",
        help="Max concurrent docker builds (default 2). "
        "On a fresh worker machine 4-8 is usually faster.",
        min=1,
    ),
):
    """Run benchmark suites across problems.

    Build each problem's solver images once, then run all requested suites in
    sequence.  Individual experiment failures are caught and logged; execution
    always continues with the next (problem, suite) pair.

    Examples::

        mosaic run -p ns-grid --suites forward
        mosaic run -p ns-grid -e gradient/fd_check
        mosaic run --plots-only
        mosaic run -p ns-grid -e forward/agreement/tgv
        mosaic run -o /tmp/bench-results -p ns-grid --suites forward

    Summary table legend:
      ok        all experiments completed
      N/M       partial — N of M experiments completed
      skip      suite not configured for this problem
      error     suite could not start (e.g. build failed, import error)
    """
    if output_dir is not None:
        os.environ[RESULTS_DIR_ENV] = str(output_dir.resolve())

    problem_list = (
        PROBLEMS if problems == "all" else [p.strip() for p in problems.split(",")]
    )

    # Parse the unified --experiments selector. When suite/exp segments are
    # present they implicitly narrow --suites and pin the experiment within
    # each suite; the third segment, if present, populates ic_names below.
    try:
        suite_seg, exp_seg, _ = _parse_experiments_path(experiments)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if suite_seg is not None:
        # --experiments takes precedence over --suites for suite selection.
        suite_list = [suite_seg]
    else:
        suite_list = (
            _ALL_SUITES if suites == "all" else [s.strip() for s in suites.split(",")]
        )

    # Validate suite names early — before any builds start.
    _run_validate_suites(suite_list)

    to_run = [exp_seg] if exp_seg is not None else None

    # status[(problem, suite)] = ("ok" | "partial" | "skip" | "error", detail)
    run_status: dict[tuple[str, str], tuple[str, str]] = {}

    for problem in problem_list:
        print_rule(f"problem: {problem}")
        try:
            cfg, tags, gpus = _run_prepare_problem(
                problem,
                suite_list,
                plots_only=plots_only,
                no_build=no_build,
                solvers=solvers,
                gpus=gpus,
                hardware=hardware,
                jobs=jobs,
            )
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
                    _plots_only(
                        cfg, to_run, plot_fns_fn(), suite, verbose_errors=traceback
                    )
                    run_status[(problem, suite)] = ("ok", "plots-only")
                    continue
                _overrides = _run_build_overrides(
                    cfg,
                    exps,
                    debug=debug,
                    gpus=gpus,
                    experiment=experiments,
                )
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
    _run_print_summary(problem_list, suite_list, run_status)
