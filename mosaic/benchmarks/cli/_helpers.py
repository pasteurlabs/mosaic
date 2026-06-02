# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-command helpers for the mosaic CLI.

Helpers live here when they're either (a) used by more than one subcommand
module or (b) tightly coupled to the ``run`` command but factored out for
readability. All helpers preserve the exact behaviour of the pre-split
``cli.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import typer

from mosaic.benchmarks.core.console import console, print_rule, print_skip, print_warn
from mosaic.benchmarks.core.runner import build_all, image_tags_no_build
from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.shared import SUITES

_ALL_SUITES = list(SUITES)


def _repo_root() -> Path:
    # _helpers.py is at <repo>/mosaic/benchmarks/cli/_helpers.py
    return Path(__file__).resolve().parents[3]


def _suite_components(suite: str, cfg: Any) -> tuple[dict, callable]:
    """Return (experiments_dict, plot_fns_factory) for ``suite`` in ``cfg``.

    Filters ``cfg.experiments`` and ``cfg.plot_fns`` to the entries whose
    full key starts with ``"<suite>/"`` and strips the prefix. The plot
    factory is a no-arg callable that returns the (prefix-stripped) plot dict.
    """
    if suite not in SUITES:
        from difflib import get_close_matches

        suggestion = get_close_matches(suite, _ALL_SUITES, n=1, cutoff=0.5)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(f"Unknown suite {suite!r}. Choose from: {_ALL_SUITES}.{hint}")

    prefix = f"{suite}/"
    extra_prefix = f"_extra/{suite}/"
    exps = {
        k[len(prefix) :]: v.fn
        for k, v in cfg.experiments.items()
        if k.startswith(prefix)
    }
    plot_fns: dict = {}
    for k, v in cfg.plot_fns.items():
        if k.startswith(prefix):
            plot_fns[k[len(prefix) :]] = v
        elif k.startswith(extra_prefix):
            # Suite-scoped bonus plots (e.g. "_extra/forward/agreement")
            # — preserved under the "_extra/<name>" key so the runner can
            # call them unconditionally (no associated experiment needed).
            plot_fns[f"_extra/{k[len(extra_prefix) :]}"] = v
        elif k.startswith("_extra/") and not any(
            k.startswith(f"_extra/{s}/") for s in SUITES
        ):
            # Bare extras (e.g. "_extra/cost_overview") — not scoped to any
            # suite, so include them in every suite's plot set.
            plot_fns[k] = v
    return exps, lambda: plot_fns


def _apply_solver_filter(cfg: Any, solvers_csv: str | None) -> Any:
    """Return cfg restricted to the requested solver names, or ``None`` to signal "skip".

    Accepts two forms:

    * **Flat CSV** — ``"XLB,jax-cfd,Firedrake"`` — a union set applied to
      every problem. Each problem keeps only the solvers whose name is in
      the set. A name that doesn't exist on the current problem is fine
      (it presumably belongs to another problem) and is silently ignored;
      a name that doesn't exist on *any* problem still surfaces upstream
      as a typo via :func:`_validate_solver_names`. A problem with zero
      matches is **skipped**, not run with all solvers.
    * **Per-problem map** — ``"<problem>=<csv>;<problem>=<csv>"`` — looks
      up ``cfg.name`` in the map and filters to that problem's list.
      Problems not listed in the map pass through unchanged (all solvers
      kept), so you only spell out the problems you want to restrict.

    Empty/None passes through unchanged. Applied BEFORE build_all so that
    excluded or broken solvers aren't built when the user passes -s to
    restrict the run.
    """
    if not solvers_csv:
        return cfg
    if "=" in solvers_csv:
        per_problem = _parse_per_problem_solver_map(solvers_csv)
        if cfg.name not in per_problem:
            return cfg
        requested = set(per_problem[cfg.name])
        # Per-problem map: an unknown name *is* a typo, because the user
        # explicitly addressed this problem.
        unknown = requested - set(cfg.solver_names)
        if unknown:
            print_warn(
                f"{cfg.name}: unknown solver(s) in -s map: "
                f"{', '.join(sorted(unknown))} — skipping"
            )
    else:
        requested = {s.strip() for s in solvers_csv.split(",") if s.strip()}
        # Flat CSV: silently ignore names that don't apply here — they
        # presumably belong to another problem.
    keep = [s for s in cfg.solvers if s.name in requested]
    if not keep:
        print_warn(f"{cfg.name}: no solvers in -s match this problem — skipping")
        return None
    return dataclasses.replace(cfg, solvers=keep)


def _parse_per_problem_solver_map(s: str) -> dict[str, list[str]]:
    """Parse ``"<problem>=<csv>;<problem>=<csv>"`` into ``{problem: [solvers]}``.

    Raises ``ValueError`` on malformed entries (missing ``=``); empty
    segments between ``;`` are skipped.
    """
    out: dict[str, list[str]] = {}
    for entry in s.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"--solvers per-problem entry missing '=': {entry!r}. "
                f"Expected '<problem>=<solver,solver,...>'."
            )
        prob, csv = entry.split("=", 1)
        out[prob.strip()] = [v.strip() for v in csv.split(",") if v.strip()]
    return out


def _resolve_gpu_pool(cfg: Any, gpus: str | None) -> tuple[Any, str | None]:
    """Translate a `--gpus` string into a (cfg, gpus_csv) pair.

    Special value ``"none"`` (or ``"cpu"``): filter cfg to solvers with
    ``uses_gpu=False`` and pass ``gpus=None`` so the runner doesn't try to
    pin containers to a GPU. Used on CPU-only hosts that can't expose any
    GPU IDs at all.

    Anything else (a real comma-separated list, or None) passes through.
    """
    if isinstance(gpus, str) and gpus.lower() in ("none", "cpu", "cpu-only"):
        cpu_only = [s for s in cfg.solvers if not getattr(s, "uses_gpu", True)]
        if not cpu_only:
            print_warn(
                "no CPU-only solvers in this problem — --gpus none would run nothing"
            )
            return cfg, None
        # Return sentinel "cpu-only" string so CLI can pass gpu_ids=[] to the runner.
        # gpu_ids=[] in run_with_gpu_pool means: no GPU flags (CPU-only host).
        return dataclasses.replace(cfg, solvers=cpu_only), "cpu-only"
    return cfg, gpus


def _filter_hardware(cfg: Any, hardware: str | None) -> Any:
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
            gpu_solvers = [s.name for s in cfg.solvers if getattr(s, "uses_gpu", True)]
            if gpu_solvers:
                print_warn(
                    f"no GPU detected — skipping GPU solvers: {', '.join(gpu_solvers)}. "
                    "Pass --hardware all to override."
                )
                cpu_only = [s for s in cfg.solvers if not getattr(s, "uses_gpu", True)]
                if cpu_only:
                    return dataclasses.replace(cfg, solvers=cpu_only)
                print_warn("no CPU-only solvers either — keeping all solvers")
        return cfg
    if hardware.lower() == "cpu":
        filtered = [s for s in cfg.solvers if not getattr(s, "uses_gpu", True)]
        if not filtered:
            print_warn(
                "no CPU-only solvers in this problem — --hardware cpu would run nothing"
            )
            return cfg
        return dataclasses.replace(cfg, solvers=filtered)
    if hardware.lower() == "gpu":
        filtered = [s for s in cfg.solvers if getattr(s, "uses_gpu", True)]
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
    max_build_workers: int = 2,
):
    cfg = get_config(problem)
    # --hardware cpu/gpu filters solvers by target hardware BEFORE build.
    had_gpu_solvers = any(getattr(s, "uses_gpu", True) for s in cfg.solvers)
    cfg = _filter_hardware(cfg, hardware)
    no_gpu_solvers_left = not any(getattr(s, "uses_gpu", True) for s in cfg.solvers)
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
    # running `-s ins_jl,lettuce` on ns-grid). ``_apply_solver_filter``
    # returns ``None`` when -s leaves zero solvers for this problem — we
    # propagate that as the same skip-signal triple so the caller can mark
    # the problem skipped without entering the build step.
    cfg = _apply_solver_filter(cfg, solvers_csv)
    if cfg is None:
        return None, {}, gpus
    # --plots-only must never trigger an image build: the operation is purely
    # filesystem-level (load result.json, re-render PNG/PDF), so a rebuild is
    # pure overhead and can wedge the process behind a multi-minute docker
    # buildx stage.  Treat plots_only as implying no_build.
    if no_build or plots_only:
        return cfg, image_tags_no_build(cfg), gpus
    print_rule("build")
    tags = build_all(cfg, max_workers=max_build_workers)
    return cfg, tags, gpus


def _filter_solvers(cfg: Any, tags: dict, solvers_csv: str | None) -> tuple[Any, dict]:
    """Backward-compat wrapper: returns (cfg, filtered_tags).

    cfg is already filtered by _resolve_cfg_and_tags, but callers may still
    invoke this to get matching tags. A no-op when cfg is already narrowed.
    """
    if not solvers_csv:
        return cfg, tags
    keep_names = set(cfg.solver_names)
    filtered_tags = {k: v for k, v in tags.items() if k in keep_names}
    return cfg, filtered_tags


def _plots_only(
    cfg: Any,
    to_run: list | None,
    plot_fns: dict | None,
    suite: str,
    verbose_errors: bool = False,
) -> None:
    if plot_fns is None:
        console.print("No plot functions registered for this suite.")
        return
    print_rule("plots")
    # Plot against the FULL (unfiltered) config — mirrors run_suite's plot
    # step. The incoming cfg may have been hardware-filtered (e.g. GPU
    # solvers dropped on a CPU-only runner), but result trees / baselines
    # still carry every solver; styling those rows needs the complete
    # solver list, or per-solver style lookups raise KeyError.
    try:
        from mosaic.benchmarks.problems import get_config

        cfg = get_config(cfg.name)
    except Exception:
        pass
    if to_run:
        names = to_run
    else:
        # Default: only attempt plots for experiments actually configured for
        # this cfg+suite (avoids noisy [SKIP]s for variants that belong to a
        # different problem, e.g. stokes-only jacobian_svd_mu* on ns-3d-grid).
        # Also include _extra/* plots unconditionally — they are suite-wide
        # bonus plots that don't have an associated experiment result.
        prefix = f"{suite}/"
        configured = {
            k[len(prefix) :]
            for k, exp in cfg.experiments.items()
            if k.startswith(prefix) and exp.params
        }
        names = [n for n in plot_fns if n in configured or n.startswith("_extra/")]
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


def _run_validate_suites(suite_list: list[str]) -> None:
    """Validate suite names against the registry; raise typer.Exit(1) on unknown.

    Done up-front before any builds — a typo'd suite name should fail fast
    with a 'did you mean' hint rather than after a multi-minute build.
    """
    unknown_suites = [s for s in suite_list if s not in SUITES]
    if not unknown_suites:
        return
    from difflib import get_close_matches

    for s in unknown_suites:
        suggestion = get_close_matches(s, _ALL_SUITES, n=1, cutoff=0.5)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        console.print(f"[red]Unknown suite {s!r}.{hint}[/red]")
    console.print(f"Available suites: {', '.join(_ALL_SUITES)}")
    raise typer.Exit(1)


def _run_prepare_problem(
    problem: str,
    suite_list: list[str],
    *,
    plots_only: bool,
    no_build: bool,
    solvers: str | None,
    gpus: str | None,
    hardware: str | None,
    jobs: int,
):
    """Build (or skip building) for one problem and return (cfg, tags, gpus).

    Encapsulates the two cfg-prep paths: full build via
    :func:`_resolve_cfg_and_tags` when any non-ics suite is requested, and
    the no-build path used for ics-only / plots-only runs.
    """
    _needs_build = (not plots_only) and any(s != "ics" for s in suite_list)
    if _needs_build:
        cfg, tags, gpus = _resolve_cfg_and_tags(
            problem,
            no_build,
            solvers_csv=solvers,
            gpus=gpus,
            hardware=hardware,
            max_build_workers=jobs,
        )
        if cfg is None:
            return None, {}, gpus
        cfg, tags = _filter_solvers(cfg, tags, solvers)
        return cfg, tags, gpus
    cfg = get_config(problem)
    had_gpu = any(getattr(s, "uses_gpu", True) for s in cfg.solvers)
    cfg = _filter_hardware(cfg, hardware)
    no_gpu_left = not any(getattr(s, "uses_gpu", True) for s in cfg.solvers)
    if had_gpu and no_gpu_left and not gpus:
        gpus = "none"
    cfg, gpus = _resolve_gpu_pool(cfg, gpus)
    if solvers:
        cfg = _apply_solver_filter(cfg, solvers)
        if cfg is None:
            return None, {}, gpus
    return cfg, {}, gpus


def _validate_solver_csv(solvers_csv: str | None, problem_list: list[str]) -> None:
    """Up-front typo check: every name in a flat -s CSV must exist on at least one problem.

    Raises ``typer.Exit(1)`` when an unknown name is found so the user gets
    immediate feedback instead of "no solvers matched" warnings on every problem.

    Per-problem maps (``<problem>=<csv>;...``) skip this check — the
    per-problem dispatcher already warns about unknown names against the
    specific cfg.
    """
    if not solvers_csv or "=" in solvers_csv:
        return
    requested = {s.strip() for s in solvers_csv.split(",") if s.strip()}
    if not requested:
        return
    all_names: set[str] = set()
    for problem in problem_list:
        try:
            all_names |= set(get_config(problem).solver_names)
        except Exception:
            # Skip problems that fail to import — they'll surface as
            # "build failed" later in the loop and aren't relevant to a
            # name-typo check.
            continue
    unknown = requested - all_names
    if unknown:
        from difflib import get_close_matches

        for name in sorted(unknown):
            suggestion = get_close_matches(name, sorted(all_names), n=1, cutoff=0.5)
            hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
            console.print(f"[red]Unknown solver {name!r}.{hint}[/red]")
        console.print(f"Available solvers: {', '.join(sorted(all_names))}")
        raise typer.Exit(1)


def _parse_experiments_path(
    experiment: str,
) -> tuple[str | None, str | None, str | None]:
    """Split an ``--experiments`` value into (suite, exp, ic) segments.

    The flag accepts:

      * ``"all"`` → ``(None, None, None)``: run everything.
      * ``"<suite>"`` → ``(suite, None, None)``: suite-wide (no exp filter).
      * ``"<suite>/<exp>"`` → ``(suite, exp, None)``: pick one experiment.
      * ``"<suite>/<exp>/<ic>"`` → ``(suite, exp, ic)``: pick one experiment
        and filter its runs to the named IC.

    Experiment names themselves can contain ``/`` (e.g. sub-experiments like
    ``physical_laws/vs_N``). For ≥3-segment inputs, the suite is the first
    segment and the *remainder* is returned as ``exp`` verbatim — callers
    that need to disambiguate "deep experiment name" vs "exp + IC" use
    :func:`_resolve_experiment_target` with the problem's experiment
    registry to pick the right interpretation.

    The third segment, when present, is interpreted as an IC-name filter
    that is fed to :func:`mosaic.benchmarks.core.utils.iter_runs` via the
    ``cli_overrides["ic_names"]`` channel.
    """
    if experiment == "all":
        return None, None, None
    parts = experiment.split(
        "/", 2
    )  # at most: suite, exp, ic (exp may itself contain '/')
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        return parts[0], parts[1], None
    return parts[0], parts[1], parts[2]


def _resolve_experiment_target(
    experiment: str, exps: dict
) -> tuple[list[str] | None, str | None]:
    """Resolve ``--experiments`` against an experiment registry.

    Returns ``(to_run, ic_name)``:

      * ``to_run`` — list of experiment keys to pass to ``run_suite``, or
        ``None`` to run the whole suite.
      * ``ic_name`` — single IC name filter, or ``None`` for no IC narrowing.

    A literal multi-segment experiment name (e.g. ``physical_laws/vs_N``) is
    matched against the registry first; only when that key isn't registered
    does the parser fall back to the ``suite/exp/ic`` interpretation. This
    lets users target nested experiments by their literal slash-separated
    name without losing the IC-filter ergonomics of the 3-segment form.
    """
    if experiment == "all":
        return None, None
    _, exp, ic = _parse_experiments_path(experiment)
    if exp is None:
        return None, None
    # Try the literal "exp[/ic]" form against the registry first.
    literal = f"{exp}/{ic}" if ic else exp
    if literal in exps:
        return [literal], None
    if ic is None:
        # 2-segment form: caller still wants an exp filter even if the key
        # isn't in this suite's registry (run_suite emits an "unknown
        # experiment" warning, preserving current behaviour).
        return [exp], None
    return [exp], ic


def _run_build_overrides(
    cfg: Any,
    exps: dict,
    *,
    debug: bool,
    gpus: str | None,
    experiment: str,
) -> dict:
    """Assemble the ``overrides`` dict passed to :func:`run_suite`.

    Bundles debug toggle, gpu pool resolution, and IC filtering derived
    from the third segment of ``--experiments`` (e.g. ``forward/agreement/tgv``).
    Unknown IC names emit a warning and are dropped.
    """
    overrides: dict = {}
    if debug:
        overrides["debug"] = True
    if gpus == "cpu-only":
        overrides["gpu_ids"] = []
    elif gpus:
        overrides["gpu_ids"] = [g.strip() for g in gpus.split(",")]
    _, ic_segment = _resolve_experiment_target(experiment, exps)
    if ic_segment:
        if ic_segment not in cfg.make_ic:
            print_warn(f"unknown IC(s): {ic_segment} — skipping")
        else:
            overrides["ic_names"] = [ic_segment]
    return overrides


def _run_print_summary(
    problem_list: list[str],
    suite_list: list[str],
    run_status: dict[tuple[str, str], tuple[str, str]],
) -> None:
    """Render the (problem × suite) summary table at the end of `run`."""
    from rich.table import Table

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
