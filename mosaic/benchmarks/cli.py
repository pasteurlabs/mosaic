"""Unified CLI entrypoint for Mosaic benchmarks."""

from __future__ import annotations

import dataclasses
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import typer

from mosaic.benchmarks.core.console import console, print_rule, print_skip, print_warn
from mosaic.benchmarks.core.runner import build_all, image_tags_no_build, run_suite
from mosaic.benchmarks.problems import PROBLEMS, get_config


def _all_known_solver_images() -> dict[str, str]:
    """Return {image_tag: problem/solver} for every solver across all problems."""
    import yaml

    known: dict[str, str] = {}
    for prob in PROBLEMS:
        try:
            cfg = get_config(prob)
        except Exception:
            continue
        for solver_name, spec in cfg.solvers.items():
            config_path = cfg.tesseract_dir / spec.dir / "tesseract_config.yaml"
            if not config_path.exists():
                continue
            try:
                with open(config_path) as f:
                    tconfig = yaml.safe_load(f)
                image_name = tconfig.get("name", spec.dir)
                tag = f"{image_name}:latest"
                known[tag] = f"{prob}/{solver_name}"
            except Exception:
                pass
    return known


def _running_container_images() -> set[str]:
    """Return the set of image tags / IDs currently used by running containers."""
    import subprocess

    result = subprocess.run(
        ["docker", "ps", "--no-trunc", "--format", "{{.Image}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


app = typer.Typer(name="mosaic", rich_markup_mode="rich", add_completion=False)

_ALL_SUITES = ["ics", "forward", "cost", "gradient", "optimization"]


def _repo_root() -> Path:
    # cli.py is at <repo>/mosaic/benchmarks/cli.py
    return Path(__file__).resolve().parents[2]


# ── `mosaic status --watch` helpers ────────────────────────────────────────


def _cpu_snapshot() -> float:
    """Current overall CPU utilisation percent (0..100). 0 on any failure.

    Uses psutil if available (fast, non-blocking). Falls back to /proc/stat
    delta in case psutil isn't importable on some odd host.
    """
    try:
        import psutil

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        pass
    try:
        with open("/proc/stat") as f:
            first = f.readline().split()[1:8]
        vals = list(map(int, first))
        total = sum(vals)
        idle = vals[3]
        time.sleep(0.05)
        with open("/proc/stat") as f:
            second = f.readline().split()[1:8]
        vals2 = list(map(int, second))
        total2 = sum(vals2)
        idle2 = vals2[3]
        dt = total2 - total
        di = idle2 - idle
        return 100.0 * (dt - di) / dt if dt > 0 else 0.0
    except Exception:
        return 0.0


_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"


def _sparkline(values, width: int = 40, lo: float = 0.0, hi: float = 100.0) -> str:
    """Render a unicode sparkline of the last *width* samples in values."""
    if not values:
        return "[dim]" + " " * width + "[/]"
    vs = list(values)[-width:]
    out = []
    n = len(_SPARK_BLOCKS) - 1
    for v in vs:
        norm = 0.0 if hi <= lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
        out.append(_SPARK_BLOCKS[int(round(norm * n))])
    # Pad left so the newest sample is always flush right.
    pad = width - len(out)
    return ("[dim]" + " " * pad + "[/]" if pad else "") + "".join(out)


def _gpu_snapshot() -> list[tuple[str, str, str, str]]:
    """(idx, util%, mem_used, mem_total) per visible GPU; empty list on failure."""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                out.append(tuple(parts))
        return out
    except Exception:
        return []


def _docker_snapshot(limit: int = 12) -> list[tuple[str, str, str]]:
    """(name, image, uptime) for every running container."""
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode != 0:
            return []
        rows: list[tuple[str, str, str]] = []
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
        return rows[:limit]
    except Exception:
        return []


def _recent_result_activity(
    since_secs: int = 600, limit: int = 8
) -> list[tuple[str, float]]:
    """Most recently modified result.json files, newest first.

    Scans benchmarks/results/**/result.json with mtime in the last
    *since_secs* seconds. Truncates to *limit* most-recent entries. Uses
    ``rglob`` which is cheap at our repo size (<200 result files).
    """
    root = Path(__file__).parent / "results"
    if not root.is_dir():
        return []
    now = time.time()
    hits: list[tuple[str, float]] = []
    for p in root.rglob("result.json"):
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if now - mt < since_secs:
            rel = p.relative_to(root).parent.as_posix()
            hits.append((rel, mt))
    hits.sort(key=lambda t: t[1], reverse=True)
    return hits[:limit]


def _run_watch_loop(
    problem_list: list[str], suite_list: list[str], interval: float
) -> None:
    """Live-refresh the status summary + GPU/docker/activity panels."""
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    from mosaic.benchmarks.core.status import (
        ANOMALY,
        EXCL_PERMANENT,
        EXCL_UNSPECIFIED,
        EXCLUDED,
        FAILED,
        NOT_RUN,
        OK,
        collect_status,
        compute_score,
        format_score,
        weight_color,
    )

    def _watch_score(score: float | None) -> str:
        return f"[bold {weight_color(score)}]{format_score(score)}[/]"

    def _summary_table() -> Table:
        t = Table(title="Status summary", show_header=True, header_style="bold")
        t.add_column("problem", style="bold", no_wrap=True)
        t.add_column("ok", justify="right")
        t.add_column("anom", justify="right")
        t.add_column("fail", justify="right")
        t.add_column("miss", justify="right")
        t.add_column("excl·w", justify="right")
        t.add_column("excl·p", justify="right")
        t.add_column("stale", justify="right")
        t.add_column("progress", no_wrap=True)
        t.add_column("score", justify="right")
        # counts layout: [fresh_ok, anom, fail, miss, excl_w, excl_p, stale, stale_ok]
        total = [0] * 8
        score_num = 0.0
        score_den = 0
        for problem in problem_list:
            try:
                cfg = get_config(problem)
            except Exception:
                continue
            st = collect_status(cfg, suites=suite_list)
            counts = [0] * 8
            all_cells = []
            for row in st.rows:
                for solver in st.solvers:
                    c = row.cells.get(solver)
                    if c is None:
                        continue
                    all_cells.append(c)
                    is_stale = getattr(c, "stale", False)
                    if is_stale:
                        counts[6] += 1
                    if c.status == OK:
                        if is_stale:
                            counts[7] += 1  # stale-ok (work-to-do, not ok)
                        else:
                            counts[0] += 1
                    elif c.status == ANOMALY:
                        counts[1] += 1
                    elif c.status == FAILED:
                        counts[2] += 1
                    elif c.status == NOT_RUN:
                        counts[3] += 1
                    elif c.status == EXCLUDED:
                        cat = c.category or EXCL_UNSPECIFIED
                        if cat in EXCL_PERMANENT:
                            counts[5] += 1
                        else:
                            counts[4] += 1
            for i in range(8):
                total[i] += counts[i]
            p_score, p_score_n = compute_score(all_cells)
            if p_score is not None:
                score_num += p_score * p_score_n
                score_den += p_score_n
            t.add_row(
                problem,
                f"[green]{counts[0]}[/]",
                f"[dark_orange]{counts[1]}[/]" if counts[1] else "0",
                f"[red]{counts[2]}[/]" if counts[2] else "0",
                f"[dim]{counts[3]}[/]" if counts[3] else "0",
                f"[yellow]{counts[4]}[/]" if counts[4] else "0",
                f"[dim yellow]{counts[5]}[/]" if counts[5] else "0",
                f"[dim]{counts[6]}[/]" if counts[6] else "[dim]0[/]",
                _watch_progress_bar(p_score, counts),
                _watch_score(p_score),
            )
        overall_score = (score_num / score_den) if score_den else None
        t.add_row(
            "[bold]overall[/]",
            f"[green]{total[0]}[/]",
            f"[dark_orange]{total[1]}[/]" if total[1] else "0",
            f"[red]{total[2]}[/]" if total[2] else "0",
            f"[dim]{total[3]}[/]" if total[3] else "0",
            f"[yellow]{total[4]}[/]" if total[4] else "0",
            f"[dim yellow]{total[5]}[/]" if total[5] else "0",
            f"[dim]{total[6]}[/]" if total[6] else "[dim]0[/]",
            _watch_progress_bar(overall_score, total),
            _watch_score(overall_score),
        )
        return t

    # Rolling per-device history of utilisation percentages, rendered as
    # unicode sparklines. Width keeps last 60 samples (1 hour @ 60s cadence).
    history: dict[str, deque] = {"cpu": deque(maxlen=60)}

    def _gpu_table() -> Table:
        snaps = _gpu_snapshot()
        t = Table(
            title="CPU & GPU usage (last 60 samples)",
            show_header=True,
            header_style="bold",
        )
        t.add_column("#", justify="right")
        t.add_column("util %", justify="right")
        t.add_column("mem", justify="right")
        t.add_column("trend", no_wrap=True)
        # CPU row — capture sample, append to history, render sparkline.
        cpu = _cpu_snapshot()
        history["cpu"].append(cpu)
        cpu_colour = "green" if cpu > 30 else "yellow" if cpu > 5 else "dim"
        t.add_row(
            "cpu",
            f"[{cpu_colour}]{cpu:.0f}%[/]",
            "-",
            _sparkline(history["cpu"]),
        )
        if not snaps:
            t.add_row("[dim]nvidia-smi unavailable[/]", "", "", "")
            return t
        for idx, util, mem_used, mem_total in snaps:
            util_int = int(util) if util.isdigit() else 0
            util_colour = (
                "green" if util_int > 30 else "yellow" if util_int > 5 else "dim"
            )
            key = f"gpu{idx}"
            history.setdefault(key, deque(maxlen=60)).append(float(util_int))
            t.add_row(
                f"gpu{idx}",
                f"[{util_colour}]{util}%[/]",
                f"{mem_used}/{mem_total} MiB",
                _sparkline(history[key]),
            )
        return t

    def _docker_table() -> Table:
        rows = _docker_snapshot()
        t = Table(
            title=f"Docker ({len(rows)} running)", show_header=True, header_style="bold"
        )
        t.add_column("name", style="bold", no_wrap=True)
        t.add_column("image", no_wrap=True)
        t.add_column("uptime", justify="right", no_wrap=True)
        if not rows:
            t.add_row("[dim]no containers[/]", "", "")
            return t
        for name, image, status in rows:
            # Trim "Up X minutes" → "X min" style
            up = status.replace("Up ", "")
            t.add_row(name[:22], image[:40], up)
        return t

    def _activity_table() -> Table:
        hits = _recent_result_activity(since_secs=900, limit=10)
        t = Table(
            title="Recent result.json writes (last 15 min)",
            show_header=True,
            header_style="bold",
        )
        t.add_column("experiment", style="bold", no_wrap=True)
        t.add_column("age", justify="right")
        if not hits:
            t.add_row("[dim]no recent writes[/]", "")
            return t
        now = time.time()
        for rel, mt in hits:
            age_s = int(now - mt)
            if age_s < 60:
                age = f"{age_s}s ago"
            else:
                age = f"{age_s // 60}m {age_s % 60}s ago"
            t.add_row(rel, age)
        return t

    def _frame():
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = Panel(
            f"[bold]mosaic status --watch[/]  ·  refreshed {ts}  ·  interval {interval:.0f}s  "
            f"·  [dim]Ctrl-C to exit[/]",
            style="bold blue",
        )
        return Group(
            header,
            _summary_table(),
            Group(_gpu_table(), _docker_table()),
            _activity_table(),
        )

    # Prime psutil CPU baseline so the first reading isn't artificially 0.
    try:
        import psutil as _ps

        _ps.cpu_percent(interval=None)
    except Exception:
        pass

    try:
        with Live(_frame(), console=console, refresh_per_second=4, screen=True) as live:
            while True:
                time.sleep(max(0.5, interval))
                live.update(_frame())
    except KeyboardInterrupt:
        console.print("[dim]watch stopped[/]")


def _stacked_bar(segs: list[tuple[int, float]], width: int) -> str:
    """Stacked colour bar.  segs = [(count, weight), ...] ordered best → worst.

    Each segment occupies width × count/total chars; colour comes from
    weight_color(weight).  Returns dim dashes when total == 0.
    """
    from mosaic.benchmarks.core.status import weight_color as _wc

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
    for (_, w), n_chars in zip(segs, chars):
        if n_chars > 0:
            bar += f"[{_wc(w)}]{'█' * n_chars}[/]"
    return bar


def _watch_progress_bar(score: float | None, counts: list[int], width: int = 18) -> str:
    """Stacked health bar for the watch table.

    counts layout: [fresh_ok, anom, fail, miss, excl_w, excl_p, stale, stale_ok]
    Categorical exclusions (excl_p) are out of the denominator and not shown.
    """
    stale_ok = counts[7] if len(counts) > 7 else 0
    segs: list[tuple[int, float]] = [
        (counts[0], 1.00),  # fresh ok
        (stale_ok, 0.67),  # stale ok
        (counts[1], 0.53),  # anomaly
        (counts[3] + counts[4], 0.33),  # missing + excl·work
        (counts[2], 0.00),  # fail
    ]
    return _stacked_bar(segs, width)


def _watch_pct(pct: float) -> str:
    if pct >= 95:
        return f"[bold green]{pct:.0f}%[/]"
    if pct >= 80:
        return f"[yellow]{pct:.0f}%[/]"
    return f"[bold red]{pct:.0f}%[/]"


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


def _filter_hardware(cfg, hardware: str | None):
    """Filter solvers by hardware target.

    Returns a (possibly filtered) cfg.  ``hardware`` is one of:
      "cpu"  — keep only ``uses_gpu=False`` solvers
      "gpu"  — keep only ``uses_gpu=True``  solvers
      "all"  — keep everything (default)
      None   — keep everything
    """
    if not hardware or hardware.lower() == "all":
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
    cfg = _filter_hardware(cfg, hardware)
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
def forward(
    problem: str = typer.Option(
        ..., "--problem", "-p", help="Problem config to benchmark"
    ),
    experiment: str = typer.Option(
        "all", "--experiment", "-e", help="Experiment to run (default: all)"
    ),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip plot generation"),
    plots_only: bool = typer.Option(
        False, "--plots-only", help="Skip execution; regenerate plots only"
    ),
    no_build: bool = typer.Option(
        False, "--no-build", help="Skip building solver images"
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass solver output cache"
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
        "Pass 'none' to skip GPU pinning on a CPU-only host.",
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
):
    """Run forward benchmarks (agreement, convergence, diagnostics, stability)."""
    from mosaic.benchmarks.suites.forward import _EXPERIMENTS, _plot_fns

    cfg, tags, gpus = _resolve_cfg_and_tags(
        problem,
        no_build,
        plots_only=plots_only,
        solvers_csv=solvers,
        gpus=gpus,
        hardware=hardware,
    )
    cfg, tags = _filter_solvers(cfg, tags, solvers)
    to_run = None if experiment == "all" else [experiment]
    if plots_only:
        _plots_only(cfg, to_run, _plot_fns(), verbose_errors=traceback)
        return
    _overrides: dict = {}
    if debug:
        _overrides["debug"] = True
    if gpus == "cpu-only":
        _overrides["gpu_ids"] = []  # empty list = CPU-only sentinel (no GPU flags)
    elif gpus:
        _overrides["gpu_ids"] = [g.strip() for g in gpus.split(",")]
    if ics:
        requested = [s.strip() for s in ics.split(",") if s.strip()]
        unknown = [ic for ic in requested if ic not in cfg.make_ic]
        if unknown:
            print_warn(f"unknown IC(s): {', '.join(sorted(unknown))} — skipping")
        valid = [ic for ic in requested if ic in cfg.make_ic]
        if valid:
            _overrides["ic_names"] = valid
    run_suite(
        cfg,
        tags,
        _EXPERIMENTS,
        to_run=to_run,
        plots=not no_plots,
        plot_fns=_plot_fns() if not no_plots else None,
        suite_name="forward",
        verbose_errors=traceback,
        overrides=_overrides or None,
    )


@app.command()
def cost(
    problem: str = typer.Option(
        ..., "--problem", "-p", help="Problem config to benchmark"
    ),
    experiment: str = typer.Option(
        "all", "--experiment", "-e", help="Experiment to run (default: all)"
    ),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip plot generation"),
    plots_only: bool = typer.Option(
        False, "--plots-only", help="Skip execution; regenerate plots only"
    ),
    no_build: bool = typer.Option(
        False, "--no-build", help="Skip building solver images"
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass solver output cache"
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
        "Pass 'none' to skip GPU pinning on a CPU-only host.",
    ),
    hardware: str | None = typer.Option(
        None,
        "--hardware",
        help="Filter solvers by hardware target: 'cpu', 'gpu', or 'all' (default).",
    ),
):
    """Run cost benchmarks (forward and VJP wall-clock timing)."""
    from mosaic.benchmarks.suites.cost import _EXPERIMENTS, _plot_fns

    cfg, tags, gpus = _resolve_cfg_and_tags(
        problem,
        no_build,
        plots_only=plots_only,
        solvers_csv=solvers,
        gpus=gpus,
        hardware=hardware,
    )
    cfg, tags = _filter_solvers(cfg, tags, solvers)
    to_run = None if experiment == "all" else [experiment]
    if plots_only:
        _plots_only(cfg, to_run, _plot_fns(), verbose_errors=traceback)
        return
    _overrides: dict = {}
    if debug:
        _overrides["debug"] = True
    if gpus == "cpu-only":
        _overrides["gpu_ids"] = []  # empty list = CPU-only sentinel (no GPU flags)
    elif gpus:
        _overrides["gpu_ids"] = [g.strip() for g in gpus.split(",")]
    run_suite(
        cfg,
        tags,
        _EXPERIMENTS,
        to_run=to_run,
        plots=not no_plots,
        plot_fns=_plot_fns() if not no_plots else None,
        suite_name="cost",
        verbose_errors=traceback,
        overrides=_overrides or None,
    )


@app.command()
def gradient(
    problem: str = typer.Option(
        ..., "--problem", "-p", help="Problem config to benchmark"
    ),
    experiment: str = typer.Option(
        "all", "--experiment", "-e", help="Experiment to run (default: all)"
    ),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip plot generation"),
    plots_only: bool = typer.Option(
        False, "--plots-only", help="Skip execution; regenerate plots only"
    ),
    no_build: bool = typer.Option(
        False, "--no-build", help="Skip building solver images"
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass solver output cache"
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
        "Pass 'none' to skip GPU pinning on a CPU-only host.",
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
):
    """Run gradient evaluation benchmarks (FD verification, parameter sweep, horizon sweep)."""
    from mosaic.benchmarks.suites.gradient import _EXPERIMENTS, _plot_fns

    cfg, tags, gpus = _resolve_cfg_and_tags(
        problem,
        no_build,
        plots_only=plots_only,
        solvers_csv=solvers,
        gpus=gpus,
        hardware=hardware,
    )
    cfg, tags = _filter_solvers(cfg, tags, solvers)
    to_run = None if experiment == "all" else [experiment]
    if plots_only:
        _plots_only(cfg, to_run, _plot_fns(), verbose_errors=traceback)
        return
    _overrides: dict = {}
    if debug:
        _overrides["debug"] = True
    if gpus == "cpu-only":
        _overrides["gpu_ids"] = []  # empty list = CPU-only sentinel (no GPU flags)
    elif gpus:
        _overrides["gpu_ids"] = [g.strip() for g in gpus.split(",")]
    if ics:
        requested = [s.strip() for s in ics.split(",") if s.strip()]
        unknown = [ic for ic in requested if ic not in cfg.make_ic]
        if unknown:
            print_warn(f"unknown IC(s): {', '.join(sorted(unknown))} — skipping")
        valid = [ic for ic in requested if ic in cfg.make_ic]
        if valid:
            _overrides["ic_names"] = valid
    run_suite(
        cfg,
        tags,
        _EXPERIMENTS,
        to_run=to_run,
        plots=not no_plots,
        plot_fns=_plot_fns() if not no_plots else None,
        suite_name="gradient",
        verbose_errors=traceback,
        overrides=_overrides or None,
    )


@app.command()
def optimization(
    problem: str = typer.Option(
        ..., "--problem", "-p", help="Problem config to benchmark"
    ),
    experiment: str = typer.Option(
        "all", "--experiment", "-e", help="Experiment to run (default: all)"
    ),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip plot generation"),
    plots_only: bool = typer.Option(
        False, "--plots-only", help="Skip execution; regenerate plots only"
    ),
    no_build: bool = typer.Option(
        False, "--no-build", help="Skip building solver images"
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass solver output cache"
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
        "Pass 'none' to skip GPU pinning on a CPU-only host.",
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
):
    """Run optimization benchmarks (IC recovery and parameter optimization via gradient descent)."""
    from mosaic.benchmarks.suites.optimization import _EXPERIMENTS, _plot_fns

    cfg, tags, gpus = _resolve_cfg_and_tags(
        problem,
        no_build,
        plots_only=plots_only,
        solvers_csv=solvers,
        gpus=gpus,
        hardware=hardware,
    )
    cfg, tags = _filter_solvers(cfg, tags, solvers)
    to_run = None if experiment == "all" else [experiment]
    if plots_only:
        _plots_only(cfg, to_run, _plot_fns(), verbose_errors=traceback)
        return
    _overrides: dict = {}
    if debug:
        _overrides["debug"] = True
    if gpus == "cpu-only":
        _overrides["gpu_ids"] = []  # empty list = CPU-only sentinel (no GPU flags)
    elif gpus:
        _overrides["gpu_ids"] = [g.strip() for g in gpus.split(",")]
    if ics:
        requested = [s.strip() for s in ics.split(",") if s.strip()]
        unknown = [ic for ic in requested if ic not in cfg.make_ic]
        if unknown:
            print_warn(f"unknown IC(s): {', '.join(sorted(unknown))} — skipping")
        valid = [ic for ic in requested if ic in cfg.make_ic]
        if valid:
            _overrides["ic_names"] = valid
    # When -e is a sub-run path (e.g. "drag_opt/re100"), inject a run_names
    # filter so iter_runs only yields the matching named run.  The top-level
    # experiment key (e.g. "drag_opt") is looked up in _EXPERIMENTS and the
    # sub-run name (e.g. "re100") gates iter_runs iteration.
    if experiment != "all" and "/" in experiment:
        _parts = experiment.split("/", 1)
        _top, _sub = _parts[0], _parts[1]
        # Only inject run_names when the top key *and* the sub-run alias both
        # exist in _EXPERIMENTS (avoids breaking genuine nested names).
        if _top in _EXPERIMENTS and experiment in _EXPERIMENTS:
            _overrides["run_names"] = [_sub]
    run_suite(
        cfg,
        tags,
        _EXPERIMENTS,
        to_run=to_run,
        plots=not no_plots,
        plot_fns=_plot_fns() if not no_plots else None,
        suite_name="optimization",
        verbose_errors=traceback,
        overrides=_overrides or None,
    )


@app.command()
def run_all(
    problems: str = typer.Option(
        "all", "--problems", help="Comma-separated problems or 'all'"
    ),
    suites: str = typer.Option(
        "all", "--suites", help="Comma-separated suites or 'all'"
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
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass solver output cache"
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
        "Pass 'none' to skip GPU pinning on a CPU-only host.",
    ),
    hardware: str | None = typer.Option(
        None,
        "--hardware",
        help="Filter solvers by hardware target: 'cpu', 'gpu', or 'all' (default).",
    ),
):
    """Run every suite across every problem; print a summary table at the end.

    Build each problem's solver images once, then run all requested suites in
    sequence.  Individual experiment failures are caught and logged; execution
    always continues with the next (problem, suite) pair.

    Pass [bold]--plots-only[/] to skip all solver runs and just refresh every
    plot across every suite and problem — useful after a plot-code edit or
    after a partial rerun of a few cells. No Docker, no GPU; runs in seconds.

    Summary table legend:
      ok        all experiments completed
      N/M       partial — N of M experiments completed
      skip      suite not configured for this problem
      error     suite could not start (e.g. build failed, import error)
    """
    from rich.table import Table

    problem_list = (
        PROBLEMS if problems == "all" else [p.strip() for p in problems.split(",")]
    )
    suite_list = (
        _ALL_SUITES if suites == "all" else [s.strip() for s in suites.split(",")]
    )

    # status[(problem, suite)] = ("ok" | "partial" | "skip" | "error", detail)
    status: dict[tuple[str, str], tuple[str, str]] = {}

    for problem in problem_list:
        print_rule(f"problem: {problem}")
        # --plots-only never needs Docker; same for 'ics' suite. Resolve cfg
        # without building unless a solver run is actually going to happen.
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
                cfg = _filter_hardware(cfg, hardware)
                cfg, gpus = _resolve_gpu_pool(cfg, gpus)
                if solvers:
                    cfg = _apply_solver_filter(cfg, solvers)
                tags = {}
        except Exception as exc:
            msg = f"build failed: {exc}"
            console.print(f"  [red]{msg}[/]")
            for suite in suite_list:
                status[(problem, suite)] = ("error", msg)
            continue

        for suite in suite_list:
            print_rule(f"  suite: {suite}")
            try:
                exps, plot_fns_fn = _suite_components(suite, cfg=cfg)
                if plots_only:
                    # Regenerate plots for every registered experiment using
                    # whatever result.json already exists on disk. Mirrors
                    # the per-suite `_plots_only(cfg, ...)` helper used by
                    # the single-suite commands.
                    _plots_only(cfg, None, plot_fns_fn(), verbose_errors=traceback)
                    status[(problem, suite)] = ("ok", "plots-only")
                    continue
                _overrides: dict = {}
                if debug:
                    _overrides["debug"] = True
                if gpus:
                    _overrides["gpu_ids"] = [g.strip() for g in gpus.split(",")]
                results = run_suite(
                    cfg,
                    tags,
                    exps,
                    plots=not no_plots,
                    plot_fns=plot_fns_fn() if not no_plots else None,
                    suite_name=suite,
                    verbose_errors=traceback,
                    overrides=_overrides or None,
                )
                n_total = len(exps)
                n_ok = len(results)
                if n_ok == n_total:
                    status[(problem, suite)] = ("ok", "")
                elif n_ok > 0:
                    status[(problem, suite)] = ("partial", f"{n_ok}/{n_total}")
                else:
                    status[(problem, suite)] = ("skip", "no experiments ran")
            except Exception as exc:
                if traceback:
                    console.print_exception()
                status[(problem, suite)] = ("error", str(exc))
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
            state, detail = status.get((problem, suite), ("skip", "not run"))
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
):
    """Generate initial-condition visualisations (no solver builds needed)."""
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
def clean(
    images: bool = typer.Option(
        False,
        "--images",
        help="Also remove known solver images not used by a running container.",
    ),
    problems: str = typer.Option(
        "all",
        "--problems",
        "-p",
        help="Comma-separated problems whose solver images to target (with --images). Default: all.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be removed without actually removing anything.",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help=(
            "Deep clean: restart the Docker daemon (releases stale cache locks) then "
            "run 'docker system prune -a'. Kills all running containers. Requires sudo."
        ),
    ),
):
    """Remove Docker build cache, stopped containers, and dangling images.

    Safe to run at any time — running containers and their images are never
    touched.  Use [bold]--images[/bold] to also remove solver images that are
    not currently in use by a running container.

    Use [bold]--full[/bold] when the normal clean reports 0B freed despite a
    large cache — this restarts the Docker daemon to release stale locks (from
    killed containers) and then runs [bold]docker system prune -a[/bold].
    [red]Warning:[/red] kills all running containers and requires sudo.

    Examples::

        mosaic clean                          # cache + dangling only
        mosaic clean --images                 # + all idle solver images
        mosaic clean --images -p thermal-mesh # + idle thermal-mesh images only
        mosaic clean --full                   # deep clean (restarts Docker daemon)
        mosaic clean --dry-run                # preview without removing
    """
    import subprocess

    def _docker(*args: str) -> tuple[str, int]:
        r = subprocess.run(["docker", *args], capture_output=True, text=True)
        return r.stdout.strip(), r.returncode

    tag = "[dim](dry-run)[/dim] " if dry_run else ""

    # ── 0. Full clean (daemon restart + system prune) ─────────────────────────
    if full:
        print_rule("full clean")
        running_out, _ = _docker("ps", "--format", "{{.ID}}\t{{.Image}}\t{{.Names}}")
        running_containers = [l for l in running_out.splitlines() if l.strip()]
        if running_containers:
            console.print(
                f"  [yellow]warning:[/yellow] {len(running_containers)} running container(s) will be killed:"
            )
            for line in running_containers:
                parts = line.split("\t")
                cid = parts[0][:12]
                img = parts[1] if len(parts) > 1 else "?"
                name = parts[2] if len(parts) > 2 else ""
                console.print(f"    [dim]{cid}[/dim]  {img}  {name}")
        if dry_run:
            console.print(f"  {tag}would run: sudo systemctl restart docker")
            console.print(f"  {tag}would run: docker system prune -a -f")
            return
        print_rule("restarting docker daemon")
        r = subprocess.run(
            ["sudo", "systemctl", "restart", "docker"], capture_output=True, text=True
        )
        if r.returncode != 0:
            print_warn(f"daemon restart failed: {r.stderr.strip()}")
            return
        console.print("  [green]docker daemon restarted[/green]")
        print_rule("docker system prune -a")
        r = subprocess.run(
            ["docker", "system", "prune", "-a", "-f"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print_warn(f"system prune failed: {r.stderr.strip()}")
        else:
            freed_line = next(
                (
                    l
                    for l in r.stdout.splitlines()
                    if "reclaimed" in l.lower() or l.strip().lower().startswith("total")
                ),
                "",
            )
            console.print(
                f"  freed: {freed_line.strip()}"
                if freed_line
                else "  [green]done (0B reclaimed)[/green]"
            )
        return

    # ── 1. Build cache ────────────────────────────────────────────────────────
    print_rule("build cache")
    out, _ = _docker("builder", "du")
    # Summarise total cache size for the user before pruning
    size_line = next(
        (l for l in (out or "").splitlines() if "Total" in l or "Build cache" in l), ""
    )
    if size_line:
        console.print(f"  current: {size_line.strip()}")
    if not dry_run:
        freed_out, rc = _docker("builder", "prune", "--all", "-f")
        if rc != 0:
            print_warn("builder prune failed")
        else:
            freed_line = next(
                (
                    l
                    for l in (freed_out or "").splitlines()
                    if "reclaimed" in l.lower() or l.strip().lower().startswith("total")
                ),
                "",
            )
            console.print(
                f"  freed: {freed_line.strip()}"
                if freed_line
                else "  [green]done (0B reclaimed)[/green]"
            )
    else:
        console.print(f"  {tag}would run: docker builder prune --all -f")

    # ── 2. Stopped containers ─────────────────────────────────────────────────
    print_rule("stopped containers")
    stopped_out, _ = _docker(
        "ps",
        "-a",
        "--filter",
        "status=exited",
        "--filter",
        "status=created",
        "--format",
        "{{.ID}}\t{{.Image}}\t{{.Status}}",
    )
    stopped = [l for l in stopped_out.splitlines() if l.strip()]
    if stopped:
        console.print(f"  {len(stopped)} stopped container(s):")
        for line in stopped:
            parts = line.split("\t")
            cid, img = parts[0][:12], parts[1] if len(parts) > 1 else "?"
            console.print(f"    [dim]{cid}[/dim]  {img}")
        if not dry_run:
            _, rc = _docker("container", "prune", "-f")
            console.print(
                f"  {'[green]removed[/green]' if rc == 0 else '[red]failed[/red]'}"
            )
        else:
            console.print(f"  {tag}would run: docker container prune -f")
    else:
        console.print("  no stopped containers")

    # ── 3. Dangling images (untagged) ─────────────────────────────────────────
    print_rule("dangling images")
    dangling_out, _ = _docker(
        "images", "--filter", "dangling=true", "--format", "{{.ID}}\t{{.Size}}"
    )
    dangling = [l for l in dangling_out.splitlines() if l.strip()]
    if dangling:
        console.print(f"  {len(dangling)} dangling image(s)")
        if not dry_run:
            _, rc = _docker("image", "prune", "-f")
            console.print(
                f"  {'[green]removed[/green]' if rc == 0 else '[red]failed[/red]'}"
            )
        else:
            console.print(f"  {tag}would run: docker image prune -f")
    else:
        console.print("  no dangling images")

    # ── 4. Known solver images (only with --images) ───────────────────────────
    if not images:
        console.print(
            "\n[dim]Tip: run [bold]mosaic clean --images[/bold] to also remove "
            "idle solver images.[/dim]"
        )
        return

    print_rule("solver images")
    all_known = _all_known_solver_images()  # {tag: "problem/solver"}
    running = _running_container_images()

    # Filter to requested problems
    if problems != "all":
        prob_set = {p.strip() for p in problems.split(",")}
        all_known = {
            t: v
            for t, v in all_known.items()
            if any(v.startswith(p + "/") for p in prob_set)
        }

    # List all local images once
    local_out, _ = _docker("images", "--format", "{{.Repository}}:{{.Tag}}\t{{.Size}}")
    local_tags = {}
    for line in local_out.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            local_tags[parts[0]] = parts[1]

    to_remove: list[tuple[str, str, str]] = []  # (tag, size, problem/solver)
    protected: list[tuple[str, str]] = []  # (tag, reason)

    for img_tag, label in sorted(all_known.items()):
        if img_tag not in local_tags:
            continue  # not present locally
        if img_tag in running:
            protected.append((img_tag, "running container"))
            continue
        to_remove.append((img_tag, local_tags[img_tag], label))

    if protected:
        console.print("  [green]keeping (in use):[/green]")
        for tag_name, reason in protected:
            console.print(f"    [green]{tag_name}[/green]  [dim]({reason})[/dim]")

    if to_remove:
        console.print(
            f"  {'[dim](dry-run) would remove[/dim]' if dry_run else 'removing'}:"
        )
        for img_tag, size, label in to_remove:
            console.print(f"    [yellow]{img_tag}[/yellow]  {size}  ({label})")
        if not dry_run:
            for img_tag, _, label in to_remove:
                _, rc = _docker("rmi", img_tag)
                status = "[green]ok[/green]" if rc == 0 else "[red]failed[/red]"
                console.print(f"    {img_tag}  {status}")
    else:
        console.print("  no solver images to remove")


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
    watch: bool = typer.Option(
        False,
        "--watch",
        "-w",
        help="Live-refresh mode: redraw the status table plus GPU / docker / recent-activity "
        "panels every --interval seconds. Ctrl-C to exit.",
    ),
    interval: float = typer.Option(
        60.0,
        "--interval",
        help="Refresh interval in seconds for --watch mode (default: 60).",
    ),
):
    """Report per-solver completion status of every experiment on disk.

    Walks ``benchmarks/results/<problem>/<suite>/<experiment>/`` and, for each
    solver in the problem config, classifies the cell as:

      [green]ok[/]       solver produced valid data
      [dark_orange]anom[/]     valid but an outlier per the problem's status_checks
      [red]fail[/]      solver was attempted but its entry is empty/invalid/NaN
      [dim]—[/]         no result file, or solver absent from the result

    Exclusions carry a category (rendered as a distinct glyph):
      [dim yellow]perm[/]     categorical — method-intrinsic, out of % denominator
      [yellow]todo[/]      not_implemented — fixable code gap
      [cyan]slow[/]      infeasible — runs but too slow at scale
      [magenta]unst[/]      unstable — regime-specific numerical limit
      [red]bug[/]        upstream_bug — external or accepted defect
      [yellow]excl[/]      legacy string exclusion (unspecified category)

    Pass [bold]--failures[/] to list every failed or anomalous cell with its
    reason.  Use [bold]--format md[/] or [bold]--format json[/] to emit a
    PR-friendly snapshot; pair [bold]--format md[/] with
    [bold]--diff-against[/] to render a regression/improvement diff.
    """
    from rich.table import Table

    from mosaic.benchmarks.core.status import (
        ANOMALY,
        EXCL_CATEGORICAL,
        EXCL_INFEASIBLE,
        EXCL_NOT_IMPLEMENTED,
        EXCL_PERMANENT,
        EXCL_UNSPECIFIED,
        EXCL_UNSTABLE,
        EXCL_UPSTREAM_BUG,
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

    # ── --watch mode: live-refresh the status summary + resource panels ──
    if watch:
        _run_watch_loop(problem_list, suite_list, interval)
        return

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

    # Text labels per (status, category). Colour is NOT baked into the label
    # text — colours come from ``cell_color(cell)`` which reads the cell's
    # weight from ``SCORE_WEIGHTS``. That way a fresh ok, a stale ok, and a
    # fresh anomaly each get a distinct hue on the same green→red ladder that
    # drives the score and the progress bar.
    _EXCL_LABELS = {
        EXCL_CATEGORICAL: "perm",
        EXCL_NOT_IMPLEMENTED: "todo",
        EXCL_INFEASIBLE: "slow",
        EXCL_UNSTABLE: "unst",
        EXCL_UPSTREAM_BUG: "bug",
        EXCL_UNSPECIFIED: "excl",
    }

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
            label = _EXCL_LABELS.get(
                cell.category or EXCL_UNSPECIFIED, _EXCL_LABELS[EXCL_UNSPECIFIED]
            )
        else:
            return "?"
        # Stale flag: trailing `*`. Excluded cells never go stale (nothing
        # to re-run), so the asterisk only applies to OK / ANOMALY / FAILED.
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
                    if (cell.category or EXCL_UNSPECIFIED) in EXCL_PERMANENT:
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
            "[dim yellow]perm[/] categorical (out of %) · "
            "[yellow]todo[/] not_implemented · "
            "[cyan]slow[/] infeasible · "
            "[magenta]unst[/] unstable · "
            "[red]bug[/] upstream_bug · "
            "[yellow]excl[/] unspecified · "
            "[bold]*[/] stale (predates current tesseract/harness source)"
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
            """Stacked colour bar: each segment width ∝ cell count, coloured
            by its score weight.  Categorical exclusions are not shown."""
            segs: list[tuple[int, float]] = [
                (n_ok, 1.00),
                (n_stale_ok, 0.67),
                (n_anom, 0.53),
                (n_missing + n_excl_work, 0.33),
                (n_fail, 0.00),
            ]
            return _stacked_bar(segs, width)

        from mosaic.benchmarks.core.status import (
            compute_score,
            format_score,
            weight_color,
        )

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


if __name__ == "__main__":
    app()
