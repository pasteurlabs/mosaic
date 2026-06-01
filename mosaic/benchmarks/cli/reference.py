# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`mosaic reference` — generate precomputed reference solutions.

Generates trimmed-mean reference NPZ files for consensus-based
experiments (structural-mesh, thermal-mesh). These references decouple
single-solver CI runs from needing all peers present.

Two modes:

1. **From existing results** (``--from-results``): reads field snapshots
   from a ``mosaic-results/`` directory and extracts the consensus
   arrays. Fast, no solver runs needed — just needs prior results.

2. **By running solvers** (default): builds + runs all solvers for the
   target experiments, computes the trimmed mean, and writes the NPZ.
"""

from __future__ import annotations

from pathlib import Path

import typer

from mosaic.benchmarks.cli import app
from mosaic.benchmarks.core.console import console
from mosaic.benchmarks.core.reference import (
    PRECOMPUTED_EXPERIMENTS,
    extract_references_from_fields,
    save_reference,
)


@app.command()
def reference(
    problems: str = typer.Option(
        "all",
        "--problems",
        "-p",
        help="Comma-separated domain(s) or 'all'. "
        "Only domains with consensus experiments are processed.",
    ),
    experiments: str = typer.Option(
        "all",
        "--experiments",
        "-e",
        help="Comma-separated experiment keys (e.g. 'forward/baseline') or 'all'.",
    ),
    from_results: str | None = typer.Option(
        None,
        "--from-results",
        help="Path to a mosaic-results/ directory. Extract references from "
        "existing fields.npz snapshots instead of running solvers.",
    ),
) -> None:
    r"""Generate precomputed reference solutions for consensus experiments.

    References are trimmed-mean fields computed across all solvers. Once
    checked in, single-solver CI runs can compute errors without needing
    the full solver ensemble.

    \b
    Examples:
        # Extract from existing benchmark results
        mosaic reference --from-results mosaic-results/

        # Extract for one domain only
        mosaic reference -p structural-mesh --from-results mosaic-results/

        # Run solvers to generate fresh references (requires built images)
        mosaic reference -p thermal-mesh
    """
    # Determine which domains to process.
    if problems == "all":
        domains = list(PRECOMPUTED_EXPERIMENTS.keys())
    else:
        domains = [p.strip() for p in problems.split(",")]
        # Filter to domains that actually have precomputed experiments.
        unknown = [d for d in domains if d not in PRECOMPUTED_EXPERIMENTS]
        if unknown:
            console.print(
                f"[yellow]WARN[/] No precomputed-reference experiments for: "
                f"{', '.join(unknown)}."
            )
        domains = [d for d in domains if d in PRECOMPUTED_EXPERIMENTS]

    if not domains:
        console.print("[yellow]No domains with precomputed references to process.[/]")
        raise typer.Exit()

    # Determine which experiments to process.
    exp_filter = None if experiments == "all" else set(experiments.split(","))

    if from_results is not None:
        _generate_from_results(Path(from_results), domains, exp_filter)
    else:
        _generate_by_running(domains, exp_filter)


def _generate_from_results(
    results_dir: Path,
    domains: list[str],
    exp_filter: set[str] | None,
) -> None:
    """Extract references from existing mosaic-results/ field snapshots."""
    if not results_dir.is_dir():
        console.print(f"[red]Results directory not found:[/] {results_dir}")
        raise typer.Exit(code=1)

    total = 0
    for domain in domains:
        exps = PRECOMPUTED_EXPERIMENTS[domain]
        if exp_filter is not None:
            exps = [e for e in exps if e in exp_filter]

        for exp_key in exps:
            # The fields.npz lives at:
            #   results_dir/<domain>/<suite>/<exp_name>/fields.npz
            suite, _, exp_name = exp_key.partition("/")
            fields_path = results_dir / domain / suite / exp_name / "fields.npz"

            if not fields_path.exists():
                console.print(
                    f"  [yellow]SKIP[/] {domain}/{exp_key}: "
                    f"no fields.npz at {fields_path}"
                )
                continue

            # Determine the number of sweep values from the result.json.
            result_path = fields_path.parent / "result.json"
            n_sweep = _count_sweep_values(result_path, fields_path)
            if n_sweep == 0:
                console.print(
                    f"  [yellow]SKIP[/] {domain}/{exp_key}: "
                    "could not determine sweep values"
                )
                continue

            refs = extract_references_from_fields(fields_path, n_sweep)
            if not refs:
                console.print(
                    f"  [yellow]SKIP[/] {domain}/{exp_key}: "
                    "no consensus arrays found in fields.npz"
                )
                continue

            # Read sweep values from the fields.npz for provenance.
            sweep_values = _read_sweep_values(fields_path)
            path = save_reference(domain, exp_key, refs, sweep_values)
            console.print(
                f"  [green]OK[/] {domain}/{exp_key}: "
                f"{len(refs)} reference(s) → "
                f"{path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}"
            )
            total += 1

    console.print(f"\n[bold]Generated {total} reference file(s).[/]")


def _count_sweep_values(result_path: Path, fields_path: Path) -> int:
    """Determine the number of sweep values for an experiment.

    Tries the result.json first (schema_version=1 has sweep.values),
    then falls back to counting consensus_* keys in the fields.npz.
    """
    import json

    import numpy as np

    if result_path.exists():
        try:
            with open(result_path) as f:
                result = json.load(f)
            sweep = result.get("sweep")
            if sweep and "values" in sweep:
                return len(sweep["values"])
        except Exception:
            pass

    # Fallback: count consensus_* keys in the NPZ.
    try:
        with np.load(str(fields_path), allow_pickle=False) as data:
            return sum(1 for k in data if k.startswith("consensus_"))
    except Exception:
        return 0


def _read_sweep_values(fields_path: Path) -> list | None:
    """Read the sweep_values array from a fields.npz, if present."""
    import numpy as np

    try:
        with np.load(str(fields_path), allow_pickle=False) as data:
            if "sweep_values" in data:
                return data["sweep_values"].tolist()
    except Exception:
        pass
    return None


def _generate_by_running(
    domains: list[str],
    exp_filter: set[str] | None,
) -> None:
    """Generate references by running all solvers and computing trimmed means."""
    import numpy as np

    from mosaic.benchmarks.core.utils import trimmed_mean
    from mosaic.benchmarks.problems import get_config

    console.print("[bold]Generating references by running solvers...[/]\n")

    for domain in domains:
        cfg = get_config(domain)
        exps = PRECOMPUTED_EXPERIMENTS[domain]
        if exp_filter is not None:
            exps = [e for e in exps if e in exp_filter]

        for exp_key in exps:
            console.print(f"  [bold]{domain}/{exp_key}[/]")

            # Look up the experiment registration.
            full_key = exp_key
            if full_key not in cfg.experiments:
                console.print(
                    f"    [yellow]SKIP[/] experiment {full_key!r} not registered"
                )
                continue

            exp = cfg.experiments[full_key]
            run_params = exp.params

            # Extract sweep info.
            sweep_cfg = run_params.get("sweep", {})
            sweep_key = sweep_cfg.get("key")
            sweep_values = sweep_cfg.get("values", [])

            if not sweep_key or not sweep_values:
                console.print("    [yellow]SKIP[/] no sweep configured")
                continue

            # Get IC.
            ic_cfg = run_params.get("ic", {})
            ic_name = ic_cfg.get("name", next(iter(cfg.make_ic)))
            seed = ic_cfg.get("seed", 0)
            phys = run_params.get("physics", {})

            # Build + run each solver at each sweep value.
            console.print("    Building solvers...")
            from mosaic.benchmarks.core.runner import build_all

            tags = build_all(cfg)

            solver_outputs: dict[str, dict] = {}  # solver -> {sweep_idx: array}
            for s in cfg.solvers:
                solver_outputs[s.name] = {}

            for i, val in enumerate(sweep_values):
                curr_phys = {
                    **phys,
                    sweep_key: val,
                    "domain_extent": cfg.domain_extent,
                }
                ic = cfg.make_ic[ic_name](L=cfg.domain_extent, seed=seed, **curr_phys)

                for s in cfg.solvers:
                    try:
                        from mosaic.benchmarks.core.runner import safe_apply

                        inputs_s = cfg.make_inputs(s.name, ic, **curr_phys)
                        t = tags[s.name]
                        result = safe_apply(t, inputs_s, cfg.output_key)
                        if result is not None:
                            norm = s.normalize_output
                            out = norm(result) if norm is not None else result
                            solver_outputs[s.name][i] = np.asarray(out)
                            console.print(
                                f"    [green]✓[/] {s.name} @ {sweep_key}={val}"
                            )
                        else:
                            console.print(
                                f"    [yellow]✗[/] {s.name} @ {sweep_key}={val}: apply failed"
                            )
                    except Exception as e:
                        console.print(
                            f"    [red]✗[/] {s.name} @ {sweep_key}={val}: {e}"
                        )

            # Compute trimmed mean per sweep value.
            refs: dict[int, np.ndarray] = {}
            for i, val in enumerate(sweep_values):
                arrays = [
                    solver_outputs[s.name][i]
                    for s in cfg.solvers
                    if i in solver_outputs[s.name]
                ]
                if len(arrays) < 2:
                    console.print(
                        f"    [yellow]WARN[/] sweep {sweep_key}={val}: "
                        f"only {len(arrays)} solver(s), need >= 2 for consensus"
                    )
                    if arrays:
                        refs[i] = np.asarray(arrays[0])
                    continue
                refs[i] = np.asarray(trimmed_mean(arrays))

            if refs:
                path = save_reference(domain, exp_key, refs, sweep_values)
                console.print(f"    [green]OK[/] {len(refs)} reference(s) → {path}")
            else:
                console.print("    [red]FAIL[/] no references generated")

    console.print("\n[bold]Done.[/]")
